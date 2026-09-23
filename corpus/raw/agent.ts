/**
 * core/agent.ts — Turn orchestration: Talk (glance/reflect) and Journal features.
 *
 * Talk flow (two phases):
 *   startTurn  — undo-check → NER → questions → inference → Gemma (if no questions)
 *   completeTurn — write answers → inference → Gemma → store
 *   If no questions, startTurn completes inline and returns { done: true }.
 *
 * Journal flow (single phase):
 *   runJournalTurn — NER → inference → store, no Gemma response
 *
 * Model: Gemma 4-E2B-IT Q4_K_M (singleton, managed here)
 * NLU / undo detection: Llama 3.2-1B in nlu.ts (shared context)
 */

import { initLlama, loadLlamaModelInfo, type LlamaContext } from 'llama.rn';
import * as FileSystem from 'expo-file-system/legacy';
import type { DB }                      from '@op-engineering/op-sqlite';

import { runNer, detectUndoIntent, classifyIntent, resolveNodeValue, maybeResolveComposite, getCompositeSourceCol, releaseNlu, type IntentResult, type NluEntity } from './nlu';
import {
  dispatchTool,
  setCurrentUserMessage,
  setRecentTopics,
  setDirectEvidenceNodes,
  overrideTurnStart,
  resetSession,
  MCP_TOOLS,
}                                               from './mcp';
import { buildFollowUps, buildCascade }         from './questionCascade';
import { storeMemory }                          from './embed';
import type { BeliefResult }                    from './inferenceEngine';

// ── Types ─────────────────────────────────────────────────────────────────────

export type AgentMode = 'glance' | 'reflect';

/**
 * Structured Chain-of-Thought fields parsed from Gemma output in the ReAct loop.
 * Present when Gemma emits CAUSE:/EFFECT:/LINK: labels in the same generation as RESPONSE:.
 * Null when any label is missing (additive — does not change existing behaviour).
 */
export interface StructuredSynthesis {
  ack:      string;
  cause:    string;
  effect:   string;
  link:     string;
  solution: string;
  question: string;
}

/** Unified question shape returned to UI (covers both follow-ups and cascade). */
export interface DisplayQuestion {
  original_col: string;
  node_name:    string;
  question:     string;
  kind:         'followup' | 'cascade';
  opts?:        { v: number; l: string }[];
  range?:       { min: number; max: number; unit: string };
}

/** One answered question from the UI. */
export interface QuestionAnswer {
  original_col: string;
  node_name:    string;
  raw_value:    number | null;
  raw_text:     string;
}

/** startTurn returned when questions exist — UI must call completeTurn next. */
export interface TurnStart {
  done:      false;
  turnId:    string;
  questions: DisplayQuestion[];
}

export type TurnResult = TurnStart | { done: true; response: string; synthesis: StructuredSynthesis | null };

// ── Gemma singleton ───────────────────────────────────────────────────────────

let _ctx:       LlamaContext | null = null;
let _modelPath: string | null       = null;

export function setAgentPath(modelPath: string): void {
  _modelPath = modelPath;
}

export async function initAgent(modelPath: string): Promise<void> {
  if (_ctx && _modelPath === modelPath) return;
  if (_ctx) { await _ctx.release(); _ctx = null; }
  _modelPath = modelPath;
  // Pre-validate: loadLlamaModelInfo reads GGUF metadata without allocating KV cache.
  // If this throws, the file is corrupted or unreadable. If it passes but initLlama
  // fails below, the issue is context/KV-cache initialisation, not the model file.
  try {
    await loadLlamaModelInfo(modelPath);
  } catch (metaErr) {
    const info = await FileSystem.getInfoAsync(modelPath, { size: true } as any).catch(() => null);
    const sizeMB = info && (info as any).size ? `${((info as any).size / 1024 / 1024).toFixed(1)} MB` : 'unknown size';
    throw new Error(`agent file invalid [${sizeMB} on disk] — re-download the model [path=${modelPath}]: ${metaErr instanceof Error ? metaErr.message : String(metaErr)}`);
  }
  try {
    _ctx = await initLlama({ model: modelPath, n_ctx: 6144, n_threads: 4, use_mlock: false, n_parallel: 1 });
  } catch (e) {
    throw new Error(`agent ctx init failed (file OK, context allocation failed) [path=${modelPath}]: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export async function releaseAgent(): Promise<void> {
  if (!_ctx) return;
  await _ctx.release();
  _ctx = null; _modelPath = null;
}

export function isAgentReady(): boolean { return _ctx !== null; }
export function isAgentPathSet(): boolean { return _modelPath !== null; }

/** Returns true for errors that indicate the native LlamaContext was evicted by the OS. */
function isNativeContextError(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return /context.*null|llamarpc|native.*invalid|llama.*not.*init/i.test(msg);
}

/** Reinitialise from stored path if context was evicted by OS. */
export async function ensureAgent(): Promise<void> {
  if (_ctx) return;
  if (!_modelPath) throw new Error('agent: initAgent() has not been called — no model path stored');
  await initAgent(_modelPath);
}

// ── Memory buffer ─────────────────────────────────────────────────────────────

/** Number of recent user messages kept as raw pairs in Gemma context. Tune here. */
const MEMORY_BUFFER_SIZE = 4;

interface RecentPair { topic: string; content: string; }

function getRecentPairs(db: DB, sessionId: string): RecentPair[] {
  const rows = db.executeSync(
    `SELECT topic, content FROM chat_messages
     WHERE session_id = ? AND role = 'user' AND is_active = 1 AND evicted = 0
     ORDER BY created_at DESC LIMIT ?`,
    [sessionId, MEMORY_BUFFER_SIZE],
  ).rows as { topic: string | null; content: string }[];
  return rows.reverse().map(r => ({ topic: r.topic ?? '', content: r.content }));
}

/**
 * After each storeChatMessage call: if non-evicted user message count exceeds
 * MEMORY_BUFFER_SIZE, compress and evict the oldest one into memory_summaries.
 * Silently skips if embed model is not initialised.
 */
async function maybeEvictOldest(db: DB, sessionId: string): Promise<void> {
  const countRow = db.executeSync(
    `SELECT COUNT(*) AS cnt FROM chat_messages
     WHERE session_id = ? AND role = 'user' AND is_active = 1 AND evicted = 0`,
    [sessionId],
  ).rows[0] as { cnt: number } | undefined;
  if ((countRow?.cnt ?? 0) <= MEMORY_BUFFER_SIZE) return;

  const oldest = db.executeSync(
    `SELECT id, topic, content FROM chat_messages
     WHERE session_id = ? AND role = 'user' AND is_active = 1 AND evicted = 0
     ORDER BY created_at ASC LIMIT 1`,
    [sessionId],
  ).rows[0] as { id: number; topic: string | null; content: string } | undefined;
  if (!oldest) return;

  try {
    const summaryText = oldest.topic ? `${oldest.topic}: ${oldest.content}` : oldest.content;
    await storeMemory(db, sessionId, summaryText, 1);
  } catch {
    // embed model not initialised — skip silently, message stays non-evicted
    return;
  }

  db.executeSync(`UPDATE chat_messages SET evicted = 1 WHERE id = ?`, [oldest.id]);
}

// ── Session tracking (for resetSession on new session) ───────────────────────

let _lastSessionId: string = '';

// ── Pending turn storage (between startTurn and completeTurn) ─────────────────

interface PendingTurn {
  sessionId:          string;
  userMessage:        string;
  isUndoTurn:         boolean;
  memorySummaries:    string[];
  recentPairs:        RecentPair[];
  topic:              string;
  isEarlyInteraction: boolean;
  intentResult:       IntentResult;
  trendSummary:       string;
  nerEntityNodes:     Set<string>;
  createdAt:          number;
}
const _pendingTurns = new Map<string, PendingTurn>();
const PENDING_TURN_TTL_MS = 30 * 60 * 1000; // 30 minutes

function evictStalePendingTurns(): void {
  const cutoff = Date.now() - PENDING_TURN_TTL_MS;
  for (const [id, turn] of _pendingTurns) {
    if (turn.createdAt < cutoff) _pendingTurns.delete(id);
  }
}

// ── Ack-only detection (Stage 0) ──────────────────────────────────────────────

const ACK_ONLY_WORDS = new Set([
  'ok', 'okay', 'k', 'got it', 'thanks', 'thank you',
  'cool', 'alright', 'sure', 'noted', '👍', '👌',
]);

function isAckOnly(text: string): boolean {
  return ACK_ONLY_WORDS.has(text.trim().toLowerCase());
}

function cleanGemmaOutput(text: string): string {
  // Strip thought blocks (emitted even when thinking is disabled on E2B/E4B)
  return text
    .replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '')
    .replace(/<\|turn>|<turn\|>|<\|think\|>|<\|channel>|<channel\|>/g, '')
    .trim();
}

const ACK_RESPONSES = [
  "Of course! Feel free to share anything whenever you're ready.",
  "I'm here whenever you want to chat.",
  "Anytime! Just let me know how you're doing.",
  "No problem at all — I'm here.",
];
function ackResponse(): string {
  return ACK_RESPONSES[Math.floor(Math.random() * ACK_RESPONSES.length)];
}

const SOCIAL_RESPONSES: string[] = [
  "Hey! Really glad you're here — how have you been feeling lately?",
  "Hi there! Always good to see you. What's been going on with you today?",
  "Hey, good to hear from you! How are you doing — honestly?",
  "Morning! Hope you're settling in okay. What's on your mind?",
  "Hey you. No rush at all — just checking in. How are you feeling these days?",
  "Hi! I was just thinking about you. How's everything going on your end?",
];

const THIRD_PARTY_TEMPLATES: string[] = [
  "It sounds like {topic} has been on your mind lately. I'm really only set up to support you personally, but I'd love to hear — how has all of this been sitting with you?",
  "I can hear that {topic} is something you care about. I'm not the right place for advice about others, but your concern says a lot — how are you holding up through it?",
  "Worrying about {topic} can be really draining. I'm best at focusing on how you're feeling day to day — is any of this weighing on you too?",
  "It means a lot that you're thinking about {topic}. I'm not able to offer guidance there, but I'm genuinely curious — what's been going on with you lately?",
  "Sounds like {topic} has been taking up some space in your head. I'll leave the advice on that to the right people, but I'd really like to hear how you've been doing yourself.",
  "I can tell {topic} matters to you. I'm focused on your wellbeing rather than others', but sometimes these worries reflect something we're carrying too — how have you been feeling?",
];

function thirdPartyResponse(userMessage: string): string {
  const possessivePattern =
    /\b(?:my\s+)?(\w+(?:\s+\w+)?)'s\s+([\w\s]{1,40}?)(?:\?|,|\.|$)/i;
  const relationTopicPattern =
    /\bmy\s+((?:dad|mom|mother|father|sister|brother|friend|partner|wife|husband|son|daughter|uncle|aunt|grandma|grandpa|colleague|coworker)(?:\s+\w+)?)\s+(?:has|have|is|are|with|about)?\s*([\w\s]{1,40}?)(?:\?|,|\.|$)/i;
  const aboutPattern =
    /(?:about|understand|help with|explain)\s+([\w\s''-]{2,50})(?:\?|,|\.|$)/i;

  let topic: string | null = null;

  const possessiveMatch = userMessage.match(possessivePattern);
  if (possessiveMatch) {
    const relation = possessiveMatch[1].trim();
    const condition = possessiveMatch[2].trim();
    topic = condition.length > 1 ? `your ${relation}'s ${condition}` : `your ${relation}'s situation`;
  }

  if (!topic) {
    const relationMatch = userMessage.match(relationTopicPattern);
    if (relationMatch) {
      const relation = relationMatch[1].trim();
      const condition = relationMatch[2].trim();
      topic = condition.length > 1 ? `your ${relation}'s ${condition}` : `your ${relation}`;
    }
  }

  if (!topic) {
    const aboutMatch = userMessage.match(aboutPattern);
    if (aboutMatch) {
      const raw = aboutMatch[1].trim();
      const stopWords = /^(me|you|us|them|it|this|that|the|a|an)$/i;
      if (!stopWords.test(raw) && raw.length > 2) topic = raw;
    }
  }

  const resolvedTopic = topic ?? "what you mentioned";
  const idx = Math.floor(Math.random() * THIRD_PARTY_TEMPLATES.length);
  return THIRD_PARTY_TEMPLATES[idx].replace(/\{topic\}/g, resolvedTopic);
}

function socialResponse(): string {
  return SOCIAL_RESPONSES[Math.floor(Math.random() * SOCIAL_RESPONSES.length)];
}

// ── Turn ID ───────────────────────────────────────────────────────────────────

function newTurnId(): string {
  const r = () => Math.random().toString(36).slice(2, 8);
  return `${Date.now().toString(36)}-${r()}`;
}

// ── DB helpers ────────────────────────────────────────────────────────────────

function getPrevTurnId(db: DB, sessionId: string): string | null {
  const row = db.executeSync(
    `SELECT DISTINCT turn_id FROM chat_messages
     WHERE session_id = ? AND is_active = 1
     ORDER BY created_at DESC LIMIT 1`,
    [sessionId],
  ).rows[0] as { turn_id: string } | undefined;
  return row?.turn_id ?? null;
}

function getReportedTurnCount(db: DB): number {
  const row = db.executeSync(
    `SELECT COUNT(DISTINCT turn_id) AS cnt FROM user_data_sensorless
     WHERE is_active = 1`,
  ).rows[0] as { cnt: number } | undefined;
  return row?.cnt ?? 0;
}

function getActiveTurnNodes(db: DB, turnId: string): string[] {
  return (db.executeSync(
    `SELECT DISTINCT node_name FROM user_data_sensorless
     WHERE turn_id = ? AND is_active = 1`,
    [turnId],
  ).rows as { node_name: string }[]).map(r => r.node_name);
}

function storeChatMessage(
  db: DB, sessionId: string, turnId: string,
  role: 'user' | 'model', content: string,
  topic?: string,
): void {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  db.executeSync(
    `INSERT INTO chat_messages (timestamp, session_id, turn_id, role, content, topic, is_active, created_at)
     VALUES (?, ?, ?, ?, ?, ?, 1, ?)`,
    [ts, sessionId, turnId, role, content, topic ?? null, ts],
  );
}

function writeProactiveAnswer(db: DB, turnId: string, ans: QuestionAnswer): void {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);

  // Build a minimal NluEntity so resolveNodeValue can discretize the answer
  const syntheticEnt: NluEntity = {
    original_column: ans.original_col || undefined,
    node_name:       ans.node_name,
    node_value:      null,
    raw_value:       ans.raw_value ?? undefined,
    raw_text:        ans.raw_text,
    summary_text:    ans.raw_text,
    confidence:      1,
  };
  const resolvedNodeValue = resolveNodeValue(syntheticEnt);

  db.executeSync(
    `INSERT INTO user_data_sensorless
       (timestamp, node_name, original_column, raw_text, raw_value, node_value,
        data_source, merge_mode, temporal_flag, report_date, turn_id,
        was_proactive, answered)
     VALUES (?, ?, ?, ?, ?, ?, 'proactive', 'scale', 'decaying', date('now'), ?, 1, 1)`,
    [ts, ans.node_name, ans.original_col, ans.raw_text, ans.raw_value ?? null, resolvedNodeValue ?? null, turnId],
  );
}

// ── Acknowledgements ──────────────────────────────────────────────────────────

const JOURNAL_ACKS = [
  "I hear you. Your thoughts are safe here.",
  "Thank you for sharing. Writing it out matters.",
  "Noted. This space is yours.",
  "Got it. It's good to put things into words.",
];
function journalAck(): string {
  return JOURNAL_ACKS[Math.floor(Math.random() * JOURNAL_ACKS.length)];
}

const UNDO_ACKS = [
  "Got it — I've updated that for you. Feel free to share more whenever you're ready.",
  "No worries at all! I've corrected that. Take your time and let me know how you're really doing.",
  "Done, I've removed the previous entry. Share whatever feels right whenever you're ready.",
];
function undoAck(): string {
  return UNDO_ACKS[Math.floor(Math.random() * UNDO_ACKS.length)];
}

// ── Prompt helpers ────────────────────────────────────────────────────────────

const PERSONA = `You are a warm, empathetic health companion — like a knowledgeable friend who genuinely cares and listens. You help the user understand their wellbeing through honest, supportive conversation. This app works by learning from what users share naturally — how they slept, felt, ate, moved, or what's been on their mind — and building a picture of their health state over time. When someone asks what the app does, explain this simply and warmly in your own words.

Rules you never break:
• Never diagnose, label conditions, or suggest medications or treatments
• Never use clinical or alarming language
• Never mention probability numbers, node names, or internal system terms
• Always lead with empathy — the user is sharing something personal
• Be warm, natural, occasionally light — a friend, not a clinician. Lightness is only appropriate when the user's tone is clearly casual or positive — never when they are distressed, venting, or reporting serious symptoms. Serious symptoms include: acute anxiety or panic, persistent low mood, fatigue from overwork or stress, sleep loss, severe pain, or any mention of feeling overwhelmed. When these appear, drop all lightness — lead with deep empathy, never minimise.
• If the user asks something outside your scope (medical prognosis, general knowledge, news, or anything unrelated to their personal health and wellbeing), acknowledge it kindly in one sentence, gently note you're focused on how they're feeling day to day, and invite them to share more about that. Never lecture or repeat the explanation.
• Never claim the user is "improving," "making progress," "trending," or "doing better/worse over time." You see current state only. If asked about trends, acknowledge honestly that you track current state and are building history over time.`;

const RESPONSE_GUIDE = `[For free-form response mode only — when SYNTHESIS RULES are present, follow those instead and ignore this section]
Structure your response naturally (only include what's relevant):
1. Acknowledge what the user shared — warmly and briefly
2. Causes: what may have contributed ("this might be connected to...", "it sounds like...")
3. Effects: what might be affected ("you may notice...", "this can sometimes...")
4. Soft suggestion: a gentle general wellness nudge — no diagnoses, no clinical advice
   (skip steps 2-4 if the user only asked a question and reported nothing new this turn)
   Compound reports (multiple emotions or symptoms reported together with a clear cause like overwork, poor sleep, or stress): synthesise into one connected picture — identify the root cause first, then trace its effects. Do not address each issue in isolation. Example: overwork + no rest → fatigue → frustration + anxiety all connect; frame them as one thread, not three separate topics.
4a. If TREND HISTORY is present in context and the user asked about trends, translate it into plain language observations — no numbers, no clinical framing, no node names.
5. Retrospective question — ONLY if something specific was surfaced this turn from causes, effects, memory, or patterns. Derived tightly from what was actually analyzed — never a generic template. Goal: send the user inward to reflect on their own experience. Their answer goes to chat naturally next turn.
   Examples of good retrospective questions (match the specificity of these, never copy them):
   • Sleep drop + fatigue surfaced → "When you're lying awake, is your mind busy — or does it feel more like your body just won't settle?"
   • Stress + loneliness co-occurring → "When things pile up, do you find yourself pulling away from people — or does it just kind of happen on its own?"
   • Exercise drop correlates with mood dips → "On those days you didn't move much — did it feel like a choice to rest, or more like something drained before you could start?"
   • Mood improved after social activity → "Does being around people usually lift you, or does it depend a lot on who's there?"
   • Recurring low mood pattern in memory → "Do you have a sense of what usually comes just before those dips — or does it catch you off guard?"
   • Pain + emotional flatness → "Does the discomfort make everything else feel heavier — or are they kind of separate for you?"
   • Appetite change + low mood → "When you're not eating much — is it that nothing sounds good, or more that you just forget?"
   • Anxiety + overwork surfaced → "When work piles up and you're running on empty, does the anxiety feel like your mind racing ahead — or more like a weight you're already carrying?"
   • Frustration + fatigue together → "On days like this, does the frustration feel like it came from the tiredness — or did it start somewhere else and the tiredness just made it worse?"
   Skip entirely if: nothing specific was surfaced, context is shallow, user is venting and needs space not reflection, CORRECTION TURN (user fixing data), or SOCIAL TURN (casual chat).`;

function selectBeliefWindow(
  beliefs:            BeliefResult | null,
  intentResult:       IntentResult,
  isEarlyInteraction: boolean,
  nerEntityNodes:     Set<string> = new Set(),
): string {
  if (!beliefs) return '(no data yet)';
  if (intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport) return '';

  type BeliefEntry = { node: string; state: string; prob: number };
  const allEntries: BeliefEntry[] = Object.entries(beliefs).map(([node, dist]) => {
    const [state, prob] = Object.entries(dist).sort((a, b) => b[1] - a[1])[0] ?? ['?', 0];
    return { node, state, prob: prob as number };
  });

  const forcedSet  = new Set(intentResult.queryNodes);
  const forced     = allEntries.filter(e => forcedSet.has(e.node));
  const notForced  = allEntries.filter(e => !forcedSet.has(e.node));

  let budget: number;
  let gate:   number;

  if (isEarlyInteraction) {
    budget = Infinity;
    gate   = 0.0;
  } else if (intentResult.hasQuery) {
    budget = 15;
    gate   = 0.35;
  } else {
    budget = 8;
    gate   = 0.55;
  }

  const remainder = notForced
    .filter(e => e.prob >= gate)
    .sort((a, b) => b.prob - a.prob)
    .slice(0, Math.max(0, budget - forced.length));

  const selected = [...forced, ...remainder].sort((a, b) => b.prob - a.prob);

  if (selected.length === 0) return '(no confident beliefs yet)';

  const lines = selected.map(e => {
    const probPct = Math.round(e.prob * 100);
    let line: string;
    if (forcedSet.has(e.node)) {
      line = `• ${e.node}: ${e.state} [${probPct}% — you asked about this]`;
    } else {
      line = `• ${e.node}: ${e.state}`;
    }
    if (isEarlyInteraction && !nerEntityNodes.has(e.node)) line += ' [prior — no user data yet]';
    return line;
  });

  return lines.join('\n');
}

type ChangedNode = { node: string; label: string; previous_state: string; new_state: string; delta: number };

function formatChangedNodes(nodes: unknown[]): string {
  if (!nodes.length) return '(no significant changes this turn)';
  return (nodes as ChangedNode[])
    .map(n => `• ${n.node} [${n.label}]: ${n.previous_state} → ${n.new_state}`)
    .join('\n');
}

function buildContextBlock(
  recentPairs:        RecentPair[],
  memorySummaries:    string[],
  beliefs:            BeliefResult | null,
  changedNodes:       unknown[],
  userMessage:        string,
  isUndoTurn:         boolean      = false,
  isEarlyInteraction: boolean      = false,
  intentResult:       IntentResult = { hasReport: true, hasQuery: false, hasTrendQuery: false, isSocial: false, isThirdPartyQuery: false, queryNodes: [] },
  trendSummary:       string       = '',
  nerEntityNodes:     Set<string>  = new Set(),
  hypothesis:         string       = '',
): string {
  const parts: string[] = [];

  // Stack all applicable injections — order: correction → first-interaction → query → mixed/social/third-party → unresolved
  if (isUndoTurn) {
    parts.push(`CORRECTION TURN: User correcting prior entry. Acknowledge warmly first. Don't re-analyze corrected data as new.`);
  }

  if (isEarlyInteraction) {
    const priorNote = nerEntityNodes.size > 0
      ? `Only this turn's reported data is real. All other state is estimated prior.`
      : `No user data yet. All beliefs are priors only.`;
    parts.push(`FIRST INTERACTION: ${priorNote} Welcome warmly, acknowledge what they shared, invite more sharing. Make no pattern claims.`);
  }

  if (intentResult.hasReport && nerEntityNodes.size === 0 && !isUndoTurn) {
    parts.push(`UNSTRUCTURED REPORT: User described health concerns (see USER MESSAGE) but no structured data was extracted this turn. In reflect mode, use store_indirect_evidence to capture the single most relevant node from the user's message, then synthesize CAUSE/EFFECT/LINK from the full picture — current beliefs, memory, and trends all apply. Do not just echo the user's words as the cause; draw from the health state already in context.`);
  }

  if (intentResult.hasQuery) {
    parts.push(`QUERY TURN: User asking about health state. Synthesize belief window warmly. No node names or probabilities to user.`);
  }

  if (intentResult.isSocial && intentResult.hasQuery) {
    parts.push(`MIXED SOCIAL+QUERY: Acknowledge social part briefly first, then answer health question.`);
  } else if (intentResult.isSocial && !intentResult.hasQuery) {
    parts.push(`SOCIAL TURN: Casual chat. Respond warmly and briefly. Don't analyze health data.`);
  }

  if (intentResult.isThirdPartyQuery) {
    parts.push(`THIRD-PARTY QUERY: About someone else. Acknowledge concern (1-2 sentences). No clinical advice. Invite user to share how THEY feel.`);
  }

  // Unresolved queryNodes — nodes asked about but not present in beliefs
  if (intentResult.queryNodes.length > 0 && beliefs) {
    const unresolved = intentResult.queryNodes.filter(n => !(n in beliefs));
    if (unresolved.length > 0) {
      parts.push(`NOTE: User asked about [${unresolved.join(', ')}]. Not currently tracked. Acknowledge kindly, redirect to tracked areas.`);
    }
  }

  if (recentPairs.length) {
    const formatted = recentPairs
      .map(p => p.topic ? `[${p.topic}] ${p.content}` : p.content)
      .join('\n');
    parts.push(`RECENT CONTEXT (last messages — raw, not summarised):\n${formatted}`);
  }

  if (memorySummaries.length) {
    parts.push(`MEMORY (older context — compressed):\n${memorySummaries.map((m, i) => `${i + 1}. ${m}`).join('\n')}`);
  }

  const beliefBlock = selectBeliefWindow(beliefs, intentResult, isEarlyInteraction, nerEntityNodes);
  if (beliefBlock) {
    parts.push(`CURRENT HEALTH STATE (internal — do not recite to user):\n${beliefBlock}`);
  }

  if (trendSummary) {
    parts.push(`TREND HISTORY (internal — use only if user asked about trends):\n${trendSummary}`);
  }

  const isSocialFastPath = intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport;
  if (!isSocialFastPath) {
    parts.push(
      `WHAT CHANGED THIS TURN (internal — use to identify causes and effects):\n${formatChangedNodes(changedNodes)}`,
    );
  }

  if (hypothesis) {
    parts.push(`CAUSAL HYPOTHESIS (pre-analyzed — use as the basis for CAUSE, EFFECT, and LINK):\n${hypothesis}`);
  }

  parts.push(`USER MESSAGE:\n${userMessage}`);
  return parts.join('\n\n');
}

function gemmaPrompt(systemContent: string, userContent: string): string {
  return (
    `<|turn>system\n${systemContent}<turn|>\n` +
    `<|turn>user\n${userContent}<turn|>\n` +
    `<|turn>model\n`
  );
}

// ── Glance: single Gemma call ─────────────────────────────────────────────────

async function runGlanceCall(
  recentPairs:        RecentPair[],
  memorySummaries:    string[],
  beliefs:            BeliefResult | null,
  changedNodes:       unknown[],
  userMessage:        string,
  isUndoTurn:         boolean      = false,
  isEarlyInteraction: boolean      = false,
  intentResult:       IntentResult = { hasReport: true, hasQuery: false, hasTrendQuery: false, isSocial: false, isThirdPartyQuery: false, queryNodes: [] },
  trendSummary:       string       = '',
  nerEntityNodes:     Set<string>  = new Set(),
): Promise<string> {
  await ensureAgent();

  const prompt = gemmaPrompt(
    `${PERSONA}\n\n${RESPONSE_GUIDE}`,
    buildContextBlock(recentPairs, memorySummaries, beliefs, changedNodes, userMessage, isUndoTurn, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes),
  );

  try {
    const result = await _ctx!.completion({
      prompt,
      n_predict:   400,
      temperature: 0.5,
      top_p:       0.9,
      stop:        ['<turn|>', '<|turn>'],
    });
    return cleanGemmaOutput(result.text);
  } catch (e) {
    if (!isNativeContextError(e)) throw e;
    _ctx = null;
    await ensureAgent();
    const result = await _ctx!.completion({
      prompt,
      n_predict:   400,
      temperature: 0.5,
      top_p:       0.9,
      stop:        ['<turn|>', '<|turn>'],
    });
    return cleanGemmaOutput(result.text);
  }
}

// ── Reflect: Gemma ReAct loop ─────────────────────────────────────────────────

const REACT_FORMAT_CAUSAL = `═══════════════════════════════════════
SYNTHESIS RULES — CAUSAL PATH
═══════════════════════════════════════

No tool calls. No THOUGHT lines. No RESPONSE label.
Write exactly these 6 labels in order, no skipping:

ACK: Exactly one warm sentence. Emotional tone only — do not summarise the analysis.
CAUSE: One sentence. Root cause from CAUSAL HYPOTHESIS if present; otherwise from CURRENT HEALTH STATE and MEMORY. Never paraphrase what the user said.
EFFECT: One sentence. Downstream impact grounded in the hypothesis, beliefs, or memory. Not a restatement of the user's words.
LINK: One sentence. The non-obvious mechanistic connection tying cause → effect → the user's experience. Must name something the user did not say.
SOLUTION: One or two gentle specific nudges grounded in the causal chain already surfaced. Never give generic wellness advice ("drink more water", "get more sleep", "try journaling", "take a break", "talk to someone") unless that exact behaviour was identified in CAUSAL HYPOTHESIS or CURRENT HEALTH STATE.
QUESTION: One empathetic question derived tightly from CAUSE + EFFECT + LINK. Ask the user to discriminate between two specific experiences — never confirm what you already said. Never generic (not "How are you feeling?" or "Have you noticed this before?").

Do not write a RESPONSE label. Write SOLUTION immediately after LINK. Write QUESTION immediately after SOLUTION. Never stop between them.

If CAUSAL HYPOTHESIS is present: ground CAUSE, EFFECT, and LINK in it.
If CAUSAL HYPOTHESIS is absent: ground CAUSE, EFFECT, and LINK in CURRENT HEALTH STATE and MEMORY. Do not invent a causal chain — synthesise only what the context supports.

═══════════════════════════════════════
EXAMPLE
═══════════════════════════════════════

Context (summarised for example):
CAUSAL HYPOTHESIS: The user stopped going to the gym three weeks ago after a stressful work period began. Stress has been rising since then. Without physical release, mood has been declining steadily. The combination of elevated stress and low mood is compounding fatigue, making it harder to re-engage with exercise.

CURRENT HEALTH STATE: stress_ema=high, mood=low, physical_activity=low

USER MESSAGE: I've just been feeling really flat and unmotivated lately.

ACK: That flatness and missing motivation make complete sense given what's been building up.
CAUSE: Stress from the work period has been accumulating since the gym stopped, with no physical outlet to discharge it.
EFFECT: The sustained stress load is pulling mood down, which in turn drains the motivation needed to re-engage with movement.
LINK: The gym wasn't just exercise — it was the valve that kept stress from compressing mood, and without it the two have been reinforcing each other.
SOLUTION: Since the gym itself feels out of reach right now, it might be worth finding one small physical thing that doesn't feel like exercise — a short walk, stretching while something plays in the background. Not to replace the gym, just to give the stress somewhere to go while things feel heavy.
QUESTION: When you imagine going back to the gym, does it feel like something you want but can't reach — or has it started to feel more distant than that?
`;

const REACT_FORMAT_SOCIAL = `═══════════════════════════════════════
SYNTHESIS RULES — SOCIAL PATH
═══════════════════════════════════════

No tool calls. No THOUGHT lines.
Write exactly these 2 labels in order:

ACK: One warm sentence acknowledging the user's tone.
RESPONSE: A brief warm reply — 1 to 2 sentences. No analysis, no health data, no advice.

Do not write CAUSE, EFFECT, LINK, SOLUTION, or QUESTION.

═══════════════════════════════════════
EXAMPLE
═══════════════════════════════════════

USER MESSAGE: Hey, how are you?

ACK: Always good to hear from you!
RESPONSE: I'm here and ready to listen — how have you been feeling lately?
`;

/**
 * Token-streaming callback type.
 *
 * Stream order (reflect / causal path):
 *   1. onThought — THOUGHT lines and [tool] <name> notifications from runReactHypothesisLoop
 *   2. onToken   — raw token fragments emitted by the final generator completion
 *                  (causal path: starts from 'ACK: '; social path: starts from 'ACK: ')
 *   3. TurnResult.response — complete assembled response returned when generation finishes
 *
 * The UI should pipe onToken into a streaming text view and replace it with
 * TurnResult.response once the promise resolves (so label prefixes are stripped).
 */
export type OnTokenCallback = (token: string) => void;

async function generator(
  recentPairs:        RecentPair[],
  memorySummaries:    string[],
  beliefs:            BeliefResult | null,
  changedNodes:       unknown[],
  userMessage:        string,
  isUndoTurn:         boolean      = false,
  isEarlyInteraction: boolean      = false,
  intentResult:       IntentResult = { hasReport: true, hasQuery: false, hasTrendQuery: false, isSocial: false, isThirdPartyQuery: false, queryNodes: [] },
  trendSummary:       string       = '',
  nerEntityNodes:     Set<string>  = new Set(),
  hypothesis:         string       = '',
  onThought?:         (thought: string) => void,
  onToken?:           OnTokenCallback,
): Promise<{ response: string; synthesis: StructuredSynthesis | null }> {
  await ensureAgent();

  // Derive routing flag: causal for any health/query turn; social only for pure social/correction.
  const isCausal = !(intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport);


  const reactFormat   = isCausal ? REACT_FORMAT_CAUSAL : REACT_FORMAT_SOCIAL;
  const systemContent = `${PERSONA}\n\n${RESPONSE_GUIDE}\n\n${reactFormat}`;
  const contextBlock  = buildContextBlock(
    recentPairs, memorySummaries, beliefs, changedNodes, userMessage,
    isUndoTurn, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes, hypothesis,
  );

  // Prime with 'ACK: ' to steer synthesis-only model directly into label output.
  let prompt = gemmaPrompt(systemContent, contextBlock) + 'ACK: ';

  let partialSynthesisSteps = 0;
  // responsePrimed: only used on the social path when all synthesis labels appeared but RESPONSE is missing.
  let responsePrimed  = false;
  const MAX_STEPS     = 3;

  const CANNED = "I'm here with you — take your time, and feel free to share more whenever you're ready.";

  let synthAccum: { ack?: string; cause?: string; effect?: string; link?: string; solution?: string; question?: string } = {};

  function parseSynthesisLabels(block: string): void {
    const ack      = block.match(/^ACK:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    const cause    = block.match(/^CAUSE:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    const effect   = block.match(/^EFFECT:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    const link     = block.match(/^LINK:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    const solution = block.match(/^SOLUTION:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    const question = block.match(/^QUESTION:\s*([\s\S]+?)(?=\n(?:CAUSE|EFFECT|LINK|SOLUTION|QUESTION|RESPONSE|THOUGHT|TOOL_CALL|ACK|OBSERVATION):|$)/m)?.[1]?.trim();
    if (ack)      synthAccum.ack      = ack;
    if (cause)    synthAccum.cause    = cause;
    if (effect)   synthAccum.effect   = effect;
    if (link)     synthAccum.link     = link;
    if (solution) synthAccum.solution = solution;
    if (question) synthAccum.question = question;
  }

  function resolveSynthesis(): StructuredSynthesis | null {
    const { ack, cause, effect, link, solution, question } = synthAccum;
    if (ack && cause && effect && link && solution && question) return { ack, cause, effect, link, solution, question };
    return null;
  }

  /** Build response from ACK+CAUSE+EFFECT+LINK paragraphs (causal path only).
   *  ACK is always included — falls back to a neutral acknowledgement if the model
   *  omitted it, so the response never starts cold with a CAUSE sentence. */
  function buildCausalResponse(): { response: string; synthesis: StructuredSynthesis | null } | null {
    if (synthAccum.cause && synthAccum.effect && synthAccum.link) {
      const synthesis = resolveSynthesis();
      const ack            = synthAccum.ack ?? "I hear you.";
      const solutionSuffix = synthAccum.solution ? `\n\n${synthAccum.solution}` : '';
      const questionSuffix = synthAccum.question ? `\n\n${synthAccum.question}` : '';
      if (__DEV__) console.log(`[agent] synthesis causal — ACK: ${ack}\n  CAUSE: ${synthAccum.cause}\n  EFFECT: ${synthAccum.effect}\n  LINK: ${synthAccum.link}\n  SOLUTION: ${synthAccum.solution}\n  QUESTION: ${synthAccum.question}`);
      return {
        response: `${ack}\n\n${synthAccum.cause}\n\n${synthAccum.effect}\n\n${synthAccum.link}${solutionSuffix}${questionSuffix}`,
        synthesis,
      };
    }
    return null;
  }

  for (let step = 0; step < MAX_STEPS; step++) {
    let result: Awaited<ReturnType<LlamaContext['completion']>>;
    // onToken only wired on the first step (step=0) — that is the main synthesis
    // generation pass where the visible response is produced (ACK → CAUSE → …).
    // Subsequent steps (re-primes) are control passes and should not stream to UI.
    const tokenCb = step === 0 && onToken ? (data: { token: string }) => onToken(data.token) : undefined;
    try {
      result = await _ctx!.completion({
        prompt,
        n_predict:   256,
        temperature: 0.3,
        top_p:       0.9,
        stop:        ['<turn|>', '<|turn>'],
      }, tokenCb);
    } catch (e) {
      if (!isNativeContextError(e)) throw e;
      _ctx = null;
      await ensureAgent();
      result = await _ctx!.completion({
        prompt,
        n_predict:   256,
        temperature: 0.3,
        top_p:       0.9,
        stop:        ['<turn|>', '<|turn>'],
      }, tokenCb);
    }

    const rawText = result.text;
    if (__DEV__) console.log(`[agent] step=${step} raw:\n${rawText}`);

    // At step=0 the prompt was primed with 'ACK: ', so prepend it for label parsing.
    const textForParsing = step === 0 ? `ACK: ${rawText}` : rawText;
    const text = cleanGemmaOutput(rawText);
    const textForParsingClean = step === 0 ? `ACK: ${text}` : text;

    // Extract thought for UI callback (if model emits one despite instructions)
    const thoughtLine = textForParsing.match(/THOUGHT:\s*(.+)/)?.[1]?.trim()
      ?? rawText.match(/<\|channel>thought([\s\S]*?)<channel\|>/)?.[1]
          ?.split('\n').map((l: string) => l.trim()).find((l: string) => l.length > 0);
    if (thoughtLine && onThought) onThought(thoughtLine);

    // ── Social path: responsePrimed branch (RESPONSE label re-prime) ──────────
    if (responsePrimed) {
      // Only reaches here on the social path.
      parseSynthesisLabels(textForParsingClean);
      const responseBody = text.trim();
      const synthesis = resolveSynthesis(); // null on social path — correct
      if (__DEV__) console.log(`[agent] synthesis social (primed) — ACK: ${synthAccum.ack}\n  RESPONSE: ${responseBody}`);
      return { response: responseBody || CANNED, synthesis };
    }

    parseSynthesisLabels(textForParsingClean);

    // ── Causal path: build response from CAUSE+EFFECT+LINK ───────────────────
    if (isCausal) {
      const causalResult = buildCausalResponse();
      if (causalResult) {
        prompt += text + '<turn|>\n';
        return causalResult;
      }
    }

    // ── Social path: look for RESPONSE label ─────────────────────────────────
    if (!isCausal) {
      const responseMatch = textForParsingClean.match(/RESPONSE:\s*([\s\S]+?)(?=\nSOLUTION:|\nQUESTION:|$)/);
      if (responseMatch) {
        const synthesis = resolveSynthesis(); // null on social path — correct
        if (__DEV__) console.log(`[agent] synthesis social — ACK: ${synthAccum.ack}\n  RESPONSE: ${responseMatch[1].trim()}`);
        prompt += text + '<turn|>\n';
        return { response: responseMatch[1].trim(), synthesis };
      }
    }

    const hasPartialSynthesis = synthAccum.ack || synthAccum.cause || synthAccum.effect || synthAccum.link || synthAccum.solution || synthAccum.question;
    if (hasPartialSynthesis) {
      partialSynthesisSteps++;
      if (partialSynthesisSteps >= 2) return { response: CANNED, synthesis: null };
      if (!isCausal) {
        // Social path: if all non-RESPONSE labels somehow appeared, re-prime RESPONSE.
        // In practice the social format only emits ACK, so this is a safety net.
        const allLabelsPresent = synthAccum.ack && synthAccum.cause && synthAccum.effect && synthAccum.link && synthAccum.solution && synthAccum.question;
        if (allLabelsPresent) {
          responsePrimed = true;
          prompt += text + '<turn|>\n' + '<|turn>model\nRESPONSE: ';
        } else {
          prompt += text + '<turn|>\n' + '<|turn>model\n';
        }
      } else {
        // Causal path: continue generating — no RESPONSE re-prime ever.
        prompt += text + '<turn|>\n' + '<|turn>model\n';
      }
      continue;
    }

    prompt += text + '<turn|>\n';
    return { response: CANNED, synthesis: null };
  }

  // MAX_STEPS exhausted — best-effort extraction from accumulated labels.
  const lastModelStart = prompt.lastIndexOf('<|turn>model\n');
  if (lastModelStart !== -1) {
    const tail = prompt.slice(lastModelStart + '<|turn>model\n'.length)
      .replace(/<turn\|>[\s\S]*/, '')
      .trim();
    parseSynthesisLabels(tail);
    if (isCausal) {
      const causalResult = buildCausalResponse();
      if (causalResult) return causalResult;
    } else {
      const resp = tail.match(/RESPONSE:\s*([\s\S]+)/)?.[1]?.trim();
      if (resp) return { response: resp, synthesis: null };
    }
  }
  return { response: CANNED, synthesis: null };
}

// ── Undo work (Stage 3b) ──────────────────────────────────────────────────────

interface UndoWorkResult {
  preUndoBeliefs: BeliefResult | null;
  isPureUndo:     boolean;
}

async function runUndoWork(
  db:        DB,
  sessionId: string,
  turnId:    string,
  nerResult: Awaited<ReturnType<typeof runNer>>,
): Promise<UndoWorkResult> {
  const prevTurnId = getPrevTurnId(db, sessionId);
  if (!prevTurnId) return { preUndoBeliefs: null, isPureUndo: false };

  const nerNodes  = nerResult.entities.map(e => e.node_name);
  const prevNodes = getActiveTurnNodes(db, prevTurnId);
  const targets   = nerNodes.filter(n => prevNodes.includes(n));

  if (nerNodes.length > 0 && targets.length === 0) {
    return { preUndoBeliefs: null, isPureUndo: false };
  }

  const undoArgs = targets.length > 0
    ? { turn_id: prevTurnId, node_names: targets }
    : { turn_id: prevTurnId };

  const res = await dispatchTool('undo_last_entry', undoArgs, db, turnId) as
    { pre_undo_beliefs: BeliefResult | null };

  const preUndoBeliefs = res.pre_undo_beliefs;

  if (preUndoBeliefs) overrideTurnStart(preUndoBeliefs);

  const correctedNodes = new Set(targets);
  const newNodes = nerResult.entities
    .map(e => e.node_name)
    .filter(n => !correctedNodes.has(n));

  // Pure undo: either no new data at all, or pure text correction (NER empty, only
  // unmatched text — user said "disregard that" with nothing new to record).
  const isPureTextCorrection = nerNodes.length === 0 && nerResult.unmatched.length > 0;
  const isPureUndo = preUndoBeliefs !== null
    && (isPureTextCorrection || (nerResult.unmatched.length === 0 && newNodes.length === 0));

  return { preUndoBeliefs, isPureUndo };
}

// ── Phase 1: startTurn ────────────────────────────────────────────────────────

/**
 * Phase 1 of the Talk flow.
 * Runs stage-ordered pipeline: ack-check → undo → classify → NER → undo-work →
 * inference → belief window → context → Gemma.
 * Returns { done: true } if no questions need answering (completes inline).
 * Returns { done: false, turnId, questions } if the UI must collect answers
 * before calling completeTurn.
 */
export async function startTurn(
  db:          DB,
  sessionId:   string,
  userMessage: string,
  mode:        AgentMode,
  onThought?:  (thought: string) => void,
  onToken?:    OnTokenCallback,
): Promise<TurnResult> {
  evictStalePendingTurns();

  // Reset MCP session state when a new session begins so stale beliefs from
  // the previous session don't bleed into the new session's first turn.
  if (sessionId !== _lastSessionId) {
    resetSession();
    _lastSessionId = sessionId;
  }

  if (__DEV__) { var _tTotal = Date.now(); console.log('[pipeline] ── startTurn ──────────────────────────'); }

  // STAGE 0: isAckOnly — exact match, no model calls
  if (isAckOnly(userMessage)) {
    if (__DEV__) console.log('[pipeline] ackOnly short-circuit');
    const response = ackResponse();
    const turnId   = newTurnId();
    storeChatMessage(db, sessionId, turnId, 'user',  userMessage);
    storeChatMessage(db, sessionId, turnId, 'model', response);
    await maybeEvictOldest(db, sessionId);
    return { done: true, response, synthesis: null };
  }

  const turnId = newTurnId();
  setCurrentUserMessage(userMessage);

  // Fetch recent pairs + memory first (needed for isEarlyInteraction and memory query)
  const recentPairs = getRecentPairs(db, sessionId);
  setRecentTopics(recentPairs.map(p => p.topic).filter(Boolean).join(' '));

  const windowDays = mode === 'glance' ? 14 : 90;
  if (__DEV__) { var _tMemory = Date.now(); console.log('[pipeline] memory fetch → start'); }
  const memRes = await dispatchTool(
    'get_user_memory', { window_days: windowDays }, db, turnId,
  ) as { summaries: string[] };
  const memorySummaries = memRes.summaries ?? [];
  if (__DEV__) console.log(`[pipeline] memory fetch → ${Date.now() - _tMemory!}ms\n  summaries(${memorySummaries.length}):\n${memorySummaries.map((s, i) => `    ${i + 1}. ${s}`).join('\n') || '    (none)'}`);

  const isEarlyInteraction = getReportedTurnCount(db) < 3;

  // STAGE 1: detectUndoIntent
  if (__DEV__) { var _tUndo = Date.now(); console.log('[pipeline] detectUndoIntent → start'); }
  const isUndoTurn = await detectUndoIntent(userMessage);
  if (__DEV__) console.log(`[pipeline] detectUndoIntent → ${Date.now() - _tUndo!}ms\n  isUndoTurn=${isUndoTurn}`);

  // STAGE 2: classifyIntent — always runs
  if (__DEV__) { var _tIntent = Date.now(); console.log('[pipeline] classifyIntent → start'); }
  const intentResult = await classifyIntent(userMessage);
  if (__DEV__) console.log(`[pipeline] classifyIntent → ${Date.now() - _tIntent!}ms\n  hasReport=${intentResult.hasReport}  hasQuery=${intentResult.hasQuery}  isSocial=${intentResult.isSocial}  hasTrendQuery=${intentResult.hasTrendQuery}  isThirdParty=${intentResult.isThirdPartyQuery}\n  queryNodes=${JSON.stringify(intentResult.queryNodes)}`);
  const socialFastPath = mode !== 'reflect' && intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport;

  let trendSummary = '';
  if (intentResult.hasTrendQuery && intentResult.queryNodes.length > 0 && !socialFastPath) {
    if (__DEV__) { var _tTrend = Date.now(); console.log('[pipeline] get_belief_trend → start'); }
    const trendRes = await dispatchTool(
      'get_belief_trend',
      { node_names: intentResult.queryNodes, window_days: windowDays, session_id: sessionId },
      db, turnId,
    ) as { trends: Record<string, string> };
    trendSummary = Object.values(trendRes.trends).filter(Boolean).join('\n');
    if (__DEV__) console.log(`[pipeline] get_belief_trend → ${Date.now() - _tTrend!}ms\n  ${trendSummary || '(no trend data)'}`);
  }

  // STAGE 3: NER — only if hasReport OR isUndoTurn
  type NerResult = Awaited<ReturnType<typeof runNer>>;
  const emptyNer: NerResult = { entities: [], unmatched: [], topics: [], raw_output: '' };
  let nerResult: NerResult = emptyNer;

  if (intentResult.hasReport || isUndoTurn) {
    if (__DEV__) { var _tNer = Date.now(); console.log('[pipeline] runNer → start'); }
    nerResult = await runNer(db, userMessage, '', turnId, mode, sessionId);
    if (__DEV__) console.log(`[pipeline] runNer → ${Date.now() - _tNer!}ms\n  entities(${nerResult.entities.length}): ${nerResult.entities.map(e => `${e.node_name}=${e.node_value ?? e.raw_value ?? '?'}(${Math.round(e.confidence * 100)}%)`).join(', ') || '(none)'}\n  unmatched(${nerResult.unmatched.length}): ${nerResult.unmatched.join(', ') || '(none)'}\n  topics: ${nerResult.topics.join(', ') || '(none)'}`);
  } else {
    if (__DEV__) console.log('[pipeline] runNer → skipped (no report, no undo)');
  }

  const topic          = nerResult.topics.length > 0
    ? nerResult.topics.join(', ')
    : intentResult.queryNodes.join(', ');
  const nerEntityNodes = new Set(nerResult.entities.map(e => e.node_name));
  setDirectEvidenceNodes(nerEntityNodes);

  // STAGE 3b: undo-specific work — only if isUndoTurn
  let preUndoBeliefs: BeliefResult | null = null;
  let isPureUndo = false;

  if (isUndoTurn) {
    if (__DEV__) { var _tUndoWork = Date.now(); console.log('[pipeline] runUndoWork → start'); }
    const undoWork = await runUndoWork(db, sessionId, turnId, nerResult);
    preUndoBeliefs = undoWork.preUndoBeliefs;
    isPureUndo     = undoWork.isPureUndo;
    if (__DEV__) console.log(`[pipeline] runUndoWork → ${Date.now() - _tUndoWork!}ms  isPureUndo=${isPureUndo}`);
  }

  // Pure undo short-circuit — no Gemma needed
  if (isPureUndo) {
    if (__DEV__) console.log('[pipeline] pureUndo short-circuit');
    const ack = undoAck();
    storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
    storeChatMessage(db, sessionId, turnId, 'model', ack);
    await maybeEvictOldest(db, sessionId);
    return { done: true, response: ack, synthesis: null };
  }

  // STAGE 5: selectBeliefWindow — computed inside buildContextBlock via params

  // isUndoTurn as flag for context block: only true when undo fired AND had prior beliefs
  const isUndoContext = isUndoTurn && preUndoBeliefs !== null;

  // STAGE 6: build question lists (not applicable on social fast path — no NER ran)
  const hasL1      = nerResult.entities.some(e => e.original_column);
  const filledCols = nerResult.entities
    .filter(e => e.original_column)
    .map(e => e.original_column!);

  const followUps  = buildFollowUps(nerResult.entities);
  const cascadeQs  = hasL1 ? buildCascade(filledCols, db).questions : [];

  const questions: DisplayQuestion[] = [
    ...followUps.map(q => ({ original_col: q.original_col, node_name: q.node_name, question: q.question, kind: 'followup' as const, opts: q.opts, range: q.range })),
    ...cascadeQs.map(q => ({ original_col: q.original_col, node_name: q.node_name, question: q.question, kind: 'cascade' as const, opts: q.opts, range: q.range })),
  ];
  if (__DEV__) console.log(`[pipeline] questions built → followUps=${followUps.length} cascadeQs=${cascadeQs.length}\n  ${[...followUps, ...cascadeQs].map(q => `[${q.node_name}] ${q.question}`).join('\n  ') || '(none)'}`);

  // If no questions, complete inline to avoid an extra UI round-trip
  if (questions.length === 0) {
    // Run inference now that all evidence (NER) is in
    let beliefs: BeliefResult | null = null;
    if (!socialFastPath) {
      if (__DEV__) { var _tInfer = Date.now(); console.log('[pipeline] run_dbn_inference → start'); }
      const inferRes = await dispatchTool(
        'run_dbn_inference', { turn_id: turnId }, db, turnId,
      ) as { beliefs?: BeliefResult; skipped?: boolean };
      beliefs = (inferRes.beliefs ?? null) as BeliefResult | null;
      if (__DEV__) console.log(`[pipeline] run_dbn_inference → ${Date.now() - _tInfer!}ms  skipped=${inferRes.skipped ?? false}\n  top nodes: ${beliefs ? Object.entries(beliefs).map(([n, d]) => { const [s, p] = Object.entries(d).sort((a,b) => (b[1] as number)-(a[1] as number))[0]; return `${n}=${s}(${Math.round((p as number)*100)}%)`; }).sort((a,b) => parseInt(b.match(/\((\d+)%\)/)?.[1]??'0') - parseInt(a.match(/\((\d+)%\)/)?.[1]??'0')).slice(0,8).join(', ') : '(none)'}`);
    } else {
      if (__DEV__) console.log('[pipeline] run_dbn_inference → skipped (socialFastPath)');
    }

    // Short-circuit: social and third-party turns need no model
    const isPureThirdParty = intentResult.isThirdPartyQuery && !intentResult.hasReport && !intentResult.hasQuery;
    if (socialFastPath) {
      if (__DEV__) console.log('[pipeline] socialFastPath short-circuit');
      const response = socialResponse();
      storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
      storeChatMessage(db, sessionId, turnId, 'model', response);
      await maybeEvictOldest(db, sessionId);
      return { done: true, response, synthesis: null };
    }
    if (isPureThirdParty) {
      if (__DEV__) console.log('[pipeline] pureThirdParty short-circuit');
      const response = thirdPartyResponse(userMessage);
      storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
      storeChatMessage(db, sessionId, turnId, 'model', response);
      await maybeEvictOldest(db, sessionId);
      return { done: true, response, synthesis: null };
    }

    if (__DEV__) { var _tChanged = Date.now(); console.log('[pipeline] get_changed_nodes → start'); }
    const changedRes = await dispatchTool(
      'get_changed_nodes', {}, db, turnId,
    ) as { changed_nodes: unknown[] };
    if (__DEV__) console.log(`[pipeline] get_changed_nodes → ${Date.now() - _tChanged!}ms\n  ${(changedRes.changed_nodes as Array<{node: string; delta: number; label: string}>).map(c => `${c.node} Δ${(c.delta*100).toFixed(1)}% [${c.label}]`).join(', ') || '(none)'}`);

    // Release NLU before loading agent to avoid OOM on constrained devices.
    // NLU (800 MB) is no longer needed after NER; agent (1.7 GB) loads next.
    // Brief pause lets Android actually reclaim the freed pages before initLlama runs.
    if (__DEV__) { var _tSwap = Date.now(); console.log('[pipeline] model swap → releaseNlu start'); }
    await releaseNlu();
    if (__DEV__) console.log(`[pipeline] model swap → releaseNlu done (${Date.now() - _tSwap!}ms), waiting 300ms`);
    await new Promise(r => setTimeout(r, 300));
    if (__DEV__) { var _tLoad = Date.now(); console.log('[pipeline] model swap → ensureAgent start'); }
    await ensureAgent();
    if (__DEV__) console.log(`[pipeline] model swap → ensureAgent done (${Date.now() - _tLoad!}ms)  totalSwap=${Date.now() - _tSwap!}ms`);

    // STAGE 7: Gemma
    if (__DEV__) console.log(`[pipeline] ── Gemma generation start (mode=${mode}) ──`);
    let response: string;
    let synthesis: StructuredSynthesis | null = null;
    if (mode === 'glance') {
      if (__DEV__) { var _tGemma = Date.now(); }
      response = await runGlanceCall(recentPairs, memorySummaries, beliefs, changedRes.changed_nodes, userMessage, isUndoContext, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes);
      if (__DEV__) console.log(`[pipeline] runGlanceCall → ${Date.now() - _tGemma!}ms\n  response: ${response}`);
    } else {
      const isSocialOnly = intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport;
      if (!isSocialOnly) {
        if (__DEV__) { var _tHypo = Date.now(); console.log('[pipeline] runReactHypothesisLoop → start'); }
      }
      const hypothesis = isSocialOnly ? '' : await runReactHypothesisLoop(db, userMessage, changedRes.changed_nodes, nerEntityNodes, onThought);
      if (__DEV__ && !isSocialOnly) console.log(`[pipeline] runReactHypothesisLoop → ${Date.now() - _tHypo!}ms\n  hypothesis: ${hypothesis || '(none)'}`);
      if (__DEV__) { var _tGen = Date.now(); console.log('[pipeline] generator → start'); }
      ({ response, synthesis } = await generator(recentPairs, memorySummaries, beliefs, changedRes.changed_nodes, userMessage, isUndoContext, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes, hypothesis, onThought, onToken));
      if (__DEV__) console.log(`[pipeline] generator → ${Date.now() - _tGen!}ms\n  response: ${response}`);
    }

    storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
    storeChatMessage(db, sessionId, turnId, 'model', response);
    await maybeEvictOldest(db, sessionId);
    if (__DEV__) console.log(`[pipeline] ── DONE  total=${Date.now() - _tTotal!}ms ──`);
    return { done: true, response, synthesis };
  }

  // Release NLU before UI collects answers; agent loads in completeTurn.
  await releaseNlu();
  await new Promise(r => setTimeout(r, 300));

  // Store pending state for completeTurn
  _pendingTurns.set(turnId, {
    sessionId, userMessage, isUndoTurn: isUndoContext,
    memorySummaries, recentPairs, topic, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes,
    createdAt: Date.now(),
  });
  return { done: false, turnId, questions };
}

// ── Phase 2: completeTurn ─────────────────────────────────────────────────────

/**
 * Phase 2 of the Talk flow. Called after UI collects question answers.
 * Writes answers, runs inference once (all evidence now in DB), then Gemma.
 */
export async function completeTurn(
  db:         DB,
  turnId:     string,
  mode:       AgentMode,
  answers:    QuestionAnswer[],
  onThought?: (thought: string) => void,
  onToken?:   OnTokenCallback,
): Promise<{ response: string; synthesis: StructuredSynthesis | null }> {
  evictStalePendingTurns();
  if (__DEV__) { var _tCTotal = Date.now(); console.log('[pipeline] ── completeTurn ──────────────────────────'); }
  await ensureAgent();

  const pending = _pendingTurns.get(turnId);
  if (!pending) throw new Error(`agent: no pending turn for id ${turnId}`);
  _pendingTurns.delete(turnId);

  const { sessionId, userMessage, isUndoTurn, memorySummaries, recentPairs, topic, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes } = pending;
  setDirectEvidenceNodes(nerEntityNodes);

  // Write proactive answers
  if (__DEV__) console.log(`[pipeline] proactive answers(${answers.length}): ${answers.map(a => `${a.node_name}=${a.raw_value ?? a.raw_text}`).join(', ') || '(none)'}`);
  for (const ans of answers) {
    writeProactiveAnswer(db, turnId, ans);
    if (ans.original_col) {
      maybeResolveComposite(db, ans.node_name, getCompositeSourceCol(ans.original_col), turnId);
    }
  }

  // Run inference once — all evidence (NER + proactive answers) is now in DB
  if (__DEV__) { var _tCInfer = Date.now(); console.log('[pipeline] run_dbn_inference → start'); }
  const inferRes = await dispatchTool(
    'run_dbn_inference', { turn_id: turnId }, db, turnId,
  ) as { beliefs?: BeliefResult };
  const beliefs = (inferRes.beliefs ?? null) as BeliefResult | null;
  if (__DEV__) console.log(`[pipeline] run_dbn_inference → ${Date.now() - _tCInfer!}ms\n  top nodes: ${beliefs ? Object.entries(beliefs).map(([n, d]) => { const [s, p] = Object.entries(d).sort((a,b) => (b[1] as number)-(a[1] as number))[0]; return `${n}=${s}(${Math.round((p as number)*100)}%)`; }).sort((a,b) => parseInt(b.match(/\((\d+)%\)/)?.[1]??'0') - parseInt(a.match(/\((\d+)%\)/)?.[1]??'0')).slice(0,8).join(', ') : '(none)'}`);

  if (__DEV__) { var _tCChanged = Date.now(); console.log('[pipeline] get_changed_nodes → start'); }
  const changedRes = await dispatchTool(
    'get_changed_nodes', {}, db, turnId,
  ) as { changed_nodes: unknown[] };
  if (__DEV__) console.log(`[pipeline] get_changed_nodes → ${Date.now() - _tCChanged!}ms\n  ${(changedRes.changed_nodes as Array<{node: string; delta: number; label: string}>).map(c => `${c.node} Δ${(c.delta*100).toFixed(1)}% [${c.label}]`).join(', ') || '(none)'}`);

  if (__DEV__) console.log(`[pipeline] ── Gemma generation start (mode=${mode}) ──`);
  let response: string;
  let synthesis: StructuredSynthesis | null = null;
  if (mode === 'glance') {
    if (__DEV__) { var _tCGemma = Date.now(); }
    response = await runGlanceCall(recentPairs, memorySummaries, beliefs, changedRes.changed_nodes, userMessage, isUndoTurn, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes);
    if (__DEV__) console.log(`[pipeline] runGlanceCall → ${Date.now() - _tCGemma!}ms\n  response: ${response}`);
  } else {
    const isSocialOnly = intentResult.isSocial && !intentResult.hasQuery && !intentResult.hasReport;
    if (!isSocialOnly) { if (__DEV__) { var _tCHypo = Date.now(); console.log('[pipeline] runReactHypothesisLoop → start'); } }
    const hypothesis = isSocialOnly ? '' : await runReactHypothesisLoop(db, userMessage, changedRes.changed_nodes, nerEntityNodes, onThought);
    if (__DEV__ && !isSocialOnly) console.log(`[pipeline] runReactHypothesisLoop → ${Date.now() - _tCHypo!}ms\n  hypothesis: ${hypothesis || '(none)'}`);
    if (__DEV__) { var _tCGen = Date.now(); console.log('[pipeline] generator → start'); }
    ({ response, synthesis } = await generator(recentPairs, memorySummaries, beliefs, changedRes.changed_nodes, userMessage, isUndoTurn, isEarlyInteraction, intentResult, trendSummary, nerEntityNodes, hypothesis, onThought, onToken));
    if (__DEV__) console.log(`[pipeline] generator → ${Date.now() - _tCGen!}ms\n  response: ${response}`);
  }

  storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
  storeChatMessage(db, sessionId, turnId, 'model', response);
  await maybeEvictOldest(db, sessionId);
  if (__DEV__) console.log(`[pipeline] ── DONE  total=${Date.now() - _tCTotal!}ms ──`);
  return { response, synthesis };
}

// ── Journal: NER + inference, no Gemma ───────────────────────────────────────

/**
 * Journal feature entry point. Runs NER and inference, stores both user message
 * and a short acknowledgement, evicts oldest if buffer exceeded.
 * Returns the acknowledgement string for the UI to display.
 */
export async function runJournalTurn(
  db:          DB,
  sessionId:   string,
  userMessage: string,
): Promise<string> {
  const turnId = newTurnId();
  setCurrentUserMessage(userMessage);

  const nerResult = await runNer(db, userMessage, '', turnId, 'reflect', sessionId);
  const topic     = nerResult.topics.join(', ');

  await dispatchTool('run_dbn_inference', { turn_id: turnId }, db, turnId);

  const ack = journalAck();
  storeChatMessage(db, sessionId, turnId, 'user',  userMessage, topic);
  storeChatMessage(db, sessionId, turnId, 'model', ack);
  await maybeEvictOldest(db, sessionId);
  return ack;
}

// ── Report generation helpers ──────────────────────────────────────────────────

let _agentBusy = false;

export function isAgentBusy(): boolean { return _agentBusy; }

/**
 * Run a single completion against the Gemma agent context with a system + user
 * prompt pair. Used by the Doctor Report Plan-and-Execute flow.
 *
 * Marks the agent context busy for the duration of the call so concurrent chat
 * turns can be deferred (see isAgentBusy()).
 */
export async function runAgentCompletion(
  systemPrompt: string,
  userPrompt:   string,
  nPredict:     number = 500,
): Promise<string> {
  _agentBusy = true;
  try {
    await ensureAgent();
    if (!_ctx) throw new Error('agent: context not available');
    const prompt = gemmaPrompt(systemPrompt, userPrompt);
    const result = await _ctx.completion({
      prompt,
      n_predict:   nPredict,
      temperature: 0.1,
      stop:        ['<turn|>', '<|turn>', '<|im_end|>', '<end_of_turn>'],
    });
    return cleanGemmaOutput(result.text);
  } finally {
    _agentBusy = false;
  }
}

// ── Report Hypothesis Loop ────────────────────────────────────────────────────

const REPORT_HYPOTHESIS_TOOLS = new Set(['get_belief_trend', 'get_user_memory']);

// Kept separate from REPORT_HYPOTHESIS_TOOLS — expected to diverge as reflect mode evolves.
const REFLECT_HYPOTHESIS_TOOLS = new Set(['get_user_memory', 'get_belief_trend']);

function buildReportToolsBlock(): string {
  return MCP_TOOLS
    .filter(t => REPORT_HYPOTHESIS_TOOLS.has(t.name))
    .map(t => `${t.name}: ${t.description}\nArguments: ${JSON.stringify(t.inputSchema.properties)}`)
    .join('\n\n');
}

function buildReflectHypothesisToolsBlock(): string {
  return MCP_TOOLS
    .filter(t => REFLECT_HYPOTHESIS_TOOLS.has(t.name))
    .map(t => `${t.name}: ${t.description}\nArguments: ${JSON.stringify(t.inputSchema.properties)}`)
    .join('\n\n');
}

/**
 * Lightweight ReAct loop that runs BEFORE the report planner.
 * Gemma probes 180-day data (get_user_memory + get_belief_trend) to discover
 * the causal chain behind the user's complaint, then outputs a HYPOTHESIS block.
 *
 * Does NOT set _agentBusy — report pipeline is never concurrent with chat.
 * Returns '' on any failure so the report pipeline degrades gracefully.
 */
export async function runReportHypothesisLoop(
  db:            DB,
  complaint:     string,
  patternList:   string,
  sensorSummary: string,
): Promise<string> {
  await ensureAgent();
  if (!_ctx) return '';
  setCurrentUserMessage(complaint);

  const systemContent =
`You are a causal analyst. You do NOT chat. You call tools, then write a HYPOTHESIS. Nothing else.

You have exactly two tools: get_user_memory and get_belief_trend.

STEP-BY-STEP SEQUENCE — follow this exact format:

THOUGHT: I need memory context first.
TOOL_CALL: {"name": "get_user_memory", "arguments": {"window_days": 180}}
OBSERVATION: {system fills this in — never write it yourself}
THOUGHT: Memory retrieved. I will check the most suspect node from DETECTED PATTERNS.
TOOL_CALL: {"name": "get_belief_trend", "arguments": {"node_names": ["node_A"], "window_days": 180}}
OBSERVATION: {system fills this in — never write it yourself}
THOUGHT: First trend retrieved. Checking second node.
TOOL_CALL: {"name": "get_belief_trend", "arguments": {"node_names": ["node_B"], "window_days": 180}}
OBSERVATION: {system fills this in — never write it yourself}
THOUGHT: Second trend retrieved. Checking third node.
TOOL_CALL: {"name": "get_belief_trend", "arguments": {"node_names": ["node_C"], "window_days": 180}}
OBSERVATION: {system fills this in — never write it yourself}
THOUGHT: Third trend retrieved. Checking fourth node.
TOOL_CALL: {"name": "get_belief_trend", "arguments": {"node_names": ["node_D"], "window_days": 180}}
OBSERVATION: {system fills this in — never write it yourself}
THOUGHT: All trends retrieved. I can now form the causal chain.
HYPOTHESIS: [causal chain in plain English, 3 to 5 sentences]

--- RULES ---

NODE SELECTION (before calling get_belief_trend):
- Read the user's complaint and the DETECTED PATTERNS list.
- Pick the 4 or 5 node names most likely to cause or worsen the complaint. Ask: "if this node is abnormal, does it directly explain the complaint?" Prefer nodes that are upstream causes over downstream effects.
- node_names values must be copied character-for-character from DETECTED PATTERNS. Do not invent or alter names.

TOOL CALL RULES:
- Call get_user_memory exactly once, always first, with window_days 180.
- Call get_belief_trend exactly once per chosen node, one node per call. Call it 4 or 5 times total.
- window_days must be 180 in every tool call.
- Never write an OBSERVATION line. Wait for the system to provide it.

HYPOTHESIS RULES:
- Write 3 to 5 sentences.
- State a causal chain: which node changed first, what it affected next, and how that led to the complaint.
- Use the actual trend direction from each OBSERVATION (rising, falling, stable). Do not guess.
- Write in plain English. No bullet points, no hedging phrases like "it seems possible that".
- HYPOTHESIS: is the final line. Do not write anything after it.

node_A, node_B, node_C, node_D above are placeholders. Replace them with names from DETECTED PATTERNS.

Available tools:
${buildReportToolsBlock()}`;

  const userContent =
`COMPLAINT: ${complaint}

DETECTED PATTERNS:
${patternList || '(none)'}

SENSOR SUMMARY: ${sensorSummary || '(none)'}

Instructions:
1. Call get_user_memory (once, now).
2. From DETECTED PATTERNS, pick the 4 or 5 nodes most directly linked to the complaint. Copy their names exactly.
3. Call get_belief_trend once per chosen node, one at a time, window_days 180.
4. After the last OBSERVATION, write HYPOTHESIS: followed by a 3 to 5 sentence causal chain using the trend data you just retrieved.`;

  let prompt = gemmaPrompt(systemContent, userContent);

  const MAX_STEPS = 6;
  for (let step = 0; step < MAX_STEPS; step++) {
    const result = await _ctx!.completion({
      prompt,
      n_predict:   400,
      temperature: 0.2,
      top_p:       0.85,
      stop:        ['<turn|>', '<|turn>', 'OBSERVATION:'],
    });
    const text = cleanGemmaOutput(result.text);
    // Fix 1: use gemmaPrompt delimiters throughout (consistent with generator)
    prompt += text + '<turn|>\n';

    const hypMatch = text.match(/HYPOTHESIS:\s*([\s\S]+)/);
    if (hypMatch) return hypMatch[1].trim();

    const toolMatch = text.match(/TOOL_CALL:\s*(\{[\s\S]*?\})(?:\s|$)/);
    if (!toolMatch) return '';

    let call: { name: string; arguments?: Record<string, unknown> };
    try   { call = JSON.parse(toolMatch[1]); }
    catch { return ''; }

    let observation: unknown;
    if (!REPORT_HYPOTHESIS_TOOLS.has(call.name)) {
      observation = { error: `Tool '${call.name}' not available in report mode` };
    } else {
      observation = await dispatchTool(call.name, call.arguments ?? {}, db, '');
    }

    prompt +=
      `<|turn>user\nOBSERVATION: ${JSON.stringify(observation)}<turn|>\n` +
      `<|turn>model\n`;
  }
  return '';
}

// ── Reflect Hypothesis Loop ───────────────────────────────────────────────────

/**
 * Lightweight locked-sequence ReAct loop for reflect mode hypothesis generation.
 * Runs BEFORE the main ReAct synthesis: calls get_user_memory then a single batched
 * get_belief_trend for up to 6 changed nodes (3 cause/evidence + 3 effect/co-influenced),
 * outputs a HYPOTHESIS causal chain.
 *
 * Does NOT set _agentBusy — runs inline within the reflect turn pipeline.
 * Returns '' on any failure so the caller degrades gracefully.
 */
export async function runReactHypothesisLoop(
  db:             DB,
  userMessage:    string,
  changedNodes:   unknown[],
  nerEntityNodes: Set<string>,
  onThought?:     (thought: string) => void,
): Promise<string> {
  try {
    await ensureAgent();
    if (!_ctx) return '';
    setCurrentUserMessage(userMessage);

    // ── Node selection ────────────────────────────────────────────────────────
    // Typed changed node from get_changed_nodes
    interface ChangedNodeEntry { node: string; label: string; }

    const typed = (changedNodes as ChangedNodeEntry[]).filter(n => n.node);

    // Separate cause/evidence nodes from effect/co-influenced nodes
    const causeNodes  = typed.filter(n => n.label === 'cause' || n.label === 'evidence').map(n => n.node).slice(0, 3);
    const effectNodes = typed.filter(n => n.label === 'effect' || n.label === 'co-influenced').map(n => n.node).slice(0, 3);

    // Fallback: if changedNodes empty, use NER nodes as generic nodes
    const allNodes = [...causeNodes, ...effectNodes];
    const nodesToUse: string[] = allNodes.length > 0
      ? allNodes
      : nerEntityNodes.size > 0
        ? [...nerEntityNodes].slice(0, 4)
        : ['stress_ema', 'mood'];

    // For the SINGLE get_belief_trend call — all nodes batched
    const trendNodeNames = nodesToUse;

    // Labels for prompt — only meaningful when changedNodes has entries
    const causeLabel  = causeNodes.length  > 0 ? causeNodes.join(', ')  : '';
    const effectLabel = effectNodes.length > 0 ? effectNodes.join(', ') : '';
    // ── End node selection ───────────────────────────────────────────────────

    const systemContent =
`You are a causal analyst. You do NOT chat. You call two tools in sequence, then write a HYPOTHESIS.
${causeLabel ? `\nCAUSE NODES: ${causeLabel}` : ''}${effectLabel ? `\nEFFECT NODES: ${effectLabel}` : ''}${!causeLabel && !effectLabel ? `\nNODES: ${nodesToUse.join(', ')}` : ''}

Copy this sequence exactly:

THOUGHT: I will retrieve memory then trend data.
TOOL_CALL: {"name": "get_user_memory", "arguments": {"window_days": 90}}
OBSERVATION:
TOOL_CALL: {"name": "get_belief_trend", "arguments": {"node_names": ${JSON.stringify(trendNodeNames)}, "window_days": 90}}
OBSERVATION:
HYPOTHESIS: [write here]

Rules:
- Write OBSERVATION: immediately after each TOOL_CALL — stop there.
- Never write the OBSERVATION content yourself.
- HYPOTHESIS is 3–5 sentences of flowing prose, no bullets.
- HYPOTHESIS sentence 1: root cause grounded in memory and${causeLabel ? ` ${causeLabel}` : ' the nodes'} early/mid/late trend.
- HYPOTHESIS sentence 2: mechanistic link — the non-obvious reason cause drives the effect.
- HYPOTHESIS sentence 3: downstream impact on${effectLabel ? ` ${effectLabel}` : ' the affected nodes'}, referencing trajectory direction (↑↓→).
- Nothing after HYPOTHESIS.

Available tools:
${buildReflectHypothesisToolsBlock()}`;

    const userContent =
`USER MESSAGE: ${userMessage}
${causeLabel ? `\nCAUSE NODES: ${causeLabel}` : ''}${effectLabel ? `\nEFFECT NODES: ${effectLabel}` : ''}${!causeLabel && !effectLabel ? `\nNODES: ${nodesToUse.join(', ')}` : ''}

Follow the exact sequence above. After OBSERVATION 2, write HYPOTHESIS.`;

    let prompt = gemmaPrompt(systemContent, userContent);

    const MAX_STEPS = 3;
    for (let step = 0; step < MAX_STEPS; step++) {
      const result = await _ctx!.completion({
        prompt,
        n_predict:   400,
        temperature: 0.2,
        top_p:       0.85,
        stop:        ['<turn|>', '<|turn>', 'OBSERVATION:'],
      });
      const text = cleanGemmaOutput(result.text);
      if (__DEV__) console.log(`[agent] reflect-hypothesis step=${step}/${MAX_STEPS - 1}`, text);

      // Fire onThought for any THOUGHT line the model emitted this step.
      if (onThought) {
        const thoughtLine = text.match(/THOUGHT:\s*(.+)/)?.[1]?.trim();
        if (thoughtLine) onThought(thoughtLine);
      }

      prompt += text + '<turn|>\n';

      const hypMatch = text.match(/HYPOTHESIS:\s*([\s\S]+)/);
      if (hypMatch) return hypMatch[1].trim();

      const toolMatch = text.match(/TOOL_CALL:\s*(\{[\s\S]*?\})(?:\s|$)/);
      if (!toolMatch) return '';

      let call: { name: string; arguments?: Record<string, unknown> };
      try   { call = JSON.parse(toolMatch[1]); }
      catch { return ''; }

      if (__DEV__) console.log(`[agent] hypothesis tool_call: ${call.name}`, call.arguments);

      let observation: unknown;
      if (!REFLECT_HYPOTHESIS_TOOLS.has(call.name)) {
        observation = { error: `Tool '${call.name}' not available in reflect hypothesis mode` };
      } else {
        observation = await dispatchTool(call.name, call.arguments ?? {}, db, '');
      }

      if (__DEV__) console.log(`[agent] hypothesis observation:`, observation);
      prompt +=
        `<|turn>user\nOBSERVATION: ${JSON.stringify(observation)}<turn|>\n` +
        `<|turn>model\n`;
    }
    return '';
  } catch {
    return '';
  }
}
