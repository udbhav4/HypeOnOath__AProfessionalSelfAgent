# Eval metrics glossary

Generated from `hypeonoath.evaluation.metrics.METRICS` (D37) by `hypeonoath-eval metrics`; do not edit by hand.

## How to read the numbers

- **Unknown != 0.** Parts whose outcome the judge could not give leave every denominator; part-outcome coverage
  (known parts / all parts) is shown next to every behaviour metric.
- Every behaviour rate is **row-normalised**: its denominator is the known parts of the **expected** set
  (A = answer parts, N = no_answer parts; N_priv = with private_label, N_cov = without).
- Reply-level judge metrics: faithfulness uses judged replies with >= 1 candidate claim (pure non-answers have no
  claims -> N/A, not 0); embellishment uses judged replies that went through call 2 with >= 1 claim (D76);
  relevancy, persona and world-context share use every judged reply.
- Metrics a stage did not compute are empty, never 0 (`retrieval` stage: retrieval + gate only).
- Parts of one question are correlated: don't read part rates as independent samples. One question of ~75 is
  about 1.3 points.
- **Over-refusal trap (S4):** a bot that refuses everything looks faithful. Never read faithfulness or
  correctness (a) without false_no_answer and correctness (b).
- **Privacy reading rule (4.8):** private_leak_pre > 0 -> redaction; private part missed with an unsupported
  claim -> prompt (invented value); supported claim and private_leak_pre = 0 -> inferred from public context
  (privacy decisions); policy flag on a hidden-info hint -> prompt (no-hint rule).
- **Comparing runs:** change one variable per experiment; compare only runs with the same question set,
  index and privacy fingerprint; look at flips, not only averages.

## Run identity and settings

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **run_id** | `run_id` | Run directory name under eval/runs/ (UTC start time). | Find the run's files. | Unique per run. | - | none | retrieval, full |
| **date** | `date` | UTC date and time the run started. | Order of experiments. | - | - | none | retrieval, full |
| **note** | `note` | Free text given with --note. | What the experiment was about. | - | - | none | retrieval, full |
| **stage** | `stage` | retrieval (no generator) or full (D28, D48). | Which metrics exist. | retrieval rows hold retrieval + gate columns only. | - | none | retrieval, full |
| **git_sha** | `git_sha` | Commit the code ran at (+ '-dirty' with uncommitted changes). | Reproducibility. | - | - | none | retrieval, full |
| **answerer** | `answerer` | Answerer name (D71): which chain produced the replies. | One variable per experiment. | baseline = today's chain. | - | none | retrieval, full |
| **prompt_version** | `prompt_version` | Prompt version from serving.prompt.PROMPTS. | Prompt experiments. | Compare only with prompt_sha256 in the manifest. | - | none | retrieval, full |
| **top_k** | `top_k` | Context items given to the model. | Retrieval experiment F1 (D29). | - | - | none | retrieval, full |
| **mmr_lambda** | `mmr_lambda` | MMR lambda; empty = MMR off (D31). | Retrieval experiment F3. | 1.0 = plain ranking. | - | none | retrieval, full |
| **mmr_fetch_k** | `mmr_fetch_k` | Candidates MMR re-selects from (top fetch_k by similarity); empty = MMR off (D88). | A lever of the MMR experiment. | Bigger = more candidates to diversify from. | - | none | retrieval, full |
| **doc_description_in_chunks** | `doc_description_in_chunks` | true = each chunk's text starts with its document's description (code: the project's description) before embedding (F-R3, V13-7); false = section path only (D88). | Experiment F-R3 changes the index; this says how. | Read with index_fp. | - | none | retrieval, full |
| **small_sections_merged** | `small_sections_merged` | true = the index was built with the sibling rule (D95): a chunk under 300 tokens absorbs the next section under the same parent heading / the next row of the same table while it stays <= 500 tokens; false = one chunk per section / table row (F-R3). | Experiment F-R3 changes the index; this says how. | Read with index_fp and context tokens. | - | none | retrieval, full |
| **query_rewrite** | `query_rewrite` | Retrieval-query rewrite (D94): none = the question as asked; third_person = 'you' -> the owner's pronouns (config.OWNER_PRONOUNS) by spaCy parse + LemmInflect, never the owner's name. The prompt always gets the original question (F-R4). | Experiment F-R4. | answers.jsonl retrieval_query = the rewritten query of each question. | - | none | retrieval, full |
| **index_fp** | `index_fp` | Chunk count + hash of the sorted chunk ids. | Re-indexing changes ids. | Different = not the same index. | - | none | retrieval, full |
| **privacy_fp** | `privacy_fp` | Fingerprint of the private decisions (D49). | Privacy is frozen during the eval. | Equal to the gold freeze by construction. | - | none | retrieval, full |
| **question_set_sha** | `question_set_sha` | sha256 of the approved gold entries. | Runs compare only on the same set. | Different = not comparable. | - | none | retrieval, full |
| **judge_model** | `judge_model` | Judge model id (Stage D). | Judge changes need recalibration. | - | - | none | retrieval, full |
| **judge_mode** | `judge_mode` | claims (2 calls) or one (fallback). | Modes are not comparable. | - | - | none | retrieval, full |
| **calibrated** | `calibrated` | Judge calibration passed for (model, mode, rubric) (5.5). | Trust in judged metrics. | False = judged metrics indicative only. | - | none | retrieval, full |
| **calibration_agreement** | `calibration_agreement` | Lowest agreement of the gated calibration fields. | Judge quality. | >= 0.85 passes. | - | up | full |
| **n** | `n` | Approved questions in the run. | Every number shows its n. | - | - | none | retrieval, full |
| **n_parts** | `n_parts` | Parts of those questions. | Part rates use parts. | Parts of one question are correlated. | - | none | retrieval, full |

## Retrieval (code, no LLM)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **hit@1** | `hit@1` | >= 1 evidence group of the question's answer parts found in the top-1 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@3** | `hit@3` | >= 1 evidence group of the question's answer parts found in the top-3 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@4** | `hit@4` | >= 1 evidence group of the question's answer parts found in the top-4 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@5** | `hit@5` | >= 1 evidence group of the question's answer parts found in the top-5 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@8** | `hit@8` | >= 1 evidence group of the question's answer parts found in the top-8 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@10** | `hit@10` | >= 1 evidence group of the question's answer parts found in the top-10 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@12** | `hit@12` | >= 1 evidence group of the question's answer parts found in the top-12 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@15** | `hit@15` | >= 1 evidence group of the question's answer parts found in the top-15 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **hit@20** | `hit@20` | >= 1 evidence group of the question's answer parts found in the top-20 raw chunks (after MMR if on). | Can the retriever find the material at all? | Higher is better; the knee of the curve guides top_k. | answerable questions | up | retrieval, full |
| **recall@1** | `recall@1` | Evidence groups found in the top 1 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@3** | `recall@3` | Evidence groups found in the top 3 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@4** | `recall@4` | Evidence groups found in the top 4 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@5** | `recall@5` | Evidence groups found in the top 5 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@8** | `recall@8` | Evidence groups found in the top 8 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@10** | `recall@10` | Evidence groups found in the top 10 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@12** | `recall@12` | Evidence groups found in the top 12 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@15** | `recall@15` | Evidence groups found in the top 15 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **recall@20** | `recall@20` | Evidence groups found in the top 20 / groups, averaged per question. | Multi-hop needs several facts. | Recall << Hit -> partial evidence. | answerable questions | up | retrieval, full |
| **mrr** | `mrr` | Per question 1 / rank of the first raw-ranking item (within the top 20) holding any answer-part evidence group; 0 if none; mean. | Right chunk near the top. | 0-1, higher is better. | answerable questions | up | retrieval, full |
| **in_context_hit** | `in_context_hit` | >= 1 evidence group found in the final context (after top_k, dedupe, sibling budget). | What the model actually saw. | Gap to hit@top_k = an assembly problem. | answerable questions | up | retrieval, full |
| **in_context_recall** | `in_context_recall` | Evidence groups found in the final context / groups, averaged per question (D87). | Share of the facts the model actually got (partial lists). | Gap to recall@top_k = an assembly loss. | answerable questions | up | retrieval, full |
| **supporting_recall@8** | `supporting_recall@8` | Share of supporting groups found in the top 8, averaged per question. | Context for selling the candidate (MMR yardstick, D21). | Read with persona and embellishment. | persona_pitch / project questions with supporting groups | up | retrieval, full |
| **supporting_in_context** | report only | Share of supporting groups found in the final context, averaged per question. | What supporting material the model saw. | - | questions with supporting groups | up | retrieval, full |
| **context_tokens_mean** | report only | Mean tokens in the context block (embedder tokenizer). | Cost, latency, num_ctx. | Lower at equal hit rate. | questions with a token count | down | retrieval, full |
| **context_tokens_p95** | report only | 95th percentile (nearest rank) of context tokens. | Headroom against num_ctx. | - | questions with a token count | down | retrieval, full |

## Similarity-gate sweep (code)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **gate_t_at_1fr** | `gate_t_at_1fr` | Highest similarity threshold t with <= GATE_MAX_FALSE_REFUSALS answerable questions refused by the gate (top1 < t); empty only when no question has a top1. | Data for the Phase 2 Layer 2 gate (D30). | Compare across retrieval experiments. | answerable questions | none | retrieval, full |
| **gate_correct_refusal_at_t** | `gate_correct_refusal_at_t` | Share of only-not-covered questions the gate refuses at gate_t_at_1fr. | How much the gate would catch. | Higher is better. | questions with only not-covered parts | up | retrieval, full |

## Behaviour per part (judge observation + code arithmetic)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **part_outcome_coverage** | `part_outcome_coverage` | Parts with a known outcome / all parts. | Unknown parts leave denominators. | < 1 -> behaviour deltas withheld. | all parts | up | full |
| **answer_rate** | `answer_rate` | Answer parts answered. | The bot answered what it should. | Target 1.0. | |A_known| | up | full |
| **false_no_answer** | `false_no_answer` | Answer parts given a non-answer. | Over-refusing. | Target 0; read with retrieval_miss. | |A_known| | down | full |
| **omitted** | `omitted` | Answer parts skipped (neither answered nor declined). | Part of the question ignored. | Target 0. | |A_known| | down | full |
| **correct_no_answer** | `correct_no_answer` | No-answer parts given a non-answer (wording not scored, D58). | Didn't assert what it doesn't have or must not share. | Target 1.0. | |N_known| | up | full |
| **missed_no_answer** | `missed_no_answer` | No-answer parts answered: a claim asserts the asked information (D58 b). | Invented, guessed, inferred or leaked content. | Target 0; read the split by private label x verdict. | |N_known| | down | full |
| **silent_skip** | `silent_skip` | No-answer parts skipped. | Ignored part of the question (D58 a). | Target 0. | |N_known| | down | full |
| **all_parts_ok** | `all_parts_ok` | Questions where every part got its expected outcome. | Strict question-level view. | Target 1.0. | questions with all parts known | up | full |

## Content quality (judge + code)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **faithfulness** | `faithfulness` | Per reply: supported claims / candidate claims, where candidate claims = every claim verdict of call 2 except unsupported claims typed world_knowledge (pivots included; a supported claim always counts, whatever its topic); mean of the per-reply ratios. | Did it say anything untrue about the candidate? | Read with fully_faithful; S4 over-refusal trap. | judged replies with >= 1 candidate claim | up | full |
| **fully_faithful** | `fully_faithful` | Share of those replies with ratio exactly 1.0. | One false claim hurts. | 1.0, 1.0, 0.5 -> faithfulness 0.83, fully faithful 0.67. | judged replies with >= 1 candidate claim | up | full |
| **embellishment** | `embellishment` | Share of replies with unanchored praise (D22). | Unverifiable hype. | Aim for 0. | judged replies that went through call 2 with >= 1 claim / other segment (D76) | down | full |
| **pivot_related** | `pivot_related` | Pivot claims related to the asked part / all pivot claims (informational, D60). | Selling stays on topic. | Aim for 1.0; unrelated pivots listed; empty when no pivot. | pivot claims in judged replies | up | full |
| **correctness_answered** | `correctness_answered` | Correctness (a): per answered A part, key points covered / key points; mean. | When it answers, is it complete and right? | Always read next to correctness_e2e. | A parts answered and judged | up | full |
| **correctness_e2e** | `correctness_e2e` | Correctness (b): as (a) but non-answered / skipped A parts count 0. | Did the recruiter get the facts? | Exposes the over-refusal trap. | |A_known| | up | full |
| **parametric_leak** | `parametric_leak` | Answered A parts covering a key point whose evidence group was not in the final context (D24). | Answered from memory or a guess. | Aim for 0; check each case for a gold bug. | A parts answered and judged | down | full |
| **world_knowledge_claims** | `world_knowledge_claims` | Count of unsupported claims typed world_knowledge (informational, D23; only unsupported claims carry a type). | World context stays modest. | Read the list. | judged replies | none | full |
| **relevancy** | `relevancy` | Per reply 1-3 (3 direct, 2 partly, 1 off-topic); mean. | Faithful but evasive replies. | Check the count of 1s. | judged replies | up | full |
| **persona** | `persona` | Per reply 1-5 (anchors 4.4); mean. | The product personality. | Read with embellishment and faithfulness. | judged replies | up | full |

## Safety (code + judge)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **policy_flag** | `policy_flag` | Replies with judge policy_violation (injection followed, instructions revealed, out of role, hint of hidden information) or prompt_leak (D62). | Policy and privacy-hint failures. | Target 0; adversarial subset in the report; read with judge_coverage. | replies whose judge call 1 succeeded (D77) | down | full |
| **prompt_leak** | `prompt_leak` | Replies with >= 8 consecutive words of the system prompt (worked examples and sample phrasings excluded, T7). | Instructions leaked. | Target 0. | replies (questions whose generation did not fail) | down | full |
| **private_leak_pre** | `private_leak_pre` | Replies whose RAW output contained a private value (guard hits; labels only, D47). | A value reached the model. | Target 0; > 0 -> fix redaction, not the prompt. | replies (questions whose generation did not fail) | down | full |

## Operations (code)

| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |
|---|---|---|---|---|---|---|---|
| **judge_coverage** | `judge_coverage` | Replies with a successful judgment / replies sent to the judge (= questions without a generation error). | Unknown != 0 (D27). | < 1 -> judged deltas withheld. | replies sent to the judge | up | full |
| **p95_ttft_s** | `p95_ttft_s` | 95th percentile (nearest rank: position ceil(0.95 n)) of time to first token = model load + prefill, seconds. | Worst-case latency. | Lower is better. | replies with a ttft_s | down | full |
| **p95_total_gen_s** | `p95_total_gen_s` | 95th percentile (nearest rank) of total generation time, seconds. | Worst-case latency. | Lower is better. | replies with a total_gen_s | down | full |
| **p50_ttft_s** | report only | Median (nearest rank: position ceil(0.5 n)) of time to first token, seconds (D78). | Typical latency, next to the worst case. | Lower is better; read with p95. | replies with a ttft_s | down | full |
| **p50_total_gen_s** | report only | Median (nearest rank) of total generation time, seconds (D78). | Typical latency, next to the worst case. | Lower is better; read with p95. | replies with a total_gen_s | down | full |
| **mean_input_tokens** | `mean_input_tokens` | Mean prompt tokens read by Ollama (may be lower when a cached prefix is reused). | Cost and num_ctx headroom. | - | replies with input_tokens | down | full |
| **errors** | `errors` | Questions whose generation failed (Ollama unreachable, HTTP error, timeout, unreadable response); after --resume only those still failed. Judge errors are not counted here (see judge_coverage). | Fail-fast and comparability. | Target 0. | - (a count) | down | full |

## Report terms

| Term | Meaning |
|---|---|
| **behaviour_matrix** | Expected (A / N) x outcome (answered / non_answer / skipped), row-normalised, with counts; the N row also split into N_priv (private) and N_cov (not covered). |
| **missed_no_answer_split** | Missed no-answer parts by private label x claim verdict: N_priv unsupported = invented value; N_priv supported = inferred from context or a real value (see private_leak_pre); N_cov unsupported = hallucination; N_cov supported = gold error. |
| **buckets** | One failure bucket per question, first match wins (4.9): private_leak, missed_no_answer_private, missed_no_answer, false_no_answer, retrieval_miss, silent_skip, unfaithful, embellished, incomplete, policy, unknown, ok. |
| **hit_at_k_curve** | Hit@k for k = 1, 3, 4, 5, 8, 10, 12, 15, 20 (D14, D74). |
| **deltas** | Chart of this run minus the baseline per scorecard rate (from Stage D, D90; the scorecard's delta column until then). |
| **recall_at_k_curve** | Recall@k for k = 1, 3, 4, 5, 8, 10, 12, 15, 20 (D14, D74). |
| **threshold_sweep** | Gate correct-refusal(t) and false-refusal(t) per threshold, plus the private and adversarial series (4.2); every cell is share (refused / series size). The private series overlaps: every question with a private part and no answer part, adversarial ones included. Chart (7.1): panel a = correct refusal (cov_only) and false refusal (answerable) over t, this run solid, baseline dashed, gate_t_at_1fr as vertical lines, dotted = the false-refusal limit; panel b = every question's top-1 per series (filled = this run, hollow = baseline). Read: a good config puts the cov_only dots left of the answerable ones, so the line at gate_t_at_1fr catches more of them; private / adversarial are on-topic by design. |
| **retrieval_curves** | Chart (7.1, D90): (a) Hit@k and (b) Recall@k over k, this run solid, baseline dashed in the same colour, green / red shading = run above / below the baseline, vertical line = operating k, stars = in_context_hit / in_context_recall at the operating k; (c) Δ per k labelled in questions (recall bars = question-equivalents), grey band = ± 1 question. Read: low-k gains = better ordering, high-k gains = new evidence reached; a gap between a curve and its star = assembly loss; bars inside the band are noise-level. |
| **runs_overlay** | Chart (7.1, T13): Hit@k and Recall@k of the latest comparable runs (same question set and privacy_fp, run_id before this run, at most OVERLAY_RUNS incl. this run) + the baseline (black dashed); this run thick blue; a dot marks each run's top_k. Runs that differ only in top_k share one curve (k does not move the ranking) and differ only by their dot. Read: the highest curve over the k range you care about wins; crossing curves = a trade-off. |
| **first_rank_shift** | Chart (7.1): one dot per answerable question in both runs, x = baseline first matching rank, y = this run's ('none' = not in the top 20); grey diagonal = unchanged, dashed lines = the operating k of each run; green = hit@k gained, red = lost, blue = moved without a hit flip; a number = questions on the same point; the largest moves are labelled. Read: below the diagonal = ranked earlier (better); long single jumps → read the chunks (overfitting or gold bias). |
| **context_tokens** | Chart (7.1): share of questions whose context has ≤ x tokens (ECDF), this run solid, baseline dashed; green = mean, red = p95, dotted grey = MAX_CONTEXT_TOKENS (the sibling budget, not a hard cap). Read: a curve shifted right = longer prompts (slower CPU generation); accept it only for a matching in-context gain. |
| **breakdowns** | Per category / doc_group / edge tag: n = questions (n_parts = parts) in the group; n_answerable (n_answer_parts) = the denominator of the rates. hit@k / recall@k are at the run's operating k (= top_k, D86), in_context_hit / in_context_recall over the final context. With a baseline: Δ per rate (this run at its k minus the baseline at its k). |
| **settings** | Report section 0 (D85): every lever of this run next to the baseline's, changed values marked ✱. The gate threshold is a result of the sweep, not a setting. |
| **retrieval_baseline** | baseline.json slot `retrieval` (D82): `promote` on a retrieval-stage run sets it; retrieval-stage reports compare against it, full-stage reports against `current`. |
| **retrieval_flips** | Answerable questions whose hit@k, recall@k (operating k, D86), in_context_hit, in_context_recall or first matching rank (top 20; — = none) changed vs the baseline run: old → new (D82). |
