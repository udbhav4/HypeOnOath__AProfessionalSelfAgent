---
date: 20.09.2026
project_name: null
---

# System Design Document — Recruiter-Facing Grounded RAG Chatbot

**Status:** v1 (derived from `plans/plan2.md`)
**Author context:** Solo build, first large end-to-end project, $0 budget, target: live within weeks.
**Source of truth for decisions below:** `plans/plan2.md`. Where this document adds structure not explicit in that plan (e.g., data flow contracts, failure modes, non-functional targets), it is marked **[Derived]**. Where a detail is explicitly unverified in the source plan, it is marked **[Assumption]**.

---

## 1. Purpose & Scope

### 1.1 Problem statement
Recruiters need a conversational interface to ask questions about a candidate (projects, experience, skills, "why hire for role X") and receive answers that are:
- Grounded strictly in the candidate's real source documents (resume, READMEs, transcripts, blog posts, code samples).
- Cited to their source.
- Honest about not knowing — the system must refuse rather than speculate when documents don't cover a question.

### 1.2 Goals
| Goal | Target | Source |
|---|---|---|
| Ship a working, honest MVP fast | Live within weeks | plan2.md:5 |
| Zero infrastructure cost | ~$0 running cost | plan2.md:5 |
| Learning value | Core RAG logic hand-written, not framework-orchestrated | plan2.md:5, 16 |
| Grounding correctness over completeness | Refuse rather than hallucinate | plan2.md:5, Section 7 below |
| Portfolio quality | Public GitHub repo recruiters may inspect directly | plan2.md:60 |

### 1.3 Non-goals (v1)
| Non-goal | Reasoning | Source |
|---|---|---|
| Multi-tenant SaaS (hosting other people's bots) | Distinct product surface — auth, billing, per-tenant isolation not addressed | plan2.md:118 |
| GraphRAG / full entity-relationship graph pipeline | Unproven need; single/two-chunk lookups dominate expected query patterns | plan2.md:66-80 |
| Conversation memory (multi-turn context) | Open item, deferred to deep-dive phase | plan2.md:135 |
| Streaming token-by-token responses | Not part of MVP request/response design | **[Derived]** — no streaming transport specified in plan2.md's architecture diagram |

---

## 2. Architecture Overview

### 2.1 High-level data flow

```
┌─────────────────────────────── OFFLINE / INGESTION (local, free) ───────────────────────────────┐
│                                                                                                     │
│  [Source documents]                                                                                │
│   resume, READMEs, transcripts,                                                                    │
│   blog posts, code samples                                                                         │
│         │                                                                                           │
│         ▼                                                                                           │
│   Clean & normalize (text/Markdown)                                                                │
│         │                                                                                           │
│         ▼                                                                                           │
│   Block-level content-type-aware chunking (Section 5.1, v1.3)                                      │
│         │                                                                                           │
│         ▼                                                                                           │
│   Embed chunks (local model, via Ollama or sentence-transformers)                                  │
│         │                                                                                           │
│         ▼                                                                                           │
│   Persist to Chroma (on-disk vector store)                                                         │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────── ONLINE / SERVING (deployed, free tier) ─────────────────────────┐
│                                                                                                     │
│  Recruiter (user)                                                                                  │
│         │  HTTP request                                                                            │
│         ▼                                                                                           │
│   FastAPI backend  ── /chat endpoint                                                                │
│         │                                                                                           │
│         ▼                                                                                           │
│   Embed incoming query (SAME embedding model as ingestion — see 2.2)                                │
│         │                                                                                           │
│         ▼                                                                                           │
│   Retrieve top-k chunks from Chroma                                                                │
│         │                                                                                           │
│         ▼                                                                                           │
│   Layer 2 gate: similarity-threshold check (Section 7) ──► below threshold → canned refusal, STOP  │
│         │  (above threshold)                                                                        │
│         ▼                                                                                           │
│   Assemble prompt (system rules + retrieved chunks + question)                                      │
│         │                                                                                           │
│         ▼                                                                                           │
│   Call hosted LLM API (Gemini Flash primary / Groq fallback)                                        │
│         │                                                                                           │
│         ▼                                                                                           │
│   Layer 3 gate: post-hoc groundedness check (Section 7, Phase 5 only)                               │
│         │                                                                                           │
│         ▼                                                                                           │
│   Parse answer + attach citations (source doc + chunk)                                              │
│         │                                                                                           │
│         ▼                                                                                           │
│   Return to frontend chat UI                                                                        │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```
Source: plan2.md:27-41, extended with Layer 2/3 gate placement per plan2.md:100-104.

### 2.2 Architectural invariants (must not be violated by any implementation)

| Invariant | Why it exists | Consequence if violated | Source |
|---|---|---|---|
| Query embedding model MUST equal document embedding model | Mismatched embedding spaces silently degrade retrieval — vectors from two different models are not comparable | Retrieval returns semantically irrelevant chunks with no error thrown — a "silent" failure, hardest kind to debug | plan2.md:43 |
| LLM receives ONLY retrieved chunks + question — never free "knowledge about the candidate" | Grounding is enforced structurally (what enters the prompt), not by instruction alone | Model can answer from parametric memory/hallucination even with a "be honest" system prompt | plan2.md:43 |
| Ingestion (offline) and serving (online) are separate pipelines/processes | Document corpus changes over months/years; querying happens continuously | Coupling them forces a full redeploy or restart on every document edit | plan2.md:20 |

---

## 3. Component Design

### 3.1 Component inventory

| Component | Responsibility | Technology | Runs where | Source |
|---|---|---|---|---|
| Document store | Raw source material (resume, READMEs, transcripts, blog posts, code) | Flat files in a version-controlled folder | Local / Git repo | plan2.md:112 |
| Ingestion pipeline | Clean → chunk → embed → index | Plain Python scripts + `tree-sitter` (code parsing only) | Local (offline, batch) | plan2.md:52 |
| Embedding model | Turns text (docs and queries) into vectors | Ollama (`nomic-embed-text`) OR `sentence-transformers` (`all-MiniLM-L6-v2`) | Local for ingestion; **also invoked inside backend at query time** | plan2.md:53, 35 |
| Vector store | Persisted, queryable chunk embeddings + metadata | Chroma (embedded, file-based) | On-disk, alongside backend deployment | plan2.md:54, 18 |
| Backend API | Orchestrates retrieval → prompting → generation → response | FastAPI | Render (or Railway) free tier | plan2.md:51, 58 |
| Production LLM | Generates the recruiter-facing answer | Gemini Flash (primary), Groq/Llama 3.3 (fallback) | Hosted, external API call | plan2.md:55 |
| Dev/local LLM | Prompt iteration without burning hosted quota | Ollama (e.g., Llama 3.1 8B) | Local | plan2.md:56 |
| Frontend | Chat interface: input, message list, citations | Plain React (Vite) or plain HTML/JS | Static hosting (Vercel/Netlify) or served by FastAPI | plan2.md:57, 59 |
| Version control | Source + history, portfolio artifact | Git + GitHub | — | plan2.md:60 |

### 3.2 Component interaction contract **[Derived]**

This is not explicit in plan2.md as a formal API, but is implied by the architecture diagram (plan2.md:34-40) and is made explicit here so any reader — including a future collaborator — has an unambiguous contract to implement against.

**`POST /chat`**

| Field | Direction | Type | Notes |
|---|---|---|---|
| `question` | request | string | Input length must be capped (plan2.md:22) — exact limit is an open item (plan2.md:136) |
| `answer` | response | string | Model-generated, grounded text |
| `sources` | response | list of `{document, chunk_id, snippet}` | Populated only when retrieval clears the Layer 2 threshold |
| `refused` | response | boolean | True when Layer 2 or Layer 3 blocks an answer |

---

## 4. Technology Decisions (Architecture Decision Records)

Each row is a discrete decision. Format: Decision → Alternatives considered → Why this choice → Trade-off accepted.

| # | Decision Area | Chosen | Alternatives Considered | Why Chosen | Trade-off Accepted | Source |
|---|---|---|---|---|---|---|
| ADR-1 | Backend framework | FastAPI | (none named — direct fit) | Async-native, minimal boilerplate, author already knows it, standard for this service shape | None significant | plan2.md:51 |
| ADR-2 | Orchestration approach | Raw Python, no framework | LangChain, LlamaIndex | Explicit learning goal — chunking/embedding/retrieval/prompting are each small enough (~20-50 LOC) to hand-write and understand fully | More code to write and maintain yourself; must revisit if multi-step agentic tool use emerges | plan2.md:16 |
| ADR-3 | Embedding execution location | Local (Ollama or sentence-transformers) | Hosted embedding API | Embedding is a one-time/occasional offline batch job — zero cost to run locally; also avoids a second hosted-API dependency | Requires local compute at ingestion time; must guarantee query-time embedding also runs the same model (Invariant, Section 2.2) | plan2.md:17, 53 |
| ADR-4 | Generation model location | Hosted (Gemini Flash / Groq) | Fully local (Ollama in production) | A model on a personal laptop cannot reliably serve public internet traffic without 24/7 uptime and port-forwarding — unprofessional and unstable | Dependent on third-party free-tier availability and limits, which change over time | plan2.md:17, 62 |
| ADR-5 | Vector store | Chroma | FAISS, pgvector, Pinecone/Weaviate (cloud) | FAISS: no built-in metadata/persistence, more plumbing for no learning benefit. pgvector: needs a running Postgres instance, unnecessary ops overhead at this scale. Pinecone/Weaviate: hosted, adds external account dependency and vendor lock-in for no benefit at this scale. Chroma is simple enough to fully understand, persists to disk, genuinely industry-used | Less horizontally scalable than a hosted vector DB — acceptable, this is a personal-scale corpus | plan2.md:18 |
| ADR-6 | Code chunking | AST-based via `tree-sitter`, function/class-level | Fixed-size chunking (same as prose) | Fixed-size chunking on code routinely cuts a function in half, breaking retrieved logic flow — a documented failure mode per RepoEval/SWE-bench-style research **[Assumption: specific benchmark figures not independently re-verified by this document, carried from plan2.md]** | Adds a parser dependency (`tree-sitter`) — accepted since it is a parsing library, not an orchestration framework, and does not replace hand-written chunk-boundary logic | plan2.md:92-94 |
| ADR-7 | Multi-hop reasoning approach (v1) | Wider top-k retrieval across chunk types | Full GraphRAG (entity/relationship graph + community detection) | GraphRAG benchmarks show large multi-hop gains (~86% vs ~32% cited) but at real extra build cost (entity extraction, graph construction, community summarization) and non-incremental update cost (adding a doc can require recomputing summaries). Most recruiter questions are single-/two-chunk lookups where this gap doesn't show up | If wider top-k proves insufficient for "why hire you" style questions, plan calls for a lightweight hand-built entity-link layer before considering full GraphRAG — not yet built, tracked as an open item | plan2.md:66-80, 137 |
| ADR-8 | Hosting (backend) | Render free tier (Railway as alternative) | (implicit: self-hosting) | Real, no-credit-card-required free tier for a Python web service | Cold-start delay of 30-60s after idle — accepted as a known MVP limitation | plan2.md:58 |
| ADR-9 | Frontend framework | Plain React (Vite) or plain HTML/JS | A styled/component framework (e.g., pre-built UI kit) | MVP UI is deliberately minimal — design investment withheld until retrieval/grounding ("the brain") is proven | Plainer visual polish at launch — explicitly accepted, not a gap | plan2.md:57 |

**[Assumption]** All free-tier claims (Gemini, Groq, Render, no credit card required) are dated "as of 2026" in the source plan and are explicitly flagged there as subject to change — this document inherits that same flag and does not re-verify current terms. (plan2.md:55, 58, 62)

---

## 5. Data Design

### 5.1 Content types and chunking strategy

The corpus is **not homogeneous** — three distinct content types require three distinct chunking rules. A single fixed-size chunking rule applied uniformly is explicitly rejected as insufficient (plan2.md:94).

**Terminology update [2026-09-13, session discussion — see Change Log v1.2]:** the content type originally named "Transcripts" below is renamed `structured_doc` to reflect that the chunking rule applies to any record-structured document (transcripts, certifications, tabular data), not transcripts specifically. The original table is preserved immediately below as an **earlier decision**, per this document's change-log convention; the current, renamed version follows it.

**Earlier decision (superseded, kept for history):**

| Content Type | Chunking Method | Rationale | Failure mode if using generic fixed-size chunking instead |
|---|---|---|---|
| Prose (resume, blog posts, README prose) | Structure-aware: split on headings/sections first, then fixed-size (~300-500 tokens) within long sections, ~10-15% overlap | Headings are free, natural semantic boundaries; overlap prevents a claim being cut across a chunk edge | Acceptable but not ideal — claims can still be split without overlap |
| Transcripts | Turn/entry-aware: one chunk per course entry or logical record | Transcripts are already discrete records (course, grade, term) | Mid-record splitting can separate a grade from its course name — the exact fact needing citation is lost |
| Code samples | AST-based via `tree-sitter`: one chunk per function/class, merge trivially small chunks, split only unusually large functions | Preserves complete, syntactically valid logic units | Fixed-size cuts routinely bisect a function, returning a broken, logic-incomplete fragment |

Source: plan2.md:88-94.

**Current (v1.2):**

| Content Type | Chunking Method | Rationale | Failure mode if using generic fixed-size chunking instead |
|---|---|---|---|
| Prose (resume, blog posts, README prose) | Structure-aware: split on headings/sections first, then fixed-size (~300-500 tokens) within long sections, ~10-15% overlap | Headings are free, natural semantic boundaries; overlap prevents a claim being cut across a chunk edge | Acceptable but not ideal — claims can still be split without overlap |
| **`structured_doc`** (e.g., academic transcripts, structured certifications, tabular records) | Turn/entry-aware: one chunk per record | These documents are already discrete records (e.g., course, grade, term) | Mid-record splitting can separate a fact from its identifying label (e.g., a grade from its course name) — the exact fact needing citation is lost |
| Code samples | AST-based via `tree-sitter`: one chunk per function/class, merge trivially small chunks, split only unusually large functions | Preserves complete, syntactically valid logic units | Fixed-size cuts routinely bisect a function, returning a broken, logic-incomplete fragment |

Source: plan2.md:88-94, renamed per `plans/phase0-document-prep-subplan-v2.md` Section 4.2 (session discussion 2026-09-13).

**Superseded 2026-09-19 — see Current (v1.3) below.** The v1.2 table above assigns `content_type` per *document* (a whole file is either `prose` or `structured_doc`). Reasoning through a real corpus file (`system-design-rag-chatbot.md` — this document itself, added to `corpus/raw/`) exposed that this is the wrong grain: that file mixes prose sections with exposition tables (167 lines of table syntax used for ADR/comparison tables, not record data), and no per-document scoring rule can correctly resolve a file that is genuinely heterogeneous. Full rationale in `plans/phase0-document-prep-subplan-v5.md`, Section 3C.

**Current (v1.3, 2026-09-19) — per-block typing, decided at chunk time (Phase 1), not per-document (Phase 0):**

A document is treated as a **sequence of typed blocks** (the standard approach used by mature document-parsing/RAG tooling, e.g. the `unstructured` library's element partitioning, LlamaParse) — the chunking rule attaches to each block's type, not to the file as a whole:

| Source type | Block-partitioning approach | Per-block chunking rule | Library |
|---|---|---|---|
| PDF / DOCX (post `docling` conversion, Section 5.4) | Reuse `docling`'s own already-typed internal document model (`TextItem`, `TableItem`, headings) directly, rather than re-parsing the flattened Markdown output | Table block → one chunk (or one chunk per row if the table is a genuine record list); prose block → structure-aware, grouped by heading, fixed ~300–500 tokens + ~10-15% overlap (same rule as v1.2's prose row, now applied per-block instead of per-document) | `docling` |
| Native `.md`/`.txt` (README, blog posts, design docs) | Walk a real Markdown block-token stream (heading, paragraph, table, list, fenced code), not regex-based whole-document detection | Same per-block rules as above; fenced code blocks kept intact, never split mid-block | `markdown-it-py` — **[Assumption]** proposed, not yet verified against a real file in this codebase |
| Code samples | Unchanged — not Markdown, no block-typing step needed | AST-based via `tree-sitter`: one chunk per function/class (unchanged from ADR-6 / v1.2's code row) | `tree-sitter` |

**Why this is a correction, not just a relabeling:** the v1.2 rule was fully automatic and well-reasoned for a homogeneous file, but real documents are usually *not* homogeneous — prose interleaved with tables/lists/code is the norm once documents beyond simple transcripts enter the corpus. Per-block typing fixes this at its actual source (the chunker), rather than trying to patch a per-document scoring rule that structurally cannot describe a mixed file correctly.

Source: `plans/phase0-document-prep-subplan-v5.md`, Section 3C (session discussion 2026-09-19).

### 5.2 Chunk metadata schema **[Derived]**

Implied by the citation requirement (plan2.md:39, 102) but not explicitly schematized in the source plan — made explicit here:

| Field | Purpose |
|---|---|
| `source_document` | Which file the chunk came from (for citation) |
| `content_type` | **Scope changed 2026-09-19 (v1.3):** now a **per-chunk** field (e.g. `prose` / `table` / `record` / `code`), assigned by the Phase 1 chunker's block-level typing (Section 5.1's Current (v1.3) design) — no longer a per-document field assigned in Phase 0. Earlier values (`prose`/`structured_doc`/`code`, assigned per-document in Phase 0) are superseded; renamed from `transcript` to `structured_doc`, 2026-09-13, before this later scope change — see Section 5.1 and Change Log v1.2/v1.3 |
| `project_name` (where applicable) | Groups chunks belonging to the same project across doc types |
| `date` | For provenance / freshness |
| `chunk_id` | Unique identifier for citation linking |

Source for the requirement (not the field list, which is derived): plan2.md:112 ("tag each with metadata (source type, project name, date)"). Renaming source: `plans/phase0-document-prep-subplan-v2.md` Section 4.2.

### 5.4 Document format conversion (pre-chunking) **[Derived — added 2026-09-13, session discussion]**

Not addressed in `plan2.md` or in this document's v1: source documents are not guaranteed to arrive in Markdown/plain text. Non-Markdown formats must be converted *before* chunking (Section 5.1) can run, since it depends on real heading syntax and clean text structure.

**Earlier decision (v1.2, superseded 2026-09-19 — kept for history):**

| Source format | Conversion approach | Why this approach (vs. plain-text extraction) |
|---|---|---|
| PDF | `pymupdf4llm` — extracts text using font-size/bold/position metadata and reconstructs heading levels (`#`, `##`) directly into Markdown output | PDF headings are visual (font size/weight/position), not syntactic — a plain-text extractor (e.g., naive `pypdf` text dump) flattens all structure, destroying the heading boundaries Section 5.1's prose-chunking rule depends on |
| DOCX / other structured formats | An equivalent structure-preserving converter (e.g., `pandoc`) rather than a plain-text extractor | Same reasoning as PDF — preserve heading/structure semantics through conversion, not just raw characters |

**[Assumption flagged]:** conversion tool output accuracy on this project's specific documents (resume/transcript formatting) is not independently verified here — spot-check output against 1-2 real files before trusting it on the full corpus.

Source: `plans/phase0-document-prep-subplan-v2.md` Step 3.1 (session discussion 2026-09-13, not in `plan2.md`).

**Current (v1.3, 2026-09-19) — library changed to `docling`, plus an added hyperlink-extraction pass:**

During real implementation (`plans/phase0-document-prep-subplan-v5.md`, carried from v4), `pymupdf4llm` was found to misjudge a text region in a real PDF (`CV_Udbhav.pdf`) as a table, swallowing a heading and concatenating words; testing `pymupdf4llm`'s own `table_strategy` parameter in all three available values did not fix it. Both PDF and DOCX conversion now use **`docling`** instead — one library for both formats, chosen for its dedicated table-structure recognition model, which directly targets that failure category. Confirmed against the real file: the previously-swallowed heading now renders correctly as its own heading.

A second, separate defect was found after that fix: `docling` groups text into layout-based chunks coarser than a PDF's own word-level link annotations, so clickable links (e.g. "LinkedIn", "GitHub") rendered as plain text with the URL silently dropped. Fix: every converted document now also gets an **independent, always-on hyperlink-extraction pass** (PyMuPDF for PDF link annotations, `python-docx` for DOCX hyperlink relationships), bypassing `docling` entirely for this, appended as a `## Links` section at the end of the converted Markdown.

**New recommendation, not yet verified (2026-09-19):** alongside the flattened `.md` output, also persist `docling`'s own structured document object (its typed block model — headings, `TextItem`, `TableItem`, etc.), so Phase 1's block-level chunker (Section 5.1's Current (v1.3)) can consume typed blocks directly instead of re-parsing flattened Markdown text and re-deriving structure `docling` already computed once.

Source: `plans/phase0-document-prep-subplan-v5.md`, Step 3.3 and Section 3B (real implementation and testing, 2026-09-15 through 2026-09-19).

### 5.5 Content-type detection — moved from a Phase 0 document-level step to a Phase 1 chunk-level step **[Derived — added 2026-09-13, session discussion; superseded 2026-09-19]**

**Earlier decision (v1, superseded):** this document's v1 did not specify a detection mechanism; Phase 0 sub-plan v1 (`plans/phase0-document-prep-subplan.md`) resolved this by manual folder sorting — a human decides prose vs. transcript by placing each file in `corpus/prose/` or `corpus/transcripts/`.

**Earlier decision (v1.2, superseded 2026-09-19 — kept for history):**

| Aspect | Design | Why |
|---|---|---|
| Classification mechanism | After Section 5.4's conversion step, a detection script scans each document for structured-record signals (e.g., repeated `Course:`/`Grade:`/`Term:`-style labels, tabular/record-like repetition). If found → tag `content_type: structured_doc`; otherwise → tag `content_type: prose`. The script writes the tag directly into the document's front-matter — no manual folder-sorting step | Fully removes the manual per-file classification step from Phase 0 sub-plan v1. Detection is heuristic (regex/pattern-based), not ML — appropriate at personal-corpus scale and keeps the "hand-written, no framework" learning goal intact (consistent with ADR-2), while still being a genuine algorithmic decision rather than manual sorting |
| Folder structure implication | `corpus/` no longer needs separate `prose/`/`structured_doc/` folders — a single flat `corpus/documents/` folder is used, since `content_type` now lives in each file's front-matter, assigned by the script, rather than being encoded by manual folder placement | Folder placement was never a functional requirement — only the metadata tag is (Section 5.2) |

Source: `plans/phase0-document-prep-subplan-v2.md` Section 3 (Steps 3.5) (session discussion 2026-09-13, not in `plan2.md`).

**Current (v1.3, 2026-09-19) — detection step removed from Phase 0 entirely; superseded by Section 5.1's per-block typing:**

Adding `system-design-rag-chatbot.md` (this document itself) to the real corpus and reasoning through how the v1.2 detection script above would score it surfaced the actual problem: the file has 167 lines of Markdown table syntax, but the tables are exposition tables (ADR trade-off comparisons), not record data — the v1.2 rule's "table presence → `structured_doc`" branch would have mislabeled the whole file. No threshold fix resolves this, because the file is genuinely mixed prose-and-tables, and a document-level label cannot correctly describe a document that isn't homogeneous.

**Resolution:** the detection *step* (the v1.2 script above, and the front-matter field it wrote) is removed from Phase 0 entirely — not replaced by a corrected document-level rule. Content-type detection now happens as part of chunking itself, at the block level, in Phase 1 — see Section 5.1's Current (v1.3) design for the mechanism. `corpus/documents/` in Phase 0 carries no `content_type` field in its front-matter (`plans/phase0-document-prep-subplan-v5.md`, Section 4.1); the field reappears only later, per-chunk, in Phase 1's output.

Source: `plans/phase0-document-prep-subplan-v5.md`, Section 3C (session discussion 2026-09-19).

### 5.3 Data lifecycle

| Stage | Trigger | Process | Source |
|---|---|---|---|
| Initial ingestion | One-time, Phase 0/1 | Collect → clean → chunk → embed → store | plan2.md:112-113 |
| Re-ingestion | Any time source documents change | Re-run the same indexing script (repeatable, not manual) | plan2.md:20 |
| Sensitive-content filtering | Before **every** indexing pass that includes new/changed documents | Manual review pass for anything not fit for public exposure (addresses, phone numbers, private grades, unfinished project details) | plan2.md:21, 112 |

**Design principle [Derived from plan2.md:20]:** the document corpus is treated as *versioned data*, not a one-time upload — the ingestion pipeline is a standing, re-runnable script for the life of the project, not a phase-0-only task.

---

## 6. Retrieval & Prompting Design

| Element | Design | Source |
|---|---|---|
| Query embedding | Same model/version as document embedding (Invariant, Section 2.2) | plan2.md:35, 43 |
| Retrieval | Top-k similarity search against Chroma | plan2.md:36 |
| Prompt assembly | System rules (grounding instructions) + retrieved chunks + user question, in that order | plan2.md:37 |
| Citation mechanism | Each chunk carries source metadata; model instructed to cite inline; backend parses and attaches structured citations | plan2.md:39, 102 |
| Multi-hop questions ("why hire you for X") | v1: retrieve wider top-k across multiple chunk types (skills + projects + experience) rather than single-chunk lookup | plan2.md:80, 117 |

---

## 7. Grounding & Anti-Hallucination Design

This is called out in the source plan as **the core hard part of the product** (plan2.md:19) and is built in three layers, incrementally, rather than all at once.

| Layer | Mechanism | Enforcement point | Build phase | Why this layer exists |
|---|---|---|---|---|
| **1 — Retrieval-gated prompting** | System prompt instructs the model to answer *only* from provided chunks; explicit refusal phrase ("I don't have information about that in my documents") when chunks don't cover the question; every chunk carries citation metadata | Inside the LLM call (prompt-level) | Phase 1 (MVP) | Baseline honesty behavior — cheapest layer to build, first line of defense | plan2.md:102 |
| **2 — Similarity-threshold refusal** | Before calling the LLM, check top retrieved chunk's similarity score; below threshold → skip LLM call entirely, return canned refusal | Pre-generation, in the backend | Phase 2/3 | Cheaper and more reliable than trusting the model to self-refuse; also saves an LLM call (cost/quota protection) | plan2.md:103 |
| **3 — Post-hoc groundedness check** | After the model answers, a second cheap check (small NLI-style model, or a second LLM call asking "is this claim actually supported by these chunks?") runs before the answer is returned; unsupported claims flagged/stripped | Post-generation, in the backend | Phase 5 (iteration) | Highest reliability, highest effort — explicitly not a launch blocker | plan2.md:104 |

**Design rule:** grounding is never enforced by asking the model to "be honest" alone — Layer 1 is prompt-level, but Layers 2 and 3 are structural/code-level checks that do not depend on model cooperation. (plan2.md:43, 98-104)

---

## 8. Non-Functional Requirements

| Category | Requirement | Design Response | Source |
|---|---|---|---|
| **Cost** | ~$0 running cost | All components chosen for genuine free tiers (Chroma embedded, Ollama local, Gemini/Groq free tier, Render/Railway free tier, Vercel/Netlify free tier) | plan2.md:5, 47-60 |
| **Availability** | Must be reachable by recruiters at any time, without the author's machine running | Generation runs on a hosted API, not local; backend deployed to Render, not self-hosted on a personal machine | plan2.md:17 |
| **Latency** | Not explicitly targeted in source plan | **[Assumption]** Cold-start delay of 30-60s after idle is an accepted, known MVP limitation of the Render free tier — no mitigation designed for v1 | plan2.md:58 |
| **Abuse resistance** | Public endpoint must not be exhaustible by scraping/abuse | Rate limiting from day one + input length caps; exact numeric limits are an open item | plan2.md:22, 136 |
| **Correctness / grounding** | No speculation, no hallucination | Three-layer grounding strategy (Section 7) | plan2.md:19, 98-104 |
| **Maintainability** | Corpus grows for years | Ingestion pipeline is a standing, repeatable script decoupled from serving | plan2.md:20 |
| **Privacy** | Public-facing system must not leak sensitive personal data | Manual sensitive-content review pass before every indexing run | plan2.md:21 |
| **Observability / testability** | Must be able to tell if a change improved or broke the system | Fixed 20-30 question eval set, re-run after every change, at every phase | plan2.md:23, 122-128 |
| **Portfolio quality** | Code will be inspected directly by recruiters | Clean Git/GitHub history treated as a first-class requirement, not incidental | plan2.md:60 |

---

## 9. Deployment View

| Environment | Purpose | Components | Trigger |
|---|---|---|---|
| Local (dev) | Ingestion, prompt iteration, CLI validation | Ingestion scripts, Ollama (embedding + local LLM), Chroma (local file), CLI test loop | Manual, Phase 0-1 |
| Production | Live recruiter-facing service | FastAPI backend (Render/Railway), Chroma (persisted alongside backend), Gemini/Groq (hosted), Frontend (static hosting or FastAPI-served) | Deploy in Phase 4 |

**Deployment dependency note [Derived]:** because Chroma is embedded/file-based rather than a separately hosted service, the vector store's persisted files must be deployed *alongside* the backend (e.g., committed/synced as part of the deployment artifact) rather than connected to over a network — this has implications for how re-indexing updates reach production. This is now resolved below (Section 9.1); previously an open gap (see Section 11.3 change log).

### 9.1 Re-indexing strategy for a deployed product **[Derived — resolved 2026-09-13, session discussion]**

**Constraint driving this decision:** Render's free tier has an ephemeral filesystem — anything Chroma writes at runtime is lost on restart/redeploy. This rules out "re-index live on the running server" as an option outright. The index must instead be baked into the deployment artifact at deploy time, not updated at runtime.

**Chosen approach — Option A: commit the index, redeploy on push.**

| Aspect | Design |
|---|---|
| What | Documents are edited locally → the existing local reindex script (already built in Phase 1, plan2.md:113) regenerates the Chroma files on disk → both the changed source docs and the regenerated Chroma folder are committed to git → push triggers Render's existing git-based auto-deploy → the new index ships as part of the deployed repo state |
| Why (vs. alternatives below) | Matches the plan's own stated principle that ingestion is a repeatable script, not a one-off (plan2.md:20). Requires no runtime disk persistence — the deploy itself *is* the persistence mechanism. Requires no embedding compute in the cloud (Ollama / sentence-transformers stay local-only, consistent with ADR-3). Zero additional cost or infrastructure. |
| Condition | Reindex script must be idempotent/repeatable (already required, plan2.md:20). Chroma folder must stay small enough to live comfortably in git (fine at personal-corpus scale — expected low single-digit MBs, not GB) |
| Steps | 1. Edit/add source document(s) in the document store folder.<br>2. Run the local reindex script — regenerates chunking → embedding → Chroma files.<br>3. `git add` the changed documents and the regenerated Chroma folder; commit.<br>4. `git push` — Render's auto-deploy picks up the new commit and redeploys with the updated index already baked in. No manual server interaction, no separate "sync" step. |
| Net effect | "Update the docs" and "update the deployed bot" become the same action: one `git push`. |

**Kept as an explicit alternative — Option B: reindex inside the Render build step.**

| Aspect | Design |
|---|---|
| What | Instead of committing the regenerated Chroma folder, the ingestion pipeline runs as part of Render's build command — the index is generated fresh inside the cloud build environment on every deploy, from source documents only |
| Why considered | Avoids committing binary index files to the git repository, keeping the portfolio repo's diffs clean (relevant since the repo is public-facing, plan2.md:60) |
| Why not chosen as primary | Requires the embedding runtime (Ollama or sentence-transformers) to be installed and run inside Render's build environment, adding build time and a dependency not otherwise needed there — against the goal of keeping the deploy path simple. Not needed unless the committed-index approach (Option A) causes real repo-size or diff-noise problems in practice |
| Status | Retained as a documented fallback, not implemented — revisit if Option A's git-blob growth becomes a genuine issue |

**Flagged as a future option — Option C: externalize the vector store.**

| Aspect | Design |
|---|---|
| What | Move the vector store off the embedded/file-based model entirely — either a Render persistent disk (paid tier) or a hosted vector-capable store (e.g., pgvector on a free-tier Postgres such as Neon/Supabase) |
| Why considered | Fully decouples "update the corpus" from "redeploy the backend" — updates could be pushed independently of a deploy cycle |
| Why not chosen now | Persistent disk costs money (violates the $0 constraint, plan2.md:5); a hosted vector store reopens ADR-5's decision (plan2.md:18) and adds a new network dependency and component to operate, unjustified at current corpus size and update frequency |
| Status | Explicitly deferred — revisit only if corpus size or update frequency grows enough that Option A's redeploy-per-update model becomes a real bottleneck, or if recruiter-facing/user-triggered content updates are ever introduced (would need updates without a redeploy) |

**Decision rationale:** at solo-developer, personal-corpus scale with infrequent document updates (adding a project or blog post every few weeks/months, not continuously), coupling "content update" to "redeploy" costs nothing in practice and avoids introducing any new component, cost, or cloud build dependency. This is treated as the resolved answer for v1; Options B and C are retained in this document as reasoned-through, deliberately-deferred alternatives rather than dropped, so a future revisit doesn't have to re-derive this trade-off from scratch.

---

## 10. Phased Build Roadmap

Reproduced from plan2.md with its ordering rationale, since the *order* is itself a design decision (walking-skeleton principle), not just a task list.

| Phase | Goal | Key Steps | Why This Order | Verification | Source |
|---|---|---|---|---|---|
| **0. Document prep** | Collect & clean source material | Gather all documents into one folder; convert to clean text/Markdown; tag metadata; manual sensitive-content review | Garbage in, garbage out; only phase with no coding, so sequence it first | — | plan2.md:112 |
| **1. Core pipeline (CLI-only)** | Prove retrieval + grounded answering works before any web app | Chunking script → embedding script → Chroma indexing script → CLI Q&A loop (local LLM fine here); build 20-30 question eval set | Walking skeleton — validate the hard part (retrieval quality) with the fastest feedback loop before adding server/UI complexity | Manually check each eval answer for correctness, correct citation, correct refusal on unanswerable questions | plan2.md:113, 124 |
| **2. Backend service** | Wrap pipeline in a real API | FastAPI `/chat` endpoint; request/response schemas; wire hosted LLM; add Layer 2 similarity-threshold refusal; add rate limiting + input validation; structured query logging | Turns CLI prototype into something callable by a frontend/Postman; logic already proven in Phase 1 | Re-run eval set via curl/Postman; confirm refusal + rate-limit behavior under bad/adversarial input | plan2.md:114, 125 |
| **3. Frontend chat UI** | Give recruiters something usable | Minimal chat interface: messages, input, citations rendered under each answer | Deliberately last of the core phases — a good UI on a bad backend is worse than a plain UI on a good backend | Manual click-through of eval questions in-browser, confirm citations render | plan2.md:115, 126 |
| **4. Deploy** | Get it live end-to-end | Deploy backend (Render), deploy frontend (static hosting), connect via env-configured API URL | First real deadline — "live within weeks" | Re-run full eval set against the *deployed* URL | plan2.md:116, 127 |
| **5. Iterate** (ongoing) | Strengthen grounding, add capabilities, polish | Add Layer 3 groundedness check; build "why hire you" multi-chunk synthesis; expand corpus over time; review real recruiter query logs; improve citation UI; consider caching common Q&A pairs | Real product quality work is prioritized by real usage data, not guesswork, once a working baseline exists | Track eval set pass rate over time as a regression check on every prompt/model/chunking change | plan2.md:117, 128 |
| **6. Productize (multi-tenant SaaS)** | *Out of scope for v1* | Not addressed | Distinct problem set (auth, billing, isolation) — revisit only after Phase 5 proves the concept | — | plan2.md:118 |

---

## 11. Risks, Open Questions & Deferred Decisions

### 11.1 Explicitly flagged assumptions (carried from source plan, not independently re-verified)

| Assumption | Risk if wrong | Source |
|---|---|---|
| Gemini Flash / Groq free tiers remain genuinely free, no-credit-card, with generous limits | Production answering path breaks or incurs cost without warning | plan2.md:55, 62 |
| Render/Railway free tier remains available, no-credit-card-required | Deployment path unavailable at zero cost | plan2.md:58, 62 |
| Wider top-k retrieval will be sufficient for "why hire you" style multi-hop questions | May need to build the lightweight entity-graph fallback sooner than planned | plan2.md:80 |
| Cited GraphRAG benchmark figures (~32% vs ~86% on multi-hop) | Decision to defer GraphRAG rests partly on these — if they don't hold for this corpus's scale, the trade-off calculus changes | plan2.md:76 |

### 11.2 Open items deferred to implementation deep-dive (verbatim from source plan)

| Open item | Source |
|---|---|
| Exact prompt wording for citation formatting | plan2.md:134 |
| Whether to add conversation memory (multi-turn context) or keep each question stateless for v1 | plan2.md:135 |
| Specific rate-limit numbers and abuse-handling behavior | plan2.md:136 |
| If Phase 5's "why hire you" multi-chunk retrieval proves weak, the specific shape of the lightweight entity-graph fallback | plan2.md:137 |

### 11.3 Gaps identified during system design review **[Derived — not in source plan]**

These are structural gaps this document surfaces by attempting to give every component a complete contract. One has since been resolved (see status column and Section 12 change log); the rest remain open items alongside 11.2.

| Gap | Why it matters | Recommended resolution point | Status |
|---|---|---|---|
| No formal `/chat` request/response schema in source plan | Section 3.2 above proposes one, but it is derived, not sourced — needs explicit confirmation | Phase 2 (backend service build) | Open |
| ~~No specified mechanism for propagating re-indexed Chroma data to the deployed backend~~ | Because Chroma is embedded/file-based, updating the corpus in production is not a simple "call an API" — needed a defined deploy/sync step | — | **Resolved 2026-09-13 — see Section 9.1** |
| No numeric rate-limit or input-length cap decided | Already flagged as open (11.2), listed here to ensure it isn't lost among smaller items | Phase 2 | Open |
| No named testing framework for API/unit-level correctness (distinct from the eval set, which tests answer *quality*, not code correctness) | Eval set alone won't catch e.g. schema bugs or chunking crashes on edge-case documents | Phase 1 onward | Open |

---

## 12. Document Change Log

| Version | Date | Change | Reason |
|---|---|---|---|
| v1 | 2026-09-13 | Initial system design document created from `plans/plan2.md` | First formal system design artifact for this project; no prior system design document existed |
| v1.1 | 2026-09-13 | Added Section 9.1 (Re-indexing strategy for a deployed product): chose Option A (commit regenerated index, redeploy on push) as the resolved design; retained Option B (reindex inside Render build step) and Option C (externalize vector store — persistent disk or hosted pgvector) as documented, deliberately-deferred alternatives. Closed the corresponding gap in Section 11.3. | User asked how re-indexing would be handled in a fully-deployed product; answered in chat (session discussion, not in `plans/plan2.md`), then explicitly requested this be appended to the system design document with Option B kept as an alternative and Option C flagged as a future option |
| v1.2 | 2026-09-13 | Three changes to Section 5 (Data Design), all sourced from `plans/phase0-document-prep-subplan-v2.md`: (1) added Section 5.4, Document format conversion — non-Markdown sources (PDF via `pymupdf4llm`, DOCX/other via `pandoc`) converted to Markdown before chunking or content-type detection; (2) added Section 5.5, Automatic prose vs. `structured_doc` detection — a fully automatic heuristic script tags `content_type` in front-matter, no manual folder-sorting step, and the corpus folder structure collapses to a flat `corpus/documents/` (+ separate `corpus/code/`); (3) renamed the `content_type` value `transcript` to `structured_doc` throughout Sections 5.1 and 5.2, since the chunking rule applies to any record-structured document, not transcripts specifically. Prior wording kept as "earlier decision" blocks in Sections 5.1/5.2 rather than deleted. | User explicitly requested these three items be appended to the system design doc, referencing `plans/phase0-document-prep-subplan-v2.md` as source; user also explicitly corrected an initial hybrid (auto-detect + manual low-confidence override) framing — the finalized design is fully automatic detection, no manual override step |
| v1.3 | 2026-09-19 | Four changes to Section 5 (Data Design), sourced from `plans/phase0-document-prep-subplan-v5.md`: (1) Section 5.4 updated — PDF/DOCX conversion library changed from `pymupdf4llm`/`pandoc` to `docling` (a real, confirmed table/heading-misdetection bug in `pymupdf4llm` on a real PDF, not fixed by its own `table_strategy` parameter), plus a new always-on independent hyperlink-extraction pass (a second, separate bug: `docling` silently dropped word-level PDF link annotations because its text grouping is coarser than the PDF's own link rectangles); (2) Section 5.5's document-level `prose`/`structured_doc` detection step is **removed from Phase 0 entirely**, not replaced by a corrected document-level rule — triggered by adding this system design document itself to the real corpus and finding that its 167 lines of exposition-table syntax would falsely trigger the old rule's `structured_doc` branch, exposing that whole-document binary classification is the wrong grain for real (heterogeneous) documents; (3) Section 5.1 gains a Current (v1.3) design — content-type/structural typing moves to a **per-block** decision made at chunk time in Phase 1 (block partitioning via `docling`'s own typed document model for PDF/DOCX-origin content, `markdown-it-py` for native Markdown, `tree-sitter` for code, unchanged), the standard approach used by tools like `unstructured` and LlamaParse; (4) Section 5.2's `content_type` chunk-metadata field changes scope from per-document (Phase 0) to per-chunk (Phase 1). All prior wording kept as "earlier decision" blocks per this document's convention, not deleted. | Corpus file addition and follow-up discussion (session, 2026-09-19) surfaced the classification-grain flaw directly; user then explicitly requested this system design document be updated to match both the discussion and the finalized `phase0-document-prep-subplan-v5.md` |

---

## 13. Traceability Index

Every section of this document maps back to a specific part of `plans/plan2.md` unless marked **[Derived]** (a structural addition implied by, but not explicit in, the source plan) or **[Assumption]** (a detail flagged as unverified). This index exists so a reader can audit any claim in this document back to its origin without re-reading the entire source plan.

| This document | Source in plan2.md |
|---|---|
| Section 1 (Purpose & Scope) | Lines 5, 7, 118, 66-80, 135 |
| Section 2 (Architecture Overview) | Lines 27-43 |
| Section 3 (Component Design) | Lines 51-60, 34-40 |
| Section 4 (ADRs) | Lines 16-18, 43, 51-62, 66-94 |
| Section 5 (Data Design) | Lines 20-21, 88-94, 112; Sections 5.4-5.5 and the `structured_doc` renaming sourced from `plans/phase0-document-prep-subplan-v2.md` (session discussion 2026-09-13, not in plan2.md) — see Change Log v1.2. Sections 5.1/5.2/5.4/5.5 Current (v1.3) content sourced from `plans/phase0-document-prep-subplan-v5.md` (session discussion 2026-09-19, not in plan2.md) — see Change Log v1.3 |
| Section 6 (Retrieval & Prompting) | Lines 35-39, 80, 102, 117 |
| Section 7 (Grounding & Anti-Hallucination) | Lines 19, 43, 98-104 |
| Section 8 (Non-Functional Requirements) | Lines 5, 17-23, 47-62, 122-128 |
| Section 9 (Deployment View) | Lines 58-59; Section 9.1 sourced from session discussion 2026-09-13 (not in plan2.md) — see Change Log v1.1 |
| Section 10 (Phased Roadmap) | Lines 108-128 |
| Section 11 (Risks & Open Items) | Lines 62, 76, 80, 134-137 |
