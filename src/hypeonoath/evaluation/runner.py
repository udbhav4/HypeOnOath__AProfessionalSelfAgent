"""
Run every question through an Answerer (eval plan v15, 6.1, 8, 9; D25, D26, D49, D52, D71).

Per question: answerer.retrieve -> (stop here in the `retrieval` stage) -> answerer.generate
(or the pinned context of an earlier run, --pin-contexts) -> serving.guard.guard_answer IN THE
RUNNER -> store the guarded text, the hit labels (private_hits_pre) and guard_declined.
A raw reply is never written; error messages are redacted before they are stored.

Writes eval/runs/<run_id>/answers.jsonl one line per question (appended, so an interrupted run
resumes by skipping finished ids) and manifest.json. Evidence groups are matched at run time and
stored per ranked / context item, so scoring never needs the index.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypeonoath.core import paths as core_paths
from hypeonoath.core.private_values import PrivateValue, redact
from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.evaluation import paths
from hypeonoath.evaluation.answerer import Answerer, Generated
from hypeonoath.evaluation.evidence import Corpus, group_found
from hypeonoath.evaluation.questions import Group, Question
from hypeonoath.serving.guard import guard_answer

STAGES = ("retrieval", "full")
WARMUP_QUESTION = "Hello"


class RunAborted(RuntimeError):
    """Generation stopped (Ollama down at warm-up, or MAX_CONSECUTIVE_LLM_ERRORS errors in a row)."""


# --- Records -----------------------------------------------------------------------------

def question_groups(q: Question) -> dict[str, Group]:
    """Every evidence / placeholder / supporting group of a question, by a stable key."""
    groups: dict[str, Group] = {}
    for p in q.parts:
        for g in p.evidence:
            groups[f"{p.id}:{'p' if g.point is None else g.point}"] = g
        for i, g in enumerate(p.supporting):
            groups[f"{p.id}:s{i}"] = g
    return groups


def _matched(item: RetrievedItem, groups: dict[str, Group], corpus: Corpus) -> list[str]:
    return [key for key, g in groups.items() if group_found(g, item.ordered_chunks, corpus)]


def ranking_record(item: RetrievedItem, groups: dict[str, Group], corpus: Corpus) -> dict[str, Any]:
    c = item.chunk
    return {"chunk_id": c.chunk_id, "document": c.source_document, "line_start": c.line_start,
            "line_end": c.line_end, "score": round(item.score, 6), "groups": _matched(item, groups, corpus)}


def context_record(item: RetrievedItem, groups: dict[str, Group], corpus: Corpus) -> dict[str, Any]:
    """Full chunk data (the index holds no private values), so --pin-contexts can replay it."""
    return {"chunk": item.chunk.to_dict(), "score": round(item.score, 6),
            "parts": [c.to_dict() for c in item.parts], "groups": _matched(item, groups, corpus)}


_CHUNK_FIELDS = {f.name for f in fields(Chunk)}


def item_from_record(rec: dict[str, Any]) -> RetrievedItem:
    def chunk(d: dict[str, Any]) -> Chunk:
        return Chunk(**{k: v for k, v in d.items() if k in _CHUNK_FIELDS})
    return RetrievedItem(chunk(rec["chunk"]), float(rec["score"]), [chunk(p) for p in rec.get("parts") or []])


def derived_timings(stats: dict[str, int]) -> dict[str, float | None]:
    """Ollama ns / counts -> seconds and rates (4.7). ttft = model load + prefill."""
    def s(key: str) -> float | None:
        return stats[key] / 1e9 if key in stats else None

    load, prefill, decode, total = s("load_duration"), s("prompt_eval_duration"), s("eval_duration"), s("total_duration")
    n_in, n_out = stats.get("prompt_eval_count"), stats.get("eval_count")
    return {
        "model_load_s": load, "prefill_s": prefill, "decode_s": decode, "total_gen_s": total,
        "ttft_s": (load or 0.0) + prefill if prefill is not None else None,
        "input_tokens": n_in, "output_tokens": n_out,
        "prefill_tok_per_s": n_in / prefill if n_in is not None and prefill else None,
        "decode_tok_per_s": n_out / decode if n_out is not None and decode else None,
    }


def read_answers(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "answers.jsonl"
    if not path.exists():
        return []
    records = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:                 # only a torn LAST line (interrupted write) is tolerated
                raise
    return records


# --- Fingerprints and the manifest --------------------------------------------------------

def index_fingerprint(chunk_ids: list[str]) -> str:
    """Chunk count + hash of the sorted chunk ids (6.1): changes whenever re-indexing changes ids."""
    digest = hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()[:16]
    return f"{len(chunk_ids)}:{digest}"


def git_state(root: Path = core_paths.PROJECT_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def new_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


RESUME_KEYS = ("index_fp", "question_set_sha256", "privacy_fp")


def resume_problem(manifest: dict[str, Any], current: dict[str, str]) -> str | None:
    """D26: --resume refuses when the index, the question set or the private decisions changed."""
    changed = [k for k in RESUME_KEYS if manifest.get(k) != current.get(k)]
    return f"cannot resume: {', '.join(changed)} changed since the run started" if changed else None


# --- The run ------------------------------------------------------------------------------

def run_questions(
    questions: list[Question],
    answerer: Answerer,
    run_dir: Path,
    stage: str,
    values: list[PrivateValue],
    corpus: Corpus | None = None,
    pinned: dict[str, dict[str, Any]] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    warmup: bool = True,
    log: Callable[[str], None] = print,
) -> str:
    """
    Answer every question not yet in answers.jsonl. Returns the generation status:
    "retrieval_only" | "complete". Raises RunAborted (the finished lines stay -> --resume).
    `pinned` = question id -> answers.jsonl record of an earlier run (--pin-contexts).
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    corpus = corpus or Corpus()
    run_dir.mkdir(parents=True, exist_ok=True)
    kept = [r for r in read_answers(run_dir) if not r.get("error")]       # errored questions are retried
    if (run_dir / "answers.jsonl").exists():                             # rewrite without the errored lines
        (run_dir / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                               encoding="utf-8")
    done = {r["id"] for r in kept}
    todo = [q for q in questions if q.id not in done]
    if done:
        log(f"resume: {len(done)} done, {len(todo)} to go")

    if stage == "full" and warmup and todo:
        try:
            answerer.generate(WARMUP_QUESTION, [])                      # untimed: loads the model
        except Exception as exc:  # noqa: BLE001 -- any warm-up failure means Ollama is not usable
            raise RunAborted(f"warm-up failed: {type(exc).__name__}: {redact(str(exc), values)[0]}") from None

    errors_in_a_row = 0
    with (run_dir / "answers.jsonl").open("a", encoding="utf-8") as out:
        for n, q in enumerate(todo, start=1):
            groups = question_groups(q)
            record: dict[str, Any] = {"id": q.id, "question": q.question, "stage": stage}
            if pinned is not None:
                old = pinned.get(q.id)
                if old is None:
                    raise RunAborted(f"--pin-contexts: question {q.id} is missing in the pinned run")
                record["ranking"], record["context"] = old["ranking"], old["context"]
                context = [item_from_record(c) for c in old["context"]]
                record["pinned"] = True
            else:
                retrieved = answerer.retrieve(q.question)
                record["ranking"] = [ranking_record(i, groups, corpus) for i in retrieved.ranking]
                record["context"] = [context_record(i, groups, corpus) for i in retrieved.context]
                context = retrieved.context
            record["top1"] = record["ranking"][0]["score"] if record["ranking"] else None
            if count_tokens is not None:
                record["context_tokens"] = sum(count_tokens(c.text) for i in context for c in i.ordered_chunks)

            if stage == "full":
                errors_in_a_row = _generate_one(answerer, q, context, values, record, errors_in_a_row)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            log(f"[{n}/{len(todo)}] {q.id}" + (f" error: {record['error']}" if record.get("error") else ""))
            if errors_in_a_row >= paths.MAX_CONSECUTIVE_LLM_ERRORS:
                raise RunAborted(f"{errors_in_a_row} LLM errors in a row; fix Ollama, then: hypeonoath-eval run "
                                 f"--resume {run_dir.name}")
    return "retrieval_only" if stage == "retrieval" else "complete"


def _generate_one(answerer: Answerer, q: Question, context: list[RetrievedItem], values: list[PrivateValue],
                  record: dict[str, Any], errors_in_a_row: int) -> int:
    """Generate + guard one reply into `record`; returns the updated consecutive-error count."""
    started = time.perf_counter()
    try:
        generated: Generated = answerer.generate(q.question, context)
    except Exception as exc:  # noqa: BLE001 -- any generator failure is recorded, never fatal per question
        record.update(prompt=None, reply=None, private_hits_pre=[], guard_declined=False, ollama_stats={},
                      timings={}, error=f"{type(exc).__name__}: {redact(str(exc), values)[0]}"[:500])
        return errors_in_a_row + 1
    wall_s = round(time.perf_counter() - started, 3)                     # the generate call only (4.7)
    guarded = guard_answer(generated.text, values)                       # D52: the guard runs in the runner
    record.update(
        prompt=generated.prompt,
        reply=guarded.text,
        private_hits_pre=list(guarded.hits),                             # labels only
        guard_declined=guarded.declined,
        ollama_stats=generated.stats,
        timings={**derived_timings(generated.stats), "wall_s": wall_s},
        error=None,
    )
    return 0
