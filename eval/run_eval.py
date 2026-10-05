"""
Run the fixed eval set through the same retrieve -> prompt -> generate chain
chat.py uses, and write one result record per question for MANUAL review
(guideline v4, Section 8; design doc Section 10, Phase 1 verification).

There is no automated pass/fail: correctness, citation and refusal are judged
by hand. Results: eval/results/<UTC timestamp>.jsonl, each line
{question, expected_answerable, notes, answer, sources, top_score, error}.

Run from the project root (after `pip install -e ".[dev]"`):   python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.serving import llm_client
from hypeonoath.serving.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        questions = yaml.safe_load(f) or []
    for n, q in enumerate(questions):
        if not isinstance(q, dict) or not q.get("question") or "expected_answerable" not in q:
            raise ValueError(f"questions.yaml entry {n} needs 'question' and 'expected_answerable'")
    return questions


def source_records(items: list[RetrievedItem]) -> list[dict]:
    out = []
    for item in items:
        chunks = item.ordered_chunks
        out.append({
            "document": chunks[0].source_document,
            "chunk_id": item.chunk.chunk_id,
            "content_type": item.chunk.content_type,
            "line_start": chunks[0].line_start,
            "line_end": chunks[-1].line_end,
            "chunk_method": item.chunk.chunk_method,
            "parts": [c.part for c in chunks] if len(chunks) > 1 else None,
            "score": round(item.score, 4),
            "snippet": item.chunk.text[:200],
        })
    return out


def run(retriever, questions: list[dict], generate=llm_client.generate, k: int = config.DEFAULT_TOP_K) -> list[dict]:
    results = []
    for q in questions:
        items = retriever.retrieve(q["question"], k=k)
        record = {
            "question": q["question"],
            "expected_answerable": q["expected_answerable"],
            "notes": q.get("notes", ""),
            "answer": None,
            "error": None,
            "sources": source_records(items),
            "top_score": round(items[0].score, 4) if items else None,
        }
        try:
            record["answer"] = generate(build_prompt(q["question"], items))
        except llm_client.LLMUnavailableError as exc:
            record["error"] = str(exc)   # retrieval results are still worth reviewing
        results.append(record)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval set (results go to eval/results/ for manual review)")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    from hypeonoath.serving.retriever import Retriever

    results = run(Retriever(), load_questions(), k=args.k)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if r["error"])
    print(f"Wrote {len(results)} results to {out}" + (f" ({errors} without an answer: LLM unavailable)" if errors else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
