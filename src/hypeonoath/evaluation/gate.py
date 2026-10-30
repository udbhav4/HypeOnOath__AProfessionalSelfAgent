"""
Similarity-gate sweep (eval plan v15, 4.2; D30). Code only, from the raw top-1 similarity.

would_refuse(q, t) = top1(q) < t, for every distinct top-1 value t. Series:
  cov_only    questions whose parts are all not-covered no_answer   -> gate correct-refusal(t)
  answerable  questions with >= 1 answer part                       -> gate false-refusal(t)
  private     private no_answer parts and no answer part            -> shown apart
  adversarial category adversarial                                  -> shown apart
gate_t_at_1fr = highest t with <= GATE_MAX_FALSE_REFUSALS answerable questions refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypeonoath.core import config
from hypeonoath.evaluation.questions import ANSWER, NOT_COVERED, PRIVATE, Question


@dataclass(frozen=True)
class GateRow:
    qid: str
    series: str          # cov_only | answerable | private | adversarial | other
    top1: float | None


def series_of(q: Question) -> str:
    if q.category == "adversarial":
        return "adversarial"
    if ANSWER in q.kinds:
        return "answerable"
    if q.kinds == {NOT_COVERED}:
        return "cov_only"
    if PRIVATE in q.kinds:
        return "private"
    return "other"


def _share(refused: int, total: int) -> float | None:
    return refused / total if total else None


def sweep(rows: list[GateRow], max_false_refusals: int = config.GATE_MAX_FALSE_REFUSALS) -> dict[str, Any]:
    scored = [r for r in rows if r.top1 is not None]
    by_series: dict[str, list[float]] = {}
    for r in scored:
        by_series.setdefault(r.series, []).append(r.top1)
    points = []
    for t in sorted({r.top1 for r in scored}):
        refused = {s: sum(v < t for v in vals) for s, vals in by_series.items()}
        points.append({
            "t": t,
            "correct_refusal": _share(refused.get("cov_only", 0), len(by_series.get("cov_only", []))),
            "false_refusals": refused.get("answerable", 0),
            "false_refusal": _share(refused.get("answerable", 0), len(by_series.get("answerable", []))),
            "private_refused": _share(refused.get("private", 0), len(by_series.get("private", []))),
            "adversarial_refused": _share(refused.get("adversarial", 0), len(by_series.get("adversarial", []))),
        })
    ok = [p for p in points if p["false_refusals"] <= max_false_refusals]
    best = max(ok, key=lambda p: p["t"]) if ok else None
    return {
        "points": points,
        "n": {s: len(v) for s, v in by_series.items()},
        "gate_t_at_1fr": best["t"] if best else None,
        "gate_correct_refusal_at_t": best["correct_refusal"] if best else None,
    }
