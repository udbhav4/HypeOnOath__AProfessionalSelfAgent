"""
Part outcomes and behaviour metrics -- arithmetic over the judge's output (eval plan v15, 4.0, 4.3, 5.4;
D42, D58, D60, D61). No judging happens here.

Judgment record (one per reply in judgments.jsonl; produced by the Stage D judge):
  {"id": qid, "status": "ok" | "error",
   "call1": {"segments": [{"id", "text", "type": claim|non_answer|other, "about"}], "relevancy", "persona",
             "persona_reason", "policy_violation", "policy_reason"},
   "call2": null | {"verdicts": [{"segment_id", "supported", "unsupported_type", "evidence_ref", "role":
             answers|related_to|unrelated, "role_part_id", "reason"}], "links": [{"segment_id", "part_id"}],
             "key_points_covered": {part_id: [bool, ...]}, "embellished": bool, "embellishment_spans": [str]}}

Outcome of a part (5.4): answered | non_answer | skipped | unknown.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from hypeonoath.evaluation.questions import ANSWER, NOT_COVERED, PRIVATE, Question

ANSWERED, NON_ANSWER, SKIPPED, UNKNOWN = "answered", "non_answer", "skipped", "unknown"
OUTCOMES = (ANSWERED, NON_ANSWER, SKIPPED)
PIVOT_ROLES = ("related_to", "unrelated")


def _segments(judgment: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [s for s in (judgment.get("call1") or {}).get("segments") or [] if s.get("type") == kind]


def part_outcomes(q: Question, judgment: dict[str, Any] | None) -> dict[str, str]:
    """5.4 table for every part of the question."""
    if not judgment or judgment.get("status") != "ok" or not judgment.get("call1"):
        return {p.id: UNKNOWN for p in q.parts}
    call2 = judgment.get("call2")
    if call2 is None:
        if len(q.parts) != 1 or _segments(judgment, "claim"):
            return {p.id: UNKNOWN for p in q.parts}           # call 2 was required (D43) but is missing
        outcome = NON_ANSWER if _segments(judgment, "non_answer") else SKIPPED
        return {q.parts[0].id: outcome}

    non_answer_ids = {s["id"] for s in _segments(judgment, "non_answer")}
    out: dict[str, str] = {}
    for p in q.parts:
        if any(v.get("role") == "answers" and v.get("role_part_id") == p.id for v in call2.get("verdicts") or []):
            out[p.id] = ANSWERED
        elif any(link.get("part_id") == p.id and link.get("segment_id") in non_answer_ids
                 for link in call2.get("links") or []):
            out[p.id] = NON_ANSWER
        else:
            out[p.id] = SKIPPED
    return out


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def behaviour(questions: list[Question], outcomes: dict[str, dict[str, str]],
              judgments: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Row-normalised behaviour metrics (4.3), coverage, matrix, the missed_no_answer split, all_parts_ok."""
    judgments = judgments or {}
    rows = {"A": Counter(), "N": Counter(), "N_priv": Counter(), "N_cov": Counter()}
    split = Counter()                      # (private | not_covered, supported | unsupported) -> count
    total = known = 0
    all_ok = all_known_q = 0
    for q in questions:
        per = outcomes.get(q.id) or {p.id: UNKNOWN for p in q.parts}
        q_known, q_ok = True, True
        for p in q.parts:
            o = per.get(p.id, UNKNOWN)
            total += 1
            if o == UNKNOWN:
                q_known = False
                continue
            known += 1
            if p.kind == ANSWER:
                rows["A"][o] += 1
                q_ok &= o == ANSWERED
            else:
                rows["N"][o] += 1
                rows["N_priv" if p.kind == PRIVATE else "N_cov"][o] += 1
                q_ok &= o == NON_ANSWER
                if o == ANSWERED:
                    supported = any(v.get("supported") for v in _answering(judgments.get(q.id), p.id))
                    split[(p.kind, "supported" if supported else "unsupported")] += 1
        if q_known:
            all_known_q += 1
            all_ok += q_ok
    a, n = rows["A"], rows["N"]
    a_known, n_known = sum(a.values()), sum(n.values())
    return {
        "answer_rate": _rate(a[ANSWERED], a_known),
        "false_no_answer": _rate(a[NON_ANSWER], a_known),
        "omitted": _rate(a[SKIPPED], a_known),
        "correct_no_answer": _rate(n[NON_ANSWER], n_known),
        "missed_no_answer": _rate(n[ANSWERED], n_known),
        "silent_skip": _rate(n[SKIPPED], n_known),
        "all_parts_ok": _rate(all_ok, all_known_q),
        "part_outcome_coverage": _rate(known, total),
        "n_parts": total, "n_parts_known": known, "n_questions_all_known": all_known_q,
        "n_a_known": a_known, "n_n_known": n_known,
        "matrix": {row: {o: c[o] for o in OUTCOMES} for row, c in rows.items()},
        "missed_split": {f"{kind}/{verdict}": split[(kind, verdict)]
                         for kind in (PRIVATE, NOT_COVERED) for verdict in ("unsupported", "supported")},
    }


def _answering(judgment: dict[str, Any] | None, part_id: str) -> list[dict[str, Any]]:
    call2 = (judgment or {}).get("call2") or {}
    return [v for v in call2.get("verdicts") or [] if v.get("role") == "answers" and v.get("role_part_id") == part_id]


def pivots(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """pivot_related = related_to pivots / all pivots (informational, D60); unrelated pivots listed."""
    related, unrelated = 0, []
    for j in judgments:
        if j.get("status") != "ok":
            continue
        texts = {s["id"]: s.get("text", "") for s in _segments(j, "claim")}
        for v in ((j.get("call2") or {}).get("verdicts") or []):
            if v.get("role") == "related_to":
                related += 1
            elif v.get("role") == "unrelated":
                unrelated.append({"id": j.get("id"), "text": texts.get(v.get("segment_id"), "")})
    total = related + len(unrelated)
    return {"pivot_related": _rate(related, total), "n_pivots": total, "unrelated": unrelated}
