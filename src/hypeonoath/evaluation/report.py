"""
report.md (eval plan v15, Section 7): readable by anyone, numbers + deltas vs the baseline + n,
behaviour matrix, flips, breakdowns, worst failures, lists, glossary. Tables only in Stage B;
the PNG charts (charts.py) come with Stage D. Replies shown are the GUARDED text; private
leaks are listed by label only.
"""
from __future__ import annotations

from typing import Any

from hypeonoath.evaluation import glossary, paths
from hypeonoath.evaluation.metrics import METRICS, MetricDef
from hypeonoath.evaluation.outcomes import OUTCOMES, UNKNOWN
from hypeonoath.evaluation.questions import ANSWER, Question

BUCKET_ORDER = ["private_leak", "missed_no_answer_private", "missed_no_answer", "false_no_answer", "retrieval_miss",
                "silent_skip", "unfaithful", "embellished", "incomplete", "policy", "unknown", "ok"]
WORST_N = 10
EXCERPT_CHARS = 300


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _arrow(m: MetricDef, delta: float | None) -> str:
    if delta is None or delta == 0 or m.direction == "none":
        return ""
    return "↑ better" if (delta > 0) == (m.direction == "up") else "↓ worse"


JUDGED_SECTIONS = ("behaviour", "content")          # deltas withheld below 100 % coverage (S3, 4.0)


def _n(m: MetricDef, scores: dict[str, Any]) -> Any:
    """The numeric n behind a metric (4.0: every number shows its own n)."""
    r, beh, content = scores["retrieval"], scores.get("behaviour") or {}, scores.get("content") or {}
    gate_n, counts = (scores.get("gate") or {}).get("n") or {}, scores.get("counts") or {}
    a_rows = ("answer_rate", "false_no_answer", "omitted")
    n_rows = ("correct_no_answer", "missed_no_answer", "silent_skip")
    sources = {                                      # metric -> (dict, key) of its own denominator (plan 4.12)
        **{name: (beh, "n_a_known") for name in a_rows}, **{name: (beh, "n_n_known") for name in n_rows},
        "all_parts_ok": (beh, "n_questions_all_known"), "part_outcome_coverage": (beh, "n_parts"),
        "faithfulness": (content, "n_judged_claims"), "fully_faithful": (content, "n_judged_claims"),
        "embellishment": (content, "n_embellishment"), "correctness_answered": (content, "n_answered"),
        "parametric_leak": (content, "n_answered"), "correctness_e2e": (content, "n_a_known"),
        "relevancy": (content, "n_relevancy"), "persona": (content, "n_persona"),
        "world_knowledge_claims": (content, "n_judged"), "pivot_related": (scores.get("pivots") or {}, "n_pivots"),
        "gate_t_at_1fr": (gate_n, "answerable"), "gate_correct_refusal_at_t": (gate_n, "cov_only"),
        "supporting_recall@8": (r, "n_supporting"), "supporting_in_context": (r, "n_supporting"),
        "context_tokens_mean": (r, "n_tokens"), "context_tokens_p95": (r, "n_tokens"),
        "prompt_leak": (counts, "replies"), "private_leak_pre": (counts, "replies"), "judge_coverage": (counts, "replies"),
        "policy_flag": (counts, "judged_call1"), "p95_ttft_s": (counts, "ttft"), "p50_ttft_s": (counts, "ttft"),
        "p95_total_gen_s": (counts, "total_gen"), "p50_total_gen_s": (counts, "total_gen"),
        "mean_input_tokens": (counts, "input_tokens"),
    }
    if m.name in sources:
        source, key = sources[m.name]
        return source.get(key)
    if m.section == "retrieval":                     # hit@k, recall@8, mrr, in_context_hit
        return r.get("n_answerable")
    return scores["metrics"].get("n")                # errors: a count over the run's questions


def _covered(metrics: dict[str, Any]) -> bool:
    return all(metrics.get(k) == 1.0 for k in ("judge_coverage", "part_outcome_coverage"))


def _scorecard(manifest: dict[str, Any], scores: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    metrics, n = scores["metrics"], scores["metrics"].get("n") or 0
    base = (baseline or {}).get("metrics") or {}
    full_coverage = _covered(metrics) and (not base or _covered(base))
    lines = ["| Metric | This run | Baseline | Δ | | n | Denominator | Note |", "|---|---|---|---|---|---|---|---|"]
    for m in METRICS:
        if m.section == "setting" or manifest.get("stage") not in m.stages:
            continue
        value, old = metrics.get(m.name, scores["retrieval"].get(m.name)), base.get(m.name)
        delta = value - old if isinstance(value, (int, float)) and isinstance(old, (int, float)) else None
        note = ""
        if delta is not None and m.section in JUDGED_SECTIONS and not full_coverage:
            delta, note = None, "Δ withheld (coverage < 100 %)"
        elif delta and m.aggregation == "rate" and n and abs(delta) * n <= 1.0:
            note = "within 1 question"
        lines.append(f"| {m.name} | {_fmt(value)} | {_fmt(old)} | {_fmt(delta)} | {_arrow(m, delta)} | "
                     f"{_fmt(_n(m, scores))} | {m.denominator} | {note} |")
    if manifest.get("stage") == "full":
        content = scores.get("content") or {}
        lines += ["", f"- policy_flag on the adversarial subset: {_fmt(scores.get('policy_adversarial'))}",
                  f"- relevancy distribution (1-3): {content.get('relevancy_distribution') or {}}",
                  f"- persona distribution (1-5): {content.get('persona_distribution') or {}}"]
    return lines


def _matrix(beh: dict[str, Any]) -> list[str]:
    lines = ["| Expected | " + " | ".join(OUTCOMES) + " | n |", "|---|" + "---|" * (len(OUTCOMES) + 1)]
    for row in ("A", "N", "N_priv", "N_cov"):
        counts = beh["matrix"][row]
        total = sum(counts.values())
        cells = [f"{counts[o]} ({counts[o] / total:.0%})" if total else "0" for o in OUTCOMES]
        lines.append(f"| {row} | " + " | ".join(cells) + f" | {total} |")
    lines += ["", "**missed_no_answer split** (private label × claim verdict):", "",
              "| | unsupported | supported |", "|---|---|---|"]
    split = beh["missed_split"]
    lines.append(f"| N_priv (private) | {split['private/unsupported']} (invented / guessed) | "
                 f"{split['private/supported']} (inferred or real value → check private_leak_pre) |")
    lines.append(f"| N_cov (not covered) | {split['not_covered/unsupported']} (hallucination) | "
                 f"{split['not_covered/supported']} (gold error → review the entry) |")
    return lines


def _curves(scores: dict[str, Any]) -> list[str]:
    r = scores["retrieval"]
    lines = ["| k | " + " | ".join(str(k) for k in paths.K_VALUES) + " |", "|---|" + "---|" * len(paths.K_VALUES),
             "| Hit@k | " + " | ".join(_fmt(r.get(f"hit@{k}")) for k in paths.K_VALUES) + " |",
             "| Recall@k | " + " | ".join(_fmt(r["recall_curve"].get(k)) for k in paths.K_VALUES) + " |", ""]
    gate = scores["gate"]
    lines += [f"Gate sweep (series sizes: {gate['n']}); gate_t_at_1fr = {_fmt(gate['gate_t_at_1fr'])}:", "",
              "| t | correct refusal | false refusals | private refused | adversarial refused |", "|---|---|---|---|---|"]
    lines += [f"| {p['t']:.3f} | {_fmt(p['correct_refusal'])} | {p['false_refusals']} | {_fmt(p['private_refused'])} "
              f"| {_fmt(p['adversarial_refused'])} |" for p in gate["points"]]
    return lines


def _breakdown(title: str, rows: dict[str, dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    keys = list(next(iter(rows.values())).keys())
    lines = [f"**{title}**", "", "| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {name} | " + " | ".join(_fmt(row.get(k)) for k in keys) + " |" for name, row in rows.items()]
    return lines + [""]


def _flips(scores: dict[str, Any], baseline_scores: dict[str, Any] | None) -> list[str]:
    if not baseline_scores or "buckets" not in baseline_scores or "buckets" not in scores:
        return ["No baseline run scores to compare with."]
    lines = ["| Question | Baseline | This run |", "|---|---|---|"]
    for qid, b in sorted(scores["buckets"].items()):
        old = baseline_scores["buckets"].get(qid)
        if old and (old["bucket"] == "ok") != (b["bucket"] == "ok"):
            lines.append(f"| {qid} | {old['bucket']} | {b['bucket']} |")
    return lines if len(lines) > 2 else ["No pass ↔ fail flips."]


def _worst(questions: dict[str, Question], answers: dict[str, dict[str, Any]], judgments: dict[str, dict[str, Any]],
           scores: dict[str, Any]) -> list[str]:
    failing = [(BUCKET_ORDER.index(b["bucket"]), qid, b) for qid, b in scores["buckets"].items() if b["bucket"] != "ok"]
    lines: list[str] = []
    for _, qid, b in sorted(failing)[:WORST_N]:
        q, a, j = questions[qid], answers.get(qid) or {}, judgments.get(qid) or {}
        reply = (a.get("reply") or a.get("error") or "").replace("\n", " ")
        lines += [f"**{qid}** — `{b['bucket']}` ({b.get('family') or '—'}) — {q.question}", "",
                  f"> {reply[:EXCERPT_CHARS]}{'…' if len(reply) > EXCERPT_CHARS else ''}", ""]
        context_groups = {g for c in a.get("context") or [] for g in c.get("groups") or []}
        covered = ((j.get("call2") or {}).get("key_points_covered") or {})
        for p in q.parts:
            observed = scores["outcomes"].get(qid, {}).get(p.id, UNKNOWN)
            line = f"- part {p.id} ({p.ask}): expected {p.expected}, observed {observed}"
            if p.kind == ANSWER:
                seen = ["✓" if f"{p.id}:{g.point}" in context_groups else "✗" for g in p.evidence]
                flags = list(covered.get(p.id) or [])
                flags += [False] * (len(p.key_points) - len(flags))     # missing entry = not covered (4.12 E)
                missing = [k for k, c in zip(p.key_points, flags) if not c]
                line += f"; in context per group {' '.join(seen)}"
                if missing:
                    line += f"; missing key points: {', '.join(missing)}"
            lines.append(line)
        texts = {s["id"]: s.get("text", "") for s in (j.get("call1") or {}).get("segments") or []}
        for v in (j.get("call2") or {}).get("verdicts") or []:
            if not v.get("supported") or v.get("role") != "answers":
                lines.append(f"- claim ({v.get('role')} {v.get('role_part_id') or ''}, "
                             f"{'supported' if v.get('supported') else 'unsupported'}): {texts.get(v.get('segment_id'), '')}")
        for span in (j.get("call2") or {}).get("embellishment_spans") or []:
            lines.append(f"- embellishment: {span}")
        lines.append("")
    return lines or ["No failures."]


def _list(title: str, items: list[Any]) -> list[str]:
    return [f"**{title}** ({len(items)})", ""] + ([f"- {i}" for i in items] or ["- none"]) + [""]


def render(manifest: dict[str, Any], scores: dict[str, Any], questions: list[Question],
           answers: list[dict[str, Any]], judgments: list[dict[str, Any]],
           baseline: dict[str, Any] | None = None, baseline_scores: dict[str, Any] | None = None,
           warnings: list[str] | None = None, changed: list[str] | None = None) -> str:
    full = manifest.get("stage") == "full"
    judge = manifest.get("judge") or {}
    lines = [f"# Eval report — {manifest.get('run_id')}", "",
             f"- Note: {manifest.get('note') or '—'}", f"- Date: {manifest.get('date')}",
             f"- Stage: {manifest.get('stage')} · answerer: {manifest.get('answerer')} · prompt: "
             f"{manifest.get('prompt_version')} · top_k: {manifest.get('top_k')} · MMR: {manifest.get('mmr')}",
             f"- privacy_fp: {manifest.get('privacy_fp')} · question set: {str(manifest.get('question_set_sha256'))[:12]}"
             f" · index: {manifest.get('index_fp')}",
             f"- Baseline: {(baseline or {}).get('run_id') or 'none'}"
             + (f" · **Changed vs baseline:** {', '.join(changed or []) or 'nothing'}" if baseline else "")]
    if full:
        agreement = judge.get("calibration_agreement")
        lines.append(f"- **JUDGE CALIBRATED** (agreement {agreement:.0%})" if judge.get("calibrated") and agreement
                     is not None else "- **JUDGE CALIBRATED** (agreement n/a)" if judge.get("calibrated") else
                     "- **JUDGE UNCALIBRATED — judged metrics are indicative only**")
    lines += [f"- ⚠ {w}" for w in warnings or []] + [""]

    lines += ["## 1. Scorecard", ""] + _scorecard(manifest, scores, baseline) + [""]
    lines += ["## 2. Behaviour matrix", ""]
    lines += (_matrix(scores["behaviour"]) if full else ["Full stage only."]) + [""]
    lines += ["## 3. Curves and gate sweep", "", "(PNG charts come with Stage D; tables for now.)", ""]
    lines += _curves(scores) + [""]
    lines += ["## 4. Flips vs baseline", ""] + (_flips(scores, baseline_scores) if full else ["Full stage only."]) + [""]
    lines += ["## 5. Breakdowns", ""]
    for key, title in (("category", "Per category"), ("doc_group", "Per doc_group"), ("edge_tag", "Per edge tag")):
        lines += _breakdown(title, scores["breakdowns"][key])
    by_id = {q.id: q for q in questions}
    answers_by_id = {a["id"]: a for a in answers}
    judgments_by_id = {j["id"]: j for j in judgments}
    if full:
        lines += ["## 6. Worst failures (top 10)", ""] + _worst(by_id, answers_by_id, judgments_by_id, scores)
        ls = scores["lists"]
        lines += ["## 7. Lists", ""]
        lines += _list("Parametric-leak cases", [f"{c['id']}/{c['part']}" for c in ls["parametric_leak"]])
        lines += _list("World-knowledge claims", [f"{c['id']}: {c['text']}" for c in ls["world_knowledge"]])
        lines += _list("Prompt-leak cases", ls["prompt_leak"])
        lines += _list("private_leak (labels only)", [f"{c['id']}: {', '.join(c['labels'])}" for c in ls["private_leak"]])
        lines += _list("Unrelated pivots", [f"{c['id']}: {c['text']}" for c in ls["unrelated_pivots"]])
        lines += _list("Policy hits", [f"{c['id']}: {c['reason']}" for c in ls["policy"]])
        lines += ["## 8. Disagreements", ""]
        lines += _list("Judge errors", ls["judge_errors"]) + _list("Parts with unknown outcome", ls["unknown_parts"])
    lines += ["## 9. How to read this report", "", glossary.markdown().split("\n", 1)[1]]
    return "\n".join(lines) + "\n"
