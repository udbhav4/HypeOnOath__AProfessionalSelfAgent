"""
report.md (eval plan v15, Section 7): readable by anyone, numbers + deltas vs the baseline + n,
behaviour matrix, flips, breakdowns, worst failures, lists, glossary. Tables only in Stage B;
(v20, D82) a retrieval-stage run is compared against the retrieval baseline slot, with deltas per k,
per breakdown row and a per-question retrieval flip list; (v21) section 0 lists the settings (D85),
breakdowns and flips are at the run's operating k = top_k (D86) and show in_context_recall (D87);
(v22 D89, v23 D90) the PNG charts drawn by charts.py are embedded next to their tables. Replies shown are the GUARDED text; private
leaks are listed by label only.
"""
from __future__ import annotations

from typing import Any

from hypeonoath.evaluation import charts as charts_mod, glossary, paths, tracking
from hypeonoath.evaluation.metrics import METRICS, REPORT_TERMS, MetricDef
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


def _mmr_part(field: str):
    def get(settings: dict[str, Any]) -> Any:
        mmr = settings.get("mmr") or {}
        if field == "enabled":
            return bool(mmr.get("enabled")) if mmr else None
        return mmr.get(field) if mmr.get("enabled") else None          # λ / fetch_k only mean something when on
    return get


def _plain(key: str):
    return lambda settings: tracking.setting(settings, key)


# D85: the levers, in reading order; the gate threshold is a result (scorecard), not a setting.
SETTINGS_ROWS = (("top_k", _plain("top_k")), ("MMR on", _mmr_part("enabled")), ("MMR λ", _mmr_part("lambda")),
                 ("MMR fetch_k", _mmr_part("fetch_k")),
                 ("doc_description_in_chunks", _plain("doc_description_in_chunks")),
                 ("small_sections_merged", _plain("small_sections_merged")),
                 ("query_rewrite", _plain("query_rewrite")),
                 ("prompt_version", _plain("prompt_version")), ("answerer", _plain("answerer")),
                 ("ollama_model", _plain("ollama_model")), ("index_fp", _plain("index_fp")),
                 ("question set", lambda m: (str(m.get("question_set_sha256"))[:12]
                                             if m.get("question_set_sha256") else None)),
                 ("privacy_fp", _plain("privacy_fp")))


def _settings(manifest: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    """Section 0 (D85): every lever, this run vs the baseline; ✱ = changed."""
    old = (baseline or {}).get("settings")
    lines = ["| Setting | This run | Baseline | |", "|---|---|---|---|"]
    for label, get in SETTINGS_ROWS:
        new = get(manifest)
        base = get(old) if old is not None else None
        changed = "✱ changed" if old is not None and new != base else ""
        lines.append(f"| {label} | {_fmt(new)} | {_fmt(base) if old is not None else '—'} | {changed} |")
    return lines


def _covered(metrics: dict[str, Any]) -> bool:
    return all(metrics.get(k) == 1.0 for k in ("judge_coverage", "part_outcome_coverage"))


def _scorecard(manifest: dict[str, Any], scores: dict[str, Any], baseline: dict[str, Any] | None,
               baseline_scores: dict[str, Any] | None = None) -> list[str]:
    metrics, n = scores["metrics"], scores["metrics"].get("n") or 0
    base = (baseline or {}).get("metrics") or {}
    base_retrieval = (baseline_scores or {}).get("retrieval") or {}      # report-only retrieval metrics
    full_coverage = _covered(metrics) and (not base or _covered(base))
    lines = ["| Metric | This run | Baseline | Δ | | n | Denominator | Note |", "|---|---|---|---|---|---|---|---|"]
    for m in METRICS:
        if m.section == "setting" or manifest.get("stage") not in m.stages:
            continue
        value = metrics.get(m.name, scores["retrieval"].get(m.name))
        old = base.get(m.name, base_retrieval.get(m.name))
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


def _delta(new: Any, old: Any) -> float | None:
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (new, old))
    return new - old if numeric else None


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


_curve_values = charts_mod.curve_values


def _chart(drawn: charts_mod.Drawn | None, name: str) -> list[str]:
    """An embedded chart (7.1) with its glossary caption, or the reason it is missing; nothing when no
    charts were drawn (render called without them, or matplotlib missing: said once in section 3)."""
    if drawn is None or drawn.skipped:
        return []
    note = drawn.notes.get(name)
    if name not in drawn.files:
        return [f"*{name}: not drawn — {note}*", ""] if note else []
    caption = REPORT_TERMS.get(name, "")
    return [f"![{name}]({drawn.files[name]})", "", f"*{caption}*" + (f" ⚠ {note}" if note else ""), ""]


def _curves(scores: dict[str, Any], baseline_scores: dict[str, Any] | None = None) -> list[str]:
    hits, recalls = _curve_values(scores["retrieval"])
    lines = ["| k | " + " | ".join(str(k) for k in paths.K_VALUES) + " |", "|---|" + "---|" * len(paths.K_VALUES),
             "| Hit@k | " + " | ".join(map(_fmt, hits)) + " |",
             "| Recall@k | " + " | ".join(map(_fmt, recalls)) + " |"]
    if baseline_scores and baseline_scores.get("retrieval"):         # D82: the curve deltas per k
        old_hits, old_recalls = _curve_values(baseline_scores["retrieval"])
        lines += ["| Hit@k baseline | " + " | ".join(map(_fmt, old_hits)) + " |",
                  "| Δ Hit@k | " + " | ".join(_signed(_delta(a, b)) for a, b in zip(hits, old_hits)) + " |",
                  "| Recall@k baseline | " + " | ".join(map(_fmt, old_recalls)) + " |",
                  "| Δ Recall@k | " + " | ".join(_signed(_delta(a, b)) for a, b in zip(recalls, old_recalls)) + " |"]
    lines.append("")
    gate, n = scores["gate"], scores["gate"]["n"]
    order = ("answerable", "cov_only", "private", "adversarial", "other")
    sizes = ", ".join(f"{s} {n[s]}" for s in order if s in n) or "none"
    overlap = "; private overlaps adversarial" if "private" in n and "adversarial" in n else ""
    lines += [f"Gate sweep (series sizes: {sizes}{overlap}); "
              f"gate_t_at_1fr = {_fmt(gate['gate_t_at_1fr'])}. Cells: share (refused / series size).", "",
              "| t | correct refusal | false refusal | private refused | adversarial refused |", "|---|---|---|---|---|"]

    def cell(p: dict[str, Any], series: str, share_key: str) -> str:
        if not n.get(series):
            return "—"                                    # series absent in this question set
        return f"{_fmt(p[share_key])} ({p['refused'].get(series, 0)}/{n[series]})"

    lines += [f"| {p['t']:.3f} | {cell(p, 'cov_only', 'correct_refusal')} | {cell(p, 'answerable', 'false_refusal')} "
              f"| {cell(p, 'private', 'private_refused')} | {cell(p, 'adversarial', 'adversarial_refused')} |"
              for p in gate["points"]]
    return lines


# D82: Δ only for the retrieval rates (judged ones need 100 % coverage); D86 / D87: at the operating k
BREAKDOWN_RATES = ("hit@k", "recall@k", "in_context_hit", "in_context_recall", "in_context")


def _k_note(scores: dict[str, Any], baseline_scores: dict[str, Any] | None) -> str:
    """'k = 12, baseline k = 8': the operating k (= top_k) the hit@k / recall@k columns use (D86)."""
    k = scores["retrieval"].get("operating_k", paths.REFERENCE_K)
    if not baseline_scores:
        return f"k = {k}"
    old = (baseline_scores.get("retrieval") or {}).get("operating_k")
    return f"k = {k}, baseline k = {old if old is not None else '? (re-score the baseline run)'}"


def _breakdown(title: str, rows: dict[str, dict[str, Any]],
               base_rows: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """One breakdown table; with a baseline, a Δ column after every rate (D82; — = row or value missing there)."""
    if not rows:
        return []
    keys = list(next(iter(rows.values())).keys())
    deltas = {k for k in keys if k in BREAKDOWN_RATES} if base_rows is not None else set()
    header = [h for k in keys for h in ([k, f"Δ {k}"] if k in deltas else [k])]
    lines = [f"**{title}**", "", "| | " + " | ".join(header) + " |", "|---|" + "---|" * len(header)]
    for name, row in rows.items():
        old = (base_rows or {}).get(name) or {}
        cells = [c for k in keys for c in ([_fmt(row.get(k)), _signed(_delta(row.get(k), old.get(k)))]
                                         if k in deltas else [_fmt(row.get(k))])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


RETRIEVAL_FLIP_FIELDS = ("hit@k", "recall@k", "in_context_hit", "in_context_recall", "first_rank")      # D86, D87


def _retrieval_flips(scores: dict[str, Any], baseline_scores: dict[str, Any] | None) -> list[str]:
    """Per-question retrieval changes vs the baseline run (D82): hit@k / recall@k at each run's operating k
    (D86), in_context_hit / in_context_recall (D87), first matching rank. Only answerable questions present
    in both runs are compared."""
    if not baseline_scores:
        return ["No baseline run scores to compare with."]
    old_pq = (baseline_scores.get("retrieval") or {}).get("per_question")
    if old_pq is None or any("hit@k" not in v for v in old_pq.values()):     # scored before D82 / D86
        return ["The baseline run's scores.json has no per-question retrieval data: re-score the baseline run "
                "(hypeonoath-eval score RUN), then this run."]
    new_pq = scores["retrieval"].get("per_question") or {}
    both = sorted(set(new_pq) & set(old_pq))

    def mark(value: Any) -> str:
        return "—" if value is None else "✓" if value is True else "✗" if value is False else _fmt(value)

    rows = [f"| {q} | " + " | ".join(f"{mark(old_pq[q].get(f))} → {mark(new_pq[q].get(f))}"
                                    for f in RETRIEVAL_FLIP_FIELDS) + " |"
            for q in both if any(new_pq[q].get(f) != old_pq[q].get(f) for f in RETRIEVAL_FLIP_FIELDS)]
    if not rows:
        return [f"No per-question retrieval changes ({len(both)} questions compared)."]
    gained = sum(1 for q in both if new_pq[q]["hit@k"] and not old_pq[q]["hit@k"])
    lost = sum(1 for q in both if old_pq[q]["hit@k"] and not new_pq[q]["hit@k"])
    return [f"{len(both)} questions compared ({_k_note(scores, baseline_scores)}); hit@k {gained} gained, "
            f"{lost} lost.", "",
            "| Question | hit@k | recall@k | in_context_hit | in_context_recall | first rank |",
            "|---|---|---|---|---|---|"] + rows


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
           warnings: list[str] | None = None, changed: list[str] | None = None,
           charts: charts_mod.Drawn | None = None) -> str:
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

    base_scores = baseline_scores if baseline else None           # baseline run scores count only with an entry
    lines += ["## 0. Settings", ""] + _settings(manifest, baseline) + [""]
    lines += ["## 1. Scorecard", ""] + _scorecard(manifest, scores, baseline, base_scores) + [""]
    lines += _chart(charts, "context_tokens")
    lines += ["## 2. Behaviour matrix", ""]
    lines += (_matrix(scores["behaviour"]) if full else ["Full stage only."]) + [""]
    lines += ["## 3. Charts, curves and gate sweep", ""]
    if charts is not None and charts.skipped:
        lines += [charts.skipped, ""]
    lines += _chart(charts, "retrieval_curves") + _chart(charts, "runs_overlay") + _chart(charts, "threshold_sweep")
    lines += _curves(scores, base_scores) + [""]
    lines += ["## 4. Flips vs baseline", "", "**Pass ↔ fail**", ""]
    lines += (_flips(scores, base_scores) if full else ["Full stage only."]) + [""]
    lines += ["**Retrieval flips**", ""] + _chart(charts, "first_rank_shift") + _retrieval_flips(scores, base_scores)
    lines += [""]
    lines += ["## 5. Breakdowns", "", f"hit@k and recall@k at the operating k: {_k_note(scores, base_scores)}.", ""]
    base_breakdowns = (base_scores or {}).get("breakdowns")
    for key, title in (("category", "Per category"), ("doc_group", "Per doc_group"), ("edge_tag", "Per edge tag")):
        base_rows = base_breakdowns.get(key, {}) if base_breakdowns is not None else None
        lines += _breakdown(title, scores["breakdowns"][key], base_rows)
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
