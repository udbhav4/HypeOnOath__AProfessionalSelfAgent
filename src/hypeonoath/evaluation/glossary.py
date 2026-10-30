"""
Metric glossary from the METRICS registry (eval plan v15, D18, D37): Markdown for report section 9
and eval/METRICS.md, plain text for `hypeonoath-eval metrics [NAME]`.
"""
from __future__ import annotations

from hypeonoath.evaluation.metrics import METRICS, METRICS_BY_NAME, REPORT_TERMS, MetricDef

SECTIONS = {
    "setting": "Run identity and settings",
    "retrieval": "Retrieval (code, no LLM)",
    "gate": "Similarity-gate sweep (code)",
    "behaviour": "Behaviour per part (judge observation + code arithmetic)",
    "content": "Content quality (judge + code)",
    "safety": "Safety (code + judge)",
    "operations": "Operations (code)",
}

DENOMINATOR_RULES = """\
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
  index and privacy fingerprint; look at flips, not only averages."""


def markdown_entry(m: MetricDef) -> str:
    column = f"`{m.csv_column}`" if m.csv_column else "report only"
    return (f"| **{m.name}** | {column} | {m.definition} | {m.why} | {m.how_to_read} | {m.denominator} | "
            f"{m.direction} | {', '.join(m.stages)} |")


def markdown() -> str:
    lines = ["# Eval metrics glossary", "",
             "Generated from `hypeonoath.evaluation.metrics.METRICS` (D37) by `hypeonoath-eval metrics`; "
             "do not edit by hand.", "", "## How to read the numbers", "", DENOMINATOR_RULES, ""]
    for key, title in SECTIONS.items():
        rows = [m for m in METRICS if m.section == key]
        if not rows:
            continue
        lines += [f"## {title}", "",
                  "| Metric | runs.csv | Definition | Why it matters | How to read it | Denominator | Better | Stages |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [markdown_entry(m) for m in rows] + [""]
    lines += ["## Report terms", "", "| Term | Meaning |", "|---|---|"]
    lines += [f"| **{name}** | {text} |" for name, text in REPORT_TERMS.items()] + [""]
    return "\n".join(lines)


def text(name: str | None = None) -> str:
    """Plain-text definition(s); unknown name -> ValueError."""
    if name is None:
        return "\n\n".join(text(m.name) for m in METRICS)
    m = METRICS_BY_NAME.get(name)
    if m is None:
        if name in REPORT_TERMS:
            return f"{name}\n  {REPORT_TERMS[name]}"
        raise ValueError(f"unknown metric {name!r}")
    return (f"{m.name}  (runs.csv: {m.csv_column or '-'}; better: {m.direction}; stages: {', '.join(m.stages)})\n"
            f"  What: {m.definition}\n  Why: {m.why}\n  Read: {m.how_to_read}\n  Denominator: {m.denominator}")
