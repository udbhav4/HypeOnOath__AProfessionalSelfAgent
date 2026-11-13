"""
Report charts (eval plan v22 D89, v23 D90, Section 7.1): PNGs in eval/runs/<id>/charts/, drawn on every
`run` / `score` and embedded in report.md next to their tables. Every chart shows this run and the
baseline in the same graph: same colour, this run solid / filled, the baseline dashed / hollow (T10).

Retrieval charts (both stages, from F-R): retrieval_curves, runs_overlay, first_rank_shift,
threshold_sweep, context_tokens. The judged charts (deltas, behaviour_matrix, buckets) come with Stage D.

Crowding rules: threshold_sweep panel a draws only the two series the gate is tuned on (correct /
false refusal; private and adversarial stay in panel b and the table), runs_overlay draws at most
OVERLAY_RUNS runs + the baseline, first_rank_shift merges questions on the same point into one
counted marker and labels only the largest moves.

matplotlib is imported lazily (T9): without it, or when one chart fails, the report still renders
and says why. The pure data helpers below carry the logic and are tested without drawing.
"""
from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hypeonoath.core import config
from hypeonoath.evaluation import paths

NONE_LANE = paths.EVAL_DEPTH + 2         # first_rank_shift: "no matching chunk in the top 20" (gap after 20)
MAX_RANK_LABELS = 12                     # first_rank_shift: label only the largest moves
SKIPPED_NO_MATPLOTLIB = "Charts skipped: install the `[eval]` extra (matplotlib)."
RESCORE = "re-score the baseline run"

# Okabe-Ito, colour-blind safe (T10)
BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#999999"
OVERLAY_COLOURS = (ORANGE, GREEN, PURPLE, SKY, RED)
STRIP_SERIES = ("answerable", "cov_only", "private", "adversarial")    # threshold_sweep panel b rows
DPI = 120


# --- pure data helpers ------------------------------------------------------------------------

def curve_values(r: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Hit@k and Recall@k per K_VALUES; recall_curve keys are ints in memory, strings once read from disk."""
    curve = r.get("recall_curve") or {}
    return ([r.get(f"hit@{k}") for k in paths.K_VALUES],
            [curve.get(k, curve.get(str(k))) for k in paths.K_VALUES])


def deltas(new: list[Any], old: list[Any]) -> list[float | None]:
    return [b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None for b, a in zip(new, old)]


def in_questions(delta: float | None, n: int) -> float | None:
    """A rate delta as a number of questions (T11): +0.037 at n = 54 -> +2.0."""
    return None if delta is None or not n else delta * n


def noise_band(n: int) -> float | None:
    """± one question as a share (T11; the scorecard's 'within 1 question' rule)."""
    return 1 / n if n else None


def rank_lane(rank: int | None) -> int:
    return rank if rank is not None else NONE_LANE


@dataclass(frozen=True)
class RankPoint:
    qid: str
    old: int        # lane (rank, or NONE_LANE)
    new: int
    old_hit: bool   # hit@k at the baseline's operating k
    new_hit: bool   # hit@k at this run's operating k

    @property
    def change(self) -> str:
        if self.new_hit and not self.old_hit:
            return "gained"
        if self.old_hit and not self.new_hit:
            return "lost"
        return "moved" if self.old != self.new else "same"


def rank_points(new_pq: dict[str, Any], old_pq: dict[str, Any]) -> list[RankPoint]:
    """One point per answerable question present in both runs (as the D82 flip list)."""
    return [RankPoint(q, rank_lane(old_pq[q]["first_rank"]), rank_lane(new_pq[q]["first_rank"]),
                      bool(old_pq[q]["hit@k"]), bool(new_pq[q]["hit@k"]))
            for q in sorted(set(new_pq) & set(old_pq))]


def labelled_moves(points: list[RankPoint], limit: int = MAX_RANK_LABELS) -> tuple[list[RankPoint], int]:
    """The points to label: hit flips first, then the longest rank moves; returns (labelled, not labelled)."""
    moved = [p for p in points if p.old != p.new]
    moved.sort(key=lambda p: (p.change not in ("gained", "lost"), -abs(p.new - p.old), p.qid))
    return moved[:limit], max(0, len(moved) - limit)


def _float(cell: Any) -> float | None:
    try:
        return float(cell) if cell not in (None, "") else None
    except (TypeError, ValueError):
        return None


def overlay_runs(rows: list[dict[str, str]], manifest: dict[str, Any], baseline_id: str | None,
                 limit: int = paths.OVERLAY_RUNS) -> tuple[list[dict[str, str]], int]:
    """T13: the other runs of runs_overlay.png — rows with run_id < this run, the same question set and
    privacy_fp, not this run or the baseline (both drawn from their scores), the latest limit - 1 of them.
    Returns (rows, number of earlier rows skipped as not comparable)."""
    run_id = str(manifest.get("run_id"))
    earlier = [r for r in rows if r.get("run_id") and r["run_id"] < run_id and r["run_id"] != baseline_id]
    same = [r for r in earlier if r.get("question_set_sha") == manifest.get("question_set_sha256")
            and r.get("privacy_fp") == manifest.get("privacy_fp")
            and any(v is not None for v in row_curves(r)[0])]                  # nothing to draw -> no slot
    same.sort(key=lambda r: r["run_id"])
    return same[-(limit - 1):] if limit > 1 else [], len(earlier) - len(same)


def row_curves(row: dict[str, str]) -> tuple[list[float | None], list[float | None]]:
    return ([_float(row.get(f"hit@{k}")) for k in paths.K_VALUES],
            [_float(row.get(f"recall@{k}")) for k in paths.K_VALUES])


def experiment_tags(settings: dict[str, Any] | None) -> tuple[str, ...]:
    """Legend tags of the F-R3 / F-R4 levers that are on (risk 12: index variants stay in the overlay but their
    legend shows the setting). Reads a manifest / baseline settings (bools) or a runs.csv row (strings)."""
    settings = settings or {}

    def on(key: str) -> bool:
        value = settings.get(key)
        return value is True or str(value).lower() == "true"

    tags = []
    if settings.get("query_rewrite") not in (None, "", "none"):
        tags.append("3rd-person" if settings["query_rewrite"] == "third_person" else str(settings["query_rewrite"]))
    if on("small_sections_merged"):
        tags.append("merged")
    if on("doc_description_in_chunks"):
        tags.append("desc")
    return tuple(tags)


def settings_label(top_k: Any, mmr_lambda: Any, tags: tuple[str, ...] = ()) -> str:
    """Short lever summary for legends, e.g. 'k12 λ0.7' or 'k12 3rd-person' (lambda None / empty = MMR off)."""
    label = f"k{top_k}" if top_k not in (None, "") else "k?"
    lam = _float(mmr_lambda)
    return " ".join([label + (f" λ{lam:g}" if lam is not None else ""), *tags])


def mmr_lambda(settings: dict[str, Any] | None) -> Any:
    """The MMR lambda of a manifest / baseline settings, None when MMR is off."""
    mmr = (settings or {}).get("mmr") or {}
    return mmr.get("lambda") if mmr.get("enabled") else None


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    return xs, [(i + 1) / len(xs) for i in range(len(xs))]


def _short(text: Any, n: int = 32) -> str:
    text = " ".join(str(text or "").split())          # a note can't add title lines
    return text if len(text) <= n else text[: n - 1] + "…"


# --- drawing ---------------------------------------------------------------------------------

@dataclass
class Context:
    """Everything a chart may read (T12: scores of this run and the baseline run, runs.csv for the overlay)."""
    manifest: dict[str, Any]
    scores: dict[str, Any]
    baseline: dict[str, Any] | None          # baseline.json entry (run_id, note, settings)
    baseline_scores: dict[str, Any] | None
    changed: list[str] | None
    runs_rows: list[dict[str, str]]

    @property
    def n(self) -> int:
        return int(self.scores["retrieval"].get("n_answerable") or 0)

    @property
    def k(self) -> int:
        return int(self.scores["retrieval"].get("operating_k") or self.manifest.get("top_k") or paths.REFERENCE_K)

    @property
    def base_r(self) -> dict[str, Any] | None:
        """The baseline run's retrieval scores; the only place that ignores scores without a baseline entry."""
        return (self.baseline_scores or {}).get("retrieval") if self.baseline else None

    @property
    def base_k(self) -> int | None:
        """The baseline's operating k; scores from before D86 have none -> its top_k setting."""
        if not self.base_r:
            return None
        k = self.base_r.get("operating_k") or (self.baseline.get("settings") or {}).get("top_k")
        return int(k) if k is not None else None

    @property
    def base_note(self) -> str | None:
        """Why a comparison is empty or trivial (T12), for the charts that compare."""
        if not self.baseline:
            return None
        if not self.base_r:
            return f"baseline {self.baseline.get('run_id')} has no scores.json: {RESCORE}"
        if self.baseline.get("run_id") == self.manifest.get("run_id"):
            return "this run is the baseline: every Δ is 0"
        return None


def _title(fig, ctx: Context, name: str) -> None:
    """Chart name, this run vs the baseline, what changed and n — one line each (T10)."""
    m = ctx.manifest
    base = (f"baseline {ctx.baseline.get('run_id')} · {_short(ctx.baseline.get('note'), 40) or '—'}"
            if ctx.baseline else "no baseline")
    keys = list(ctx.changed or [])
    changed = ", ".join(keys[:5]) + (f" +{len(keys) - 5} more" if len(keys) > 5 else "")
    changed = changed or ("nothing" if ctx.baseline else "—")
    fig.suptitle(f"{name}\nthis run {m.get('run_id')} · {_short(m.get('note'), 40) or '—'}   vs   {base}\n"
                 + textwrap.fill(f"changed: {changed} · n answerable = {ctx.n}", 100), fontsize=9)


def _curve_panel(ax, ks, new, old, colour, label, k, base_k, ctx_new, ctx_old, ctx_label) -> None:
    ax.plot(ks, new, "-o", color=colour, lw=2, ms=5, zorder=3, label=f"{label} this run")
    if old is not None:      # hollow (mfc none) and below: equal values show this run's filled marker inside
        ax.plot(ks, old, "--o", color=colour, lw=1.5, ms=8, mfc="none", zorder=2, label=f"{label} baseline")
        ok = [(x, a, b) for x, a, b in zip(ks, new, old) if a is not None and b is not None]
        if ok:
            xs, a, b = zip(*ok)
            ax.fill_between(xs, a, b, where=[p >= q for p, q in zip(a, b)], color=GREEN, alpha=0.12, interpolate=True)
            ax.fill_between(xs, a, b, where=[p < q for p, q in zip(a, b)], color=RED, alpha=0.12, interpolate=True)
    ax.axvline(k, color="#444444", lw=1, label=f"operating k = {k}")
    if base_k is not None and base_k != k:
        ax.axvline(base_k, color="#444444", lw=1, ls=":", label=f"baseline k = {base_k}")
    if k in ks and new[ks.index(k)] is not None:
        ax.annotate(f"{new[ks.index(k)]:.3f}", (k, new[ks.index(k)]), textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color=colour)
    if ctx_new is not None:
        ax.plot([k], [ctx_new], "*", color=colour, ms=11, mec="black", mew=0.5, zorder=5,
                label=f"{ctx_label} this run")
    if ctx_old is not None and base_k is not None:
        ax.plot([base_k], [ctx_old], "*", color="none", ms=16, mec=colour, mew=1.2, zorder=4,
                label=f"{ctx_label} baseline")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)


def draw_retrieval_curves(plt, ctx: Context):
    r, base = ctx.scores["retrieval"], ctx.base_r
    ks = list(paths.K_VALUES)
    hits, recalls = curve_values(r)
    old_hits, old_recalls = curve_values(base) if base else (None, None)
    notes = [ctx.base_note] if ctx.base_note else []
    if old_recalls is not None and all(v is None for v in old_recalls):
        old_recalls = None                               # nothing to draw: no empty legend entry
        notes.append(f"baseline has no recall values: {RESCORE}")
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [3, 3, 2.4]})
    _title(fig, ctx, "Retrieval curves")
    _curve_panel(a, ks, hits, old_hits, BLUE, "Hit@k", ctx.k, ctx.base_k, r.get("in_context_hit"),
                 (base or {}).get("in_context_hit"), "in_context_hit")
    _curve_panel(b, ks, recalls, old_recalls, ORANGE, "Recall@k", ctx.k, ctx.base_k, r.get("in_context_recall"),
                 (base or {}).get("in_context_recall"), "in_context_recall")
    if base:
        band = noise_band(ctx.n) or 0.0
        both = (deltas(hits, old_hits), deltas(recalls, old_recalls or [None] * len(ks)))
        # symmetric y range with headroom for the labels; never smaller than the band, so ± 1 q stays readable
        top = max([abs(v) for vs in both for v in vs if v is not None] + [band, 0.01]) * 2.0
        c.set_ylim(-top, top)
        if band:
            c.axhspan(-band, band, color=GREY, alpha=0.18, label="± 1 question (noise level)")
        # k = 3, 4, 5 are one unit apart: narrow bars so neighbouring k's never overlap
        # recall labels sit one line further out, so equal-height bars don't print on top of each other
        for offset, values, colour, label, fmt, gap in ((-0.13, both[0], BLUE, "Δ Hit@k", "{:+.0f} q", 0),
                                                        (0.13, both[1], ORANGE, "Δ Recall@k", "{:+.1f} q", 9)):
            xs = [k + offset for k, v in zip(ks, values) if v is not None]
            vs = [v for v in values if v is not None]
            if not vs:
                continue
            c.bar(xs, vs, width=0.24, color=colour, label=label)
            for x, v in zip(xs, vs):
                q = in_questions(v, ctx.n)
                if q is not None and abs(q) >= 0.05:            # 0 q bars stay unlabelled (less clutter)
                    c.annotate(fmt.format(q), (x, v), textcoords="offset points", xytext=(0, 2 + gap if v >= 0 else -9 - gap),
                               ha="center", fontsize=6.5, color=colour)
        if all(abs(v) < 1e-12 for vs in both for v in vs if v is not None):
            c.text(0.5, 0.75, "no change at any k", ha="center", transform=c.transAxes, color="#444444")
        c.axhline(0, color="black", lw=0.8)
        c.set_title("Δ = this run − baseline (labels in questions; recall = question-equivalents)", fontsize=8)
        if c.get_legend_handles_labels()[0]:              # n = 0 and no Δ: nothing to list
            c.legend(fontsize=7, loc="best", ncol=3, framealpha=0.9)
    else:
        c.text(0.5, 0.5, ctx.base_note or "no baseline: no Δ", ha="center", va="center", transform=c.transAxes,
               color="#666666")
        if not ctx.baseline:
            notes.append("no baseline: curves of this run only")
    c.set_ylabel("Δ (share)")
    c.set_xlabel("k")
    c.set_xticks(ks)
    c.grid(alpha=0.3)
    return fig, "; ".join(notes) or None


def draw_runs_overlay(plt, ctx: Context):
    others, skipped = overlay_runs(ctx.runs_rows, ctx.manifest, (ctx.baseline or {}).get("run_id"))
    ks = list(paths.K_VALUES)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 6), sharey=True, layout="constrained")
    _title(fig, ctx, "Runs overlay")
    no_recall = 0
    for row, colour in zip(others, OVERLAY_COLOURS):
        hits, recalls = row_curves(row)
        label = " · ".join(x for x in (row["run_id"], _short(row.get("note"), 24),
                                       settings_label(row.get("top_k"), row.get("mmr_lambda"),
                                                      experiment_tags(row))) if x)
        _overlay_line(a, ks, hits, colour, label, _float(row.get("top_k")), "-", 1.2)
        if any(v is not None for v in recalls):
            _overlay_line(b, ks, recalls, colour, label, _float(row.get("top_k")), "-", 1.2)
        else:
            no_recall += 1
    if ctx.base_r:
        hits, recalls = curve_values(ctx.base_r)
        base_settings = ctx.baseline.get("settings")
        lever = settings_label(ctx.base_k, mmr_lambda(base_settings), experiment_tags(base_settings))
        label = f"baseline {ctx.baseline.get('run_id')} · {lever}"
        _overlay_line(a, ks, hits, "black", label, ctx.base_k, "--", 1.5)
        _overlay_line(b, ks, recalls, "black", label, ctx.base_k, "--", 1.5)
    hits, recalls = curve_values(ctx.scores["retrieval"])
    lever = settings_label(ctx.k, mmr_lambda(ctx.manifest), experiment_tags(ctx.manifest))
    label = f"this run {ctx.manifest.get('run_id')} · {lever}"
    _overlay_line(a, ks, hits, BLUE, label, ctx.k, "-", 2.8)
    _overlay_line(b, ks, recalls, BLUE, label, ctx.k, "-", 2.8)
    for ax, name in ((a, "Hit@k"), (b, "Recall@k")):
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("k")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    notes = [ctx.base_note] if ctx.base_note else []
    if skipped:
        notes.append(f"{skipped} earlier run(s) left out: other question set or privacy_fp")
    if no_recall:
        notes.append(f"{no_recall} run(s) without recall columns left out of the Recall panel (re-score them)")
    return fig, "; ".join(notes) or None


def _overlay_line(ax, ks, values, colour, label, top_k, style, width) -> None:
    pts = [(k, v) for k, v in zip(ks, values) if v is not None]
    if not pts:
        return
    xs, vs = zip(*pts)
    ax.plot(xs, vs, style, color=colour, lw=width, label=label)
    if top_k is not None and int(top_k) in ks and values[ks.index(int(top_k))] is not None:
        ax.plot([int(top_k)], [values[ks.index(int(top_k))]], "o", color=colour, ms=7, mec="black", mew=0.6)


def draw_first_rank_shift(plt, ctx: Context):
    new_pq = ctx.scores["retrieval"].get("per_question") or {}
    old_pq = (ctx.base_r or {}).get("per_question")
    if not ctx.base_r:
        return None, ctx.base_note or "needs a baseline"
    if old_pq is None or any("hit@k" not in v for v in old_pq.values()):
        return None, f"no per-question data in the baseline scores: {RESCORE}"
    points = rank_points(new_pq, old_pq)
    if not points:
        return None, "no answerable question in both runs"
    k, bk = ctx.k, ctx.base_k or paths.REFERENCE_K
    fig, ax = plt.subplots(figsize=(8, 8.4), layout="constrained")
    _title(fig, ctx, "First matching rank")
    colours = {"gained": GREEN, "lost": RED, "moved": BLUE, "same": "#BBBBBB"}
    groups: dict[tuple[int, int, str], list[str]] = {}
    for p in points:
        groups.setdefault((p.old, p.new, p.change), []).append(p.qid)
    for (x, y, change), qids in groups.items():
        ax.scatter([x], [y], s=40 + 30 * (len(qids) - 1), color=colours[change], edgecolor="black", lw=0.4, zorder=3,
                   alpha=0.9 if change != "same" else 0.6)
        if len(qids) > 1:
            ax.annotate(str(len(qids)), (x, y), ha="center", va="center", fontsize=6, zorder=4)
    labelled, rest = labelled_moves(points)
    labels: dict[tuple[int, int, str], list[str]] = {}
    for p in labelled:                                  # one label per point, ids joined (no stacked text)
        labels.setdefault((p.old, p.new, p.change), []).append(p.qid)
    for (x, y, change), qids in labels.items():
        left = x > paths.EVAL_DEPTH                     # the "none" column is at the right edge: label leftwards
        ax.annotate(", ".join(qids), (x, y), textcoords="offset points", xytext=(-6 if left else 6, 4), fontsize=7,
                    ha="right" if left else "left", color=colours[change])
    ax.plot([0.5, NONE_LANE + 0.5], [0.5, NONE_LANE + 0.5], color=GREY, lw=1, zorder=1)
    ax.axvline(bk + 0.5, color="#444444", ls="--", lw=0.8)
    ax.axhline(k + 0.5, color="#444444", ls="--", lw=0.8)
    count = Counter(p.change for p in points)
    kept = sum(1 for p in points if p.old_hit and p.new_hit)
    both_missed = sum(1 for p in points if not p.old_hit and not p.new_hit)
    ax.set_title(f"hit@k: kept {kept} · gained {count['gained']} (green) · lost {count['lost']} (red) · "
                 f"missed in both {both_missed} · moved without a flip {count['moved']} (blue)", fontsize=8)
    ticks = [t for t in paths.K_VALUES] + [NONE_LANE]
    tick_labels = [str(t) for t in paths.K_VALUES] + ["none"]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_ticklabels(tick_labels)
    ax.set_xlim(0.5, NONE_LANE + 0.5)
    ax.set_ylim(0.5, NONE_LANE + 0.5)
    ax.set_xlabel(f"baseline first rank (dashed: baseline k = {bk})")
    ax.set_ylabel(f"this run first rank (dashed: k = {k})")
    ax.grid(alpha=0.2)
    notes = [ctx.base_note] if ctx.base_note else []
    if rest:
        notes.append(f"{rest} more moved question(s) not labelled: see the retrieval flip table")
    return fig, "; ".join(notes) or None


def draw_threshold_sweep(plt, ctx: Context):
    gate = ctx.scores["gate"]
    base_gate = (ctx.baseline_scores or {}).get("gate") if ctx.baseline else None
    if not gate.get("points"):
        return None, "no top-1 scores: nothing to sweep"
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, layout="constrained",
                               gridspec_kw={"height_ratios": [3, 2]})
    _title(fig, ctx, "Similarity-gate sweep")
    for g, style, who in ((gate, "-", "this run"), (base_gate, "--", "baseline")):
        if not g or not g.get("points"):
            continue
        ts = [p["t"] for p in g["points"]]
        for key, colour, name in (("correct_refusal", RED, "correct refusal (cov_only)"),
                                  ("false_refusal", BLUE, "false refusal (answerable)")):
            vals = [p.get(key) for p in g["points"]]
            if any(v is not None for v in vals):
                a.step(ts, [v if v is not None else float("nan") for v in vals], where="post", ls=style, color=colour,
                       lw=2 if who == "this run" else 1.4, label=f"{name} — {who}")
        if g.get("gate_t_at_1fr") is not None:
            a.axvline(g["gate_t_at_1fr"], color="#444444", ls=style if who == "baseline" else "-", lw=1,
                      label=f"gate_t_at_1fr {g['gate_t_at_1fr']:.3f} — {who}")
    n_ans = (gate.get("n") or {}).get("answerable")
    if n_ans:
        a.axhline(config.GATE_MAX_FALSE_REFUSALS / n_ans, color=BLUE, ls=":", lw=1,
                  label=f"false-refusal limit ({config.GATE_MAX_FALSE_REFUSALS} q)")
    t, n_cov = gate.get("gate_t_at_1fr"), (gate.get("n") or {}).get("cov_only")
    if t is not None and gate.get("gate_correct_refusal_at_t") is not None:
        at = next(p for p in gate["points"] if p["t"] == t)
        a.annotate(f"{gate['gate_correct_refusal_at_t']:.3f} ({at['refused'].get('cov_only', 0)}/{n_cov})",
                   (t, gate["gate_correct_refusal_at_t"]), textcoords="offset points", xytext=(6, 4), fontsize=8,
                   color=RED)
    a.set_ylim(0, 1.02)
    a.set_ylabel("share refused")
    a.grid(alpha=0.3)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)   # outside: curves stay clear
    a.set_title("a. gate rates over t (private / adversarial: panel b and the table)", fontsize=8, loc="left")

    note = None
    rows = gate.get("top1_by_question")
    base_rows = (base_gate or {}).get("top1_by_question") if base_gate else None
    if rows is None:
        b.text(0.5, 0.5, "no per-question top-1 in these scores: re-score this run", ha="center",
               transform=b.transAxes)
    else:
        for y, series in enumerate(STRIP_SERIES):
            for data, offset, filled in ((rows, 0.14, True), (base_rows, -0.14, False)):
                if not data:
                    continue
                xs = [v["top1"] for v in data.values() if _in_series(v, series)]
                ys = [y + offset + _jitter(i) for i in range(len(xs))]
                b.scatter(xs, ys, s=18, color=_SERIES_COLOUR[series] if filled else "white",
                          edgecolor=_SERIES_COLOUR[series], lw=0.9, zorder=3)
        b.set_yticks(range(len(STRIP_SERIES)))
        b.set_yticklabels([f"{s} ({sum(_in_series(v, s) for v in rows.values())})" for s in STRIP_SERIES], fontsize=8)
        b.set_ylim(-0.6, len(STRIP_SERIES) - 0.4)
        for g, style in ((gate, "-"), (base_gate, "--")):
            if g and g.get("gate_t_at_1fr") is not None:
                b.axvline(g["gate_t_at_1fr"], color="#444444", ls=style, lw=1)
        if ctx.baseline and base_rows is None:
            note = f"panel b: no per-question top-1 in the baseline scores ({RESCORE})"
    b.set_title("b. top-1 similarity per question (filled = this run, hollow = baseline)", fontsize=8, loc="left")
    b.set_xlabel("top-1 similarity t (a question is refused when top1 < t)")
    b.grid(alpha=0.3, axis="x")
    return fig, note


_SERIES_COLOUR = {"answerable": BLUE, "cov_only": RED, "private": PURPLE, "adversarial": ORANGE}


def _in_series(row: dict[str, Any], series: str) -> bool:
    """Strip rows as in the gate table: private overlaps adversarial (D79)."""
    if series == "private":
        return bool(row.get("private_only")) or row.get("series") == "private"
    return row.get("series") == series


def _jitter(i: int) -> float:
    """Deterministic vertical spread so equal scores don't hide each other (no randomness: same PNG each time)."""
    return ((i * 37) % 11 - 5) * 0.012


def draw_context_tokens(plt, ctx: Context):
    new = list((ctx.scores["retrieval"].get("context_tokens_by_question") or {}).values())
    if not new:
        return None, "no context token counts in this run"
    old_map = (ctx.base_r or {}).get("context_tokens_by_question")
    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    _title(fig, ctx, "Context tokens")
    note = None
    series = [(new, ctx.scores["retrieval"], "-", "this run", 2.2)]
    if ctx.base_r and old_map:
        series.append((list(old_map.values()), ctx.base_r, "--", "baseline", 1.5))
    elif ctx.base_r:
        note = f"baseline distribution missing: {RESCORE}"
    for values, r, style, who, width in series:
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", ls=style, color=BLUE, lw=width,
                label=f"{who}: mean {r.get('context_tokens_mean') or 0:,.0f} · p95 {r.get('context_tokens_p95') or 0:,.0f}")
        for value, colour in ((r.get("context_tokens_mean"), GREEN), (r.get("context_tokens_p95"), RED)):
            if value is not None:
                ax.axvline(value, color=colour, ls=style, lw=1)
    ax.axvline(config.MAX_CONTEXT_TOKENS, color=GREY, ls=":", lw=1.2,
               label=f"MAX_CONTEXT_TOKENS {config.MAX_CONTEXT_TOKENS:,} (sibling budget, not a cap)")
    ax.set_xlabel("context tokens per question (green = mean, red = p95)")
    ax.set_ylabel("share of questions ≤ x")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    return fig, note


@dataclass(frozen=True)
class ChartDef:
    name: str                                   # file stem = glossary term (D37)
    draw: Callable[..., tuple[Any, str | None]]


# 7.1 (D90): the retrieval charts, drawn in both stages; report.render places each next to its table.
# Judged charts are added with Stage D.
CHARTS = (
    ChartDef("context_tokens", draw_context_tokens),
    ChartDef("retrieval_curves", draw_retrieval_curves),
    ChartDef("runs_overlay", draw_runs_overlay),
    ChartDef("threshold_sweep", draw_threshold_sweep),
    ChartDef("first_rank_shift", draw_first_rank_shift),
)


@dataclass
class Drawn:
    """What draw_all produced: files {name: relative path}, notes {name: why missing / caveat}."""
    files: dict[str, str]
    notes: dict[str, str]
    skipped: str | None = None


def draw_all(run_dir: Path, manifest: dict[str, Any], scores: dict[str, Any], baseline: dict[str, Any] | None,
             baseline_scores: dict[str, Any] | None, changed: list[str] | None,
             runs_rows: list[dict[str, str]]) -> Drawn:
    """Redraw every chart of the run (T14: old PNGs removed first). Never raises because of a chart (T9)."""
    out_dir = run_dir / paths.CHARTS_DIRNAME
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return Drawn({}, {}, SKIPPED_NO_MATPLOTLIB)
    except Exception as exc:                     # a broken install must not lose the report either (T9)
        return Drawn({}, {}, f"Charts skipped: matplotlib failed to load ({type(exc).__name__}: {exc}).")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Drawn({}, {}, f"Charts skipped: cannot create {out_dir.name}/ ({exc}).")
    notes: dict[str, str] = {}
    for old in out_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError as exc:                   # e.g. open in a viewer on Windows; savefig may still replace it
            notes[old.stem] = f"old chart not removed: {exc}"
    ctx = Context(manifest, scores, baseline, baseline_scores, changed, runs_rows)
    files: dict[str, str] = {}
    for chart in CHARTS:
        try:
            fig, note = chart.draw(plt, ctx)
            if fig is not None:
                fig.savefig(out_dir / f"{chart.name}.png", dpi=DPI, metadata={"Software": None})
                files[chart.name] = f"{paths.CHARTS_DIRNAME}/{chart.name}.png"
            if note:
                notes[chart.name] = "; ".join(filter(None, (notes.get(chart.name), note)))
        except Exception as exc:                 # a broken chart must not lose the report (T9)
            notes[chart.name] = f"chart failed: {type(exc).__name__}: {exc}"
        finally:
            plt.close("all")                     # also a figure created before the error
    return Drawn(files, notes)
