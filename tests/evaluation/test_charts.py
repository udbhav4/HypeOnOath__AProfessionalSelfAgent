"""Report charts (eval plan v22 D89, v23 D90, 7.1): data helpers without drawing, then the PNGs (Agg)."""
from __future__ import annotations

import builtins
import json
import subprocess
import sys

import pytest

from hypeonoath.evaluation import charts, paths, tracking
from hypeonoath.evaluation.metrics import REPORT_TERMS, score_run
from hypeonoath.evaluation.report import render
from tests.evaluation.test_outcomes_metrics import NC, A, Q, rec
from tests.evaluation.test_tracking_report import ROOT, manifest, two_retrieval_runs

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def baseline_entry(run_id="r0", **settings):
    return {"run_id": run_id, "note": "ref", "settings": {k: tracking.setting(manifest(**settings), k)
                                                          for k in tracking.SETTING_KEYS}}


def csv_row(run_id, qs="qs", fp="fp", note="", top_k="8", mmr_lambda="", hit="0.5", recall="0.4"):
    row = {"run_id": run_id, "note": note, "question_set_sha": qs, "privacy_fp": fp, "top_k": top_k,
           "mmr_lambda": mmr_lambda}
    row.update({f"hit@{k}": hit for k in paths.K_VALUES})
    row.update({f"recall@{k}": recall for k in paths.K_VALUES})
    return row


# --- data helpers ------------------------------------------------------------------------------

def test_deltas_in_questions_and_noise_band():
    assert charts.deltas([0.5, None, 0.9], [0.4, 0.3, None]) == pytest.approx([0.1, None, None])
    assert charts.in_questions(2 / 54, 54) == pytest.approx(2.0)
    assert charts.in_questions(None, 54) is None and charts.in_questions(0.1, 0) is None
    assert charts.noise_band(54) == pytest.approx(1 / 54) and charts.noise_band(0) is None


def test_curve_values_read_int_and_string_recall_keys():
    _, _, scores, base = two_retrieval_runs()                       # base went through json: string keys
    for r in (scores["retrieval"], base["retrieval"]):
        hits, recalls = charts.curve_values(r)
        assert hits == [r[f"hit@{k}"] for k in paths.K_VALUES]
        assert recalls == [r[f"recall@{k}"] for k in paths.K_VALUES] and None not in recalls


def test_rank_points_lanes_and_hit_changes():
    old = {"q1": {"hit@k": True, "first_rank": 1}, "q2": {"hit@k": False, "first_rank": None},
           "q3": {"hit@k": True, "first_rank": 2}, "q4": {"hit@k": True, "first_rank": 3}, "only_old": {}}
    new = {"q1": {"hit@k": False, "first_rank": None}, "q2": {"hit@k": True, "first_rank": 4},
           "q3": {"hit@k": True, "first_rank": 5}, "q4": {"hit@k": True, "first_rank": 3}, "only_new": {}}
    points = {p.qid: p for p in charts.rank_points(new, old)}
    assert set(points) == {"q1", "q2", "q3", "q4"}                      # questions in both runs only
    assert (points["q1"].old, points["q1"].new, points["q1"].change) == (1, charts.NONE_LANE, "lost")
    assert (points["q2"].old, points["q2"].change) == (charts.NONE_LANE, "gained")
    assert points["q3"].change == "moved" and points["q4"].change == "same"
    assert charts.NONE_LANE > paths.EVAL_DEPTH


def test_labelled_moves_puts_flips_first_then_longest_moves():
    P = charts.RankPoint
    points = [P("a", 1, 2, True, True), P("b", 1, 15, True, True), P("c", 9, 3, False, True), P("d", 4, 4, True, True)]
    labelled, rest = charts.labelled_moves(points, limit=2)
    assert [p.qid for p in labelled] == ["c", "b"] and rest == 1          # unchanged d never counted


def test_overlay_runs_filters_orders_and_caps():
    rows = [csv_row("r1"), csv_row("r2"), csv_row("r3", qs="other"), csv_row("r4", fp="other"), csv_row("r5"),
            csv_row("r6"), csv_row("r7"), csv_row("r9")]
    m = manifest(run_id="r8")
    others, skipped = charts.overlay_runs(rows, m, baseline_id="r6", limit=4)
    assert [r["run_id"] for r in others] == ["r2", "r5", "r7"]       # latest 3 before r8, baseline + later out
    assert skipped == 2                                              # r3 (question set), r4 (privacy_fp)
    assert charts.overlay_runs(rows, m, None, limit=1) == ([], 2)


def test_settings_label_and_row_curves():
    assert charts.settings_label("12", "0.7000") == "k12 λ0.7"
    assert charts.settings_label(8, "") == "k8" and charts.settings_label("", None) == "k?"
    assert charts.mmr_lambda({"mmr": {"enabled": False, "lambda": 0.5}}) is None
    assert charts.mmr_lambda({"mmr": {"enabled": True, "lambda": 0.5}}) == 0.5
    hits, recalls = charts.row_curves(csv_row("r1", recall=""))
    assert hits == [0.5] * len(paths.K_VALUES) and recalls == [None] * len(paths.K_VALUES)


def test_ecdf():
    assert charts.ecdf([3, 1, 2]) == ([1, 2, 3], pytest.approx([1 / 3, 2 / 3, 1.0]))


def test_scores_carry_the_per_question_maps_for_the_baseline_charts():
    qs = [Q("q1", A("a")), Q("q2", NC("a"))]
    scores = score_run(qs, [rec("q1", [["a:0"]], ["a:0"], tokens=40), rec("q2", [[]], tokens=7, top1=0.3)], [],
                       {"stage": "retrieval"})
    assert scores["retrieval"]["context_tokens_by_question"] == {"q1": 40, "q2": 7}
    assert scores["gate"]["top1_by_question"]["q2"] == {"top1": 0.3, "series": "cov_only", "private_only": False}


SECTION = {"context_tokens": 1, "retrieval_curves": 3, "runs_overlay": 3, "threshold_sweep": 3,
           "first_rank_shift": 4}                                              # 7.1 (D90)


def test_every_chart_has_a_glossary_term_and_a_report_section():
    assert {c.name for c in charts.CHARTS} == set(SECTION)                    # D90: 5 now
    assert all(c.name in REPORT_TERMS for c in charts.CHARTS)


# --- drawing -----------------------------------------------------------------------------------

@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


def _draw(tmp_path, with_baseline=True, base_scores=None, rows=()):
    _, _, scores, base = two_retrieval_runs()
    scores = json.loads(json.dumps(scores))
    baseline = baseline_entry() if with_baseline else None
    return charts.draw_all(tmp_path, manifest(run_id="r1"), scores, baseline,
                           base if base_scores is None else base_scores, ["top_k"] if with_baseline else None,
                           list(rows)), scores


def test_all_retrieval_charts_written_with_a_baseline(mpl, tmp_path):
    drawn, _ = _draw(tmp_path, rows=[csv_row("r0", note="ref"), csv_row("q0")])
    assert drawn.skipped is None and set(drawn.files) == {c.name for c in charts.CHARTS}, drawn.notes
    for rel in drawn.files.values():
        data = (tmp_path / rel).read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > 5000
    assert not any("failed" in n for n in drawn.notes.values())
    assert not (tmp_path / "charts" / "deltas.png").exists()           # judged charts: Stage D (D90)


def test_without_a_baseline_first_rank_is_not_drawn_and_the_rest_are(mpl, tmp_path):
    drawn, _ = _draw(tmp_path, with_baseline=False)
    assert "first_rank_shift" not in drawn.files and drawn.notes["first_rank_shift"] == "needs a baseline"
    assert {"retrieval_curves", "runs_overlay", "threshold_sweep", "context_tokens"} <= set(drawn.files)
    assert "no baseline" in drawn.notes["retrieval_curves"]


def test_old_format_baseline_scores_say_rescore(mpl, tmp_path):
    _, _, _, base = two_retrieval_runs()
    del base["retrieval"]["context_tokens_by_question"], base["gate"]["top1_by_question"]
    base["retrieval"]["per_question"] = {"q1": {"hit@8": True, "first_rank": 1}}        # D82 format
    drawn, _ = _draw(tmp_path, base_scores=base)
    assert "re-score" in drawn.notes["first_rank_shift"] and "first_rank_shift" not in drawn.files
    assert "re-score" in drawn.notes["context_tokens"] and "context_tokens" in drawn.files
    assert "re-score" in drawn.notes["threshold_sweep"] and "threshold_sweep" in drawn.files


def test_overlay_notes_runs_left_out(mpl, tmp_path):
    drawn, _ = _draw(tmp_path, rows=[csv_row("r0"), csv_row("q0", qs="other"), csv_row("p0", recall="")])
    assert "1 earlier run(s) left out" in drawn.notes["runs_overlay"]
    assert "without recall columns" in drawn.notes["runs_overlay"]


def test_rescore_removes_stale_pngs(mpl, tmp_path):
    stale = tmp_path / "charts" / "hit_at_k.png"
    stale.parent.mkdir()
    stale.write_bytes(PNG_MAGIC)
    _draw(tmp_path)
    assert not stale.exists()


def test_one_broken_chart_does_not_lose_the_others(mpl, tmp_path, monkeypatch):
    def boom(plt, ctx):
        raise RuntimeError("bad data")
    broken = tuple(charts.ChartDef(c.name, boom if c.name == "context_tokens" else c.draw)
                   for c in charts.CHARTS)
    monkeypatch.setattr(charts, "CHARTS", broken)
    drawn, _ = _draw(tmp_path)
    assert drawn.notes["context_tokens"] == "chart failed: RuntimeError: bad data"
    assert "retrieval_curves" in drawn.files


def test_missing_matplotlib_skips_charts_but_the_report_renders(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    drawn, scores = _draw(tmp_path)
    assert drawn.files == {} and drawn.skipped == charts.SKIPPED_NO_MATPLOTLIB
    qs, answers, _, base = two_retrieval_runs()
    text = render(manifest(run_id="r1"), scores, qs, answers, [], baseline_entry(), base, charts=drawn)
    assert charts.SKIPPED_NO_MATPLOTLIB in text and "![" not in text and "| Hit@k |" in text


def test_report_embeds_charts_in_their_sections_with_captions(mpl, tmp_path):
    drawn, scores = _draw(tmp_path)
    qs, answers, _, base = two_retrieval_runs()
    text = render(manifest(run_id="r1"), scores, qs, answers, [], baseline_entry(), base, charts=drawn)
    sec = {n: text.index(f"## {n}.") for n in (1, 2, 3, 4, 5)}
    for chart in charts.CHARTS:
        at = text.index(f"![{chart.name}](charts/{chart.name}.png)")
        assert sec[SECTION[chart.name]] < at < sec[SECTION[chart.name] + 1], chart.name
    assert f"*{REPORT_TERMS['retrieval_curves']}*" in text
    assert text.index("![retrieval_curves]") < text.index("| Hit@k |")       # chart above its table
    assert text.index("![first_rank_shift]") < text.index("questions compared")


def test_report_without_charts_has_no_images_or_notes():
    qs, answers, scores, base = two_retrieval_runs()
    text = render(manifest(), scores, qs, answers, [], baseline_entry(), base)
    assert "![" not in text and "not drawn" not in text and "## 3. Charts, curves and gate sweep" in text


def test_report_says_why_a_chart_is_missing():
    qs, answers, scores, _ = two_retrieval_runs()
    drawn = charts.Drawn({}, {"first_rank_shift": "needs a baseline"})
    text = render(manifest(), scores, qs, answers, [], charts=drawn)
    assert "*first_rank_shift: not drawn — needs a baseline*" in text


def test_report_import_does_not_load_matplotlib():
    probe = ("import sys; import hypeonoath.evaluation.report, hypeonoath.evaluation.cli; "
             "print(any(m == 'matplotlib' or m.startswith('matplotlib.') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"                                 # T9: lazy import


# --- tester additions: edge cases, determinism (T14), never fail a run (T9) ---------------------

def test_same_inputs_give_byte_identical_pngs(mpl, tmp_path):
    first, _ = _draw(tmp_path / "a", rows=[csv_row("r0")])
    second, _ = _draw(tmp_path / "b", rows=[csv_row("r0")])
    assert first.files == second.files
    for rel in first.files.values():
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel


def test_edge_inputs_draw_without_a_chart_failure(mpl, tmp_path):
    _, _, scores, base = two_retrieval_runs()
    scores = json.loads(json.dumps(scores))
    scores["retrieval"]["operating_k"] = 7                              # top_k not in K_VALUES
    rows = [csv_row("r0a", hit="", recall=""), csv_row("r0b", top_k="", mmr_lambda="0.7000"), {"run_id": ""},
            {"run_id": "r0c", "question_set_sha": "qs", "privacy_fp": "fp"}]
    cases = {"k7": (manifest(run_id="r1", top_k=7), scores, baseline_entry(), base, rows),
             "no_base_scores": (manifest(run_id="r1"), scores, baseline_entry(), None, []),   # scores.json missing
             "same_run": (manifest(run_id="r1"), scores, baseline_entry("r1"), scores, [])}
    for name, (m, s, b, bs, r) in cases.items():
        drawn = charts.draw_all(tmp_path / name, m, s, b, bs, [], r)
        assert not any("failed" in n for n in drawn.notes.values()), (name, drawn.notes)
        assert {"retrieval_curves", "runs_overlay", "threshold_sweep", "context_tokens"} <= set(drawn.files), name


def test_gate_without_cov_only_and_empty_sweep(mpl, tmp_path):
    qs = [Q("q1", A("a")), Q("q2", A("a"))]
    s = json.loads(json.dumps(score_run(qs, [rec("q1", [["a:0"]], ["a:0"], top1=0.8), rec("q2", [[]], top1=0.6)], [],
                                        {"stage": "retrieval"})))
    drawn = charts.draw_all(tmp_path / "nocov", manifest(run_id="r1"), s, baseline_entry(), s, [], [])
    assert "threshold_sweep" in drawn.files and not any("failed" in n for n in drawn.notes.values())
    s["gate"]["points"] = []
    drawn = charts.draw_all(tmp_path / "empty", manifest(run_id="r1"), s, None, None, None, [])
    assert "threshold_sweep" not in drawn.files and drawn.notes["threshold_sweep"] == "no top-1 scores: nothing to sweep"


def test_overlay_always_draws_the_baseline_and_this_run(mpl, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, _, scores, base = two_retrieval_runs()
    ctx = charts.Context(manifest(run_id="r1"), scores, baseline_entry("r0"), base, [],
                         [csv_row("r0"), csv_row("p0"), csv_row("z9")])          # baseline row + a later run
    fig, _ = charts.draw_runs_overlay(plt, ctx)
    labels = [t.get_text() for t in fig.legends[0].texts]
    plt.close(fig)
    assert any(t.startswith("baseline r0") for t in labels) and any(t.startswith("this run r1") for t in labels)
    assert sum(t.startswith("p0") for t in labels) == 1 and not any(t.startswith(("r0 ", "z9")) for t in labels)


def test_no_answerable_question_is_not_called_an_old_format_baseline(mpl, tmp_path):
    qs = [Q("q3", NC("a"))]
    s = json.loads(json.dumps(score_run(qs, [rec("q3", [[]], top1=0.3)], [], {"stage": "retrieval"})))
    drawn = charts.draw_all(tmp_path, manifest(run_id="r1"), s, baseline_entry(), s, [], [])
    assert "re-score" not in drawn.notes["first_rank_shift"]


def test_baseline_without_recall_says_rescore(mpl, tmp_path):
    _, _, _, base = two_retrieval_runs()
    base["retrieval"].pop("recall_curve")
    for k in paths.K_VALUES:
        base["retrieval"].pop(f"recall@{k}")
    drawn, _ = _draw(tmp_path, base_scores=base)
    assert "re-score" in (drawn.notes.get("retrieval_curves") or "")


def test_a_locked_old_png_does_not_fail_the_run(mpl, tmp_path, monkeypatch):
    stale = tmp_path / "charts" / "retrieval_curves.png"
    stale.parent.mkdir()
    stale.write_bytes(PNG_MAGIC)

    def locked(self, *args, **kwargs):
        raise PermissionError("in use")
    monkeypatch.setattr(type(stale), "unlink", locked)
    drawn, _ = _draw(tmp_path)                                           # must not raise
    assert drawn is not None


# --- developer: tester findings 3, 4, 7 -----------------------------------------------------------

def test_baseline_note_missing_scores_and_same_run():
    _, _, scores, base = two_retrieval_runs()
    ctx = charts.Context(manifest(run_id="r1"), scores, baseline_entry("r0"), None, [], [])
    assert "has no scores.json" in ctx.base_note and ctx.base_r is None and ctx.base_k is None
    same = charts.Context(manifest(run_id="r0"), scores, baseline_entry("r0"), base, [], [])
    assert same.base_note == "this run is the baseline: every Δ is 0"
    assert charts.Context(manifest(), scores, None, base, [], []).base_r is None      # no entry -> ignored


def test_baseline_k_falls_back_to_its_top_k_setting():
    _, _, scores, base = two_retrieval_runs()
    del base["retrieval"]["operating_k"]
    ctx = charts.Context(manifest(run_id="r1"), scores, baseline_entry(top_k=12), base, [], [])
    assert ctx.base_k == 12


def test_missing_baseline_scores_are_named_in_the_chart_notes(mpl, tmp_path):
    _, _, scores, _ = two_retrieval_runs()
    drawn = charts.draw_all(tmp_path, manifest(run_id="r1"), json.loads(json.dumps(scores)), baseline_entry("r0"),
                            None, [], [])
    for name in ("retrieval_curves", "runs_overlay", "first_rank_shift"):
        assert "has no scores.json" in drawn.notes[name], name


def test_long_changed_list_is_capped_and_notes_are_one_line(mpl):
    import matplotlib.pyplot as plt
    _, _, scores, base = two_retrieval_runs()
    fig = plt.figure()
    ctx = charts.Context(manifest(run_id="r1", note="a\nb"), scores, baseline_entry(), base,
                         [f"key{i}" for i in range(8)], [])
    charts._title(fig, ctx, "x")
    title = fig._suptitle.get_text()
    plt.close(fig)
    assert "key4 +3 more" in title and "key5" not in title and "a b" in title
