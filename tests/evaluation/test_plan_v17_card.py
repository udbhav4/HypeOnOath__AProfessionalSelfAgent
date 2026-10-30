"""Plan v17 4.7 / 4.12 / 6.2 / 9 checks not covered elsewhere (fixtures only).

Two tests pinned code-vs-plan mismatches found in the v17 review (now fixed); they fail loudly
(XPASS -> error) once the code is fixed, so the marker must then be removed."""
from __future__ import annotations

import json

import pytest

from hypeonoath.evaluation import cli, paths, runner, tracking
from hypeonoath.evaluation.metrics import METRICS_BY_NAME, score_run
from hypeonoath.evaluation.report import _n, render
from tests.evaluation.test_outcomes_metrics import MANIFEST, NC, A, Q, full_answer, judgment, seg, verdict
from tests.evaluation.test_tracking_report import manifest


# --- 4.7 per-reply timings ------------------------------------------------------------------

def test_derived_timings_follow_the_4_7_table():
    t = runner.derived_timings({"load_duration": 500_000_000, "prompt_eval_duration": 1_000_000_000,
                                "eval_duration": 2_000_000_000, "total_duration": 4_000_000_000,
                                "prompt_eval_count": 300, "eval_count": 100})
    assert t == {"model_load_s": 0.5, "prefill_s": 1.0, "decode_s": 2.0, "total_gen_s": 4.0, "ttft_s": 1.5,
                 "input_tokens": 300, "output_tokens": 100, "prefill_tok_per_s": 300.0, "decode_tok_per_s": 50.0}


def test_ttft_is_prefill_when_load_missing_and_empty_when_prefill_missing():
    assert runner.derived_timings({"prompt_eval_duration": 2_000_000_000})["ttft_s"] == 2.0
    t = runner.derived_timings({"load_duration": 1_000_000_000, "eval_count": 5, "eval_duration": 0})
    assert t["ttft_s"] is None and t["decode_tok_per_s"] is None and t["prefill_tok_per_s"] is None


def test_operations_figures_skip_errored_questions_and_errors_count_generation_only():
    qs = [Q("q1", A("a")), Q("q2", A("a")), Q("q3", A("a"))]
    answers = [full_answer("q1", ttft=1.0), full_answer("q2", ttft=3.0),
               {**full_answer("q3"), "reply": None, "error": "URLError: x", "timings": {}}]
    judgments = [judgment("q1", [seg("s1", "claim")], [verdict("s1")], covered={"a": [True]}),
                 {"id": "q2", "status": "error"}]
    m = score_run(qs, answers, judgments, MANIFEST)["metrics"]
    assert m["errors"] == 1                                   # the judge error on q2 is not counted (V17-2)
    assert m["judge_coverage"] == 0.5                         # 1 judged of 2 replies sent to the judge
    assert m["p95_ttft_s"] == 3.0 and m["p50_ttft_s"] == 1.0 and m["mean_input_tokens"] == 100


# --- 4.12 / 6.2 runs.csv cell format -----------------------------------------------------------

def test_csv_cells_four_decimals_booleans_and_empty_report_only_metrics():
    row = tracking.csv_row({**manifest("full"), "judge": {}}, {"answer_rate": 2 / 3, "calibrated": True,
                                                                "errors": 0, "p50_ttft_s": 1.0})
    assert row["answer_rate"] == "0.6667" and row["calibrated"] == "true" and row["errors"] == "0"
    assert "p50_ttft_s" not in row and row["faithfulness"] == ""          # D78 report only; empty, never 0
    assert row["git_sha"] == "abc-dirty"


# --- 9 / D73 -------------------------------------------------------------------------------

def test_resume_of_a_full_stage_run_is_refused_until_stage_d(capsys):
    run_dir = paths.RUNS_DIR / "20261029-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"stage": "full"}), encoding="utf-8")
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "Stage D" in capsys.readouterr().err


# --- known mismatches (v17 review, 2026-10-29) -----------------------------------------------

def _mixed_scores():
    qs = [Q("q1", A("a")), Q("q2", NC("a")), Q("q3", A("a"), NC("b"), category="adversarial")]
    answers = [full_answer("q1"), full_answer("q2", groups=()), full_answer("q3")]
    answers[1]["top1"] = 0.2
    judgments = [judgment("q1", [seg("s1", "claim")], [verdict("s1")], covered={"a": [True]}),
                 judgment("q2", [seg("s1", "non_answer")], call2=False),
                 judgment("q3", [seg("s1", "non_answer"), seg("s2", "non_answer")],
                          links=[{"segment_id": "s1", "part_id": "a"}, {"segment_id": "s2", "part_id": "b"}])]
    return qs, answers, judgments, score_run(qs, answers, judgments, MANIFEST)


def test_scorecard_n_is_each_metrics_own_denominator():
    *_, s = _mixed_scores()
    want = {"answer_rate": 2, "correct_no_answer": 2,          # |A_known| / |N_known|, not all known parts (4)
            "embellishment": 1, "correctness_answered": 1, "parametric_leak": 1, "correctness_e2e": 2,
            "gate_correct_refusal_at_t": 1, "gate_t_at_1fr": 1}  # cov_only / gate answerable series
    assert {k: _n(METRICS_BY_NAME[k], s) for k in want} == want


def test_worst_failures_list_key_points_missing_from_a_short_covered_list():
    qs = [Q("q1", A("a", points=2))]
    answers = [full_answer("q1", groups=("a:0", "a:1"))]
    j = [judgment("q1", [seg("s1", "claim")], [verdict("s1")], covered={"a": [True]})]
    s = score_run(qs, answers, j, MANIFEST)
    assert s["metrics"]["correctness_answered"] == 0.5 and s["buckets"]["q1"]["bucket"] == "incomplete"
    assert "missing key points: k1" in render(manifest("full"), s, qs, answers, j)
