"""tracking.py, report.py, glossary.py, .gitignore (eval plan v15, 6.2, 6.3, 7, D37, D50; Section 11)."""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

from hypeonoath.evaluation import glossary, tracking
from hypeonoath.evaluation.metrics import METRICS, METRICS_BY_NAME, REPORT_TERMS, RUNS_CSV_COLUMNS, score_run
from hypeonoath.evaluation.report import render
from tests.evaluation.test_outcomes_metrics import (MANIFEST, NC, A, Q, full_answer, judgment, rec, seg,
                                                    verdict)

ROOT = Path(__file__).resolve().parents[2]

PLAN_COLUMNS = (
    "run_id, date, note, stage, git_sha, answerer, prompt_version, top_k, mmr_lambda, index_fp, privacy_fp, "
    "question_set_sha, judge_model, judge_mode, calibrated, calibration_agreement, judge_coverage, "
    "part_outcome_coverage, n, n_parts, hit@1, hit@3, hit@4, hit@5, hit@8, hit@10, hit@12, hit@15, hit@20, recall@8, mrr, "
    "in_context_hit, supporting_recall@8, gate_t_at_1fr, gate_correct_refusal_at_t, answer_rate, false_no_answer, "
    "omitted, correct_no_answer, missed_no_answer, silent_skip, all_parts_ok, faithfulness, fully_faithful, "
    "embellishment, pivot_related, correctness_answered, correctness_e2e, parametric_leak, world_knowledge_claims, "
    "relevancy, persona, policy_flag, prompt_leak, private_leak_pre, p95_ttft_s, p95_total_gen_s, "
    "mean_input_tokens, errors").split(", ")


def manifest(stage="retrieval", run_id="r1", **extra):
    return {"run_id": run_id, "date": "2026-10-28T10:00:00+00:00", "stage": stage, "note": "n", "git_sha": "abc",
            "git_dirty": True, "answerer": "baseline", "prompt_version": "v1", "prompt_sha256": "p", "top_k": 8,
            "mmr": {"enabled": False, "lambda": None, "fetch_k": 20}, "index_fp": "3:x", "privacy_fp": "fp",
            "question_set_sha256": "qs", "ollama_model": "llama3.1:8b", "judge": {"status": "not_run"}, **extra}


def retrieval_scores():
    qs = [Q("q1", A("a")), Q("q2", NC("a"))]
    return qs, [rec("q1", [["a:0"]], ["a:0"]), rec("q2", [[]], top1=0.3)], score_run(
        qs, [rec("q1", [["a:0"]], ["a:0"]), rec("q2", [[]], top1=0.3)], [], {"stage": "retrieval"})


def full_scores():
    qs = [Q("q1", A("a")), Q("q2", NC("a"), category="adversarial")]
    answers = [full_answer("q1", reply="I did an MSc [cv]."), full_answer("q2", reply="Ask me in person.", groups=())]
    judgments = [judgment("q1", [seg("s1", "claim", "MSc")], [verdict("s1")], covered={"a": [True]}),
                 judgment("q2", [seg("s1", "non_answer")], call2=False)]
    return qs, answers, judgments, score_run(qs, answers, judgments, MANIFEST)


# --- runs.csv -------------------------------------------------------------------------------

def test_column_order_is_the_plan_order():
    assert RUNS_CSV_COLUMNS == PLAN_COLUMNS
    assert not any("citation" in c for c in RUNS_CSV_COLUMNS)                 # D68


def test_retrieval_stage_row_has_empty_cells_never_zero(tmp_path):
    _, _, scores = retrieval_scores()
    row = tracking.csv_row(manifest(), scores["metrics"])
    assert row["hit@1"] == "1.0000" and row["answerer"] == "baseline" and row["git_sha"] == "abc-dirty"
    for column in ("answer_rate", "faithfulness", "errors", "judge_coverage", "p95_ttft_s", "mmr_lambda"):
        assert row[column] == ""
    path = tmp_path / "runs.csv"
    tracking.write_row(row, path)
    tracking.write_row(dict(row, note="rescored"), path)                     # same run_id -> rewritten
    tracking.write_row(dict(row, run_id="r2"), path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert next(r for r in rows if r["run_id"] == "r1")["note"] == "rescored"
    assert list(rows[0].keys()) == PLAN_COLUMNS


def test_full_row_and_mmr_lambda():
    _, _, _, scores = full_scores()
    row = tracking.csv_row(manifest("full", mmr={"enabled": True, "lambda": 0.7, "fetch_k": 20}), scores["metrics"])
    assert row["mmr_lambda"] == "0.7000" and row["answer_rate"] == "1.0000" and row["errors"] == "0"
    assert row["calibrated"] == "false"


# --- baseline -------------------------------------------------------------------------------

def test_promote_refuses_retrieval_runs_and_low_coverage(tmp_path):
    _, _, scores = retrieval_scores()
    with pytest.raises(ValueError, match="full run"):
        tracking.promote(manifest(), scores, "x", tmp_path / "b.json")
    _, _, _, full = full_scores()
    for key in ("judge_coverage", "part_outcome_coverage"):
        low = {**full, "metrics": {**full["metrics"], key: 0.9}}
        with pytest.raises(ValueError, match=key):
            tracking.promote(manifest("full"), low, "x", tmp_path / "b.json")


def test_promote_keeps_history(tmp_path):
    _, _, _, full = full_scores()
    path = tmp_path / "baseline.json"
    tracking.promote(manifest("full", run_id="r1"), full, "first", path)
    data = tracking.promote(manifest("full", run_id="r2"), full, "second", path)
    assert data["current"]["run_id"] == "r2" and [h["run_id"] for h in data["history"]] == ["r1"]
    assert data["current"]["settings"]["answerer"] == "baseline"


def test_comparability_warnings_and_changed_settings():
    m = manifest("full", top_k=5, prompt_version="v2")
    base = {"settings": {k: manifest("full").get(k) for k in tracking.SETTING_KEYS}}
    assert tracking.changed_settings(m, base["settings"]) == ["prompt_version", "top_k"]
    warnings = tracking.comparability_warnings(m, {"judge_coverage": 0.9, "errors": 2}, base)
    assert any("more than one setting" in w for w in warnings) and any("judge_coverage" in w for w in warnings)
    privacy = tracking.comparability_warnings(manifest("full", privacy_fp="other"), {}, base)
    assert any("D49" in w for w in privacy)


def test_compare_only_metrics_present_in_both():
    rows = tracking.compare({"hit@8": 0.5, "answer_rate": None, "errors": 1, "run_id": "a"},
                            {"hit@8": 0.75, "answer_rate": 0.9, "errors": 0, "run_id": "b"})
    by = {r["metric"]: r for r in rows}
    assert set(by) == {"hit@8", "errors"}
    assert by["hit@8"]["better"] is True and by["errors"]["better"] is True and by["hit@8"]["delta"] == 0.25


# --- report ---------------------------------------------------------------------------------

def test_report_renders_retrieval_stage_without_baseline():
    qs, answers, scores = retrieval_scores()
    text = render(manifest(), scores, qs, answers, [])
    assert "## 1. Scorecard" in text and "| hit@8 |" in text and "| answer_rate |" not in text
    assert "Full stage only." in text and "Per edge tag" in text and "## 9. How to read this report" in text


def test_report_renders_full_stage_with_baseline_matrix_and_flips():
    qs, answers, judgments, scores = full_scores()
    base_scores = {**scores, "buckets": {"q1": {"bucket": "retrieval_miss"}, "q2": {"bucket": "ok"}}}
    baseline = {"run_id": "r0", "metrics": {**scores["metrics"], "answer_rate": 0.5}, "settings": {}}
    text = render(manifest("full"), scores, qs, answers, judgments, baseline, base_scores,
                  ["generation was aborted"], ["top_k"])
    assert "JUDGE UNCALIBRATED" in text and "Changed vs baseline:** top_k" in text and "⚠ generation" in text
    assert "| N_priv |" in text and "missed_no_answer split" in text
    assert "| q1 | retrieval_miss | ok |" in text
    assert "| answer_rate | 1.000 | 0.500 | 0.500 | ↑ better |" in text


def test_report_never_shows_a_raw_value_only_guarded_text_and_labels():
    qs, answers, judgments, scores = full_scores()
    answers[0]["private_hits_pre"] = ["phone number"]
    scores = score_run(qs, answers, judgments, MANIFEST)
    text = render(manifest("full"), scores, qs, answers, judgments)
    assert "q1: phone number" in text


# --- glossary (D37) -------------------------------------------------------------------------

def test_every_csv_column_scorecard_row_and_report_term_has_an_entry():
    assert all(c in METRICS_BY_NAME for c in RUNS_CSV_COLUMNS)
    for m in METRICS:
        assert m.definition and m.why and m.how_to_read and m.denominator and m.direction in ("up", "down", "none")
    md = glossary.markdown()
    assert all(f"**{m.name}**" in md for m in METRICS) and all(f"**{t}**" in md for t in REPORT_TERMS)
    assert "hit@8" in glossary.text("hit@8") and "behaviour_matrix" in glossary.text("behaviour_matrix")
    with pytest.raises(ValueError):
        glossary.text("nope")


def test_metrics_md_is_regenerated_and_equal_to_the_registry():
    assert (ROOT / "eval" / "METRICS.md").read_text(encoding="utf-8") == glossary.markdown()


# --- git (D50) ------------------------------------------------------------------------------

def test_gitignore_keeps_only_the_d50_eval_files_trackable(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    keep = ["runs.csv", "baseline.json", "METRICS.md", "questions.yaml", "doc_groups.yaml", "edge_tags.yaml",
            "judge.yaml", "judge_rubric.md"]
    drop = ["runs/r1/answers.jsonl", "runs/r1/report.md", "calibration/labels.yaml", "gold_work/drafts.yaml",
            ".judge_cache/x.json", "results/old.jsonl", "questions_v0.yaml", "charts/hit_at_k.png"]
    for rel in keep + drop:
        (tmp_path / "eval" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "eval" / rel).write_text("x")
    out = subprocess.run(["git", "check-ignore", *[f"eval/{r}" for r in keep + drop]], cwd=tmp_path,
                         capture_output=True, text=True).stdout.split()
    assert sorted(out) == sorted(f"eval/{r}" for r in drop)


def test_judged_deltas_are_withheld_below_full_coverage_and_n_is_numeric():
    qs, answers, judgments, scores = full_scores()
    baseline = {"run_id": "r0", "metrics": {**scores["metrics"], "answer_rate": 0.5, "judge_coverage": 0.9},
                "settings": {}}
    text = render(manifest("full"), scores, qs, answers, judgments, baseline)
    assert "| answer_rate | 1.000 | 0.500 | — |  | 1 | |A_known| | Δ withheld (coverage < 100 %) |" in text
    assert "| hit@8 | 1.000 |" in text and "adversarial subset: 1.000" not in text
    assert "policy_flag on the adversarial subset: 0.000" in text and "relevancy distribution" in text


def test_stage_change_vs_baseline_is_warned():
    base = {"settings": {k: manifest("full").get(k) for k in tracking.SETTING_KEYS}}
    warnings = tracking.comparability_warnings(manifest("retrieval"), {}, base)
    assert any("stage retrieval differs" in w for w in warnings)
    assert not any("more than one setting" in w for w in warnings)


def test_report_scorecard_shows_p50_with_its_n_and_registry_matches_plan_card():
    qs, answers, judgments, scores = full_scores()
    text = render(manifest("full"), scores, qs, answers, judgments)
    assert "| p50_ttft_s | 1.000 |" in text and "| p95_total_gen_s | 2.000 |" in text
    row = next(line for line in text.splitlines() if line.startswith("| prompt_leak |"))
    assert "| 2 | replies (questions whose generation did not fail) |" in row
    assert METRICS_BY_NAME["embellishment"].denominator.endswith("(D76)")
    assert "except unsupported claims typed world_knowledge" in METRICS_BY_NAME["faithfulness"].definition
    assert METRICS_BY_NAME["p50_ttft_s"].csv_column is None


def test_world_knowledge_empty_when_nothing_judged_and_count_otherwise():
    qs = [Q("q1", NC("a"))]
    answers = [full_answer("q1", groups=())]
    assert score_run(qs, answers, [], MANIFEST)["metrics"]["world_knowledge_claims"] is None
    j = [judgment("q1", [seg("s1", "non_answer")], call2=False)]
    assert score_run(qs, answers, j, MANIFEST)["metrics"]["world_knowledge_claims"] == 0
