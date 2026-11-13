"""tracking.py, report.py, glossary.py, .gitignore (eval plan v15, 6.2, 6.3, 7, D37, D50; Section 11)."""
from __future__ import annotations

import csv
import json
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
    "run_id, date, note, stage, git_sha, answerer, prompt_version, top_k, mmr_lambda, mmr_fetch_k, "
    "doc_description_in_chunks, small_sections_merged, query_rewrite, index_fp, privacy_fp, "
    "question_set_sha, judge_model, judge_mode, calibrated, calibration_agreement, judge_coverage, "
    "part_outcome_coverage, n, n_parts, hit@1, hit@3, hit@4, hit@5, hit@8, hit@10, hit@12, hit@15, hit@20, recall@1, "
    "recall@3, recall@4, recall@5, recall@8, recall@10, recall@12, recall@15, recall@20, mrr, in_context_hit, "
    "in_context_recall, supporting_recall@8, gate_t_at_1fr, gate_correct_refusal_at_t, answer_rate, false_no_answer, "
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

def test_promote_refuses_low_coverage_and_unknown_stage(tmp_path):
    _, _, scores = retrieval_scores()
    with pytest.raises(ValueError, match="full or retrieval run"):
        tracking.promote(manifest("partial"), scores, "x", tmp_path / "b.json")
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


def test_retrieval_promote_uses_its_own_slot_and_keeps_current(tmp_path):
    """v20, D82: retrieval runs -> slot `retrieval` (+ retrieval_history); `current` / `history` untouched."""
    _, _, scores = retrieval_scores()
    _, _, _, full = full_scores()
    path = tmp_path / "baseline.json"
    tracking.promote(manifest("full", run_id="f1"), full, "full", path)
    tracking.promote(manifest(run_id="r1"), scores, "ref", path)
    data = tracking.promote(manifest(run_id="r2"), scores, "better", path)
    assert data["current"]["run_id"] == "f1" and data["history"] == []
    assert data["retrieval"]["run_id"] == "r2" and [h["run_id"] for h in data["retrieval_history"]] == ["r1"]
    data = tracking.promote(manifest("full", run_id="f2"), full, "full 2", path)       # full keeps the slot
    assert data["retrieval"]["run_id"] == "r2" and [h["run_id"] for h in data["history"]] == ["f1"]
    assert tracking.baseline_for("retrieval", path)["run_id"] == "r2"
    assert tracking.baseline_for("full", path)["run_id"] == "f2"
    assert tracking.baseline_for("retrieval", tmp_path / "none.json") is None


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


def test_report_breakdowns_show_their_denominators_and_gate_cells_fractions():
    """v19: n_answerable / n_answer_parts next to n; every gate cell = share (refused / series size)."""
    qs, answers, scores = retrieval_scores()                     # q1 answerable, q2 not covered (top1 0.3)
    text = render(manifest(), scores, qs, answers, [])
    assert "| | n | n_answerable | hit@k | recall@k | in_context_hit | in_context_recall |" in text
    assert "| | n_parts | n_answer_parts | in_context |" in text
    cat = scores["breakdowns"]["category"]
    assert sum(r["n"] for r in cat.values()) == 2 and sum(r["n_answerable"] for r in cat.values()) == 1
    top = scores["gate"]["points"][-1]                           # highest t: q2 (0.3) is below it
    assert f"{top['correct_refusal']:.3f} (1/1)" in text
    assert "share (refused / series size)" in text
    assert "series sizes: answerable 1, cov_only 1)" in text                  # fixed order
    assert "private overlaps adversarial" not in text                        # neither series present


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


def test_score_run_private_series_overlaps_and_breakdown_denominators():
    """v19 end to end: adversarial private-only q counts in adversarial AND private (once); answer+private q is
    answerable only; adversarial n_answerable is the hit@8 denominator; edge tags carry n_answer_parts."""
    from tests.evaluation.test_outcomes_metrics import NP
    qs = [Q("x1", NP("b"), category="adversarial"), Q("x2", NC("c"), category="adversarial"),
          Q("x3", A("a"), category="adversarial"), Q("p1", NP("b"), NC("c")), Q("m1", A("a"), NP("b"))]
    answers = [rec("x1", [[]], top1=0.2), rec("x2", [[]], top1=0.3), rec("x3", [["a:0"]], ["a:0"], top1=0.9),
               rec("p1", [[]], top1=0.4), rec("m1", [["a:0"]], ["a:0"], top1=0.7)]
    scores = score_run(qs, answers, [], {"stage": "retrieval"})
    gate = scores["gate"]
    assert gate["n"] == {"adversarial": 3, "private": 2, "answerable": 1}
    by_t = {p["t"]: p for p in gate["points"]}
    assert by_t[0.9]["refused"] == {"adversarial": 2, "private": 2, "answerable": 1}
    assert by_t[0.4]["private_refused"] == 0.5 and by_t[0.4]["correct_refusal"] is None
    adv = scores["breakdowns"]["category"]["adversarial"]
    assert adv["n"] == 3 and adv["n_answerable"] == 1 and adv["hit@k"] == 1.0
    tags = scores["breakdowns"]["edge_tag"]
    assert tags["decline_direct"]["n_parts"] == 3 and tags["decline_direct"]["n_answer_parts"] == 0
    assert tags["decline_direct"]["in_context"] is None
    assert tags["single_doc"]["n_answer_parts"] == 2
    text = render(manifest(), scores, qs, answers, [])
    assert "| 0.900 | — | 1.000 (1/1) | 1.000 (2/2) | 0.667 (2/3) |" in text      # no cov_only series -> —


def test_report_gate_cells_dash_for_absent_series_and_empty_sweep():
    _, _, scores = retrieval_scores()                            # no private, no adversarial question
    assert "private" not in scores["gate"]["n"] and "adversarial" not in scores["gate"]["n"]
    lines = [ln for ln in render(manifest(), scores, *retrieval_scores()[:2], []).splitlines() if ln.startswith("| 0.")]
    assert lines and all(ln.rstrip().endswith("| — | — |") for ln in lines)
    empty = {**scores, "gate": {"points": [], "n": {}, "gate_t_at_1fr": None, "gate_correct_refusal_at_t": None}}
    assert "series sizes: none" in render(manifest(), empty, *retrieval_scores()[:2], [])


# --- v20, D82: retrieval baseline deltas + per-question retrieval flips ----------------------

def two_retrieval_runs():
    """Baseline: q1 hit at rank 1, q2 missed. New run: q1 missed, q2 hit at rank 2 (both in context)."""
    import json

    qs = [Q("q1", A("a")), Q("q2", A("a")), Q("q3", NC("a"))]
    old = [rec("q1", [["a:0"]], ["a:0"]), rec("q2", [[]]), rec("q3", [[]], top1=0.3)]
    new = [rec("q1", [[]]), rec("q2", [[], ["a:0"]], ["a:0"]), rec("q3", [[]], top1=0.3)]
    base = json.loads(json.dumps(score_run(qs, old, [], {"stage": "retrieval"})))   # as read back from disk
    return qs, new, score_run(qs, new, [], {"stage": "retrieval"}), base


def test_per_question_retrieval_record():
    _, _, scores, _ = two_retrieval_runs()
    pq = scores["retrieval"]["per_question"]
    assert set(pq) == {"q1", "q2"}                                         # answerable questions only
    assert pq["q1"] == {"hit@k": False, "recall@k": 0.0, "in_context_hit": False, "in_context_recall": 0.0,
                        "first_rank": None}
    assert pq["q2"] == {"hit@k": True, "recall@k": 1.0, "in_context_hit": True, "in_context_recall": 1.0,
                        "first_rank": 2}
    assert scores["retrieval"]["operating_k"] == 8                             # no top_k in the manifest


def test_retrieval_report_shows_deltas_per_k_breakdowns_and_flips():
    qs, answers, scores, base = two_retrieval_runs()
    baseline = {"run_id": "r0", "metrics": base["metrics"], "settings": {}}
    text = render(manifest(), scores, qs, answers, [], baseline, base, [], ["top_k"])
    assert "Baseline: r0" in text and "Changed vs baseline:** top_k" in text
    assert "| Hit@k baseline | 0.500 |" in text and "| Δ Hit@k | -0.500 | +0.000 |" in text   # k=1, k=3
    assert "| Δ Recall@k |" in text
    assert "| mrr | 0.250 | 0.500 | -0.250 | ↓ worse |" in text
    assert "2 questions compared (k = 8, baseline k = 8); hit@k 1 gained, 1 lost." in text
    assert "| q1 | ✓ → ✗ | 1.000 → 0.000 | ✓ → ✗ | 1.000 → 0.000 | 1 → — |" in text
    assert "| q2 | ✗ → ✓ | 0.000 → 1.000 | ✗ → ✓ | 0.000 → 1.000 | — → 2 |" in text
    assert ("| | n | n_answerable | hit@k | Δ hit@k | recall@k | Δ recall@k | in_context_hit | Δ in_context_hit "
            "| in_context_recall | Δ in_context_recall |") in text
    assert "hit@k and recall@k at the operating k: k = 8, baseline k = 8." in text
    assert "| n_parts | n_answer_parts | in_context | Δ in_context |" in text
    assert "| context_tokens_mean | 10 | 10 | 0 |" in text      # report-only metric: old value from baseline scores


def test_retrieval_flips_without_baseline_unchanged_or_old_scores():
    qs, answers, scores, base = two_retrieval_runs()
    assert "**Retrieval flips**\n\nNo baseline run scores to compare with." in render(manifest(), scores, qs, answers, [])
    same = {"run_id": "r0", "metrics": scores["metrics"], "settings": {}}
    assert "No per-question retrieval changes (2 questions compared)." in render(
        manifest(), scores, qs, answers, [], same, scores)
    legacy = {**base, "retrieval": {k: v for k, v in base["retrieval"].items() if k != "per_question"}}
    assert "no per-question retrieval data: re-score" in render(manifest(), scores, qs, answers, [], same, legacy)
    no_rows = render(manifest(), scores, qs, answers, [], same, {**base, "breakdowns": {}})
    section = no_rows.split("## 5. Breakdowns")[1].split("## 9.")[0]
    assert "Δ hit@k" in section and "| — |" in section                         # baseline lacks the rows -> —


def test_baseline_scores_ignored_without_a_baseline_entry():
    qs, answers, scores, base = two_retrieval_runs()
    text = render(manifest(), scores, qs, answers, [], None, base)
    assert "Δ Hit@k" not in text and "Δ hit@k" not in text and "No baseline run scores" in text


def test_retrieval_report_gate_deltas_come_from_the_slot_metrics():
    qs, answers, scores, base = two_retrieval_runs()
    baseline = {"run_id": "r0", "metrics": {**base["metrics"], "gate_t_at_1fr": 0.25}, "settings": {}}
    text = render(manifest(), scores, qs, answers, [], baseline, base)
    assert "| gate_t_at_1fr | 0.800 | 0.250 | 0.550 |" in text
    assert "| gate_correct_refusal_at_t | 1.000 | 1.000 | 0.000 |" in text


def test_edge_tag_breakdown_in_context_delta_and_dash_for_no_answer_parts():
    qs, answers, scores, base = two_retrieval_runs()
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": base["metrics"]}, base)
    section = text.split("**Per edge tag**")[1].split("## ")[0]
    assert "| single_doc | 2 | 2 | 0.500 | +0.000 |" in section
    assert "| absent_personal | 1 | 0 | — | — |" in section                  # in_context None -> no Δ


def test_retrieval_flips_compare_only_questions_in_both_runs():
    qs, answers, scores, base = two_retrieval_runs()
    pq = base["retrieval"]["per_question"]
    extra = {**base, "retrieval": {**base["retrieval"], "per_question": {
        **pq, "q9": {"hit@k": True, "recall@k": 1.0, "in_context_hit": True, "in_context_recall": 1.0,
                     "first_rank": 1}}}}
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": base["metrics"]}, extra)
    assert "2 questions compared (k = 8, baseline k = 8); hit@k 1 gained, 1 lost." in text and "| q9 |" not in text


def test_recall_or_rank_change_alone_is_listed():
    qs, answers, scores, base = two_retrieval_runs()
    import copy
    old = copy.deepcopy(scores)
    old["retrieval"]["per_question"]["q2"]["first_rank"] = 1                 # same hit, rank 1 -> 2
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": old["metrics"]}, old)
    assert "| q2 | ✓ → ✓ | 1.000 → 1.000 | ✓ → ✓ | 1.000 → 1.000 | 1 → 2 |" in text
    assert "hit@k 0 gained, 0 lost." in text and "| q1 |" not in text


def test_full_report_shows_retrieval_flips_against_current():
    import copy
    qs, answers, judgments, scores = full_scores()
    old = json.loads(json.dumps(scores, default=str))
    old["retrieval"]["per_question"]["q1"].update({"hit@k": False, "recall@k": 0.0, "in_context_hit": False,
                                                   "in_context_recall": 0.0, "first_rank": None})
    baseline = {"run_id": "f0", "metrics": copy.deepcopy(scores["metrics"]), "settings": {}}
    text = render(manifest("full"), scores, qs, answers, judgments, baseline, old)
    flips = text.split("**Retrieval flips**")[1].split("## 5.")[0]
    assert "1 questions compared (k = 8, baseline k = 8); hit@k 1 gained, 0 lost." in flips
    assert "| q1 | ✗ → ✓ | 0.000 → 1.000 | ✗ → ✓ | 0.000 → 1.000 | — → 1 |" in flips
    assert "**Pass ↔ fail**" in text


def test_retrieval_slot_comparability_warnings_and_changed_line():
    _, _, scores = retrieval_scores()
    path_free = {"run_id": "r0", "settings": {k: manifest().get(k) for k in tracking.SETTING_KEYS},
                 "metrics": scores["metrics"]}
    m = manifest(top_k=5)
    assert tracking.changed_settings(m, path_free["settings"]) == ["top_k"]
    assert tracking.comparability_warnings(m, scores["metrics"], path_free) == []
    two = manifest(top_k=5, index_fp="3:y")
    warnings = tracking.comparability_warnings(two, scores["metrics"], path_free)
    assert "different index than the baseline" in warnings and any("ambiguous" in w for w in warnings)
    assert not any("stage" in w for w in warnings)


# --- v21: D85 settings section, D86 operating k, D87 in_context_recall, D88 runs.csv columns ---------

def test_settings_section_marks_changes_and_comes_first():
    qs, answers, scores, base = two_retrieval_runs()
    old = manifest(top_k=8)
    new = manifest(top_k=12, mmr={"enabled": True, "lambda": 0.7, "fetch_k": 20}, doc_description_in_chunks=False)
    baseline = {"run_id": "r0", "metrics": base["metrics"],
                "settings": {k: tracking.setting(old, k) for k in tracking.SETTING_KEYS}}
    text = render(new, scores, qs, answers, [], baseline, base)
    assert text.index("## 0. Settings") < text.index("## 1. Scorecard")
    assert "| top_k | 12 | 8 | ✱ changed |" in text
    assert "| MMR on | yes | no | ✱ changed |" in text and "| MMR λ | 0.700 | — | ✱ changed |" in text
    assert "| MMR fetch_k | 20 | — | ✱ changed |" in text
    assert "| doc_description_in_chunks | no | no |  |" in text                 # old manifest without it -> false
    assert "| prompt_version | v1 | v1 |  |" in text and "gate" not in text.split("## 0. Settings")[1].split("## 1.")[0]
    alone = render(new, scores, qs, answers, [])
    assert "| top_k | 12 | — |  |" in alone                                    # no baseline: nothing marked


def test_operating_k_follows_top_k_in_breakdowns_and_flips():
    from tests.evaluation.test_outcomes_metrics import rec
    qs = [Q("q1", A("a", 2))]
    # group a:0 at rank 3, a:1 at rank 10: at k = 8 recall 0.5, at k = 12 recall 1.0
    ranking = [[], [], ["a:0"], [], [], [], [], [], [], ["a:1"]]
    answers = [rec("q1", ranking, ["a:0"])]
    k8 = json.loads(json.dumps(score_run(qs, answers, [], {"stage": "retrieval", "top_k": 8})))
    k12 = score_run(qs, answers, [], {"stage": "retrieval", "top_k": 12})
    assert k8["retrieval"]["per_question"]["q1"]["recall@k"] == 0.5
    assert k12["retrieval"]["per_question"]["q1"]["recall@k"] == 1.0
    assert k12["breakdowns"]["category"]["single_fact"]["recall@k"] == 1.0
    assert k12["retrieval"]["in_context_recall"] == 0.5                        # context holds a:0 only
    text = render(manifest(top_k=12), k12, qs, answers, [], {"run_id": "r0", "metrics": k8["metrics"]}, k8)
    assert "k = 12, baseline k = 8" in text
    assert "| q1 | ✓ → ✓ | 0.500 → 1.000 | ✓ → ✓ | 0.500 → 0.500 | 3 → 3 |" in text
    assert "| single_fact | 1 | 1 | 1.000 | +0.000 | 1.000 | +0.500 |" in text


def test_in_context_recall_metric_and_every_recall_column():
    qs, _, scores, _ = two_retrieval_runs()
    m = scores["metrics"]
    assert m["in_context_recall"] == 0.5 and m["recall@20"] == 0.5 and m["recall@1"] == 0.0
    assert all(f"recall@{k}" in m for k in (1, 3, 4, 5, 8, 10, 12, 15, 20))
    assert scores["retrieval"]["recall_curve"][20] == m["recall@20"]


def test_runs_csv_new_columns_and_old_rows_get_empty_cells(tmp_path):
    _, _, scores = retrieval_scores()
    path = tmp_path / "runs.csv"
    old_header = [c for c in PLAN_COLUMNS if c not in ("mmr_fetch_k", "doc_description_in_chunks", "in_context_recall",
                                                       "small_sections_merged", "query_rewrite")
                  and not (c.startswith("recall@") and c != "recall@8")]
    with path.open("w", newline="", encoding="utf-8") as f:                 # a pre-v21 runs.csv
        w = csv.DictWriter(f, fieldnames=old_header)
        w.writeheader()
        w.writerow({c: "x" for c in old_header} | {"run_id": "old"})
    row = tracking.csv_row(manifest(mmr={"enabled": True, "lambda": 0.5, "fetch_k": 30}), scores["metrics"])
    assert row["mmr_fetch_k"] == "30" and row["doc_description_in_chunks"] == "false"
    assert row["in_context_recall"] == "1.0000" and row["recall@20"] == "1.0000"
    assert tracking.csv_row(manifest(), scores["metrics"])["mmr_fetch_k"] == ""      # MMR off -> empty
    tracking.write_row(row, path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == PLAN_COLUMNS
    old = next(r for r in rows if r["run_id"] == "old")
    assert old["in_context_recall"] == "" and old["recall@20"] == "" and old["mmr_fetch_k"] == ""


def test_missing_doc_description_setting_reads_false_and_is_not_a_change():
    old = {k: v for k, v in manifest().items()}
    new = manifest(doc_description_in_chunks=False)
    assert tracking.setting(old, "doc_description_in_chunks") is False
    assert "doc_description_in_chunks" not in tracking.changed_settings(new, old)
    assert "doc_description_in_chunks" in tracking.changed_settings(manifest(doc_description_in_chunks=True), old)


def test_d82_format_baseline_scores_ask_for_a_rescore():
    qs, answers, scores, base = two_retrieval_runs()
    d82 = json.loads(json.dumps(base))
    d82["retrieval"]["per_question"] = {"q1": {"hit@8": True, "in_context_hit": True, "recall@8": 1.0,
                                               "first_rank": 1}}
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": base["metrics"]}, d82)
    assert "no per-question retrieval data: re-score" in text


# --- v21 QA additions: operating-k edge cases, old baselines, settings rows ------------------------

def test_operating_k_outside_k_values_and_missing_top_k():
    qs = [Q("q1", A("a", 2))]
    ranking = [[], [], ["a:0"], [], [], [], ["a:1"]]                          # a:0 rank 3, a:1 rank 7
    answers = [rec("q1", ranking, ["a:0"])]
    k6 = score_run(qs, answers, [], {"stage": "retrieval", "top_k": 6})      # 6 is not in K_VALUES
    assert k6["retrieval"]["operating_k"] == 6 and k6["retrieval"]["per_question"]["q1"]["recall@k"] == 0.5
    assert k6["breakdowns"]["category"]["single_fact"]["recall@k"] == 0.5
    assert "hit@6" not in k6["metrics"]                                       # scorecard keeps the fixed k values
    none = score_run(qs, answers, [], {"stage": "retrieval", "top_k": None})
    assert none["retrieval"]["operating_k"] == 8 and none["retrieval"]["per_question"]["q1"]["recall@k"] == 1.0
    k20 = score_run(qs, answers, [], {"stage": "retrieval", "top_k": 20})    # deeper than the ranking: no error
    assert k20["retrieval"]["per_question"]["q1"]["recall@k"] == 1.0
    text = render(manifest(top_k=6), k6, qs, answers, [])
    assert "hit@k and recall@k at the operating k: k = 6." in text


def test_in_context_recall_partial_context_is_macro_mean():
    # q1: 3 groups, context holds 1 -> 1/3; q2: 1 group in context -> 1; q3: no context -> 0. Mean = 4/9
    qs = [Q("q1", A("a", 3)), Q("q2", A("a")), Q("q3", A("a"))]
    answers = [rec("q1", [["a:0"], ["a:1"], ["a:2"]], ["a:0"]), rec("q2", [["a:0"]], ["a:0"]),
               rec("q3", [["a:0"]], [])]
    s = score_run(qs, answers, [], {"stage": "retrieval", "top_k": 8})
    assert s["metrics"]["in_context_recall"] == pytest.approx(4 / 9)
    assert s["retrieval"]["per_question"]["q1"]["in_context_recall"] == pytest.approx(1 / 3)
    assert s["metrics"]["in_context_hit"] == pytest.approx(2 / 3)
    row = tracking.csv_row(manifest(), s["metrics"])
    assert row["in_context_recall"] == "0.4444"


def test_d82_promoted_baseline_without_new_keys_still_compares():
    """baseline.json written by the D82 promote: settings without doc_description_in_chunks, metrics without
    in_context_recall / recall@12 -> no false 'changed', scorecard Δ from the re-scored baseline scores.json."""
    qs, answers, scores, base = two_retrieval_runs()
    old_settings = {k: manifest().get(k) for k in tracking.SETTING_KEYS if k != "doc_description_in_chunks"}
    old_metrics = {k: v for k, v in base["metrics"].items()
                   if k != "in_context_recall" and not (k.startswith("recall@") and k != "recall@8")}
    baseline = {"run_id": "r0", "settings": old_settings, "metrics": old_metrics}
    new = manifest(doc_description_in_chunks=False)
    assert tracking.changed_settings(new, old_settings) == []
    assert tracking.comparability_warnings(new, scores["metrics"], baseline) == []
    text = render(new, scores, qs, answers, [], baseline, base, [], [])
    assert "| in_context_recall | 0.500 | 0.500 | 0.000 |" in text
    assert "| recall@12 | 0.500 | 0.500 | 0.000 |" in text
    assert "| doc_description_in_chunks | no | no |  |" in text


def test_baseline_scores_without_operating_k_do_not_crash():
    qs, answers, scores, base = two_retrieval_runs()
    old = json.loads(json.dumps(base))
    old["retrieval"].pop("operating_k")
    for row in old["breakdowns"]["category"].values():                       # pre-D86 row keys
        row["hit@8"] = row.pop("hit@k")
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": base["metrics"]}, old)
    assert "k = 8, baseline k = ?" in text
    section = text.split("**Per category**")[1].split("**Per doc_group**")[0]
    assert "Δ hit@k" in section and "| — |" in section


def test_settings_rows_mmr_off_question_set_and_privacy_fp():
    qs, answers, scores, base = two_retrieval_runs()
    old = manifest(question_set_sha256="a" * 64)
    baseline = {"run_id": "r0", "metrics": base["metrics"],
                "settings": {k: tracking.setting(old, k) for k in tracking.SETTING_KEYS}}
    new = manifest(question_set_sha256="b" * 64, privacy_fp="other")
    section = render(new, scores, qs, answers, [], baseline, base).split("## 0. Settings")[1].split("## 1.")[0]
    assert "| MMR on | no | no |  |" in section and "| MMR λ | — | — |  |" in section
    assert "| MMR fetch_k | — | — |  |" in section                             # fetch_k hidden while MMR is off
    assert f"| question set | {'b' * 12} | {'a' * 12} | ✱ changed |" in section
    assert "| privacy_fp | other | fp | ✱ changed |" in section
    assert "| index_fp | 3:x | 3:x |  |" in section and "| ollama_model | llama3.1:8b | llama3.1:8b |  |" in section
    labels = [line.split("|")[1].strip() for line in section.strip().splitlines()[2:]]
    assert labels == ["top_k", "MMR on", "MMR λ", "MMR fetch_k", "doc_description_in_chunks", "small_sections_merged",
                      "query_rewrite", "prompt_version", "answerer", "ollama_model", "index_fp", "question set",
                      "privacy_fp"]


def test_new_metrics_and_report_terms_in_glossary_and_compare():
    md = glossary.markdown()
    for name in ("in_context_recall", "recall@12", "mmr_fetch_k", "doc_description_in_chunks", "settings",
                 "retrieval_flips"):
        assert f"**{name}**" in md
    _, _, scores, base = two_retrieval_runs()
    names = {r["metric"] for r in tracking.compare(base["metrics"], scores["metrics"])}
    assert {"in_context_recall", "recall@1", "recall@20"} <= names
    assert "doc_description_in_chunks" not in names and "mmr_fetch_k" not in names


def test_promote_snapshot_holds_doc_description_setting(tmp_path):
    _, _, scores = retrieval_scores()
    data = tracking.promote(manifest(), scores, "ref", path=tmp_path / "b.json")
    assert data["retrieval"]["settings"]["doc_description_in_chunks"] is False
    assert data["retrieval"]["metrics"]["in_context_recall"] == 1.0


def test_mmr_off_with_a_different_unused_fetch_k_is_not_a_change():
    off_20 = manifest(mmr={"enabled": False, "lambda": None, "fetch_k": 20})
    off_15 = manifest(mmr={"enabled": False, "lambda": None, "fetch_k": 15})
    assert "mmr" not in tracking.changed_settings(off_15, off_20)
    on = manifest(mmr={"enabled": True, "lambda": 0.7, "fetch_k": 15})
    assert "mmr" in tracking.changed_settings(on, off_15)
    assert "mmr" in tracking.changed_settings(on, manifest(mmr={"enabled": True, "lambda": 0.7, "fetch_k": 20}))


def test_k_note_asks_for_a_rescore_when_the_baseline_has_no_operating_k():
    qs, answers, scores, base = two_retrieval_runs()
    old = json.loads(json.dumps(base))
    del old["retrieval"]["operating_k"]
    text = render(manifest(), scores, qs, answers, [], {"run_id": "r0", "metrics": base["metrics"]}, old)
    assert "baseline k = ? (re-score the baseline run)" in text
