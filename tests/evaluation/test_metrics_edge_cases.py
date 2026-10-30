"""Edge cases of outcomes / metrics / gate / leaks / tracking / report found in the Stage B review
(eval plan v15, 4.0-4.9, 5.4, 6.2, 7; T7). Fixture records only; no network, no corpus."""
from __future__ import annotations

import pytest

from hypeonoath.core.config import GUARD_LINE
from hypeonoath.evaluation import tracking
from hypeonoath.evaluation.gate import GateRow, sweep
from hypeonoath.evaluation.leaks import leak_source, prompt_leak
from hypeonoath.evaluation.metrics import score_run
from hypeonoath.evaluation.outcomes import ANSWERED, NON_ANSWER, SKIPPED, UNKNOWN, behaviour, part_outcomes
from hypeonoath.evaluation.report import render
from hypeonoath.serving.prompt import PROMPT_EXAMPLES, PROMPT_SAMPLE_PHRASES, system_rules
from tests.evaluation.test_outcomes_metrics import (MANIFEST, NC, NP, A, Q, full_answer, judgment, rec, seg,
                                                    verdict)
from tests.evaluation.test_tracking_report import manifest


# --- 5.4: links only count from non_answer segments of that part ------------------------------

def test_link_from_other_segment_or_to_another_part_is_not_a_non_answer():
    q = Q("q", A("a"), NC("b"))
    j = judgment("q", [seg("s1", "other"), seg("s2", "non_answer")], [],
                 links=[{"segment_id": "s1", "part_id": "a"}, {"segment_id": "s2", "part_id": "b"}])
    assert part_outcomes(q, j) == {"a": SKIPPED, "b": NON_ANSWER}


def test_answers_claim_for_an_unknown_part_id_answers_nothing():
    q = Q("q", NC("a"))
    j = judgment("q", [seg("s1", "claim"), seg("s2", "non_answer")], [verdict("s1", part="zz")],
                 links=[{"segment_id": "s2", "part_id": "a"}])
    assert part_outcomes(q, j) == {"a": NON_ANSWER}


def test_missed_private_part_with_a_supported_claim_goes_to_private_supported():
    q = Q("q", NP("a"))
    j = judgment("q", [seg("s1", "claim")], [verdict("s1", supported=True)])
    b = behaviour([q], {"q": part_outcomes(q, j)}, {"q": j})
    assert b["missed_split"]["private/supported"] == 1 and b["missed_no_answer"] == 1.0


def test_behaviour_with_every_part_unknown_gives_none_not_zero():
    qs = [Q("q1", A("a")), Q("q2", NC("a"))]
    b = behaviour(qs, {"q1": {"a": UNKNOWN}, "q2": {"a": UNKNOWN}})
    for key in ("answer_rate", "false_no_answer", "omitted", "correct_no_answer", "missed_no_answer",
                "silent_skip", "all_parts_ok"):
        assert b[key] is None
    assert b["part_outcome_coverage"] == 0.0


# --- full-stage scoring on degenerate inputs ---------------------------------------------------

def test_full_run_without_any_judgment_is_all_unknown_and_still_renders():
    qs = [Q("q1", A("a")), Q("q2", NC("a"))]
    answers = [full_answer("q1"), full_answer("q2", groups=())]
    s = score_run(qs, answers, [], MANIFEST)
    m = s["metrics"]
    assert m["part_outcome_coverage"] == 0.0 and m["judge_coverage"] == 0.0
    assert m["answer_rate"] is None and m["faithfulness"] is None and m["relevancy"] is None
    assert {b["bucket"] for b in s["buckets"].values()} == {"unknown"}
    assert "**Parts with unknown outcome** (2)" in render(manifest("full"), s, qs, answers, [])


@pytest.mark.parametrize("stage", ["retrieval", "full"])
def test_empty_question_set_scores_and_renders(stage):
    s = score_run([], [], [], {"stage": stage})
    assert s["metrics"]["n"] == 0 and s["metrics"]["hit@8"] is None and s["metrics"]["gate_t_at_1fr"] is None
    assert "## 9. How to read this report" in render(manifest(stage), s, [], [], [])


def test_world_knowledge_supported_claim_counts_as_candidate():
    qs = [Q("q1", A("a"))]
    j = [judgment("q1", [seg("s1", "claim"), seg("s2", "claim")],
                  [verdict("s1"), verdict("s2", "related_to", supported=False, utype="world_knowledge")],
                  covered={"a": [True]})]
    m = score_run(qs, [full_answer("q1")], j, MANIFEST)["metrics"]
    assert m["faithfulness"] == 1.0 and m["fully_faithful"] == 1.0 and m["world_knowledge_claims"] == 1


# --- gate (4.2) ------------------------------------------------------------------------------

def test_adversarial_answerable_question_never_counts_as_a_false_refusal():
    from hypeonoath.evaluation.gate import series_of
    q = Q("q", A("a"), category="adversarial")
    assert series_of(q) == "adversarial"
    out = sweep([GateRow("q", series_of(q), 0.1), GateRow("a", "answerable", 0.9)], max_false_refusals=0)
    assert out["gate_t_at_1fr"] == 0.9                                      # q below 0.9 is not a false refusal


def test_retrieval_row_has_gate_columns_filled():
    qs = [Q("q1", A("a")), Q("q2", NC("a"))]
    answers = [rec("q1", [["a:0"]], ["a:0"], top1=0.8), rec("q2", [[]], top1=0.3)]
    row = tracking.csv_row(manifest(), score_run(qs, answers, [], {"stage": "retrieval"})["metrics"])
    assert row["gate_t_at_1fr"] == "0.8000" and row["gate_correct_refusal_at_t"] == "1.0000"
    assert row["calibration_agreement"] == "" and row["policy_flag"] == ""


# --- prompt_leak on the real v1 prompt (T7) ------------------------------------------------------

@pytest.fixture(scope="module")
def v1_source():
    return leak_source(system_rules("v1"), list(PROMPT_EXAMPLES) + list(PROMPT_SAMPLE_PHRASES))


def test_v1_sample_phrasings_examples_and_guard_line_never_fire(v1_source):
    for text in [*PROMPT_SAMPLE_PHRASES, GUARD_LINE,
                 "Kubernetes isn't something I've worked with so far, but in my HypeOnOath project I built it.",
                 "My phone number I don't have with me right now, but feel free to ask me in person."]:
        assert not prompt_leak(text, v1_source), text


def test_v1_copied_rule_fires(v1_source):
    assert prompt_leak("Sure! Never say or hint that anything is hidden, redacted, private, withheld.", v1_source)


# --- report privacy: only the guarded text is ever shown --------------------------------------

def test_worst_failures_show_the_stored_guarded_reply_only():
    qs = [Q("q1", NP("a"))]
    answers = [full_answer("q1", reply=f"Sure. {GUARD_LINE}", hits=["phone number"], groups=())]
    answers[0]["prompt"] = "SYSTEM ... CONTEXT with RAW-SECRET-VALUE"            # never rendered
    j = [judgment("q1", [seg("s1", "other", "Sure."), seg("s2", "non_answer", GUARD_LINE)], call2=False)]
    s = score_run(qs, answers, j, MANIFEST)
    text = render(manifest("full"), s, qs, answers, j)
    assert "RAW-SECRET-VALUE" not in text and GUARD_LINE in text and "q1: phone number" in text
    assert s["buckets"]["q1"]["bucket"] == "private_leak"


# --- fixed source issues (tester findings 1 and 2) -------------------------------------------

def test_embellishment_ignores_replies_without_call2():
    qs = [Q("q1", NC("a"))]
    j = [judgment("q1", [seg("s1", "other", "I'm simply brilliant")], call2=False)]
    assert score_run(qs, [full_answer("q1", groups=())], j, MANIFEST)["metrics"]["embellishment"] is None


def test_report_calibrated_banner_without_agreement_does_not_crash():
    qs = [Q("q1", NC("a"))]
    answers = [full_answer("q1", groups=())]
    s = score_run(qs, answers, [], MANIFEST)
    assert "JUDGE CALIBRATED" in render(manifest("full", judge={"calibrated": True}), s, qs, answers, [])
