"""questions.py: V1-V14 (eval plan v15, 3.1; tests Section 11 'questions' + 'privacy gate')."""
from __future__ import annotations

import json

import pytest
import yaml

from hypeonoath.core import config
from hypeonoath.evaluation import questions as qmod
from hypeonoath.evaluation.questions import (PRIVACY_CHANGED, load_file, privacy_gate, private_decisions,
                                             question_set_sha256, validate)
from tests.evaluation.gold_fixtures import (APPLICANT, CODE, CV, DECISIONS, PHONE, answer_part, context, parse,
                                            private_part,
                                            valid_raw)


@pytest.fixture
def ctx(tmp_path):
    return context(tmp_path)


def errors_for(raw, ctx, approved_only=True):
    return validate(parse(raw), ctx, approved_only)


def has(errors, code, text=""):
    return any(code in e and text in e for e in errors)


def test_valid_set_passes(ctx):
    assert errors_for(valid_raw(), ctx) == []


def test_v1_duplicate_question_and_part_ids(ctx):
    raw = valid_raw()
    raw[30]["id"] = raw[29]["id"]
    raw[0]["parts"][1]["id"] = "a"
    errors = errors_for(raw, ctx)
    assert has(errors, "V1 duplicate question id") and has(errors, "V1 duplicate part id")


@pytest.mark.parametrize("n", [69, 81])
def test_v1_question_count(ctx, n):
    raw = valid_raw(80)
    raw = raw[:n] if n <= 80 else raw + [dict(valid_raw(1)[0], id="q999")]
    if n == 81:
        raw[-1]["parts"] = [answer_part()]
    assert has(errors_for(raw, ctx), "V1", "need 70-80")


def test_v1_only_approved_count_for_the_frozen_set_but_drafts_count_unless_rejected(ctx):
    raw = valid_raw()
    for r in raw[60:]:
        r["status"] = "draft"
    assert has(errors_for(raw, ctx, approved_only=True), "V1 60 approved")
    assert not has(errors_for(raw, ctx, approved_only=False), "V1")
    raw[60]["status"] = "rejected"
    assert has(errors_for(raw, ctx, approved_only=False), "V1 69 non-rejected")


def test_v2_no_answer_share(ctx):
    raw = valid_raw()
    for r in raw[10:25]:
        r["parts"], r["doc_group"] = [answer_part()], "cv"
    assert has(errors_for(raw, ctx), "V2")


def test_v3_missing_category_and_doc_group_minimum(ctx):
    raw = valid_raw()
    for r in raw:
        if r["category"] == "code":
            r["category"] = "single_fact"
    raw[26]["doc_group"] = "cv"
    errors = errors_for(raw, ctx)
    assert has(errors, "V3 category code") and has(errors, "V3 doc group code: 1")


def test_v3_unknown_category_and_group(ctx):
    raw = valid_raw()
    raw[40]["category"], raw[41]["doc_group"] = "trivia", "nowhere"
    errors = errors_for(raw, ctx)
    assert has(errors, "V3 unknown category") and has(errors, "V3 unknown doc_group")


def test_v4_answer_part_without_key_points_evidence_or_reference(ctx):
    raw = valid_raw()
    raw[40]["parts"][0]["key_points"] = []
    raw[41]["parts"][0]["evidence"] = []
    del raw[42]["parts"][0]["reference_answer"]
    raw[43]["parts"][0]["key_points"] = ["MSc in AI", "Example University"]      # 2 points, 1 group
    errors = errors_for(raw, ctx)
    assert has(errors, "q041/a: V4 answer part without key points")
    assert has(errors, "q042/a: V4 every key point")
    assert has(errors, "q043/a: V4 answer part without reference_answer")
    assert has(errors, "q044/a: V4 every key point")


def test_v5_not_covered_part_with_evidence(ctx):
    raw = valid_raw()
    raw[12]["parts"][0]["evidence"] = [{"point": 0, "any_of": [{"document": CV, "quote": "MSc in AI"}]}]
    assert has(errors_for(raw, ctx), "q013/a: V5")


def test_v6_private_label_not_private_and_missing_placeholder(ctx):
    raw = valid_raw()
    raw[0]["parts"][1]["private_label"] = "town"                     # a PUBLIC decision
    raw[1]["parts"][1]["evidence"][0]["any_of"][0]["quote"] = "Phone:"
    errors = errors_for(raw, ctx)
    assert has(errors, "q001/b: V6 private_label 'town' is not a private decision label")
    assert has(errors, "q002/b: V6 private part needs exactly one group")


def test_v7_quote_not_found_and_document_missing(ctx):
    raw = valid_raw()
    raw[40]["parts"][0]["evidence"][0]["any_of"][0]["quote"] = "PhD in Physics"
    raw[41]["parts"][0]["evidence"][0]["any_of"][0]["document"] = "corpus/converted/nope.md"
    errors = errors_for(raw, ctx)
    assert has(errors, "q041/a: V7 quote not found") and has(errors, "q042/a: V7 document not found")


def test_v7_quote_match_ignores_whitespace_and_case(ctx):
    raw = valid_raw()
    raw[40]["parts"][0]["evidence"][0]["any_of"][0]["quote"] = "msc   IN\nai at example"
    assert errors_for(raw, ctx) == []


def test_d70_evidence_document_outside_every_group(ctx, tmp_path):
    (tmp_path / "corpus/converted/other.md").write_text("MSc in AI\n", encoding="utf-8")
    raw = valid_raw()
    raw[40]["parts"][0]["evidence"][0]["any_of"][0]["document"] = "corpus/converted/other.md"
    assert has(errors_for(raw, ctx), "D70 document belongs to no doc group")


def test_v8_supporting_on_wrong_category(ctx):
    raw = valid_raw()
    target = next(r for r in raw[27:] if r["category"] == "single_fact")
    target["parts"][0]["supporting"] = [{"any_of": [{"document": CV, "quote": "Python and SQL"}]}]
    assert has(errors_for(raw, ctx), "V8")
    target["category"] = "persona_pitch"
    assert not has(errors_for(raw, ctx), "V8")


@pytest.mark.parametrize("field", ["question", "ask", "reference_answer", "key_point", "quote", "notes"])
def test_v9_private_value_anywhere_fails_with_label_only(ctx, field):
    raw = valid_raw()
    entry, part = raw[40], raw[40]["parts"][0]
    if field == "question":
        entry["question"] = f"Is his number {PHONE}?"
    elif field == "notes":
        entry["notes"] = f"id {APPLICANT}"
    elif field == "ask":
        part["ask"] = f"is it {PHONE}"
    elif field == "reference_answer":
        part["reference_answer"] = f"Call {PHONE}."
    elif field == "key_point":
        part["key_points"] = [f"number {APPLICANT}"]
    else:
        part["evidence"][0]["any_of"][0]["quote"] = APPLICANT
    errors = errors_for(raw, ctx)
    assert has(errors, "q041: V9 private value")
    assert not any(PHONE in e or APPLICANT in e or "7788991" in e for e in errors)   # never echo a value


def test_v9_public_values_are_allowed(ctx):
    raw = valid_raw()
    raw[40]["question"] = "Do you live in Exampleton?"                 # public decision (D69)
    assert errors_for(raw, ctx) == []


@pytest.mark.parametrize("count", [9, 16])
def test_v10_mixed_count(ctx, count):
    raw = valid_raw()
    if count == 9:
        raw[0]["parts"] = [answer_part()]
    else:
        for r in raw[27:33]:
            r["parts"] = [answer_part("a"), private_part("b")]
    assert has(errors_for(raw, ctx), "V10", "mixed questions; need 10-15")


@pytest.mark.parametrize("combo,index", [("mix_answer_private", [0, 1, 2]), ("mix_answer_notcovered", [3, 4, 5]),
                                         ("mix_private_notcovered", [6, 7]), ("mix_all_three", [8, 9])])
def test_v10_every_combination_needed(ctx, combo, index):
    raw = valid_raw()
    for i in index:                                                  # replace by another combination
        raw[i]["parts"] = [answer_part("a"), private_part("b")] if combo != "mix_answer_private" else \
            [answer_part("a"), {"id": "b", "ask": "x", "expected": "no_answer", "edge": ["absent_personal"]}]
    assert has(errors_for(raw, ctx), f"V10 no mixed question of combination {combo}")


def test_v11_private_minimum_contact_and_id(ctx):
    raw = valid_raw()
    for i in (0, 1, 2, 6, 7, 8, 9):
        raw[i]["parts"] = [p for p in raw[i]["parts"] if p.get("private_label") is None] or [answer_part()]
    errors = errors_for(raw, ctx)
    assert has(errors, "V11 0 private parts") and has(errors, "V11 no private part about a contact")
    assert has(errors, "V11 no private part about an ID-like")


def test_v11_contact_detail_needed_even_with_enough_private_parts(ctx):
    raw = valid_raw()
    for r in raw:
        for p in r["parts"]:
            if p.get("private_label") == "phone number":
                p.update(private_part(p["id"], "applicant number"))
    errors = errors_for(raw, ctx)
    assert has(errors, "V11 no private part about a contact") and not has(errors, "V11 no private part about an ID")


def test_v12_required_tag_below_minimum_unknown_tag_and_untagged_part(ctx):
    raw = valid_raw()
    for r in raw:
        for p in r["parts"]:
            if p["edge"] == ["decline_direct"]:
                p["edge"] = ["typo"]                                  # typo is optional (min 0)
    raw[40]["parts"][0]["edge"] = ["single_dok"]
    raw[41]["parts"][0]["edge"] = []
    errors = errors_for(raw, ctx)
    assert has(errors, "V12 edge tag decline_direct: 0")
    assert has(errors, "q041/a: V12 unknown edge tag 'single_dok'") and has(errors, "q042/a: V12 part without")


def test_v13_t6_prompt_example_overlap(ctx):
    raw = valid_raw()
    raw[40]["question"] = "<covered question> and <private item>?"
    assert has(errors_for(raw, ctx), "q041: V13")


def test_schema_errors_are_collected_not_raised(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump({"questions": [{"id": "q1", "parts": [{"id": "a", "expected": "maybe"}]}, "x"]}))
    _, qs, errors = load_file(path)
    assert [q.id for q in qs] == ["q1"]
    assert any("expected must be one of" in e for e in errors) and any("without an id" in e for e in errors)


def test_old_flat_questions_file_is_rejected_with_a_hint(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("- question: hi\n  expected_answerable: true\n")
    assert "questions_v0" in load_file(path)[2][0]


def test_hash_is_stable_under_key_reordering_and_ignores_non_approved():
    raw = valid_raw()
    reordered = [dict(reversed(list(r.items()))) for r in reversed(raw)]
    assert question_set_sha256(parse(raw)) == question_set_sha256(parse(reordered))
    raw[5]["status"] = "rejected"
    changed = question_set_sha256(parse(raw))
    raw.append(dict(valid_raw(1)[0], id="q500", status="draft"))
    assert question_set_sha256(parse(raw)) == changed != question_set_sha256(parse(valid_raw()))


def test_mix_tags_and_kinds():
    qs = parse(valid_raw())
    assert [q.mix_tag for q in qs[:10]] == ["mix_answer_private"] * 3 + ["mix_answer_notcovered"] * 3 + \
        ["mix_private_notcovered"] * 2 + ["mix_all_three"] * 2
    assert qs[20].mix_tag is None and not qs[20].answerable and qs[30].answerable


def test_private_decisions_reads_labels_types_kinds_only(tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"schema_version": 2, "entries": DECISIONS}))
    info = private_decisions(path)
    assert set(info) == {"phone number", "applicant number"}                 # public 'town' left out
    assert info["phone number"].entity_types == {"PHONE_NUMBER"} and info["phone number"].match_kinds == {"id"}
    assert PHONE not in repr(info) and "491512345678" not in repr(info)


def test_privacy_gate_d49():
    assert privacy_gate({"privacy_fp": "abc"}, "abc") is None
    assert privacy_gate({"privacy_fp": "abc"}, "xyz") == PRIVACY_CHANGED
    assert "not frozen" in privacy_gate({}, "abc")


def test_current_privacy_fp_follows_the_decisions_file():
    config.PRIVACY_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PRIVACY_DECISIONS_PATH.write_text(json.dumps({"schema_version": 2, "entries": DECISIONS}))
    first = qmod.current_privacy_fp()
    entries = [dict(DECISIONS[0], label="mobile")] + DECISIONS[1:]
    config.PRIVACY_DECISIONS_PATH.write_text(json.dumps({"schema_version": 2, "entries": entries}))
    assert qmod.current_privacy_fp() != first


def test_real_config_files_are_consistent():
    """eval/doc_groups.yaml lists every corpus/ evidence file of plan 3.2 (D70); edge_tags.yaml has every 3.4 tag."""
    groups = qmod.load_doc_groups()
    assert set(groups) == {"cv", "cover_letter", "certificates", "admission_letter", "system_design", "readme", "code"}
    assert {"corpus/converted/CV_Udbhav.md", "corpus/converted/CV_Udbhav2.md",
            "corpus/converted/CV_Udbhav3.md"} == set(groups["cv"].files)
    assert len(groups["certificates"].files) == 5 and groups["certificates"].minimum == 4
    assert {g.minimum for g in groups.values()} == {None, 2, 3, 4, 7}
    tags = qmod.load_edge_tags()
    assert len(tags) == 40 and all(t.required for t in tags.values())
    assert tags["single_doc"].minimum == 8 and tags["decline_synonym"].minimum == 2
    assert not any(t.startswith("mix_") for t in tags)
    assert CODE and CV                                                       # fixture paths stay fixture paths


def test_v9_scans_every_string_of_the_entry_unknown_keys_included(ctx):
    """freeze writes the whole YAML entry, so a value under any key must fail V9 (tester finding 2)."""
    raw = valid_raw()
    raw[40]["parts"][0]["comment"] = f"call {PHONE}"
    raw[41]["extra"] = {"nested": [APPLICANT]}
    errors = errors_for(raw, ctx)
    assert has(errors, "q041: V9 private value (phone number) in parts.0.comment")
    assert has(errors, "q042: V9 private value (applicant number) in extra.nested.0")


def test_messages_never_echo_a_value_typed_into_a_label_tag_or_category(ctx):
    raw = valid_raw()
    raw[0]["parts"][1]["private_label"] = PHONE
    raw[40]["category"] = APPLICANT
    raw[41]["parts"][0]["edge"] = [PHONE]
    errors = errors_for(raw, ctx)
    assert errors and not any(s in e for e in errors for s in (PHONE, APPLICANT, "2345678", "7788991"))
    assert has(errors, "V6") and has(errors, "V3 unknown category") and has(errors, "V12 unknown edge tag")


def test_entry_without_id_is_not_echoed(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump({"questions": [{"question": f"call {PHONE}"}]}))
    errors = load_file(path)[2]
    assert errors == ["an entry without an id (its content is not shown)"]
