"""serving.guard: answer-time guard over owner-marked values (privacy plan v3, 6.9; P0 guideline, eval D63)."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core import config
from hypeonoath.core.private_values import PrivateValue, find_private_spans
from hypeonoath.serving.guard import guard_answer

ID = PrivateValue("pv_id", "applicant number", "id", "BEW20234711", ("BEW-2023-4711",))
DOB = PrivateValue("pv_dob", "date of birth", "date", "1999-03-12", ("12.03.1999",))
ADDRESS = PrivateValue("pv_addr", "address", "text", "lauteschlaegerstrasse 12", ("Lauteschlägerstraße 12",))
PHONE = PrivateValue("pv_phone", "phone number", "id", "491512345678", ("+49 151 2345678",))
SPLIT = PrivateValue("pv_split", "nickname", "text", "jane doe. main street", ("Jane Doe. Main Street",))
VALUES = [ID, DOB, ADDRESS, PHONE, SPLIT]
LONG = ("I built an on-device health app, led analytics projects across FinTech and travel, "
        "and study AI at TU Darmstadt. ")

# Every answer that holds a private value; the guarded text must never hold one (T-P0-4).
HIT_ANSWERS = [
    LONG + "Their application BEW-2023-4711 was accepted.",
    "I studied at TU Darmstadt [cv.md]. I was born on 12.03.1999 [cv.md].",
    "- Python\n- Phone: +49 151 2345678\n- SQL and dashboards",
    "Call me Jane Doe. Main Street is where I live. I code Python every day.",
    LONG + "Born 1999-03-12.",
    "BEW-2023-4711",
    "BEW-2023-4711. I code SQL.",
    "Sure. Born 12.03.1999, see [corpus/converted/cv.md].",
    "[corpus/converted/cv.md] Yes. Born on 12.03.1999 in a small town far away from here.",
    LONG + "Born 12.03.1999.",
    "12.03.1999, Lauteschlaegerstr. 12, BEW-2023-4711",
]


def test_no_hits_returns_answer_unchanged():
    result = guard_answer(LONG, VALUES)
    assert (result.text, result.hits, result.declined) == (LONG, (), False)


def test_one_id_in_a_long_answer_drops_only_its_sentence():
    result = guard_answer(LONG + "Their application BEW-2023-4711 was accepted.", VALUES)
    assert result.text == LONG.strip() + "\n\n" + config.GUARD_LINE
    assert result.hits == ("applicant number",) and not result.declined


def test_mixed_answer_keeps_the_public_part():
    result = guard_answer("I studied at TU Darmstadt [cv.md]. I was born on 12.03.1999 [cv.md].", VALUES)
    assert result.text == "I studied at TU Darmstadt [cv.md].\n\n" + config.GUARD_LINE
    assert "1999" not in result.text and not result.declined


def test_bullet_line_is_dropped_alone():
    result = guard_answer("- Python\n- Phone: +49 151 2345678\n- SQL and dashboards", VALUES)
    assert result.text == "- Python\n- SQL and dashboards\n\n" + config.GUARD_LINE
    assert result.hits == ("phone number",)


def test_span_across_a_sentence_boundary_drops_both_sentences():
    result = guard_answer("Call me Jane Doe. Main Street is where I live. I code Python every day.", VALUES)
    assert result.text == "I code Python every day.\n\n" + config.GUARD_LINE
    assert "Jane" not in result.text and "Main Street" not in result.text


def test_reformatted_date_leak_is_caught():
    result = guard_answer(LONG + "Born 1999-03-12.", VALUES)
    assert "1999" not in result.text and result.hits == ("date of birth",)


def test_answer_mostly_the_value_is_the_line_alone():
    result = guard_answer("BEW-2023-4711", VALUES)
    assert result.declined and result.text == config.GUARD_LINE


def test_share_exactly_at_the_threshold_is_the_line_alone():
    """T-P0-2: >= the share. 13 of 26 chars; with > the 3 public words would be kept."""
    answer = "BEW-2023-4711. I code SQL."
    assert len(answer) == 2 * len("BEW-2023-4711")
    result = guard_answer(answer, VALUES)
    assert result.declined and result.text == config.GUARD_LINE


def test_fewer_than_min_words_left_is_the_line_alone():
    result = guard_answer("Sure. Born 12.03.1999, see [corpus/converted/cv.md].", VALUES)
    assert result.declined and result.text == config.GUARD_LINE


def test_citations_do_not_count_as_words():
    result = guard_answer("[corpus/converted/cv.md] Yes. Born on 12.03.1999 in a small town far away from here.",
                          VALUES)
    assert result.declined and result.text == config.GUARD_LINE


def test_small_share_is_not_the_line_alone_even_when_asked_for():
    """P0-3: the "question names the label" trigger is gone; the public part stays."""
    result = guard_answer(LONG + "Born 12.03.1999.", VALUES)
    assert not result.declined and result.text == LONG.strip() + "\n\n" + config.GUARD_LINE


def test_several_labels_in_first_appearance_order():
    result = guard_answer("12.03.1999, Lauteschlaegerstr. 12, BEW-2023-4711", VALUES)
    assert result.hits == ("date of birth", "address", "applicant number")
    assert result.declined and result.text == config.GUARD_LINE


@pytest.mark.parametrize("answer", HIT_ANSWERS)
def test_guarded_text_holds_no_private_value_and_no_hint(answer):
    """T-P0-4 invariant + D56: no value, no label, no word saying something is withheld."""
    result = guard_answer(answer, VALUES)
    assert result.hits and find_private_spans(result.text, VALUES) == []
    lowered = result.text.lower()
    for word in ("private", "redacted", "hidden", *(v.label for v in VALUES)):
        assert word not in lowered


def test_recheck_on_kept_text_forces_the_line_alone(monkeypatch):
    """T-P0-4: if the kept text still matches (e.g. a fuzzy join), the reply is the line alone."""
    from hypeonoath.serving import guard

    real = guard.find_private_spans
    calls = []

    def fake(text, values):
        calls.append(text)
        return real(LONG + "Born 12.03.1999.", values) if len(calls) == 2 else real(text, values)

    monkeypatch.setattr(guard, "find_private_spans", fake)
    result = guard.guard_answer(LONG + "Born 12.03.1999.", VALUES)
    assert len(calls) == 2 and result.declined and result.text == config.GUARD_LINE


def test_no_values_means_guard_inactive():
    result = guard_answer("BEW-2023-4711", [])
    assert (result.text, result.hits, result.declined) == ("BEW-2023-4711", (), False)


def test_log_record_has_labels_only(isolated_logs):
    guard_answer(LONG + "Born 12.03.1999.", VALUES)
    raw = (isolated_logs / "privacy_log_path.jsonl").read_text(encoding="utf-8")
    record = json.loads(raw.strip())
    assert set(record) == {"timestamp", "event", "labels", "declined"}
    assert record["labels"] == ["date of birth"] and record["declined"] is False
    assert "1999" not in raw and "Darmstadt" not in raw
