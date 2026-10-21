"""serving.guard: answer-time guard over owner-marked values (privacy plan v3, 6.9; tests 14.8)."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core import config
from hypeonoath.core.private_values import PrivateValue
from hypeonoath.serving.guard import guard_answer

ID = PrivateValue("pv_id", "applicant number", "id", "BEW20234711", ("BEW-2023-4711",))
DOB = PrivateValue("pv_dob", "date of birth", "date", "1999-03-12", ("12.03.1999",))
ADDRESS = PrivateValue("pv_addr", "address", "text", "lauteschlaegerstrasse 12", ("Lauteschlägerstraße 12",))
VALUES = [ID, DOB, ADDRESS]
LONG = ("The candidate built an on-device health app, led analytics projects across FinTech and travel, "
        "and studies AI at TU Darmstadt. ")


def test_no_hits_returns_answer_unchanged():
    result = guard_answer(LONG, "What did they build?", VALUES)
    assert (result.text, result.hits, result.declined) == (LONG, (), False)


def test_one_id_in_a_long_answer_only_the_span_is_replaced():
    answer = LONG + "Their application BEW-2023-4711 was accepted."
    result = guard_answer(answer, "Tell me about the candidate", VALUES)
    assert not result.declined and result.hits == ("applicant number",)
    assert result.text == (LONG + f"Their application {config.GUARD_SPAN_REPLACEMENT} was accepted.\n\n"
                           + config.GUARD_NOTE)


def test_reformatted_date_leak_is_caught():
    result = guard_answer(LONG + "Born 1999-03-12.", "Tell me more", VALUES)
    assert "1999-03-12" not in result.text and result.hits == ("date of birth",)


def test_answer_mostly_the_value_declines():
    result = guard_answer("BEW-2023-4711", "Anything else?", VALUES)
    assert result.declined and result.text == "The candidate prefers not to share their applicant number here."


def test_question_asking_for_the_label_declines_even_with_a_small_share():
    result = guard_answer(LONG + "Born 12.03.1999.", "What's his date of birth?", VALUES)
    assert result.declined and result.text == config.DECLINE_TEMPLATES["third"].format(label="date of birth")


@pytest.mark.parametrize("person, expected", [
    ("third", "The candidate prefers not to share their date of birth and address here."),
    ("first", "I'd rather not share my date of birth and address here."),
])
def test_two_labels_are_named_in_the_right_person(person, expected):
    answer = "12.03.1999, Lauteschlaegerstr. 12"
    assert guard_answer(answer, "where and when?", VALUES, person=person).text == expected


def test_three_labels_join():
    answer = "12.03.1999, Lauteschlaegerstr. 12, BEW-2023-4711"
    assert guard_answer(answer, "x", VALUES).text.endswith(
        "their date of birth, address and applicant number here.")


def test_no_values_means_guard_inactive():
    assert guard_answer("BEW-2023-4711", "id?", []).text == "BEW-2023-4711"


def test_log_record_has_labels_only(isolated_logs):
    guard_answer(LONG + "Born 12.03.1999.", "Tell me more", VALUES)
    raw = (isolated_logs / "privacy_log_path.jsonl").read_text(encoding="utf-8")
    record = json.loads(raw.strip())
    assert set(record) == {"timestamp", "event", "labels", "declined"}
    assert record["labels"] == ["date of birth"] and record["declined"] is False
    assert "1999" not in raw and "candidate" not in raw
