"""core.private_values: normalisation, dates, value keys, matching, redaction,
loading and the fingerprint (privacy plan v3, 6.4.1, 6.6; tests 14.3)."""
from __future__ import annotations

import datetime
import json

import pytest

from hypeonoath.core import config
from hypeonoath.core import private_values as pv

DOB = datetime.date(1999, 3, 12)


def value(kind: str, normalized: str, *surface: str, label: str = "label", vid: str = "pv_1") -> pv.PrivateValue:
    return pv.PrivateValue(vid, label, kind, normalized, tuple(surface))


PHONE = value("id", "491512345678", "+49 151 2345678", label="phone number", vid="pv_phone")
DATE = value("date", "1999-03-12", "12.03.1999", label="date of birth", vid="pv_dob")
ADDRESS = value("text", pv.normalize("Lauteschlägerstraße 12", "text"), "Lauteschlägerstraße 12",
                label="address", vid="pv_addr")
EMAIL = value("email", "jane.doe@example.com", "Jane.Doe@example.com", label="email address", vid="pv_mail")


# --- normalisation -----------------------------------------------------------------------

@pytest.mark.parametrize("raw, kind, expected", [
    ("+49 151 234-5678", "id", "491512345678"),
    ("DE89 3704 0044", "id", "DE8937040044"),
    ("  Jane.Doe@Example.COM ", "email", "jane.doe@example.com"),
    ("Lauteschlägerstraße 12", "text", "lauteschlaegerstrasse 12"),
    ("Lauteschlaegerstr. 12", "text", "lauteschlaegerstrasse 12"),
    ("221B Baker St.", "text", "221b baker street"),
    ("Main Rd., Ave.", "text", "main road avenue"),
    ("  Udbhav   Narayan,  Sharma! ", "text", "udbhav narayan sharma"),
    ("Müller", "text", "mueller"),                     # decomposed umlaut -> NFKC
    ("12. März 1999", "date", "1999-03-12"),
    ("not a date", "date", ""),
])
def test_normalize(raw, kind, expected):
    assert pv.normalize(raw, kind) == expected


def test_first_dot_word_is_not_a_street():
    assert pv.normalize("first.", "text") == "first"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        pv.normalize("x", "nope")


@pytest.mark.parametrize("text", [
    "12.03.1999", "12.3.1999", "12. 03. 1999", "12/03/1999", "1999-03-12", "12 March 1999", "12. März 1999",
    "12 Mrz 1999", "12 Mar 1999", "March 12, 1999", "Mar 12th 1999", "12 maerz 1999", " 12.03.1999 ",
])
def test_parse_date_formats(text):
    assert pv.parse_date(text) == DOB


@pytest.mark.parametrize("text", ["12.03.99", "31.02.1999", "1999", "March 1999", "12.03.1999 extra", ""])
def test_parse_date_rejects(text):
    assert pv.parse_date(text) is None


def test_parse_date_day_first_then_month_first():
    assert pv.parse_date("03/12/1999") == datetime.date(1999, 12, 3)     # day-first wins
    assert pv.parse_date("12/31/1999") == datetime.date(1999, 12, 31)    # only month-first is valid


# --- value keys (6.4.1) -----------------------------------------------------------------------

def test_value_keys_phone():
    keys = pv.value_keys("+49 151 234-5678")
    assert "id:491512345678" in keys and any(k.startswith("text:") for k in keys)


def test_value_keys_same_date_in_two_formats_share_a_key():
    assert pv.value_keys("12. März 1999") & pv.value_keys("1999-03-12") == {"date:1999-03-12"}


def test_value_keys_short_word_has_no_id_key():
    assert pv.value_keys("Ali") == frozenset({"text:ali"})


def test_value_keys_email():
    assert "email:udbhav@x.de" in pv.value_keys("Udbhav@X.de")


# --- entity type mapping (5.3) -------------------------------------------------------------------

@pytest.mark.parametrize("etype, val, kind", [
    ("EMAIL_ADDRESS", "a@b.de", "email"),
    ("PHONE_NUMBER", "+49 1", "id"),
    ("DATE_TIME", "12.03.1999", "date"),
    ("DATE_TIME", "last spring", "text"),
    ("DATE_OF_BIRTH", "March 12, 1999", "date"),
    ("MONEY", "€65,000", "id"),
    ("PERSON", "Jane", "text"),
    ("DE_TAX_ID", "12345678995", "id"),
    ("IN_PAN", "ABCDE1234F", "id"),
    ("SECRET:AWS Access Key", "AKIA...", "id"),
    ("SOMETHING_NEW", "ab 1234", "id"),
    ("SOMETHING_NEW", "abc", "text"),
])
def test_match_kind_for(etype, val, kind):
    assert pv.match_kind_for(etype, val) == kind


@pytest.mark.parametrize("etype, label", [
    ("PHONE_NUMBER", "phone number"), ("IN_AADHAAR", "Aadhaar number"), ("DE_TAX_ID", "tax ID"),
    ("SECRET:AWS Access Key", "secret"), ("MANUAL", "private detail"), ("DATE_OF_BIRTH", "date of birth"),
])
def test_default_label(etype, label):
    assert pv.default_label(etype) == label


# --- matching (6.6) ---------------------------------------------------------------------------------

def spans(text, *values):
    return pv.find_private_spans(text, list(values))


def test_surface_form_case_insensitive_with_word_boundaries():
    name = value("text", "ali khan", "Ali Khan")
    assert [(s.start, s.end, s.how) for s in spans("hi ALI KHAN!", name)] == [(3, 11, "surface")]
    assert spans("Alice Khanna", value("text", "ali", "Ali")) == []


def test_exact_id_after_reformatting():
    text = "Phone: 0049-151-2345678 or +49 (151) 2345678"
    found = spans(text, PHONE)
    assert len(found) == 2 and all(s.label == "phone number" for s in found)


def test_partial_id_without_country_code():
    # Settles the plan 14.3 assumption: "1512345678" is a >= 6-char run of the stored ID.
    text = "call 0151 2345678 today"
    (span,) = spans(text, PHONE)
    assert span.how == "partial_id" and text[span.start:span.end] == "0151 2345678"


def test_partial_id_needs_six_chars_and_four_digits():
    assert spans("code 12345 here", PHONE) == []          # 5 chars only
    letters = value("id", "ABCDEFGH1", "ABCDEFGH1")
    assert spans("see ABCDEFG 2024 now", letters) == []   # shared run is letters, not digits


def test_unrelated_numbers_are_not_glued():
    stored = value("id", "123456789", "123456789")
    assert spans("years 12345 67890", stored) == []        # exact must align to tokens; partial splits groups


def test_spaced_id_with_two_long_groups_still_matches_exactly():
    indian = value("id", "919876543210", label="phone number")
    text = "mobile +91 98765 43210."
    (span,) = spans(text, indian)
    assert span.how == "exact" and text[span.start:span.end] == "91 98765 43210"


@pytest.mark.parametrize("text", ["born 1999-03-12", "born 12 March 1999", "born 12. März 1999",
                                  "born 03/12/1999"])
def test_date_in_any_format(text):
    (span,) = spans(text, DATE)
    assert span.how == "date" and text[span.start:span.end] == text[5:]


def test_other_date_is_not_a_match():
    assert spans("born 13.03.1999", DATE) == []


def test_address_matches_after_normalisation():
    text = "Lives at Lauteschlaegerstr. 12, Darmstadt."
    (span,) = spans(text, ADDRESS)
    assert span.how == "exact" and text[span.start:span.end] == "Lauteschlaegerstr. 12"


def test_fuzzy_text_match_for_long_values_only():
    (span,) = spans("at Lauteschlagerstrasse 12 x", ADDRESS)     # one letter off
    assert span.how == "fuzzy"
    short = value("text", "berlin", "Berlin")
    assert spans("Merlin", short) == []                          # < GUARD_FUZZY_MIN_CHARS


def test_email_case_insensitive():
    (span,) = spans("Mail JANE.DOE@EXAMPLE.COM now", EMAIL)
    assert span.label == "email address"


def test_overlapping_spans_merge_and_positions_refer_to_original():
    text = "DOB: 12.03.1999 (+49 151 2345678)"
    found = spans(text, DATE, PHONE)
    assert [text[s.start:s.end] for s in found] == ["12.03.1999", "+49 151 2345678"]
    overlapping = value("text", "dob 12.03.1999", "DOB: 12.03.1999", label="dob line")
    (merged,) = spans(text[:15], DATE, overlapping)
    assert (merged.start, merged.end, merged.label) == (0, 15, "dob line")       # label of the longest span


def test_no_values_or_text_is_cheap_noop():
    assert spans("", PHONE) == [] and spans("anything", ) == []


# --- redaction -----------------------------------------------------------------------------------------

def test_redact_placeholder_and_idempotent():
    text = "Born 12.03.1999, phone +49 151 2345678."
    once, found = pv.redact(text, [DATE, PHONE])
    assert once == "Born [REDACTED: date of birth], phone [REDACTED: phone number]."
    assert len(found) == 2
    twice, again = pv.redact(once, [DATE, PHONE])
    assert twice == once and again == []


def test_redact_custom_replacement():
    assert pv.redact("x 12.03.1999", [DATE], lambda s: "[private]")[0] == "x [private]"


def test_placeholder_format():
    assert pv.placeholder("date of birth") == "[REDACTED: date of birth]"


# --- loading + fingerprint ----------------------------------------------------------------------------

def entry(**over):
    base = {"id": "pv_a", "decision": "private", "label": "phone number", "entity_type": "PHONE_NUMBER",
            "match_kind": "id", "value": "+49 151 2345678", "normalized": "491512345678",
            "aliases": [], "surface_forms": ["+49 151 2345678"]}
    base.update(over)
    return base


def test_values_from_entries_private_only_one_per_form():
    entries = [
        entry(aliases=[{"entity_type": "DE_TAX_ID", "match_kind": "id", "normalized": "491512345678"},
                       {"entity_type": "DATE_TIME", "match_kind": "text", "normalized": "49 151 2345678"}]),
        entry(id="pv_k", decision="keep", label=None),
    ]
    values = pv.values_from_entries(entries)
    assert [(v.id, v.match_kind, v.normalized) for v in values] == [
        ("pv_a", "id", "491512345678"), ("pv_a", "text", "49 151 2345678")]
    assert all(v.label == "phone number" and v.surface_forms == ("+49 151 2345678",) for v in values)


def test_load_private_values_missing_file_is_empty(tmp_path):
    assert pv.load_private_values(tmp_path / "nope.json") == []


def test_load_private_values_default_path(tmp_path):
    config.PRIVACY_DECISIONS_PATH.parent.mkdir(parents=True)
    config.PRIVACY_DECISIONS_PATH.write_text(json.dumps({"schema_version": 2, "entries": [entry()]}))
    assert [v.id for v in pv.load_private_values()] == ["pv_a"]


@pytest.mark.parametrize("content", ["{not json", '["list"]', '{"entries": 3}'])
def test_unreadable_file_error_never_quotes_content(tmp_path, content):
    path = tmp_path / "d.json"
    path.write_text(content + " SECRET-VALUE-123")
    with pytest.raises(ValueError) as info:
        pv.load_private_values(path)
    assert "SECRET-VALUE-123" not in str(info.value)


def test_fingerprint_rules():
    base = pv.values_from_entries([entry()])
    assert pv.privacy_fingerprint([]) == "none"
    fp = pv.privacy_fingerprint(base)
    assert len(fp) == 16
    assert pv.privacy_fingerprint(base + pv.values_from_entries([entry(id="pv_k", decision="keep")])) == fp
    assert pv.privacy_fingerprint(pv.values_from_entries([entry(label="mobile")])) != fp
    same_form = entry(aliases=[{"entity_type": "DE_TAX_ID", "match_kind": "id", "normalized": "491512345678"}])
    assert pv.privacy_fingerprint(pv.values_from_entries([same_form])) == fp       # known form: no change
    new_form = entry(aliases=[{"entity_type": "X", "match_kind": "text", "normalized": "49 151 2345678"}])
    assert pv.privacy_fingerprint(pv.values_from_entries([new_form])) != fp       # new form: changes
