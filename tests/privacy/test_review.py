"""privacy.review: per-file screens with fake stdin/stdout (privacy plan v10, 6.5, P-D43 – P-D45; tests 14.5, 14.10)."""
from __future__ import annotations

import io

import pytest

from hypeonoath.core import config
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy import review

from tests.factories import candidate

PHONE = "+49 151 2345678"
KEY = "AKIAIOSFODNN7EXAMPLE"
IBAN = "DE89 3704 0044 0532 0130 00"
CV, LETTER = "corpus/converted/CV.md", "corpus/converted/Letter.md"
CV_TEXT = f"Jane Doe\nBerlin\nworked at X\nPhone: {PHONE}\n"
LETTER_TEXT = f"Dear Jane Doe\nIBAN: {IBAN}\ntoken = {KEY}\nPhone: {PHONE}\n"


@pytest.fixture(autouse=True)
def corpus_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    for rel, text in ((CV, CV_TEXT), (LETTER, LETTER_TEXT)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)


def stored(value, etype, *places, decision="keep", label=None, source="detected"):
    """A decision for `value` located at (file, line) places, as the scan stores it."""
    cands = [candidate(value, etype, file=f, line=n) for f, n in places]
    (group,) = dec.group_candidates(cands)
    made = dec.new_decision(group, decision, label, source=source)
    return made


def sample():
    return dec.DecisionFile([
        stored("Jane Doe", "PERSON", (CV, 1), (LETTER, 1)),
        stored("Berlin", "LOCATION", (CV, 2)),
        stored(PHONE, "PHONE_NUMBER", (CV, 4), (LETTER, 4)),
        stored(IBAN, "IBAN_CODE", (LETTER, 2)),
        stored(KEY, "SECRET:AWS Access Key", (LETTER, 3), decision="private", label="secret", source="default_secret"),
    ])


def run(screens, data, answers):
    stdin, stdout = io.StringIO(answers), io.StringIO()
    saves = []
    outcome = review.review(screens, data, stdin=stdin, stdout=stdout,
                            save=lambda d: saves.append([x.to_dict() for x in d.decisions]))
    return outcome, stdout.getvalue(), saves


def state(data, etype):
    return next(d for d in data.decisions if d.entity_type == etype).decision


# --- categories and order (P-D44) ----------------------------------------------------------------------------

@pytest.mark.parametrize("etype, value, expected", [
    ("SECRET:AWS Access Key", KEY, review.CRITICAL), ("IBAN_CODE", IBAN, review.CRITICAL),
    ("DE_TAX_ID", "12345678995", review.CRITICAL), ("US_SSN", "536-22-1234", review.CRITICAL),
    ("PHONE_NUMBER", PHONE, review.HIGH), ("EMAIL_ADDRESS", "a@b.de", review.HIGH),
    ("DATE_OF_BIRTH", "12.03.1999", review.HIGH), ("APPLICANT_ID", "658336", review.HIGH),
    ("LOCATION", "Hauptstraße 5", review.HIGH), ("PERSON", "Jane Doe", review.MEDIUM),
    ("IP_ADDRESS", "10.0.0.1", review.MEDIUM), ("LOCATION", "Berlin", review.LOW),
    ("DATE_TIME", "2023", review.LOW), ("URL", "https://x.org", review.LOW), ("NRP", "Indian", review.LOW),
])
def test_type_categories(etype, value, expected):
    assert review.type_category(etype, value) == expected


def test_upload_order_puts_pending_files_first_access_order_by_counts():
    data = sample()
    upload = review.build_reviews(data, pending=[CV])
    assert [s.file for s in upload] == [CV, LETTER] and upload[0].pending
    access = review.build_reviews(data, pending=[CV], order="access")
    assert [s.file for s in access] == [LETTER, CV]                   # 2 critical values beat none
    assert [s.file for s in review.build_reviews(data, pending=[CV], pending_only=True)] == [CV]
    assert [s.file for s in review.build_reviews(data, selection=LETTER)] == [LETTER]


def test_screen_groups_by_category_and_highlights_the_value_in_its_line():
    (letter,) = review.build_reviews(sample(), pending=[LETTER], selection=LETTER)
    assert [i.decision.entity_type for i in letter.items] == ["IBAN_CODE", "SECRET:AWS Access Key", "PHONE_NUMBER",
                                                             "PERSON"]
    text = review.render(letter, 1, 1)
    assert "PENDING REVIEW" in text and "⚠ critical values in this file · public for all: off" in text
    assert text.index("  critical") < text.index("  high") < text.index("  medium")
    assert f"IBAN: »{IBAN}«" in text and "secret (default)" in text and "(also in 1 file(s))" in text


# --- private values shown in their original line (privacy-review-line-restore-plan-v1) ---------------------

def write(rel, text):
    (config.PROJECT_ROOT / rel).write_text(text, encoding="utf-8")


def item_text(data, file, etype):
    (screen,) = review.build_reviews(data, selection=file)
    n = next(i for i, item in enumerate(screen.items) if item.decision.entity_type == etype)
    return review.render(screen, 1, 1).splitlines(), screen.items[n]


def test_private_value_redacted_on_disk_is_shown_in_its_line():
    write(LETTER, "Dear Jane Doe\nIBAN: [REDACTED: bank account number] was given to the office\n")
    data = sample()
    iban = next(d for d in data.decisions if d.entity_type == "IBAN_CODE")
    iban.decision, iban.label = "private", "bank account number"
    lines, item = item_text(data, LETTER, "IBAN_CODE")
    assert item.line_text == f"IBAN: {IBAN} was given to the office"
    assert any(f"IBAN: »{IBAN}« was given to the office" in line for line in lines)
    assert not any("redacted on disk" in line or "[REDACTED" in line for line in lines)


def test_two_private_values_with_different_labels_in_one_line():
    write(CV, "Contact: [REDACTED: phone number] or [REDACTED: email address] today\n")
    data = dec.DecisionFile([
        stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="phone number"),
        stored("jane@x.de", "EMAIL_ADDRESS", (CV, 1), decision="private", label="email address"),
    ])
    (screen,) = review.build_reviews(data, selection=CV)
    assert {i.line_text for i in screen.items} == {f"Contact: {PHONE} or jane@x.de today"}
    text = review.render(screen, 1, 1)
    assert f"»{PHONE}« or jane@x.de" in text and f"{PHONE} or »jane@x.de«" in text


def test_same_label_twice_is_filled_in_order():
    write(CV, "Phones: [REDACTED: phone number], [REDACTED: phone number]\n")
    data = dec.DecisionFile([
        stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="phone number"),
        stored("+49 30 1234567", "PHONE_NUMBER", (CV, 1), decision="private", label="phone number"),
    ])
    (screen,) = review.build_reviews(data, selection=CV)
    assert screen.items[0].line_text == f"Phones: {PHONE}, +49 30 1234567"


def test_relabelled_value_is_restored_by_order():
    write(CV, "Phone: [REDACTED: phone number]\n")
    data = dec.DecisionFile([stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="office phone")])
    (screen,) = review.build_reviews(data, selection=CV)
    assert screen.items[0].line_text == f"Phone: {PHONE}" and screen.items[0].start == len("Phone: ")


def test_placeholder_of_a_value_located_elsewhere_stays():
    write(CV, "Phone: [REDACTED: phone number]\nJane Doe\n")
    data = dec.DecisionFile([stored(PHONE, "PHONE_NUMBER", (CV, 2), decision="private", label="phone number"),
                             stored("Jane Doe", "PERSON", (CV, 1))])
    (screen,) = review.build_reviews(data, selection=CV)
    person = next(i for i in screen.items if i.decision.entity_type == "PERSON")
    assert person.line_text == "Phone: [REDACTED: phone number]" and person.start == -1


def test_public_value_sharing_a_line_with_a_redacted_private_one():
    write(CV, "Jane Doe, phone [REDACTED: phone number]\n")
    data = dec.DecisionFile([stored("Jane Doe", "PERSON", (CV, 1)),
                             stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="phone number")])
    lines, item = item_text(data, CV, "PERSON")
    assert any(f"»Jane Doe«, phone {PHONE}" in line for line in lines)


def test_unknown_line_falls_back_to_the_value_only():
    data = dec.DecisionFile([stored("Demo app", "PERSON", ("corpus/code/_metadata.yaml", None))])
    (screen,) = review.build_reviews(data)
    assert screen.items[0].start == -1 and "│ value »Demo app«" in review.render(screen, 1, 1)


def test_restore_line_without_placeholders_is_unchanged():
    decision = stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="phone number")
    assert review._restore_line("plain text", [decision]) == "plain text"
    assert review._restore_line("x [REDACTED: phone number]", []) == "x [REDACTED: phone number]"


def test_building_screens_writes_nothing():
    write(CV, "Phone: [REDACTED: phone number]\n")
    data = dec.DecisionFile([stored(PHONE, "PHONE_NUMBER", (CV, 1), decision="private", label="phone number")])
    dec.save(data)
    files = [config.PROJECT_ROOT / CV, config.PRIVACY_DECISIONS_PATH]
    before = [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
    review.render(review.build_reviews(data)[0], 1, 1)
    assert [(f.stat().st_mtime_ns, f.read_bytes()) for f in files] == before


# --- keys --------------------------------------------------------------------------------------------------

def test_numbers_switch_value_wide_and_save_at_once(monkeypatch):
    from hypeonoath.privacy import restore

    monkeypatch.setattr(restore, "raw_sources_for", lambda files: ["CV.pdf"])
    data = sample()
    screens = review.build_reviews(data, pending=[CV])                  # CV: [1] phone (high), [2] person, [3] Berlin
    outcome, out, saves = run(screens, data, "1 2\n1\nq\n")
    phone = next(d for d in data.decisions if d.entity_type == "PHONE_NUMBER")
    assert outcome == review.ReviewOutcome(True, 2) and len(saves) == 2
    assert (phone.decision, state(data, "PERSON")) == ("keep", "private")      # phone switched back, person private
    assert "[1] also changes 1 other file(s)" in out and "1 document(s) will be rebuilt from raw" in out
    assert data.pending_restore == ["CV.pdf"]
    person = next(d for d in data.decisions if d.entity_type == "PERSON")
    assert person.label == "name"                                      # default label


def test_public_for_all_needs_a_confirmation_when_secrets_are_inside():
    data = sample()
    screens = review.build_reviews(data, selection=LETTER)
    _, out, saves = run(screens, data, "a\nno\na\nyes\nq\n")
    assert out.count("This makes 1 secret(s) public everywhere") == 2 and "Nothing changed." in out
    assert len(saves) == 1 and all(not d.is_private for d in data.decisions)
    assert "public for all: on" in out


def test_private_for_all_and_public_for_all_without_secrets_asks_nothing():
    data = sample()
    screens = review.build_reviews(data, selection=CV)
    _, out, _ = run(screens, data, "x\n")
    assert {state(data, t) for t in ("PERSON", "LOCATION", "PHONE_NUMBER")} == {"private"}
    _, out, _ = run(screens, data, "a\nq\n")
    assert "secret(s) public" not in out and {state(data, t) for t in ("PERSON", "LOCATION", "PHONE_NUMBER")} == {
        "keep"}


def test_label_edit_and_refusals():
    data = sample()
    screens = review.build_reviews(data, selection=LETTER)              # [2] = the private secret, [3] phone
    _, out, saves = run(screens, data, f"l3\nl2\n{KEY} key\nl2\naws key\nq\n")
    assert "Only private values have a label" in out and "must not contain the private value" in out
    assert next(d for d in data.decisions if d.entity_type.startswith("SECRET")).label == "aws key"
    assert len(saves) == 1


def test_enter_walks_the_files_and_wraps_q_is_done_eof_is_not():
    data = sample()
    screens = review.build_reviews(data, pending=[CV])
    outcome, out, _ = run(screens, data, "\n\n\nq\n")
    assert outcome.done and out.count(f"File 1/2   {CV}") == 2 and out.count(f"File 2/2   {LETTER}") == 2
    outcome, _, _ = run(screens, data, "")
    assert outcome == review.ReviewOutcome(False, 0)


def test_invalid_input_changes_nothing():
    data = sample()
    _, out, saves = run(review.build_reviews(data), data, "z\n9\n0-1\nq\n")
    assert out.count("Please press Enter") == 3 and saves == []


def test_no_screens_is_done():
    assert review.review([], dec.DecisionFile()) == review.ReviewOutcome(True, 0)


def test_ansi_only_on_a_tty(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    data = sample()
    screens = review.build_reviews(data, selection=CV)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stdout = Tty()
    review.review(screens, data, stdin=io.StringIO("q\n"), stdout=stdout, save=lambda d: None)
    assert "\x1b[7m" in stdout.getvalue()
    monkeypatch.setenv("NO_COLOR", "1")
    stdout = Tty()
    review.review(screens, data, stdin=io.StringIO("q\n"), stdout=stdout, save=lambda d: None)
    assert "\x1b[7m" not in stdout.getvalue()
