"""
privacy.scan on the tiny build corpus (fake detector, no spaCy models): only
new / changed files are detected, every value is stored with its default
state, pending files, detector-setup re-scan, deleted files, vanished public
values, no values in the record (privacy plan v10, P-D37 – P-D39, P-D48;
tests 14.10).
"""
from __future__ import annotations

import json
import re

import pytest

from hypeonoath.core import config
from hypeonoath.indexing import build_index
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy import detector, scan

from tests.indexing.test_build_index import EMAIL, FAKE_KEY, FAKE_PII, MARKER, add_document, corpus  # noqa: F401

NEW_KEY = "AKIAJ4EXAMPLE7ABCDEF"
PERSON = "Bob Example"


@pytest.fixture
def calls(corpus, monkeypatch):
    """Fake detector that also knows a PERSON; records which files it was asked to read."""
    seen: list[list[str]] = []

    def fake(documents, code_blocks, projects):
        seen.append(sorted({d.source_document for d in documents} | {b.source_document for b in code_blocks}))
        found = []
        for item in detector._block_texts(documents, code_blocks):
            hits = [detector._Hit(m.start(), m.end(), etype, 0.9, "Fake")
                    for value, etype in {**FAKE_PII, PERSON: "PERSON"}.items()
                    for m in re.finditer(re.escape(value), item.text)]
            hits += detector._secret_hits(item.text, item.source)
            found += detector._to_candidates(item.text, hits, item.source, item.first_line)
        return found

    monkeypatch.setattr(scan, "_detect", fake)
    return seen


def run_scan(**kwargs):
    return scan.run(build_index.load_corpus(), **kwargs)


def by_type(entity_type):
    return [d for d in dec.load().decisions if d.entity_type == entity_type]


def test_first_scan_stores_every_value_with_its_default_and_marks_pending(calls):
    result = run_scan()
    assert result.new_values == 3 and result.pending == ["corpus/code/fake_secret.py", "corpus/documents/notes.md"]
    states = {d.entity_type: (d.decision, d.source) for d in dec.load().decisions}
    assert states == {"EMAIL_ADDRESS": ("keep", "detected"), "US_PASSPORT": ("keep", "detected"),
                      "SECRET:AWS Access Key": ("private", "default_secret")}
    record = config.PRIVACY_SCAN_STATE_PATH.read_text(encoding="utf-8")
    assert all(v not in record for v in (EMAIL, MARKER, FAKE_KEY))        # paths + hashes only
    assert "2 file(s) pending review" in result.summary()


def test_unchanged_corpus_detects_nothing(calls):
    run_scan()
    calls.clear()
    result = run_scan()
    assert result.scanned == [] and result.new_values == 0 and calls == []
    assert scan.blockers() == (["corpus/code/fake_secret.py", "corpus/documents/notes.md"], [])


def test_only_the_changed_file_is_detected(calls):
    run_scan()
    scan.mark_reviewed()
    calls.clear()
    notes = config.DOCUMENTS_DIR / "notes.md"
    notes.write_text(notes.read_text(encoding="utf-8") + f"\nkey = {NEW_KEY}\n", encoding="utf-8")
    assert scan.blockers() == ([], ["corpus/documents/notes.md"])
    result = run_scan()
    assert calls == [["corpus/documents/notes.md"]] and result.scanned == ["corpus/documents/notes.md"]
    assert result.new_values == 1 and result.pending == ["corpus/documents/notes.md"]


def test_known_value_in_a_new_file_is_not_pending(calls, corpus):
    run_scan()
    scan.mark_reviewed()
    add_document(corpus, "more.md", f"# More\n\nAgain: {EMAIL}\n")
    result = run_scan()
    assert result.scanned == ["corpus/documents/more.md"] and result.new_values == 0 and result.pending == []
    (email,) = by_type("EMAIL_ADDRESS")
    assert sorted(email.files()) == ["corpus/documents/more.md", "corpus/documents/notes.md"]


def test_detector_setup_change_and_full_scan_rescan_everything(calls, monkeypatch):
    first = run_scan()
    monkeypatch.setattr(scan, "setup_hash", lambda: "another-setup")
    assert sorted(scan.blockers()[1]) == sorted(first.scanned)
    assert run_scan().scanned == first.scanned                       # setup changed -> every file
    assert run_scan().scanned == []
    assert run_scan(full=True).scanned == first.scanned


def test_deleted_file_drops_its_record_public_values_and_keeps_private_ones(calls, corpus):
    add_document(corpus, "more.md", f"# More\n\n{PERSON} key {NEW_KEY}\n")
    run_scan()
    assert by_type("PERSON") and len(by_type("SECRET:AWS Access Key")) == 2
    (config.DOCUMENTS_DIR / "more.md").unlink()
    manifest = json.loads((corpus / "manifest.json").read_text())
    del manifest["more.md"]
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    assert "corpus/documents/more.md" in scan.blockers()[1]          # a gone file needs a scan too
    result = run_scan()
    assert result.deleted == ["corpus/documents/more.md"]
    assert "corpus/documents/more.md" not in scan.load_state().files
    assert by_type("PERSON") == []                                    # public, no location left -> dropped
    assert len(by_type("SECRET:AWS Access Key")) == 2                 # private -> always kept


def test_public_value_removed_from_a_file_loses_that_location(calls):
    run_scan()
    notes = config.DOCUMENTS_DIR / "notes.md"
    notes.write_text(notes.read_text(encoding="utf-8").replace(EMAIL, "someone"), encoding="utf-8")
    run_scan()
    assert by_type("EMAIL_ADDRESS") == [] and len(by_type("US_PASSPORT")) == 1


def test_mark_reviewed_clears_only_the_given_files(calls):
    run_scan()
    scan.mark_reviewed(["corpus/documents/notes.md"])
    assert scan.load_state().pending() == ["corpus/code/fake_secret.py"]
    scan.mark_reviewed()
    assert scan.load_state().pending() == []
