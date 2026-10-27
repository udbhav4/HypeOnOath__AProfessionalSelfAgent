"""
build_index.py end to end on a tiny isolated corpus (real tokenizer, fake
fast embeddings, fake detector -- no spaCy models needed): run order,
stop-on-missing-project, refusal while files are pending review or not
scanned (exit 3, --force), no detection in the build, redaction, locations
from the redaction matches, verify = find-and-fix, fresh-folder purge,
rebuild from raw when a value goes public, the build lock, leak safety
(privacy plan v10, 4.2, 6.7.1, 7.1; tests 14.1, 14.7, 14.10).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time

import pytest
import yaml

from hypeonoath.core import config, embedder
from hypeonoath.core import private_values as pv
from hypeonoath.core import registry
from hypeonoath.indexing import build_index, indexer
from hypeonoath.ingestion import config as ingestion_config
from hypeonoath.ingestion import router
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy import detector, redactor, scan
from hypeonoath.serving.guard import guard_answer

from tests.conftest import FIXTURES

PROJECT = {"demo": {"name": "Demo", "description": "Demo project."}}
EMAIL = "jane.doe@example.com"
MARKER = "Testvalue-918273645"            # plan 14.1: a known fake value
FAKE_KEY = "AKIAIOSFODNN7EXAMPLE"
FAKE_PII = {EMAIL: "EMAIL_ADDRESS", MARKER: "US_PASSPORT"}
NOTES = f"# Notes\n\nReach me at {EMAIL}.\n\nPassport {MARKER} is mine.\n"


def fake_detect(documents, code_blocks, projects):
    """Literal 'recognizers' for the fake PII + the real detect-secrets, through the real Candidate plumbing."""
    found = []
    for item in detector._block_texts(documents, code_blocks):
        hits = [detector._Hit(m.start(), m.end(), etype, 0.9, "FakeRecognizer")
                for value, etype in FAKE_PII.items() for m in re.finditer(re.escape(value), item.text)]
        hits += detector._secret_hits(item.text, item.source)
        found += detector._to_candidates(item.text, hits, item.source, item.first_line)
    return found


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    code_dir, docs_dir, raw_dir = corpus_dir / "code", corpus_dir / "documents", corpus_dir / "raw"
    for folder in (code_dir, docs_dir, raw_dir):
        folder.mkdir(parents=True)
    shutil.copy(FIXTURES / "fake_secret.py", code_dir / "fake_secret.py")
    shutil.copy(FIXTURES / "small_functions.py", code_dir / "small_functions.py")
    (raw_dir / "notes.md").write_text(NOTES, encoding="utf-8")                       # the untouched original
    (docs_dir / "notes.md").write_text("---\ndate: 01.01.2026\n---\n\n" + NOTES, encoding="utf-8")
    meta = {rel: {"content_type": "code", "date": "01.01.2026", "project_key": "demo",
                  "project_name": "Demo", "project_description": "Demo project."}
            for rel in ("fake_secret.py", "small_functions.py")}
    (code_dir / "_metadata.yaml").write_text(yaml.safe_dump(meta))
    manifest = {"notes.md": {"destination_category": "document", "converted_output": None}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(ingestion_config, "CODE_PROJECTS", PROJECT)
    monkeypatch.setattr(ingestion_config, "CODE_PROJECT_MAP", {"fake_secret.py": "demo", "small_functions.py": "demo"})
    monkeypatch.setattr(registry, "DOCUMENTS", {"notes.md": {"description": "Test notes.", "project": None}})
    for name, value in {"CORPUS_DIR": corpus_dir, "RAW_DIR": raw_dir, "DOCUMENTS_DIR": docs_dir,
                        "CODE_DIR": code_dir, "CONVERTED_DIR": corpus_dir / "converted",
                        "MANIFEST_PATH": tmp_path / "manifest.json"}.items():
        monkeypatch.setattr(ingestion_config, name, value)
    for name in ("ROUTING_LOG_PATH", "CONVERSION_LOG_PATH", "CLEANING_LOG_PATH", "TAGGING_LOG_PATH"):
        monkeypatch.setattr(ingestion_config, name, tmp_path / "logs" / f"{name.lower()}.jsonl")
    monkeypatch.setattr(router, "_CATEGORY_DIRS", {"code": code_dir, "document": docs_dir,
                                                   "quarantine": corpus_dir / "quarantine"})
    for name, value in {"PROJECT_ROOT": tmp_path, "CORPUS_DIR": corpus_dir, "CODE_DIR": code_dir,
                        "DOCUMENTS_DIR": docs_dir, "CONVERTED_DIR": corpus_dir / "converted",
                        "CODE_METADATA_PATH": code_dir / "_metadata.yaml",
                        "CHROMA_DIR": tmp_path / "index" / "chroma",
                        "SCAN_ALLOWLIST_PATH": tmp_path / "allow.yaml"}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr("hypeonoath.indexing.code_chunker.config.CODE_DIR", code_dir)
    monkeypatch.setattr(embedder, "embed_documents", lambda texts: [[float(len(t) % 7), 1.0, 0.5] for t in texts])
    monkeypatch.setattr(scan, "_detect", fake_detect)
    return tmp_path


def scan_and_done():
    """What `hypeonoath-privacy review` does before the build: scan, then Done without switches."""
    result = scan.run(build_index.load_corpus())
    scan.mark_reviewed()
    return result


def decide_all(email="private", marker="private", secret="keep"):
    """Scan, then decide every fake finding as the owner would in review, then Done."""
    scan_and_done()
    choices = {"EMAIL_ADDRESS": (email, "email address"), "US_PASSPORT": (marker, "passport number"),
               "SECRET:AWS Access Key": (secret, "secret")}
    data = dec.load()
    for decision in data.decisions:
        choice, label = choices[decision.entity_type]
        if choice == "private":
            dec.set_private(decision, label, data)
        else:
            dec.set_public(decision, data)
    dec.save(data)
    return data


def run():
    return build_index.main([])


def collection():
    return indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME)


def ids():
    return sorted(collection().get()["ids"])


def texts():
    return "\n".join(collection().get()["documents"])


def add_document(corpus, name, text):
    (corpus / "corpus" / "documents" / name).write_text(text, encoding="utf-8")
    manifest = json.loads((corpus / "manifest.json").read_text())
    manifest[name] = {"destination_category": "document", "converted_output": None}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    registry.DOCUMENTS[name] = {"description": f"Test document {name}.", "project": None}  # fixture's dict


# --- review required (P-D40, P-D42) ---------------------------------------------------------------------

def test_unscanned_files_refuse_the_build(corpus, capsys):
    assert run() == build_index.EXIT_NOT_REVIEWED
    err = capsys.readouterr().err
    assert "Privacy review needed" in err and "not scanned: corpus/documents/notes.md" in err
    assert "hypeonoath-privacy review" in err and not config.CHROMA_DIR.exists()


def test_pending_files_refuse_the_build_until_done(corpus, capsys):
    result = scan.run(build_index.load_corpus())
    assert result.pending == ["corpus/code/fake_secret.py", "corpus/documents/notes.md"]
    assert run() == build_index.EXIT_NOT_REVIEWED
    assert "pending:     corpus/documents/notes.md" in capsys.readouterr().err
    scan.mark_reviewed()
    assert run() == 0
    assert EMAIL in texts() and MARKER in texts()                    # values: public by default
    assert FAKE_KEY not in texts() and "[REDACTED: secret]" in texts()   # secret: private by default


def test_force_builds_with_the_defaults_and_leaves_files_pending(corpus, capsys):
    assert build_index.main(["--force"]) == 0
    out = capsys.readouterr().out
    assert FAKE_KEY not in texts() and EMAIL in texts()
    assert "--force: 2 file(s) still pending review" in out
    record = json.loads(config.PRIVACY_LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "build" and record["forced"] is True and record["pending_files"] == 2


def test_build_never_detects(corpus, monkeypatch):
    decide_all()

    def must_not_run(*a, **k):
        raise AssertionError("the build must not detect (P-D37)")

    monkeypatch.setattr(scan, "_detect", must_not_run)
    assert run() == 0
    assert run() == 0                                             # its own disk redaction is not a change


def test_build_takes_no_review_flags(corpus):
    with pytest.raises(SystemExit):
        build_index.main(["--no-review"])


def test_earlier_private_decision_applies_to_a_new_file(corpus):
    decide_all()
    add_document(corpus, "more.md", f"# More\n\nAgain: {EMAIL}\n")
    assert run() == build_index.EXIT_NOT_REVIEWED                 # the new file is not scanned yet
    assert scan_and_done().pending == []                          # only a known value: nothing new to review
    assert run() == 0
    assert EMAIL not in texts() and texts().count("[REDACTED: email address]") == 2
    email = next(d for d in dec.load().decisions if d.entity_type == "EMAIL_ADDRESS")
    assert sorted(email.files()) == ["corpus/documents/more.md", "corpus/documents/notes.md"]


# --- decided corpus ----------------------------------------------------------------------------------------

def test_all_decided_builds_with_placeholders_and_fingerprint(corpus):
    data = decide_all()
    assert run() == 0
    got = collection().get(include=["documents", "metadatas"])
    joined = "\n".join(got["documents"])
    assert "[REDACTED: email address]" in joined and "[REDACTED: passport number]" in joined
    assert EMAIL not in joined and MARKER not in joined
    assert FAKE_KEY in joined                                     # the owner KEPT the secret explicitly
    assert collection().metadata["privacy_fp"] == dec.privacy_fingerprint(data.decisions) != "none"
    assert {"code", "prose", "project_summary"} <= {m["content_type"] for m in got["metadatas"]}
    notes = (config.DOCUMENTS_DIR / "notes.md").read_text(encoding="utf-8")
    assert EMAIL not in notes and "[REDACTED: email address]" in notes and notes.startswith("---\ndate: 01.01.2026")


def test_summary_counts_and_labels_only(corpus, capsys):
    decide_all()
    assert run() == 0
    out = capsys.readouterr().out
    assert "fake_secret.py:2: AWS Access Key" in out                  # kept secret listed in the scan report
    assert "Privacy: 2 private value occurrence(s) redacted (labels: email address ×1, passport number ×1)" in out
    assert "redacted 1 corpus file(s) on disk" in out
    assert out.rstrip().splitlines()[-1].startswith("Privacy:")
    for value in (EMAIL, MARKER, FAKE_KEY):
        assert value not in out


def test_no_private_decisions_records_privacy_fp_none(corpus):
    decide_all(email="keep", marker="keep", secret="keep")
    assert run() == 0
    assert collection().metadata["privacy_fp"] == "none"


# --- verification = find-and-fix (P-D46) + locations from redaction (P-D49) ----------------------------------

def test_verification_fixes_a_redaction_hole(corpus, monkeypatch, capsys):
    decide_all()
    assert run() == 0
    monkeypatch.setattr(redactor, "redact_blocks", lambda *a: redactor.RedactionStats())   # a redaction hole
    monkeypatch.setattr(redactor, "redact_files_on_disk", lambda values: [])
    (config.DOCUMENTS_DIR / "notes.md").write_text(f"# Notes\n\nmail {EMAIL}\n", encoding="utf-8")
    scan_and_done()
    assert run() == 0
    out = capsys.readouterr().out
    assert "verify fixed 1 occurrence(s) the chunker re-joined (labels: email address ×1)" in out
    assert EMAIL not in texts() and "[REDACTED: email address]" in texts() and EMAIL not in out


def test_newly_private_value_gets_its_locations_from_the_redaction(corpus):
    data = decide_all()
    (group,) = dec.group_candidates([detector.Candidate("x", None, "MANUAL", "manual", 1.0, "is mine", "", 0)])
    manual = dec.set_private(group, "private detail", data, source="manual")
    manual.locations = []                                         # known nowhere yet
    dec.save(data)
    assert run() == 0
    stored = dec.load().by_id(manual.id)
    assert stored.locations == [{"file": "corpus/documents/notes.md", "line": 9}]


# --- fresh-folder purge (P-D12) + rebuild from raw (P-D33) -------------------------------------------------

def test_changed_decisions_rebuild_into_a_fresh_folder(corpus):
    decide_all()
    assert run() == 0
    marker_file = config.CHROMA_DIR / "old-index-marker"
    marker_file.write_text("stale")
    assert run() == 0                                             # unchanged -> same folder
    assert marker_file.exists()
    data = dec.load()
    dec.set_private(next(d for d in data.decisions if d.entity_type.startswith("SECRET:")), "secret", data)
    dec.save(data)
    assert run() == 0
    assert not marker_file.exists()                               # old folder deleted
    assert sorted(p.name for p in config.CHROMA_DIR.parent.iterdir()) == ["chroma"]
    assert FAKE_KEY not in texts()


def test_private_to_public_rebuilds_the_document_from_raw(corpus, capsys):
    decide_all()
    assert run() == 0
    notes = config.DOCUMENTS_DIR / "notes.md"
    assert EMAIL not in notes.read_text(encoding="utf-8")         # redacted on disk
    data = dec.load()
    email = next(d for d in data.decisions if d.entity_type == "EMAIL_ADDRESS")
    assert dec.set_public(email, data) == ["notes.md"]            # queued via the manifest
    dec.save(data)

    assert run() == 0
    out = capsys.readouterr().out
    on_disk = notes.read_text(encoding="utf-8")
    assert EMAIL in on_disk and MARKER not in on_disk             # value back, the other still redacted
    assert EMAIL in texts() and "[REDACTED: passport number]" in texts()
    assert dec.load().pending_restore == [] and "restored 1 document(s) from raw" in out
    assert (corpus / "corpus" / "raw" / "notes.md").read_text(encoding="utf-8") == NOTES   # raw untouched


def test_restore_failure_exits_with_queue_kept_and_index_untouched(corpus, monkeypatch):
    decide_all()
    assert run() == 0
    before = ids()
    data = dec.load()
    dec.set_public(next(d for d in data.decisions if d.entity_type == "EMAIL_ADDRESS"), data)
    dec.save(data)

    def broken(*a, **k):
        raise RuntimeError("conversion crashed")

    monkeypatch.setattr("hypeonoath.ingestion.cleaner.run_cleaning", broken)
    with pytest.raises(RuntimeError):
        run()
    assert dec.load().pending_restore == ["notes.md"] and ids() == before


def test_rerun_produces_identical_chunk_ids(corpus):
    decide_all()
    assert run() == 0
    first = ids()
    assert run() == 0
    assert ids() == first


# --- unchanged behaviour ------------------------------------------------------------------------------

def test_missing_project_stops_before_anything_and_keeps_index(corpus, capsys, monkeypatch):
    decide_all()
    assert run() == 0
    before = ids()
    shutil.copy(FIXTURES / "query.sql", config.CODE_DIR / "q1.sql")
    shutil.copy(FIXTURES / "query.sql", config.CODE_DIR / "q2.sql")

    def must_not_run(*a, **k):
        raise AssertionError("restore/loading/detection must not start when the project check fails")

    monkeypatch.setattr(build_index, "_restore_from_raw", must_not_run)
    monkeypatch.setattr(build_index, "load_documents", must_not_run)
    monkeypatch.setattr(scan, "_detect", must_not_run)
    assert run() == 2
    err = capsys.readouterr().err
    assert "q1.sql" in err and "q2.sql" in err
    assert ids() == before


def test_document_without_registry_entry_stops_the_build_and_keeps_index(corpus, capsys):
    """Document-registry plan v2, DR-9 / T-7: every indexed document needs a description."""
    decide_all()
    assert run() == 0
    before = ids()
    add_document(corpus, "extra.md", "# Extra\n\nMore text.\n")
    scan_and_done()
    del registry.DOCUMENTS["extra.md"]
    assert run() == build_index.EXIT_PROJECT_CHECK
    err = capsys.readouterr().err
    assert "document-registry problem" in err and "corpus/documents/extra.md: no entry" in err
    assert ids() == before


def test_embedding_failure_keeps_previous_index(corpus, monkeypatch):
    decide_all()
    assert run() == 0
    before = ids()

    def boom(texts):
        raise RuntimeError("embedding crashed")

    monkeypatch.setattr(embedder, "embed_documents", boom)
    with pytest.raises(RuntimeError):
        run()
    assert ids() == before


# --- one build at a time (P-D47) -------------------------------------------------------------------------

def test_second_build_waits_for_the_lock(corpus, capsys):
    order = []
    holding, release = threading.Event(), threading.Event()

    def first():
        with build_index.build_lock():
            order.append("first in")
            holding.set()
            release.wait(5)
            order.append("first out")

    thread = threading.Thread(target=first)
    thread.start()
    holding.wait(5)
    def second_build():
        with build_index.build_lock(poll_seconds=0.01):
            order.append("second in")

    second = threading.Thread(target=second_build)
    second.start()
    time.sleep(0.2)
    assert order == ["first in"]                                  # still waiting
    release.set()
    thread.join(5)
    second.join(5)
    assert order == ["first in", "first out", "second in"]
    assert "Another index build is running" in capsys.readouterr().err


# --- leak safety (14.1) ----------------------------------------------------------------------------------

def test_known_fake_value_appears_only_in_decisions_json(corpus, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    decide_all()
    assert run() == 0
    values = pv.load_private_values()
    guard_answer(f"The passport is {MARKER}.", values)
    guard_answer(f"Lots of text about the candidate and projects. {MARKER}", values)

    out = capsys.readouterr()
    assert MARKER not in out.out and MARKER not in out.err
    assert all(MARKER not in record.getMessage() for record in caplog.records)
    holders = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*")
                     if p.is_file() and MARKER.encode() in p.read_bytes())
    # decisions.json and the untouched raw original (corpus/raw/ is never modified, P-D11)
    assert holders == ["corpus/raw/notes.md", config.PRIVACY_DECISIONS_PATH.relative_to(corpus).as_posix()]
