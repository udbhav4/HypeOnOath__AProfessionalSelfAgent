"""Document registry: lookup by stem, project fields, all-failures validation, privacy check, and the real
registry against the real corpus (document-registry plan v2, 3.2 / 6, T-1 - T-5, T-10)."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core import config, document_registry, registry
from hypeonoath.ingestion import config as ingestion_config

FAKE_EMAIL = "jane.doe@mail.example"
HYPE_WORDS = ("expert", "best", "world-class", "production-grade", "led")


def write_decisions(*entries):
    config.PRIVACY_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PRIVACY_DECISIONS_PATH.write_text(json.dumps({"schema_version": 2, "entries": list(entries)}))


# --- T-1 / T-2 lookup ------------------------------------------------------------------------------------

def test_document_key_maps_indexed_paths_by_stem(doc_registry):
    assert document_registry.document_key("corpus/converted/cv.md") == "cv.pdf"
    assert document_registry.document_key("corpus/documents/design.md") == "design.md"
    assert document_registry.document_key("corpus/code/design.md") is None          # code is not a document
    assert document_registry.document_key("corpus/code/_projects/chat") is None    # nor a project summary
    assert document_registry.document_key("corpus/documents/unknown.md") is None


def test_document_info_with_and_without_project(doc_registry):
    info = document_registry.document_info("corpus/documents/design.md")
    assert (info.description, info.project_key, info.project_name, info.project_description) == (
        "Design document of the chatbot.", "chat", "Chatbot", "A chatbot project.")
    cv = document_registry.document_info("corpus/converted/cv.md")
    assert cv.description == "My CV." and cv.project_key is None and cv.project_name is None
    assert document_registry.document_info("corpus/documents/unknown.md") is None


# --- T-3 / T-4 validation --------------------------------------------------------------------------------

def test_validate_documents_returns_every_failure_at_once(doc_registry):
    doc_registry.DOCUMENTS.update({
        "empty.md": {"description": "  ", "project": None},
        "long.md": {"description": "x" * 301, "project": None},
        "orphan.md": {"description": "Fine.", "project": "nope"},
        "cv.txt": {"description": "Same stem as cv.pdf.", "project": None},
    })
    failures = document_registry.validate_documents([
        "corpus/documents/design.md", "corpus/documents/empty.md", "corpus/documents/long.md",
        "corpus/documents/orphan.md", "corpus/documents/missing.md"])
    text = "\n".join(failures)
    assert len(failures) == 5, text
    for needle in ("corpus/documents/missing.md: no entry", "'empty.md']: empty description",
                   "'long.md']: description over 300", "project 'nope' is not defined", "same file stem"):
        assert needle in text, needle


def test_valid_documents_pass(doc_registry):
    assert document_registry.validate_documents(["corpus/documents/design.md", "corpus/converted/cv.md"]) == []


def test_private_value_in_description_is_named_but_never_printed(doc_registry):
    write_decisions({"id": "pv_mail", "decision": "private", "label": "email address", "entity_type": "EMAIL_ADDRESS",
                     "match_kind": "email", "value": FAKE_EMAIL, "normalized": FAKE_EMAIL, "aliases": [],
                     "surface_forms": [FAKE_EMAIL]})
    doc_registry.DOCUMENTS["cv.pdf"]["description"] = f"My CV, reach me at {FAKE_EMAIL}."
    doc_registry.CODE_PROJECTS["chat"]["description"] = f"Built by {FAKE_EMAIL}."
    failures = document_registry.validate_documents(["corpus/converted/cv.md", "corpus/documents/design.md"])
    assert failures == ["DOCUMENTS['cv.pdf']: description contains a private value",
                        "DOCUMENTS['design.md']: project description contains a private value"]
    assert FAKE_EMAIL not in "\n".join(failures)


# --- T-5 / T-10 the real registry ------------------------------------------------------------------------

def test_real_registry_entries_are_short_factual_and_linked():
    assert ingestion_config.DOCUMENTS is registry.DOCUMENTS                     # one registry, re-exported
    assert ingestion_config.CODE_PROJECTS is registry.CODE_PROJECTS
    assert registry.CODE_PROJECTS["hypeonoath"]["name"] == "HypeOnOath: A Privacy-Preserving Professional RAG Agent"
    for key, entry in registry.DOCUMENTS.items():
        description = entry["description"]
        assert description.strip() and len(description) <= document_registry.DESCRIPTION_MAX_CHARS, key
        assert entry["project"] is None or entry["project"] in registry.CODE_PROJECTS, key
        words = description.lower().replace(",", " ").replace(".", " ").split()
        assert not [w for w in HYPE_WORDS if w in words], key
    assert registry.DOCUMENTS["system-design-rag-chatbot.md"]["project"] == "hypeonoath"
    assert registry.DOCUMENTS["README.md"]["project"] == "gliimr"
    assert all(e["project"] is None for k, e in registry.DOCUMENTS.items()
               if k not in ("system-design-rag-chatbot.md", "README.md"))


def test_real_corpus_documents_all_have_a_valid_entry(monkeypatch):
    """Every non-code raw file has an entry; the real check passes, private values included (read-only;
    failure messages never hold a value)."""
    if not ingestion_config.RAW_DIR.exists():
        pytest.skip("corpus/raw not present")
    raw = sorted(p.name for p in ingestion_config.RAW_DIR.iterdir()
                 if p.is_file() and p.suffix.lower() in ingestion_config.DOCUMENT_EXTENSIONS)
    assert raw and set(raw) <= set(registry.DOCUMENTS), sorted(set(raw) - set(registry.DOCUMENTS))
    real_decisions = config.CORPUS_DIR / ".privacy" / "decisions.json"
    if real_decisions.exists():
        monkeypatch.setattr(config, "PRIVACY_DECISIONS_PATH", real_decisions)
    indexed = [f"corpus/converted/{name.rsplit('.', 1)[0]}.md" for name in raw]
    assert document_registry.validate_documents(indexed) == []
