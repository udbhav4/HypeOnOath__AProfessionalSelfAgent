"""privacy.restore: derived file -> raw source mapping and the rebuild queue (plan v8, 6.7.1; P-D33)."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core import config
from hypeonoath.ingestion import config as ingestion_config
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy import restore


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    for name in ("raw", "documents", "converted"):
        (corpus_dir / name).mkdir(parents=True)
    manifest = {
        "CV.pdf": {"destination_category": "document", "converted_output": "CV.md"},
        "notes.md": {"destination_category": "document", "converted_output": None},
        "Failed.pdf": {"destination_category": "document", "converted_output": None},
        "agent.ts": {"destination_category": "code"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (corpus_dir / "documents" / "notes.md").write_text("x")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for name, value in {"MANIFEST_PATH": tmp_path / "manifest.json", "CONVERTED_DIR": corpus_dir / "converted",
                        "DOCUMENTS_DIR": corpus_dir / "documents", "RAW_DIR": corpus_dir / "raw"}.items():
        monkeypatch.setattr(ingestion_config, name, value)
    return tmp_path


def test_raw_sources_for_maps_converted_and_native_documents(corpus):
    files = ["corpus/converted/CV.md", "corpus/documents/notes.md", "corpus/code/agent.ts", "other.md"]
    assert restore.raw_sources_for(files) == ["CV.pdf", "notes.md"]
    assert restore.raw_sources_for([]) == []


def test_restore_pending_drops_vanished_raw_files(corpus, isolated_logs):
    data = dec.DecisionFile([], ["Gone.pdf"])
    assert restore.restore_pending(data) == [] and data.pending_restore == []


def test_restore_pending_nothing_queued(corpus):
    assert restore.restore_pending(dec.DecisionFile()) == []
