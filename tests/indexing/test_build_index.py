"""
build_index.py end to end on a tiny isolated corpus (real tokenizer,
fake fast embeddings): run order, stop-on-missing-project, warn-only scan.
"""
from __future__ import annotations

import shutil

import pytest
import yaml

from hypeonoath.core import config, embedder
from hypeonoath.indexing import indexer, build_index
from hypeonoath.ingestion import config as ingestion_config

from tests.conftest import FIXTURES

PROJECT = {"demo": {"name": "Demo", "description": "Demo project."}}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    code_dir = tmp_path / "corpus" / "code"
    docs_dir = tmp_path / "corpus" / "documents"
    code_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "fake_secret.py", code_dir / "fake_secret.py")
    shutil.copy(FIXTURES / "small_functions.py", code_dir / "small_functions.py")
    (docs_dir / "notes.md").write_text("---\ndate: 01.01.2026\n---\n\n# Notes\n\nReach me at jane.doe@example.com.\n")
    meta = {rel: {"content_type": "code", "date": "01.01.2026", "project_key": "demo",
                  "project_name": "Demo", "project_description": "Demo project."}
            for rel in ("fake_secret.py", "small_functions.py")}
    (code_dir / "_metadata.yaml").write_text(yaml.safe_dump(meta))
    manifest = {"notes.md": {"destination_category": "document", "converted_output": None}}

    monkeypatch.setattr(ingestion_config, "CODE_PROJECTS", PROJECT)
    monkeypatch.setattr(ingestion_config, "CODE_PROJECT_MAP", {"fake_secret.py": "demo", "small_functions.py": "demo"})
    monkeypatch.setattr(ingestion_config, "DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(ingestion_config, "MANIFEST_PATH", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest))
    for name, value in {"PROJECT_ROOT": tmp_path, "CODE_DIR": code_dir, "CODE_METADATA_PATH": code_dir / "_metadata.yaml",
                        "CHROMA_DIR": tmp_path / "chroma", "SCAN_ALLOWLIST_PATH": tmp_path / "allow.yaml"}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr("hypeonoath.indexing.code_chunker.config.CODE_DIR", code_dir)
    monkeypatch.setattr(embedder, "embed_documents", lambda texts: [[float(len(t) % 7), 1.0, 0.5] for t in texts])
    return tmp_path


def run():
    return build_index.main()


def test_full_run_warns_but_completes(corpus, capsys):
    assert run() == 0
    out = capsys.readouterr().out
    collection = indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME)
    metas = collection.get(include=["metadatas"])["metadatas"]
    kinds = {m["content_type"] for m in metas}
    assert {"code", "prose", "project_summary"} <= kinds
    assert all(m.get("project_key") and m.get("project_description") for m in metas if m["content_type"] == "code")
    # warnings printed LAST, values never shown
    assert out.rstrip().endswith("=" * 72)
    assert "fake_secret.py:2: AWS Access Key" in out and "email_address" in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out and "jane.doe@example.com" not in out
    report = config.SCAN_WARNINGS_PATH.read_text()
    assert "AKIA" not in report and '"finding_type": "AWS Access Key"' in report


def test_rerun_produces_identical_chunk_ids(corpus):
    assert run() == 0
    first = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])
    assert run() == 0
    second = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])
    assert first == second


def test_missing_project_stops_before_anything_and_keeps_index(corpus, capsys, monkeypatch):
    assert run() == 0
    before = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])
    shutil.copy(FIXTURES / "query.sql", config.CODE_DIR / "q1.sql")
    shutil.copy(FIXTURES / "query.sql", config.CODE_DIR / "q2.sql")

    def must_not_run(*a, **k):
        raise AssertionError("chunking/embedding must not start when the project check fails")

    monkeypatch.setattr(build_index, "load_documents", must_not_run)
    assert run() == 2
    err = capsys.readouterr().err
    assert "q1.sql" in err and "q2.sql" in err
    after = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])
    assert after == before


def test_embedding_failure_keeps_previous_index(corpus, monkeypatch):
    assert run() == 0
    before = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])

    def boom(texts):
        raise RuntimeError("embedding crashed")

    monkeypatch.setattr(embedder, "embed_documents", boom)
    with pytest.raises(RuntimeError):
        run()
    after = sorted(indexer.get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).get()["ids"])
    assert after == before
