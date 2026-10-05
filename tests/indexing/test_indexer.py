"""indexer.rebuild_index: explicit embeddings, staging + delete-last swap,
duplicate / mismatch guards (3A.8, Section 5)."""
from __future__ import annotations

import pytest

from hypeonoath.indexing import indexer

from tests.factories import code_chunk, doc_chunk


def test_rebuild_writes_explicit_embeddings_and_swaps(tmp_path):
    chunks = [code_chunk(), doc_chunk()]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert indexer.rebuild_index(chunks, vectors, tmp_path, "test") == 2
    client = indexer.get_client(tmp_path)
    names = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
    assert names == {"test"}                                   # staging collection gone
    got = client.get_collection("test").get(include=["embeddings", "documents", "metadatas"])
    assert sorted(got["ids"]) == ["c1", "d1"]
    by_id = dict(zip(got["ids"], got["embeddings"]))
    assert list(by_id["c1"]) == [1.0, 0.0, 0.0]               # our vectors, not Chroma's

    # second rebuild fully replaces (no stale chunks)
    indexer.rebuild_index([doc_chunk()], [[0.0, 0.0, 1.0]], tmp_path, "test")
    assert client.get_collection("test").count() == 1


def test_failed_write_keeps_previous_index(tmp_path, monkeypatch):
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], tmp_path, "test")

    def broken(chunk):
        raise TypeError("boom")

    monkeypatch.setattr(indexer, "flatten_metadata", broken)
    with pytest.raises(TypeError):
        indexer.rebuild_index([code_chunk()], [[0.0, 1.0]], tmp_path, "test")
    assert indexer.get_client(tmp_path).get_collection("test").get()["ids"] == ["d1"]


def test_rejects_duplicates_and_mismatch(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        indexer.rebuild_index([doc_chunk(), doc_chunk()], [[1.0], [1.0]], tmp_path, "t")
    with pytest.raises(ValueError, match="vectors"):
        indexer.rebuild_index([doc_chunk()], [], tmp_path, "t")
