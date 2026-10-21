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


# --- privacy plan v3, 7.3: privacy_fp + fresh-directory rebuild ----------------------------------

def _metadata(chroma_dir, name="test"):
    return indexer.get_client(chroma_dir).get_collection(name).metadata


def test_collection_metadata_always_has_privacy_fp(tmp_path):
    tmp_path = tmp_path / "idx"
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], tmp_path / "chroma", "test")
    assert _metadata(tmp_path / "chroma")["privacy_fp"] == "none"
    assert indexer.current_privacy_fp(tmp_path / "chroma", "test") == "none"
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], tmp_path / "chroma", "test", privacy_fp="abc123")
    assert indexer.current_privacy_fp(tmp_path / "chroma", "test") == "abc123"


def test_current_privacy_fp_without_index(tmp_path):
    tmp_path = tmp_path / "idx"
    assert indexer.current_privacy_fp(tmp_path / "missing", "test") is None
    indexer.get_client(tmp_path / "empty")
    assert indexer.current_privacy_fp(tmp_path / "empty", "test") is None


def test_fresh_dir_replaces_the_folder_even_with_open_clients(tmp_path):
    tmp_path = tmp_path / "idx"
    chroma = tmp_path / "chroma"
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], chroma, "test")
    old_files = {p.name for p in chroma.rglob("*")}
    held = indexer.get_client(chroma)                           # e.g. a reader elsewhere in the process
    held.get_collection("test").count()
    indexer.rebuild_index([code_chunk()], [[0.0, 1.0]], chroma, "test", privacy_fp="fp1", fresh_dir=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chroma"]          # no __new / __old left
    assert indexer.get_client(chroma).get_collection("test").get()["ids"] == ["c1"]
    assert _metadata(chroma)["privacy_fp"] == "fp1"
    # The old HNSW segment folder is gone with the old directory.
    assert {p.name for p in chroma.iterdir() if p.is_dir()}.isdisjoint(
        {n for n in old_files if "-" in n and len(n) == 36})


def test_fresh_dir_failure_leaves_old_index_and_no_new_folder(tmp_path, monkeypatch):
    tmp_path = tmp_path / "idx"
    chroma = tmp_path / "chroma"
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], chroma, "test")

    def broken(chunk):
        raise TypeError("boom")

    monkeypatch.setattr(indexer, "flatten_metadata", broken)
    with pytest.raises(TypeError):
        indexer.rebuild_index([code_chunk()], [[0.0, 1.0]], chroma, "test", fresh_dir=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chroma"]
    assert indexer.get_client(chroma).get_collection("test").get()["ids"] == ["d1"]


def test_fresh_dir_first_build_and_leftover_cleanup(tmp_path):
    tmp_path = tmp_path / "idx"
    chroma = tmp_path / "chroma"
    (tmp_path / ("chroma" + indexer.NEW_SUFFIX)).mkdir(parents=True)   # crash leftover from an earlier run
    (tmp_path / ("chroma" + indexer.NEW_SUFFIX) / "junk").write_text("x")
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], chroma, "test", fresh_dir=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chroma"]
    assert indexer.get_client(chroma).get_collection("test").count() == 1


def test_swap_failure_moves_the_old_index_back(tmp_path, monkeypatch):
    tmp_path = tmp_path / "idx"
    chroma = tmp_path / "chroma"
    indexer.rebuild_index([doc_chunk()], [[1.0, 0.0]], chroma, "test")
    real_replace = indexer.os.replace

    def flaky(src, dst):
        if str(src).endswith(indexer.NEW_SUFFIX):
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(indexer.os, "replace", flaky)
    monkeypatch.setattr(indexer.time, "sleep", lambda s: None)
    with pytest.raises(indexer.IndexSwapError, match="rename"):
        indexer.rebuild_index([code_chunk()], [[0.0, 1.0]], chroma, "test", fresh_dir=True)
    monkeypatch.setattr(indexer.os, "replace", real_replace)
    assert indexer.get_client(chroma).get_collection("test").get()["ids"] == ["d1"]   # old index back
    assert (tmp_path / ("chroma" + indexer.NEW_SUFFIX)).exists()                      # kept for the manual step
