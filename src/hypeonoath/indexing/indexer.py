"""
Step 4 -- chunks + vectors -> persisted Chroma collection (guideline v4,
Sections 3A.8 and 5; ADR-5, ADR-16 in the design doc; privacy plan v3, 7.3).

  * Embeddings are ALWAYS passed explicitly -- Chroma's default embedding
    function is a different model and would silently break the Section 2.2
    "same model at index and query time" invariant.
  * Full rebuild every run, in one of two modes:
      same directory (default)  the new collection is written under a
                     temporary name and swapped in only once fully written,
                     so a crash mid-write leaves the previous good index.
                     Keeps the committed index diff-free when nothing changed.
      fresh directory (fresh_dir=True, privacy purge P-D12)  the index is
                     built in "<dir>__new" and swapped for the old folder,
                     which is then deleted. Used when the privacy decisions
                     changed: deleting a collection may leave old text in
                     SQLite pages; a new folder certainly doesn't.
  * The collection metadata records `privacy_fp` (the private-decisions
    fingerprint the index was redacted with, "none" when nothing is private),
    so serving can warn on a mismatch and check-git can tell a privacy-era
    index from an old one (P-D20).
  * The client and the metadata flattening rules live in hypeonoath.core.store,
    shared with the serving side (which restores what is written here).
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Chunk
from hypeonoath.core.store import flatten_metadata, get_client, release_client

logger = logging.getLogger(__name__)

_ADD_BATCH = 256
NEW_SUFFIX = "__new"
OLD_SUFFIX = "__old"


class IndexSwapError(RuntimeError):
    """The fresh index was built but could not replace the old folder (e.g. a file lock)."""


def _collection_metadata(privacy_fp: str) -> dict[str, str]:
    return {"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL_NAME,
            "embed_revision": config.EMBED_MODEL_REVISION or "unpinned", "privacy_fp": privacy_fp}


def _write(collection, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    for start in range(0, len(chunks), _ADD_BATCH):
        batch = chunks[start:start + _ADD_BATCH]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[start:start + _ADD_BATCH],
            documents=[c.text for c in batch],
            metadatas=[flatten_metadata(c) for c in batch],
        )


def current_privacy_fp(chroma_dir: Path | None = None, collection_name: str | None = None) -> str | None:
    """`privacy_fp` of the existing index; "none" for a pre-privacy index; None when there is no index."""
    chroma_dir = config.CHROMA_DIR if chroma_dir is None else chroma_dir
    collection_name = collection_name or config.COLLECTION_NAME
    if not chroma_dir.exists():
        return None
    try:
        collection = get_client(chroma_dir).get_collection(collection_name)
    except Exception:  # noqa: BLE001 -- chromadb raises different types per version
        return None
    return (collection.metadata or {}).get("privacy_fp", "none")


def rebuild_index(
    chunks: list[Chunk],
    vectors: list[list[float]],
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
    *,
    privacy_fp: str = "none",
    fresh_dir: bool = False,
) -> int:
    """Replace the collection with exactly these chunks. Returns the count."""
    chroma_dir = config.CHROMA_DIR if chroma_dir is None else chroma_dir
    collection_name = collection_name or config.COLLECTION_NAME
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    duplicates = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk_ids (would overwrite each other): {duplicates[:5]}")

    if fresh_dir:
        _rebuild_fresh(chunks, vectors, chroma_dir, collection_name, privacy_fp)
    else:
        _rebuild_same_dir(chunks, vectors, chroma_dir, collection_name, privacy_fp)
    log_json_line(config.INDEXING_LOG_PATH, status="index_rebuilt", chunks=len(chunks),
                  content_types=dict(Counter(c.content_type for c in chunks)),
                  privacy_fp=privacy_fp, fresh_dir=fresh_dir)
    logger.info("Indexed %d chunks into %s/%s (fresh_dir=%s)", len(chunks), chroma_dir, collection_name, fresh_dir)
    return len(chunks)


def _rebuild_same_dir(chunks, vectors, chroma_dir: Path, collection_name: str, privacy_fp: str) -> None:
    client = get_client(chroma_dir)
    staging = f"{collection_name}__staging"
    _delete_if_exists(client, staging)
    collection = client.create_collection(
        name=staging,
        embedding_function=None,                    # never let Chroma embed
        metadata=_collection_metadata(privacy_fp),
    )
    _write(collection, chunks, vectors)
    # Swap: only now is the old index removed.
    _delete_if_exists(client, collection_name)
    collection.modify(name=collection_name)


def _rebuild_fresh(chunks, vectors, chroma_dir: Path, collection_name: str, privacy_fp: str) -> None:
    new_dir = chroma_dir.with_name(chroma_dir.name + NEW_SUFFIX)
    old_dir = chroma_dir.with_name(chroma_dir.name + OLD_SUFFIX)
    for leftover in (new_dir, old_dir):           # from an earlier interrupted run
        if leftover.exists():
            release_client(leftover)
            shutil.rmtree(leftover)
    try:
        client = get_client(new_dir)
        collection = client.create_collection(name=collection_name, embedding_function=None,
                                              metadata=_collection_metadata(privacy_fp))
        _write(collection, chunks, vectors)
    except BaseException:
        release_client(new_dir)
        shutil.rmtree(new_dir, ignore_errors=True)   # old folder untouched
        raise
    release_client(new_dir)
    release_client(chroma_dir)
    _swap(new_dir, chroma_dir, old_dir)


def _retry(action, attempts: int = 2, delay: float = 0.5) -> None:
    """Windows can hold a just-closed file for a moment: retry once."""
    for attempt in range(attempts):
        try:
            action()
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _swap(new_dir: Path, target: Path, old_dir: Path) -> None:
    """
    target -> old_dir, new_dir -> target, then delete old_dir. If the second
    rename fails the old folder is moved back, so there is always one index.
    """
    had_old = target.exists()
    try:
        if had_old:
            _retry(lambda: os.replace(target, old_dir))
        try:
            _retry(lambda: os.replace(new_dir, target))
        except OSError:
            if had_old:
                os.replace(old_dir, target)
            raise
    except OSError as exc:
        raise IndexSwapError(
            f"New index built in {new_dir} but it could not replace {target} ({exc}). "
            f"Close anything using the index, then replace the folder by hand: "
            f"delete {target} and rename {new_dir} to {target.name}."
        ) from exc
    if had_old:
        try:
            _retry(lambda: shutil.rmtree(old_dir))
        except OSError as exc:  # the new index is in place; the old copy must still go
            logger.warning("Old index folder %s could not be deleted (%s); delete it by hand -- "
                           "it may still hold text redacted since.", old_dir, exc)


def _delete_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 -- chromadb raises different types per version
        pass
