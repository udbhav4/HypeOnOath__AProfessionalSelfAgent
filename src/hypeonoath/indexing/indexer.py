"""
Step 4 -- chunks + vectors -> persisted Chroma collection (guideline v4,
Sections 3A.8 and 5; ADR-5, ADR-16 in the design doc).

  * Embeddings are ALWAYS passed explicitly -- Chroma's default embedding
    function is a different model and would silently break the Section 2.2
    "same model at index and query time" invariant.
  * Full rebuild every run. The new collection is written under a temporary
    name first and only swapped in once fully written, so even a crash
    mid-write leaves the previous good index in place (run order: delete the
    old index only after everything before it succeeded).
  * The client and the metadata flattening rules live in hypeonoath.core.store,
    shared with the serving side (which restores what is written here).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Chunk
from hypeonoath.core.store import flatten_metadata, get_client

logger = logging.getLogger(__name__)

_ADD_BATCH = 256


def rebuild_index(
    chunks: list[Chunk],
    vectors: list[list[float]],
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
) -> int:
    """Replace the collection with exactly these chunks. Returns the count."""
    chroma_dir = config.CHROMA_DIR if chroma_dir is None else chroma_dir
    collection_name = collection_name or config.COLLECTION_NAME
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    duplicates = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk_ids (would overwrite each other): {duplicates[:5]}")

    client = get_client(chroma_dir)
    staging = f"{collection_name}__staging"
    _delete_if_exists(client, staging)
    collection = client.create_collection(
        name=staging,
        embedding_function=None,                    # never let Chroma embed
        metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL_NAME,
                  "embed_revision": config.EMBED_MODEL_REVISION or "unpinned"},
    )
    for start in range(0, len(chunks), _ADD_BATCH):
        batch = chunks[start:start + _ADD_BATCH]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[start:start + _ADD_BATCH],
            documents=[c.text for c in batch],
            metadatas=[flatten_metadata(c) for c in batch],
        )

    # Swap: only now is the old index removed.
    _delete_if_exists(client, collection_name)
    collection.modify(name=collection_name)
    log_json_line(config.INDEXING_LOG_PATH, status="index_rebuilt", chunks=len(chunks),
                  content_types=dict(Counter(c.content_type for c in chunks)))
    logger.info("Indexed %d chunks into %s/%s", len(chunks), chroma_dir, collection_name)
    return len(chunks)


def _delete_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 -- chromadb raises different types per version
        pass
