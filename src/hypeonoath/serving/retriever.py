"""
Step 5 -- question -> ranked context items (guideline v4, Section 6).

  1. embed_query(): same module + model object as indexing, "search_query: "
     prefix (Section 2.2 invariant).
  2. Top-k cosine search (k=8 start, tune), returned WITH similarity scores
     so Phase 2's Layer 2 threshold gate needs no signature change.
  3. Restore the flattened metadata into Chunks (core.store.restore_chunk).
  4. Sibling-part fetch: a hit that is one part of a split function gets all
     its parts, ordered 1 -> n, fetched once per group; if the whole group
     would push the context past MAX_CONTEXT_TOKENS, only the hit and its
     immediate neighbours (part +/- 1) are included.
  5. Optional duplicate suppression: same code text (header stripped) as a
     higher-ranked hit -> dropped (README fence vs source file).
Filters (project_key / content_type / language) are available via `where`
but unused by default, so multi-hop questions see every content type.
"""
from __future__ import annotations

from collections.abc import Callable

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, RetrievedItem, sha256_hex
from hypeonoath.core.store import get_client, restore_chunk

CountTokens = Callable[[str], int]


def _code_hash(chunk: Chunk) -> str | None:
    """Hash of the code body (header line stripped) for duplicate suppression."""
    if chunk.content_type == "code":
        return sha256_hex((chunk.text.split("\n", 1)[1] if "\n" in chunk.text else chunk.text).strip())
    if chunk.content_type == "code_fence":
        body = chunk.text.split("```")
        return sha256_hex(body[1].split("\n", 1)[-1].strip()) if len(body) >= 3 else None
    return None


class Retriever:
    def __init__(self, collection=None, embed_query: Callable[[str], list[float]] | None = None,
                 count_tokens: CountTokens | None = None):
        if collection is None:
            collection = get_client().get_collection(config.COLLECTION_NAME)
        if embed_query is None or count_tokens is None:
            from hypeonoath.core import embedder
            embed_query = embed_query or embedder.embed_query
            count_tokens = count_tokens or embedder.count_tokens
        self.collection = collection
        self.embed_query = embed_query
        self.count = count_tokens

    def retrieve(self, question: str, k: int = config.DEFAULT_TOP_K,
                 where: dict | None = None) -> list[RetrievedItem]:
        result = self.collection.query(
            query_embeddings=[self.embed_query(question)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            RetrievedItem(restore_chunk(cid, doc, meta or {}), 1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
        return self._assemble(hits)

    def _assemble(self, hits: list[RetrievedItem]) -> list[RetrievedItem]:
        items: list[RetrievedItem] = []
        seen_groups: set[str] = set()
        seen_code: set[str] = set()
        used_tokens = 0

        for item in hits:
            chunk = item.chunk
            code_hash = _code_hash(chunk)
            if code_hash and code_hash in seen_code:
                continue                                   # duplicate suppression
            if chunk.group_id and (chunk.total_parts or 0) > 1:
                if chunk.group_id in seen_groups:
                    continue                               # each split function once
                seen_groups.add(chunk.group_id)
                item.parts = self._group_parts(chunk, budget=config.MAX_CONTEXT_TOKENS - used_tokens)
            if code_hash:
                seen_code.add(code_hash)
            used_tokens += sum(self.count(c.text) for c in item.ordered_chunks)
            items.append(item)
        return items

    def _group_parts(self, hit: Chunk, budget: int) -> list[Chunk]:
        got = self.collection.get(where={"group_id": hit.group_id}, include=["documents", "metadatas"])
        parts = sorted(
            (restore_chunk(cid, doc, meta or {}) for cid, doc, meta in
             zip(got["ids"], got["documents"], got["metadatas"])),
            key=lambda c: c.part or 0,
        )
        if not parts:
            return [hit]
        if sum(self.count(c.text) for c in parts) <= budget:
            return parts
        return [c for c in parts if abs((c.part or 0) - (hit.part or 0)) <= 1]
