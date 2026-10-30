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

Eval plan v15, Section 8: retrieve() = assemble(search()). search() returns the raw
ranked hits (search_with_embeddings() also their stored vectors), assemble() builds the
production context from them, so the eval harness gets the raw ranking and the exact
context from ONE query. mmr() re-ranks raw hits (default off: config.MMR_LAMBDA = None).

On start-up the index's `privacy_fp` (collection metadata) is compared with
the fingerprint of the local privacy decisions; a mismatch only WARNS -- a
host without the decisions file (Phase 2) must still serve (plan 7.4).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable

from hypeonoath.core import config
from hypeonoath.core.private_values import load_private_values, privacy_fingerprint
from hypeonoath.core.records import Chunk, RetrievedItem, sha256_hex
from hypeonoath.core.store import get_client, restore_chunk

CountTokens = Callable[[str], int]

logger = logging.getLogger(__name__)

STALE_PRIVACY_WARNING = "Index was built with different privacy decisions; rebuild: hypeonoath-index"


def check_privacy_fingerprint(collection) -> bool:
    """True when the index matches the local privacy decisions; logs a warning otherwise."""
    index_fp = (getattr(collection, "metadata", None) or {}).get("privacy_fp", "none")
    try:
        local_fp = privacy_fingerprint(load_private_values())
    except ValueError as exc:            # unreadable decisions file: serve anyway
        logger.warning("Privacy decisions unreadable (%s); index fingerprint not checked", exc)
        return False
    if index_fp != local_fp:
        logger.warning(STALE_PRIVACY_WARNING)
        return False
    return True


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
        check_privacy_fingerprint(collection)
        if embed_query is None or count_tokens is None:
            from hypeonoath.core import embedder
            embed_query = embed_query or embedder.embed_query
            count_tokens = count_tokens or embedder.count_tokens
        self.collection = collection
        self.embed_query = embed_query
        self.count = count_tokens

    def retrieve(self, question: str, k: int = config.DEFAULT_TOP_K,
                 where: dict | None = None) -> list[RetrievedItem]:
        return self.assemble(self.search(question, k, where))

    def search(self, question: str, k: int, where: dict | None = None) -> list[RetrievedItem]:
        """Raw top-k hits with cosine similarity, best first (no sibling fetch, no dedupe)."""
        return self.search_with_embeddings(question, k, where, embeddings=False)[0]

    def search_with_embeddings(self, question: str, k: int, where: dict | None = None,
                               embeddings: bool = True) -> tuple[list[RetrievedItem], list[list[float]]]:
        """search() plus each hit's stored embedding (for mmr); [] when embeddings=False."""
        result = self.collection.query(
            query_embeddings=[self.embed_query(question)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"] + (["embeddings"] if embeddings else []),
        )
        hits = [
            RetrievedItem(restore_chunk(cid, doc, meta or {}), 1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
        vectors = [[float(x) for x in v] for v in result["embeddings"][0]] if embeddings else []
        return hits, vectors

    def assemble(self, hits: list[RetrievedItem]) -> list[RetrievedItem]:
        """Production context from ranked hits: sibling parts, duplicate suppression, token budget."""
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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def mmr(hits: list[RetrievedItem], embeddings: list[list[float]], k: int, lambda_: float) -> list[RetrievedItem]:
    """
    Maximal marginal relevance re-ranking of raw hits (eval D31; Chroma has no built-in MMR,
    plan assumption): greedily pick argmax  lambda * sim(query, d) - (1 - lambda) * max sim(d, picked),
    with sim(query, d) = the hit's score. lambda = 1.0 keeps the plain ranking. Returns <= k items.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"MMR lambda must be in [0, 1], got {lambda_}")
    if len(embeddings) != len(hits):
        raise ValueError("mmr needs one embedding per hit")
    sim = [[_cosine(a, b) for b in embeddings] for a in embeddings]     # once, not per greedy step
    remaining = list(range(len(hits)))
    picked: list[int] = []
    while remaining and len(picked) < k:
        def gain(i: int) -> float:
            redundancy = max((sim[i][j] for j in picked), default=0.0)
            return lambda_ * hits[i].score - (1.0 - lambda_) * redundancy
        best = max(remaining, key=gain)          # max() keeps the first (higher-ranked) hit on ties
        picked.append(best)
        remaining.remove(best)
    return [hits[i] for i in picked]
