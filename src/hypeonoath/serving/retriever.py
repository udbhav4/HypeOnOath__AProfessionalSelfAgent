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
F-R5 (eval D109): hybrid_search() fuses the dense ranking with a BM25 keyword ranking (serving.lexical)
by rrf(); eval only (config.HYBRID = "none"), retrieve() / search() never use it (chat unchanged, D98).
F-R6 (fr6 plan v2): candidates() builds the pool a cross-encoder re-ranks (serving.reranker); eval only too.

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
        text = chunk.text
        # F-R3 index (D95): a "project name — description" line sits above the header; skip it too.
        if chunk.project_name and text.startswith(f"{' '.join(chunk.project_name.split())} — "):
            text = text.split("\n", 1)[1] if "\n" in text else text
        return sha256_hex((text.split("\n", 1)[1] if "\n" in text else text).strip())
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
        self._lex = None                  # F-R5: (BM25Index, {id: Chunk}, {id: vector}), built on first use

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

    def _lexical(self):
        """BM25 index + every chunk and stored vector of the collection, loaded once per Retriever (F-R5)."""
        if self._lex is None:
            from hypeonoath.serving.lexical import BM25Index

            got = self.collection.get(include=["documents", "metadatas", "embeddings"])
            ids = list(got["ids"])
            if not ids:
                raise ValueError("index is empty: nothing to search")
            chunks = {cid: restore_chunk(cid, doc, meta or {})
                      for cid, doc, meta in zip(ids, got["documents"], got["metadatas"])}
            vectors = {cid: [float(x) for x in v] for cid, v in zip(ids, got["embeddings"])}
            self._lex = (BM25Index.build(ids, [chunks[cid].text for cid in ids]), chunks, vectors)
        return self._lex

    def hybrid_search(self, question: str, depth: int, candidates: int = config.HYBRID_CANDIDATES,
                      rrf_k: int = config.HYBRID_RRF_K,
                      bm25_weight: float = config.HYBRID_BM25_WEIGHT) -> tuple[list[RetrievedItem], list[list[float]]]:
        """
        F-R5: top `candidates` of the dense leg (Chroma, as search()) and of the BM25 leg, fused by RRF;
        the first `depth` fused hits, each with score = cosine(query, stored vector) for every hit (also
        BM25-only ones, which have no Chroma distance; D110), plus their stored vectors (for mmr).
        Each leg returns max(candidates, depth) hits, so a pool larger than `candidates` (e.g. MMR fetch_k
        above it, D112) is still filled from both legs and the fused list is never shorter than `depth`
        while the index has that many chunks.
        """
        from hypeonoath.serving.lexical import tokenize

        bm25, chunks, vectors = self._lexical()
        per_leg = min(max(candidates, depth), len(chunks))              # each leg at least as deep as the pool
        query_vector = self.embed_query(question)                      # one embedding call, as search()
        result = self.collection.query(query_embeddings=[query_vector], n_results=per_leg,
                                       where=None, include=["distances"])
        dense_ids = list(result["ids"][0])
        lexical_ids = [cid for cid, _ in bm25.top(tokenize(question), per_leg)]
        dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
        bm25_rank = {cid: r for r, cid in enumerate(lexical_ids, 1)}
        hits = [RetrievedItem(chunks[cid], _cosine(query_vector, vectors[cid]), fused=fused,
                              dense_rank=dense_rank.get(cid), bm25_rank=bm25_rank.get(cid))
                for cid, fused in rrf([dense_ids, lexical_ids], rrf_k, [1.0, bm25_weight])[:depth]]
        return hits, [vectors[h.chunk.chunk_id] for h in hits]

    def candidates(self, question: str, depth: int,
                   sources: str = config.RERANK_SOURCES) -> tuple[list[RetrievedItem], list[list[float]]]:
        """
        F-R6 candidate pool for the cross-encoder, with each item's stored vector (for MMR) and cosine `score`.
        dense      = the dense top `depth` (as search_with_embeddings), source "dense".
        dense_bm25 = that list followed by the BM25 top `depth` not already in it, in BM25 order -- a plain
                     union, no fusion (FR6-8): the cross-encoder decides the order, not RRF. BM25-only items
                     get their cosine from the stored vector (as hybrid_search, D110); dense_rank / bm25_rank
                     and source (dense | bm25 | both) say where each came from.
        """
        if sources not in config.RERANK_SOURCE_MODES:
            raise ValueError(f"unknown candidate sources {sources!r}; known: {list(config.RERANK_SOURCE_MODES)}")
        if sources == "dense":
            hits, vectors = self.search_with_embeddings(question, depth)
            for rank, h in enumerate(hits, 1):
                h.dense_rank, h.source = rank, "dense"
            return hits, vectors

        from hypeonoath.serving.lexical import tokenize

        bm25, chunks, stored = self._lexical()
        per_leg = min(depth, len(chunks))
        query_vector = self.embed_query(question)                      # one embedding call for both legs
        result = self.collection.query(query_embeddings=[query_vector], n_results=per_leg,
                                       where=None, include=["distances"])
        dense_ids = list(result["ids"][0])
        lexical_ids = [cid for cid, _ in bm25.top(tokenize(question), per_leg)]
        dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
        bm25_rank = {cid: r for r, cid in enumerate(lexical_ids, 1)}
        ordered = list(dict.fromkeys(dense_ids + lexical_ids))          # dense first, then BM25-only
        hits = [RetrievedItem(chunks[cid], _cosine(query_vector, stored[cid]), dense_rank=dense_rank.get(cid),
                              bm25_rank=bm25_rank.get(cid),
                              source="both" if cid in dense_rank and cid in bm25_rank
                              else "dense" if cid in dense_rank else "bm25")
                for cid in ordered]
        return hits, [stored[cid] for cid in ordered]

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


def rrf(rankings: list[list[str]], k: int = config.HYBRID_RRF_K,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (F-R5, D109): score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1; an id missing
    from a ranking gets nothing from it. Best first; ties -> best single rank, then the rank in the first
    ranking (the dense leg), then the id, so the order is deterministic.
    """
    weights = [1.0] * len(rankings) if weights is None else weights
    if len(weights) != len(rankings):
        raise ValueError("rrf needs one weight per ranking")
    if k < 0:
        raise ValueError(f"rrf k must be >= 0, got {k}")
    score: dict[str, float] = {}
    ranks: dict[str, list[int]] = {}
    absent = float("inf")
    for i, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, cid in enumerate(dict.fromkeys(ranking), 1):           # a repeated id counts once
            score[cid] = score.get(cid, 0.0) + weight / (k + rank)
            ranks.setdefault(cid, [absent] * len(rankings))[i] = rank
    return sorted(score.items(), key=lambda kv: (-kv[1], min(ranks[kv[0]]), ranks[kv[0]][0], kv[0]))


def mmr(hits: list[RetrievedItem], embeddings: list[list[float]], k: int, lambda_: float,
        relevance: list[float] | None = None) -> list[RetrievedItem]:
    """
    Maximal marginal relevance re-ranking of raw hits (eval D31; Chroma has no built-in MMR,
    plan assumption): greedily pick argmax  lambda * rel(d) - (1 - lambda) * max sim(d, picked),
    with rel(d) = the hit's score, or `relevance` when given (F-R5, D111: the normalised fused score,
    so MMR keeps the fused order instead of falling back to cosine). lambda = 1.0 keeps the ranking by
    rel. Returns <= k items.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"MMR lambda must be in [0, 1], got {lambda_}")
    if len(embeddings) != len(hits):
        raise ValueError("mmr needs one embedding per hit")
    if relevance is not None and len(relevance) != len(hits):
        raise ValueError("mmr needs one relevance value per hit")
    rel = [h.score for h in hits] if relevance is None else relevance
    sim = [[_cosine(a, b) for b in embeddings] for a in embeddings]     # once, not per greedy step
    remaining = list(range(len(hits)))
    picked: list[int] = []
    while remaining and len(picked) < k:
        def gain(i: int) -> float:
            redundancy = max((sim[i][j] for j in picked), default=0.0)
            return lambda_ * rel[i] - (1.0 - lambda_) * redundancy
        best = max(remaining, key=gain)          # max() keeps the first (higher-ranked) hit on ties
        picked.append(best)
        remaining.remove(best)
    return [hits[i] for i in picked]
