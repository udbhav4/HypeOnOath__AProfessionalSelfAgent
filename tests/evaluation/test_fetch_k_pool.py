"""MMR candidate pool larger than the eval depth (D112): search max(EVAL_DEPTH, fetch_k), MMR picks EVAL_DEPTH."""
from __future__ import annotations

import pytest

from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import paths
from hypeonoath.evaluation.answerer import BaselineAnswerer
from hypeonoath.serving.retriever import mmr
from tests.evaluation.test_runner import cv_chunk

N = 60                                            # chunks in the fake index (> any fetch_k used below)


def _vector(i: int) -> list[float]:
    # Ranks 0-29 are near-duplicates (one crowd); from rank 30 on every chunk points another way.
    return [1.0, 0.0, 0.0] if i < 30 else [0.0, 1.0, float(i)]


class PoolRetriever:
    """Fake retriever: n hits with falling scores; records every requested depth."""

    def __init__(self, n: int = N):
        self.hits = [RetrievedItem(cv_chunk(f"c{i}", f"text {i}", i + 1, i + 1), 0.9 - i / 200) for i in range(n)]
        self.vectors = [_vector(i) for i in range(n)]
        self.depths: list[tuple[str, int]] = []

    def search(self, q, k):
        self.depths.append(("search", k))
        return self.hits[:k]

    def search_with_embeddings(self, q, k):
        self.depths.append(("search_with_embeddings", k))
        return self.hits[:k], self.vectors[:k]

    def hybrid_search(self, q, depth):
        self.depths.append(("hybrid_search", depth))
        items = [RetrievedItem(h.chunk, h.score, fused=1 / (61 + i), dense_rank=i + 1)
                 for i, h in enumerate(self.hits[:depth])]
        return items, self.vectors[:depth]

    def assemble(self, hits):
        return list(hits)


def ids(items):
    return [i.chunk.chunk_id for i in items]


def test_pool_above_eval_depth_searches_deeper_and_ranks_eval_depth():
    r = PoolRetriever()
    got = BaselineAnswerer(r, top_k=12, mmr_lambda=0.7, fetch_k=40).retrieve("q")
    assert r.depths == [("search_with_embeddings", 40)]
    assert len(got.ranking) == paths.EVAL_DEPTH and len(got.context) == 12
    assert got.ranking[0].chunk.chunk_id == "c0"                     # MMR's first pick = top hit: gate unchanged
    assert any(int(cid[1:]) >= 30 for cid in ids(got.ranking[:12]))  # a chunk from ranks 31-40 reaches the prompt


def test_pool_equals_mmr_choosing_eval_depth_out_of_fetch_k():
    r = PoolRetriever()
    got = BaselineAnswerer(r, mmr_lambda=0.7, fetch_k=40).retrieve("q")
    expected = mmr(r.hits[:40], r.vectors[:40], paths.EVAL_DEPTH, 0.7)
    assert ids(got.ranking) == ids(expected)


def test_first_top_k_equal_picking_top_k_directly():
    """Greedy MMR: the first k picks do not depend on how many are picked after them."""
    r = PoolRetriever()
    got = BaselineAnswerer(r, top_k=12, mmr_lambda=0.7, fetch_k=40).retrieve("q")
    assert ids(got.context) == ids(mmr(r.hits[:40], r.vectors[:40], 12, 0.7))


@pytest.mark.parametrize("fetch_k", [5, 15, 20])
def test_pool_up_to_eval_depth_is_unchanged(fetch_k):
    r = PoolRetriever()
    got = BaselineAnswerer(r, mmr_lambda=0.7, fetch_k=fetch_k).retrieve("q")
    assert r.depths == [("search_with_embeddings", paths.EVAL_DEPTH)]
    head = mmr(r.hits[:fetch_k], r.vectors[:fetch_k], fetch_k, 0.7)
    assert ids(got.ranking) == ids(head + r.hits[fetch_k:paths.EVAL_DEPTH])     # the pre-D112 result
    assert ids(got.ranking[fetch_k:]) == [f"c{i}" for i in range(fetch_k, paths.EVAL_DEPTH)]   # tail: plain order


def test_mmr_off_ignores_fetch_k():
    r = PoolRetriever()
    got = BaselineAnswerer(r, fetch_k=50).retrieve("q")
    assert r.depths == [("search", paths.EVAL_DEPTH)] and ids(got.ranking) == [f"c{i}" for i in range(20)]


def test_pool_larger_than_the_index():
    r = PoolRetriever(n=25)                                         # Chroma returns at most the index size
    got = BaselineAnswerer(r, mmr_lambda=0.7, fetch_k=50).retrieve("q")
    assert len(got.ranking) == paths.EVAL_DEPTH and len(set(ids(got.ranking))) == paths.EVAL_DEPTH


def test_hybrid_uses_the_same_pool_rule():
    r = PoolRetriever()
    got = BaselineAnswerer(r, hybrid="rrf", mmr_lambda=0.7, fetch_k=40).retrieve("q")
    assert r.depths == [("hybrid_search", 40)] and len(got.ranking) == paths.EVAL_DEPTH
    r2 = PoolRetriever()
    BaselineAnswerer(r2, hybrid="rrf").retrieve("q")                 # no MMR: depth stays EVAL_DEPTH
    assert r2.depths == [("hybrid_search", paths.EVAL_DEPTH)]


def test_settings_record_the_pool():
    assert BaselineAnswerer(PoolRetriever(), mmr_lambda=0.7, fetch_k=40).settings()["mmr"] == {
        "enabled": True, "lambda": 0.7, "fetch_k": 40}


def test_fetch_k_below_one_refused():
    with pytest.raises(ValueError, match="fetch_k must be >= 1"):
        BaselineAnswerer(PoolRetriever(), mmr_lambda=0.7, fetch_k=0)
