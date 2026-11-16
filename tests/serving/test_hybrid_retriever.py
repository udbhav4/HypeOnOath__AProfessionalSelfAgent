"""F-R5 hybrid search in the retriever (eval D109-D111): rrf(), mmr(relevance=), hybrid_search (plan T9-T17)."""
from __future__ import annotations

import math

import pytest

from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.core.store import flatten_metadata
from hypeonoath.serving.retriever import Retriever, mmr, rrf


def prose(cid, text):
    return Chunk(cid, text, "corpus/converted/doc.md", "prose")


class HybridCollection:
    """A Chroma-like collection: query() = a fixed dense order, get() = every chunk with its vector."""

    def __init__(self, chunks, vectors, dense_order):
        self.chunks, self.vectors, self.dense_order = chunks, vectors, dense_order
        self.get_calls, self.query_calls = 0, []

    def query(self, query_embeddings, n_results, where, include):
        self.query_calls.append(n_results)
        ids = self.dense_order[:n_results]
        by_id = {c.chunk_id: c for c in self.chunks}
        return {"ids": [ids], "documents": [[by_id[i].text for i in ids]],
                "metadatas": [[flatten_metadata(by_id[i]) for i in ids]], "distances": [[0.0] * len(ids)]}

    def get(self, include, where=None):
        self.get_calls += 1
        return {"ids": [c.chunk_id for c in self.chunks], "documents": [c.text for c in self.chunks],
                "metadatas": [flatten_metadata(c) for c in self.chunks],
                "embeddings": [self.vectors[c.chunk_id] for c in self.chunks]}


QUERY_VECTOR = [1.0, 0.0]


def setup(dense_order=("a", "b", "c", "d")):
    chunks = [prose("a", "career summary analytics"), prose("b", "project overview model"),
              prose("c", "glance replies use temperature 0.5"), prose("d", "doctor report temperature 0.1")]
    vectors = {"a": [1.0, 0.0], "b": [0.8, 0.6], "c": [0.0, 1.0], "d": [0.6, 0.8]}
    col = HybridCollection(chunks, vectors, list(dense_order))
    return Retriever(col, embed_query=lambda q: QUERY_VECTOR, count_tokens=lambda t: len(t.split())), col


# --- rrf (T9, T10) ------------------------------------------------------------------------------

def test_rrf_hand_computed_example():
    fused = dict(rrf([["a", "b", "c"], ["c", "x"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61) and fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)                 # in both legs: two terms
    assert fused["x"] == pytest.approx(1 / 62)                          # in one leg only: one term
    assert [cid for cid, _ in rrf([["a", "b", "c"], ["c", "x"]], k=60)][0] == "c"


def test_rrf_consensus_beats_a_single_top_rank_at_k60_not_at_k20():
    dense = [f"d{i}" for i in range(1, 30)] + ["both"]
    lexical = [f"l{i}" for i in range(1, 30)] + ["both"]
    assert rrf([dense, lexical], k=60)[0][0] == "both"                  # 2/90 > 1/61
    assert rrf([dense, lexical], k=20)[0][0] != "both"                  # 2/50 < 1/21


def test_rrf_weights_ties_and_errors():
    assert [cid for cid, _ in rrf([["a", "b"], ["b", "a"]], weights=[1.0, 0.0])] == ["a", "b"]   # BM25 off
    # equal scores: best single rank, then the dense rank, then the id
    assert [cid for cid, _ in rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]
    assert [cid for cid, _ in rrf([["z"], ["y"]])] == ["z", "y"]
    assert rrf([["a", "a", "b"]]) == rrf([["a", "b"]])                  # a repeated id counts once
    assert rrf([[], []]) == []
    with pytest.raises(ValueError, match="one weight per ranking"):
        rrf([["a"]], weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="k must be"):
        rrf([["a"]], k=-1)


# --- mmr relevance (T11) ------------------------------------------------------------------------

def _items(scores):
    return [RetrievedItem(prose(f"h{i}", "t"), s) for i, s in enumerate(scores)]


def test_mmr_without_relevance_is_unchanged_and_with_relevance_uses_it():
    hits, emb = _items([0.9, 0.8, 0.1]), [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    assert mmr(hits, emb, 3, 1.0) == hits                                # plain ranking by score
    by_rel = mmr(hits, emb, 3, 1.0, relevance=[0.1, 0.5, 1.0])
    assert [h.chunk.chunk_id for h in by_rel] == ["h2", "h1", "h0"]      # relevance, not score, decides
    with pytest.raises(ValueError, match="one relevance value per hit"):
        mmr(hits, emb, 3, 0.7, relevance=[1.0])


# --- hybrid_search (T12-T17) --------------------------------------------------------------------

def test_bm25_only_hit_enters_the_ranking_with_its_leg_ranks():
    r, _ = setup()
    hits, vectors = r.hybrid_search("glance temperature", depth=2, candidates=2)    # dense leg: a, b only
    by_id = {h.chunk.chunk_id: h for h in hits}
    assert by_id["c"].dense_rank is None and by_id["c"].bm25_rank == 1
    assert by_id["a"].dense_rank == 1 and by_id["a"].bm25_rank is None
    assert hits[0].chunk.chunk_id in ("a", "c") and len(vectors) == len(hits)
    assert all(h.fused is not None for h in hits)
    assert [h.fused for h in hits] == sorted((h.fused for h in hits), reverse=True)


def test_score_is_cosine_for_every_hit_including_bm25_only():
    r, col = setup()
    hits, vectors = r.hybrid_search("glance", depth=4, candidates=4)
    for h, v in zip(hits, vectors):
        assert v == col.vectors[h.chunk.chunk_id]
        expected = sum(x * y for x, y in zip(QUERY_VECTOR, v)) / math.hypot(*v)
        assert h.score == pytest.approx(expected)
    assert {h.chunk.chunk_id: round(h.score, 3) for h in hits}["c"] == 0.0


def test_lexical_index_built_once_per_retriever():
    r, col = setup()
    r.hybrid_search("glance", depth=4)
    r.hybrid_search("doctor report", depth=4)
    assert col.get_calls == 1


def test_candidates_and_depth_larger_than_the_index():
    r, col = setup()
    hits, _ = r.hybrid_search("glance", depth=20, candidates=50)
    assert len(hits) == 4 and col.query_calls == [4]                    # n_results capped at the chunk count


def test_depth_cuts_the_fused_list():
    r, _ = setup()
    hits, vectors = r.hybrid_search("glance", depth=2, candidates=4)
    assert len(hits) == 2 and len(vectors) == 2


def test_empty_collection_is_a_clear_error():
    col = HybridCollection([], {}, [])
    r = Retriever(col, embed_query=lambda q: QUERY_VECTOR, count_tokens=len)
    with pytest.raises(ValueError, match="index is empty"):
        r.hybrid_search("anything", depth=20)


def test_query_without_known_words_falls_back_to_the_dense_order():
    r, _ = setup()
    hits, _ = r.hybrid_search("the of you", depth=4, candidates=4)     # only stop words: BM25 leg empty
    assert [h.chunk.chunk_id for h in hits] == ["a", "b", "c", "d"]
    assert all(h.bm25_rank is None for h in hits)


def test_chat_path_never_builds_the_lexical_index():
    r, col = setup()
    col.query = lambda **kw: {"ids": [["a"]], "documents": [["career summary analytics"]],
                              "metadatas": [[flatten_metadata(prose("a", "career summary analytics"))]],
                              "distances": [[0.1]]}
    items = r.retrieve("anything", k=1)
    r.search("anything", 1)
    assert col.get_calls == 0 and r._lex is None
    assert items[0].fused is None and items[0].dense_rank is None and items[0].bm25_rank is None


def test_depth_above_candidates_deepens_both_legs():
    """D112: a pool (depth) larger than `candidates` is asked of both legs, not cut to `candidates`."""
    r, col = setup()
    hits, _ = r.hybrid_search("glance temperature", depth=3, candidates=2)
    assert col.query_calls == [3]                                         # dense leg asked for 3, not 2
    assert len(hits) == 3
    assert {h.chunk.chunk_id: h.dense_rank for h in hits}["c"] == 3       # c now found by the dense leg too


def test_candidates_above_depth_unchanged():
    r, col = setup()
    r.hybrid_search("glance", depth=2, candidates=3)
    assert col.query_calls == [3]                                         # legs still use `candidates`
