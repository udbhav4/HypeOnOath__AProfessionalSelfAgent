"""F-R6 re-ranking in serving (fr6 plan v2): Reranker, rerank(), rescale_to_cosine(), Retriever.candidates()."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.serving import reranker as reranker_mod
from hypeonoath.serving.reranker import Reranker, check_available, rerank, rescale_to_cosine
from tests.serving.test_hybrid_retriever import setup


def item(cid, score=0.5):
    return RetrievedItem(Chunk(cid, f"text of {cid}", "corpus/converted/doc.md", "prose"), score)


class FakeTokenizer:
    """Token count = words of query + text (+ 2 special tokens), as a real pair encoding."""

    def __call__(self, queries, texts, truncation, verbose):
        assert truncation is False and verbose is False
        return {"input_ids": [[0] * (len(q.split()) + len(t.split()) + 2) for q, t in zip(queries, texts)]}


class FakeCrossEncoder:
    """Score = how many query words the text holds; records every predict call."""

    def __init__(self):
        self.calls = []
        self.tokenizer = FakeTokenizer()
        self.activation_fn = type("Sigmoid", (), {})()

    def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
        self.calls.append((list(pairs), batch_size))
        return [float(sum(w in t.split() for w in q.split())) for q, t in pairs]


# --- Reranker ----------------------------------------------------------------------------------

def test_score_keeps_text_order_and_uses_the_fixed_batch_size():
    model = FakeCrossEncoder()
    r = Reranker("gte", model=model)
    assert r.score("python sql", ["sql only", "python and sql", "nothing"]) == [1.0, 2.0, 0.0]
    assert model.calls[0][1] == config.RERANK_BATCH_SIZE
    assert r.score("q", []) == [] and len(model.calls) == 1                 # no texts -> no model call
    assert r.activation == "Sigmoid"


def test_count_truncated_counts_pairs_longer_than_max_length():
    r = Reranker("gte", max_length=10, model=FakeCrossEncoder())
    # "a b" (2) + 6 words + 2 special = 10 -> fits; 7 words -> 11 -> cut
    assert r.count_truncated("a b", ["w " * 6, "w " * 7, "w"]) == 1
    assert r.count_truncated("a b", []) == 0


def test_unknown_alias_and_bad_sizes_refuse():
    with pytest.raises(ValueError, match="unknown re-ranker"):
        Reranker("bge")
    with pytest.raises(ValueError, match="unknown re-ranker"):
        check_available("bge")
    with pytest.raises(ValueError, match=">= 1"):
        Reranker("gte", max_length=0)
    check_available("gte")                                                   # known alias, package importable


def test_model_load_failure_becomes_value_error(monkeypatch):
    import sentence_transformers

    calls = []

    def broken(*args, **kwargs):
        calls.append(kwargs["local_files_only"])
        raise OSError("no network")
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", broken)
    with pytest.raises(ValueError, match="cannot load re-ranker"):
        Reranker("gte").model
    assert calls == [True, False]                                            # cache first, then one download


def test_model_loads_from_cache_first(monkeypatch):
    import sentence_transformers

    seen = []

    class Loaded:
        def __init__(self, model_id, revision, max_length, local_files_only):
            seen.append((model_id, revision, max_length, local_files_only))
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Loaded)
    Reranker("gte").model
    model_id, revision = config.RERANK_MODELS["gte"]
    assert seen == [(model_id, revision, config.RERANK_MAX_LENGTH, True)]   # pinned revision, no Hub request


# --- rerank() / rescale_to_cosine() -------------------------------------------------------------

def test_rerank_orders_by_score_ties_keep_incoming_order_and_copies():
    hits = [item("a", .9), item("b", .8), item("c", .7)]
    got = rerank(hits, [0.1, 0.5, 0.5])
    assert [h.chunk.chunk_id for h in got] == ["b", "c", "a"]                # b before c: tie -> incoming order
    assert [h.pre_rerank_rank for h in got] == [2, 3, 1]
    assert [h.rerank_score for h in got] == [0.5, 0.5, 0.1]
    assert [h.score for h in got] == [.8, .7, .9]                            # cosine untouched (FR6-2)
    assert hits[0].rerank_score is None                                      # inputs not mutated
    with pytest.raises(ValueError, match="one score per hit"):
        rerank(hits, [1.0])


def test_rescale_maps_ce_into_the_cosine_range_keeping_order_and_gaps():
    rel = rescale_to_cosine([0.99, 0.5, 0.01], [0.70, 0.88, 0.80])
    assert rel[0] == pytest.approx(0.88) and rel[2] == pytest.approx(0.70)  # extremes -> cos max / min
    assert rel[1] == pytest.approx(0.70 + (0.49 / 0.98) * 0.18)             # gap kept proportionally
    assert rel == sorted(rel, reverse=True)                                  # CE order kept


def test_rescale_edge_cases():
    assert rescale_to_cosine([0.3, 0.3], [0.9, 0.8]) == [0.9, 0.8]          # CE tells nothing -> own cosine
    assert rescale_to_cosine([], []) == []
    assert rescale_to_cosine([0.9, 0.1], [0.5, 0.5]) == [0.5, 0.5]          # flat cosines -> flat relevance
    with pytest.raises(ValueError):
        rescale_to_cosine([0.1], [])


# --- Retriever.candidates() ---------------------------------------------------------------------

def test_candidates_dense_marks_rank_and_source():
    retriever, col = setup()
    col.query_calls.clear()

    def swe(q, k):                                            # search_with_embeddings stand-in
        hits = [item("a", .9), item("b", .8)]
        return hits, [[1.0, 0.0], [0.8, 0.6]]
    retriever.search_with_embeddings = swe
    hits, vectors = retriever.candidates("q", 2, "dense")
    assert [(h.dense_rank, h.source) for h in hits] == [(1, "dense"), (2, "dense")] and len(vectors) == 2


def test_candidates_union_dense_first_then_bm25_only_without_duplicates():
    retriever, col = setup(dense_order=("a", "b", "c", "d"))
    hits, vectors = retriever.candidates("glance doctor", 2, "dense_bm25")     # each word in one chunk only
    ids = [h.chunk.chunk_id for h in hits]
    assert ids[:2] == ["a", "b"]                                             # dense top 2 first, in order
    assert set(ids[2:]) == {"c", "d"} and len(ids) == len(set(ids))          # BM25 top 2, not repeated
    by_id = {h.chunk.chunk_id: h for h in hits}
    assert by_id["a"].source == "dense" and by_id["c"].source == "bm25" and by_id["c"].dense_rank is None
    assert by_id["c"].score == pytest.approx(0.0)                            # cosine from the stored vector
    assert vectors[ids.index("c")] == [0.0, 1.0]


def test_candidates_union_marks_items_found_by_both_legs():
    retriever, _ = setup(dense_order=("c", "a", "b", "d"))
    hits, _ = retriever.candidates("glance doctor", 2, "dense_bm25")
    both = [h for h in hits if h.source == "both"]
    assert both and all(h.dense_rank and h.bm25_rank for h in both)


def test_candidates_unknown_source_refuses():
    retriever, _ = setup()
    with pytest.raises(ValueError, match="unknown candidate sources"):
        retriever.candidates("q", 2, "rrf")


def test_importing_the_module_does_not_load_sentence_transformers():
    import subprocess
    import sys

    code = ("import sys, hypeonoath.serving.reranker\n"
            "print('sentence_transformers' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr
    assert reranker_mod.Reranker("gte")._model is None                       # lazy: nothing loaded at init


# --- the real model (skipped when the pinned revision is not cached) ----------------------------

def _gte_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    model_id, revision = config.RERANK_MODELS["gte"]
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors", revision=revision), str)


@pytest.mark.model
@pytest.mark.skipif(not _gte_cached(), reason="pinned gte-reranker revision not in the HF cache")
def test_real_gte_scores_the_relevant_text_higher():
    r = Reranker("gte")
    relevant, unrelated = r.score("Which programming languages does he use?",
                                  ["He writes Python and TypeScript every day.", "Berlin had snow in November."])
    assert 0.0 <= unrelated < relevant <= 1.0                                # Sigmoid scores in [0, 1]
    assert r.activation == "Sigmoid"
    assert r.count_truncated("q", ["word " * 1500, "short"]) == 1


# --- QA review additions -------------------------------------------------------------------------

def test_rerank_all_equal_scores_keep_the_incoming_order_and_empty_input():
    hits = [item(c) for c in "abcd"]
    got = rerank(hits, [0.5] * 4)
    assert [h.chunk.chunk_id for h in got] == list("abcd") and [h.pre_rerank_rank for h in got] == [1, 2, 3, 4]
    assert rerank([], []) == []


def test_rescale_single_item_negative_logits_and_monotonicity():
    assert rescale_to_cosine([3.2], [0.61]) == [0.61]                       # one item: ce_max == ce_min
    rel = rescale_to_cosine([-4.0, 2.0, -1.0, 0.5], [0.80, 0.60, 0.75, 0.90])  # raw logits, any sign
    assert rel[1] == pytest.approx(0.90) and rel[0] == pytest.approx(0.60)  # max CE -> cos max, min CE -> cos min
    assert rel[2] == pytest.approx(0.60 + 3 / 6 * 0.30) and rel[3] == pytest.approx(0.60 + 4.5 / 6 * 0.30)
    order = sorted(range(4), key=lambda i: -[-4.0, 2.0, -1.0, 0.5][i])
    assert [rel[i] for i in order] == sorted(rel, reverse=True)             # CE order kept exactly


def test_candidates_union_depth_larger_than_the_index_is_capped_and_unique():
    retriever, col = setup()
    col.query_calls.clear()
    hits, vectors = retriever.candidates("glance doctor", 10, "dense_bm25")
    assert col.query_calls == [4]                                            # n_results capped at the index size
    ids = [h.chunk.chunk_id for h in hits]
    assert ids == ["a", "b", "c", "d"] and len(vectors) == 4                 # every chunk once, dense order
    assert {h.chunk.chunk_id: h.source for h in hits} == {"a": "dense", "b": "dense", "c": "both", "d": "both"}


def test_candidates_union_when_bm25_finds_fewer_than_depth():
    retriever, _ = setup(dense_order=("a", "b", "c", "d"))
    hits, _ = retriever.candidates("glance", 3, "dense_bm25")                # one chunk holds the word
    assert [(h.chunk.chunk_id, h.source, h.bm25_rank) for h in hits] == \
        [("a", "dense", None), ("b", "dense", None), ("c", "both", 1)]
    hits, _ = retriever.candidates("zzz", 2, "dense_bm25")                   # BM25 finds nothing
    assert [h.chunk.chunk_id for h in hits] == ["a", "b"]


def test_candidates_union_vectors_and_cosines_follow_each_hit():
    retriever, col = setup(dense_order=("b", "a", "d", "c"))
    hits, vectors = retriever.candidates("glance doctor", 2, "dense_bm25")
    for h, v in zip(hits, vectors):
        assert v == col.vectors[h.chunk.chunk_id]                            # vector i belongs to hit i (MMR)
        assert h.score == pytest.approx(v[0])                                # cosine with the query [1, 0]


@pytest.mark.model
@pytest.mark.skipif(not _gte_cached(), reason="pinned gte-reranker revision not in the HF cache")
def test_real_gte_truncates_long_inputs_at_max_length_instead_of_failing():
    r = Reranker("gte")
    assert r.model.max_seq_length == config.RERANK_MAX_LENGTH                   # FR6-6: 1,024, not GTE's 8,192
    scores = r.score("q", ["word " * 1500, "short"])                         # > 1,024 tokens: cut, still scored
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)
