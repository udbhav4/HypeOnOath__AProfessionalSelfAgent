"""Eval side of F-R6 re-ranking (fr6 plan v2): answerer paths, timing, records, metrics, settings, D13, CLI."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import cli, paths, runner, tracking
from hypeonoath.evaluation.answerer import BaselineAnswerer, Retrieved
from hypeonoath.evaluation.metrics import score_run
from hypeonoath.evaluation.report import SETTINGS_ROWS
from tests.evaluation.test_runner import FakeAnswerer, cv_chunk
from tests.evaluation.test_tracking_report import manifest

N = 60                                                     # candidates the fake retriever can return


def _vec(i):
    return [1.0, i / 100]


class RerankRetriever:
    """Fake retriever: chunk 'cI' has cosine 1 - I/100; candidates() / search_with_embeddings() record calls."""

    def __init__(self):
        self.calls = []
        self.collection = type("C", (), {"metadata": {}})()

    def _hits(self, depth):
        hits = [RetrievedItem(cv_chunk(f"c{i}", f"text {i}"), 1 - i / 100) for i in range(min(depth, N))]
        return hits, [_vec(i) for i in range(len(hits))]

    def candidates(self, q, depth, sources):
        self.calls.append(("candidates", q, depth, sources))
        hits, vectors = self._hits(depth)
        for rank, h in enumerate(hits, 1):
            h.dense_rank, h.source = rank, "dense"
        return hits, vectors

    def search(self, q, k):
        self.calls.append(("search", q, k))
        return self._hits(k)[0]

    def search_with_embeddings(self, q, k):
        self.calls.append(("search_with_embeddings", q, k))
        return self._hits(k)

    def assemble(self, hits):
        return list(hits)


class FakeReranker:
    """Score = the chunk number (higher number = better): reverses the cosine order; records every call."""

    activation = "Sigmoid"

    def __init__(self, truncated=0):
        self.calls, self.truncated = [], truncated

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [int(t.split()[1]) / 100 if t.startswith("text ") else 0.0 for t in texts]

    def count_truncated(self, query, texts):
        return self.truncated


def ids(items):
    return [h.chunk.chunk_id for h in items]


def answerer(**kw):
    kw.setdefault("reranker", FakeReranker())
    return BaselineAnswerer(RerankRetriever(), rerank=kw.pop("rerank", "gte"), **kw)


# --- answerer paths ------------------------------------------------------------------------------

def test_rerank_off_is_todays_path():
    r = RerankRetriever()
    got = BaselineAnswerer(r).retrieve("q")
    assert [c[0] for c in r.calls] == ["search"] and got.rerank_s is None and got.rerank_truncated is None
    assert ids(got.ranking) == [f"c{i}" for i in range(paths.EVAL_DEPTH)]


def test_rerank_first_scores_the_pool_and_returns_the_ce_order(monkeypatch):
    monkeypatch.setattr("hypeonoath.evaluation.answerer.rewrite_query", lambda q, mode: "REWRITTEN")
    a = answerer(query_rewrite="third_person", top_k=3, reranker=FakeReranker(truncated=2))
    got = a.retrieve("q")
    assert a.retriever.calls == [("candidates", "REWRITTEN", config.RERANK_CANDIDATES, "dense")]
    assert a.reranker.calls[0][0] == "REWRITTEN" and len(a.reranker.calls[0][1]) == config.RERANK_CANDIDATES
    assert ids(got.ranking) == [f"c{i}" for i in range(49, 29, -1)]          # best CE first, cut to 20
    assert got.ranking[0].pre_rerank_rank == 50 and got.ranking[0].score == pytest.approx(0.51)   # cosine kept
    assert ids(got.context) == ["c49", "c48", "c47"]                          # assemble(top_k) of the new order
    assert got.rerank_s is not None and got.rerank_truncated == 2 and got.query == "REWRITTEN"


def test_ce_query_follows_the_rewrite_flag_when_off():
    a = answerer()
    a.retrieve("Tell me about yourself")
    assert a.reranker.calls[0][0] == "Tell me about yourself"                 # FR6-4: flag off -> the question


def test_rerank_first_with_mmr_passes_rescaled_relevance(monkeypatch):
    seen = {}

    def fake_mmr(hits, vectors, k, lambda_, relevance=None):
        seen.update(n=len(hits), k=k, rel=relevance, first=hits[0].chunk.chunk_id, vec=vectors[0])
        return hits[:k]
    monkeypatch.setattr("hypeonoath.serving.retriever.mmr", fake_mmr)
    a = answerer(mmr_lambda=0.7, fetch_k=55, rerank_candidates=50)
    got = a.retrieve("q")
    assert a.retriever.calls[0][2] == 55                                      # depth = max(20, C, fetch_k)
    assert seen["n"] == 55 and seen["k"] == paths.EVAL_DEPTH and seen["first"] == "c54"
    assert seen["vec"] == _vec(54)                                            # vectors follow the CE order
    cos = [1 - i / 100 for i in range(55)]
    assert seen["rel"][0] == pytest.approx(max(cos)) and seen["rel"][-1] == pytest.approx(min(cos))   # FR6-3
    assert len(got.ranking) == paths.EVAL_DEPTH


def test_rerank_first_mmr_real_rescaled_relevance_keeps_the_ce_top():
    got = answerer(mmr_lambda=0.7, fetch_k=50).retrieve("q")
    assert got.ranking[0].chunk.chunk_id == "c49"                             # MMR's first pick = best relevance
    assert len(got.ranking) == paths.EVAL_DEPTH and len(set(ids(got.ranking))) == paths.EVAL_DEPTH


def test_union_sources_are_passed_through():
    a = answerer(rerank_sources="dense_bm25")
    a.retrieve("q")
    assert a.retriever.calls[0][3] == "dense_bm25"


@pytest.mark.parametrize("kwargs, message", [
    ({"rerank": "bge"}, "unknown re-ranker"),
    ({"hybrid": "rrf"}, "cannot be combined"),
    ({"rerank_candidates": 10}, "rerank_candidates must be >="),
    ({"rerank_sources": "rrf"}, "unknown rerank sources"),
])
def test_bad_levers_refuse_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        answerer(**kwargs)


def test_off_never_checks_the_rerank_levers():
    BaselineAnswerer(RerankRetriever(), rerank_candidates=1, rerank_sources="x")   # unused when off


def test_settings_and_prepare():
    off = BaselineAnswerer(RerankRetriever())
    assert off.settings()["rerank"] == {"mode": "none"} and set(off.prepare()) == {"retrieval_warmup_s"}
    on = answerer(rerank_sources="dense_bm25")
    model_id, revision = config.RERANK_MODELS["gte"]
    assert on.settings()["rerank"] == {"mode": "gte", "model": model_id, "revision": revision, "candidates": 50,
                                       "sources": "dense_bm25",
                                       "max_length": config.RERANK_MAX_LENGTH, "batch_size": config.RERANK_BATCH_SIZE}
    runtime = on.prepare()
    assert set(runtime) == {"retrieval_warmup_s", "rerank_load_s", "activation", "torch_threads"}
    assert runtime["activation"] == "Sigmoid"
    assert on.reranker.calls == [("warm-up", ["warm-up"])]                    # one dummy pair before question 1


# --- runner records ------------------------------------------------------------------------------

class RerankedAnswerer(FakeAnswerer):
    def retrieve(self, question):
        got = super().retrieve(question)
        for rank, h in enumerate(got.ranking, 1):
            h.rerank_score, h.pre_rerank_rank, h.source, h.dense_rank = 1 / rank, rank + 5, "dense", rank + 5
        return Retrieved(got.ranking, got.context, None, 0.25, 1)


def test_runner_writes_timing_and_rerank_fields_only_when_on(tmp_path):
    from tests.evaluation.gold_fixtures import VALUES, parse, valid_raw, write_corpus

    corpus, questions = write_corpus(tmp_path), parse(valid_raw())[:2]
    runner.run_questions(questions, RerankedAnswerer(), tmp_path / "r", "retrieval", VALUES, corpus, log=lambda m: None)
    runner.run_questions(questions, FakeAnswerer(), tmp_path / "p", "retrieval", VALUES, corpus, log=lambda m: None)
    on, off = runner.read_answers(tmp_path / "r")[0], runner.read_answers(tmp_path / "p")[0]
    assert on["rerank_s"] == 0.25 and on["rerank_truncated"] == 1 and on["retrieve_s"] >= 0
    first = on["ranking"][0]
    assert first["rerank_score"] == 1.0 and first["pre_rerank_rank"] == 6 and first["source"] == "dense"
    assert "rerank_s" not in off and "retrieve_s" in off                      # every run is timed (FR6-7)
    assert set(off["ranking"][0]) == {"chunk_id", "document", "line_start", "line_end", "score", "groups"}


# --- metrics -------------------------------------------------------------------------------------

def _answers(rerank=True):
    out = []
    for i in range(1, 5):
        a = {"id": f"x{i}", "ranking": [], "context": [], "top1": None, "retrieve_s": float(i)}
        if rerank:
            a.update(rerank_s=i / 10, rerank_truncated=i - 1)
        out.append(a)
    return out


def test_latency_percentiles_and_truncation_sum():
    from tests.evaluation.gold_fixtures import parse, valid_raw

    qs = parse(valid_raw())[:4]
    answers = [dict(a, id=q.id) for a, q in zip(_answers(), qs)]
    m = score_run(qs, answers, [], {"stage": "retrieval", "top_k": 12})["metrics"]
    assert m["retrieve_p50_s"] == 2.0 and m["retrieve_p95_s"] == 4.0          # nearest rank
    assert m["rerank_p50_s"] == 0.2 and m["rerank_p95_s"] == 0.4 and m["rerank_truncated_pairs"] == 6
    off = score_run(qs, [dict(a, id=q.id) for a, q in zip(_answers(False), qs)], [], {"stage": "retrieval"})
    assert off["metrics"]["rerank_p50_s"] is None and off["metrics"]["rerank_truncated_pairs"] is None
    assert off["metrics"]["retrieve_p50_s"] == 2.0 and off["retrieval"]["n_rerank_s"] == 0


# --- tracking / report / CLI ---------------------------------------------------------------------

RERANK_ON = {"mode": "gte", "model": "m", "revision": "r", "candidates": 50, "sources": "dense",
             "max_length": 1024, "batch_size": 16}


def _base():
    return {"settings": {k: tracking.setting(manifest(), k) for k in tracking.SETTING_KEYS}}


def test_old_manifest_reads_rerank_none_and_csv_columns():
    assert tracking.setting(manifest(), "rerank") == {"mode": "none"}
    row = tracking.csv_row(manifest(), {})
    assert row["rerank"] == "none" and row["rerank_candidates"] == "" and row["rerank_sources"] == ""
    assert "rerank_order" not in row                                         # mmr_first removed: no order lever
    row = tracking.csv_row(manifest(rerank=RERANK_ON), {})
    assert (row["rerank"], row["rerank_candidates"], row["rerank_sources"]) == ("gte", "50", "dense")


def test_rerank_counts_as_one_setting_for_d13():
    assert tracking.changed_settings(manifest(rerank=RERANK_ON), _base()["settings"]) == ["rerank"]
    warnings = tracking.comparability_warnings(manifest(rerank=RERANK_ON), {}, _base())
    assert not any("more than one setting" in w for w in warnings)
    assert tracking.changed_settings(manifest(rerank={"mode": "none"}), _base()["settings"]) == []
    other = dict(RERANK_ON, candidates=100)
    assert tracking.changed_settings(manifest(rerank=other), {"rerank": RERANK_ON, **{
        k: v for k, v in _base()["settings"].items() if k != "rerank"}}) == ["rerank"]


def test_report_rows_show_the_rerank_levers():
    rows = dict(SETTINGS_ROWS)
    assert "rerank order" not in rows
    mmr_on = manifest(rerank=RERANK_ON, mmr={"enabled": True, "lambda": 0.7, "fetch_k": 50})
    assert rows["rerank candidates"](mmr_on) == 50 and rows["rerank sources"](mmr_on) == "dense"
    assert rows["rerank"](manifest()) == "none" and rows["rerank max_length"](manifest()) is None
    assert rows["rerank max_length"](manifest(rerank=RERANK_ON)) == 1024


def test_cli_flags_and_resume_kwargs():
    args = cli.build_parser().parse_args(["run", "--retrieval-only", "--rerank", "gte", "--rerank-candidates", "30",
                                          "--rerank-sources", "dense_bm25"])
    assert (args.rerank, args.rerank_candidates, args.rerank_sources) == ("gte", 30, "dense_bm25")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--rerank", "gte", "--rerank-order", "mmr_first"])   # flag removed
    defaults = cli.build_parser().parse_args(["run", "--retrieval-only"])
    assert defaults.rerank == "none" and defaults.rerank_candidates == config.RERANK_CANDIDATES
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--rerank", "bge"])            # choices = RERANK_MODELS keys
    assert cli._rerank_kwargs({}) == {"rerank": "none"}
    assert cli._rerank_kwargs(RERANK_ON) == {"rerank": "gte", "rerank_candidates": 50, "rerank_sources": "dense"}


# --- QA review additions -------------------------------------------------------------------------

class PermReranker(FakeReranker):
    """Score = a fixed permutation of the chunk number: the CE order shares nothing with the cosine order."""

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [((int(t.split()[1]) * 37) % 101) / 101 for t in texts]


class ConstReranker(FakeReranker):
    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [0.5] * len(texts)


class SmallRetriever(RerankRetriever):
    """An index of 8 chunks: every pool is smaller than the eval depth."""

    def _hits(self, depth):
        return super()._hits(min(depth, 8))


def test_pre_rerank_rank_points_at_the_candidate_position():
    got = answerer(reranker=PermReranker()).retrieve("q")
    assert all(h.chunk.chunk_id == f"c{h.pre_rerank_rank - 1}" for h in got.ranking)   # 1-based, no off-by-one
    assert [h.rerank_score for h in got.ranking] == sorted((h.rerank_score for h in got.ranking), reverse=True)


def test_rerank_first_mmr_gets_each_hits_own_vector_and_relevance(monkeypatch):
    seen = {}

    def fake_mmr(hits, vectors, k, lambda_, relevance=None):
        seen.update(pairs=[(h.chunk.chunk_id, v) for h, v in zip(hits, vectors)], hits=hits, rel=relevance)
        return hits[:k]
    monkeypatch.setattr("hypeonoath.serving.retriever.mmr", fake_mmr)
    answerer(reranker=PermReranker(), mmr_lambda=0.7, fetch_k=50).retrieve("q")
    assert all(v == _vec(int(cid[1:])) for cid, v in seen["pairs"])        # vector follows its hit after CE sort
    ce = [h.rerank_score for h in seen["hits"]]
    cos = [h.score for h in seen["hits"]]
    span = (max(cos) - min(cos)) / (max(ce) - min(ce))
    assert seen["rel"] == pytest.approx([min(cos) + (s - min(ce)) * span for s in ce])  # FR6-3 per hit


def test_rerank_first_mmr_with_an_uninformative_ce_equals_the_d113_path():
    d113 = BaselineAnswerer(RerankRetriever(), mmr_lambda=0.7, fetch_k=50).retrieve("q")
    got = answerer(reranker=ConstReranker(), mmr_lambda=0.7, fetch_k=50).retrieve("q")
    assert ids(got.ranking) == ids(d113.ranking)                            # equal CE -> own cosine -> same MMR


def test_rerank_first_mmr_with_fetch_k_below_candidates_rescales_over_the_head_only(monkeypatch):
    seen = {}

    def fake_mmr(hits, vectors, k, lambda_, relevance=None):
        seen.update(n=len(hits), rel=relevance)
        return hits[:k]
    monkeypatch.setattr("hypeonoath.serving.retriever.mmr", fake_mmr)
    a = answerer(mmr_lambda=0.7, fetch_k=20, rerank_candidates=50)
    got = a.retrieve("q")
    assert a.retriever.calls[0][2] == 50 and len(a.reranker.calls[0][1]) == 50   # CE scores all 50
    assert seen["n"] == 20                                                  # MMR re-orders the CE top 20
    assert max(seen["rel"]) == pytest.approx(0.70) and min(seen["rel"]) == pytest.approx(0.51)   # c30..c49
    assert set(ids(got.ranking)) == {f"c{i}" for i in range(30, 50)}


@pytest.mark.parametrize("kw", [{}, {"mmr_lambda": 0.7, "fetch_k": 50}])
def test_pools_smaller_than_the_eval_depth(kw):
    a = BaselineAnswerer(SmallRetriever(), rerank="gte", reranker=FakeReranker(), top_k=12, **kw)
    got = a.retrieve("q")
    assert len(got.ranking) == 8 and len(set(ids(got.ranking))) == 8 and len(got.context) == 8


def test_report_n_of_latency_metrics_and_old_scores():
    from hypeonoath.evaluation.metrics import METRICS_BY_NAME
    from hypeonoath.evaluation.report import _n
    from tests.evaluation.gold_fixtures import parse, valid_raw

    qs = parse(valid_raw())[:4]
    scores = score_run(qs, [dict(a, id=q.id) for a, q in zip(_answers(), qs)], [], {"stage": "retrieval"})
    assert _n(METRICS_BY_NAME["retrieve_p95_s"], scores) == 4 and _n(METRICS_BY_NAME["rerank_p50_s"], scores) == 4
    old = {"metrics": {"n": 4}, "retrieval": {}}                            # scores.json before F-R6
    assert _n(METRICS_BY_NAME["retrieve_p50_s"], old) is None


def test_rerank_revision_or_max_length_change_is_a_change_and_counts_with_others():
    base = {"rerank": RERANK_ON, **{k: v for k, v in _base()["settings"].items() if k != "rerank"}}
    for change in ({"revision": "r2"}, {"max_length": 512}, {"sources": "dense_bm25"}):
        assert tracking.changed_settings(manifest(rerank=dict(RERANK_ON, **change)), base) == ["rerank"]
    assert tracking.changed_settings(manifest(rerank=RERANK_ON), base) == []
    warnings = tracking.comparability_warnings(manifest(rerank=RERANK_ON, top_k=12), {}, _base())
    assert any("more than one setting" in w for w in warnings)              # rerank + top_k = two settings


def _cli_rerank_run(tmp_path, monkeypatch, prepare):
    from tests.evaluation.test_cli_gold import frozen_set

    frozen_set(tmp_path, monkeypatch)
    original, seen = cli.make_answerer, []

    def make(name, **kwargs):
        seen.append(kwargs)
        fake = original(name, **kwargs)
        settings = fake.settings
        rerank = dict(RERANK_ON, candidates=kwargs["rerank_candidates"], sources=kwargs["rerank_sources"])
        fake.settings = lambda: {**settings(), "rerank": rerank}
        fake.rerank_settings = lambda: rerank
        fake.prepare = prepare
        return fake
    monkeypatch.setattr(cli, "make_answerer", make)
    return seen


def test_cli_run_and_resume_warm_up_and_record_the_runtime(tmp_path, monkeypatch):
    prepared = []

    def prepare():
        prepared.append(1)
        return {"rerank_load_s": 1.5, "activation": "Sigmoid", "torch_threads": 4}
    seen = _cli_rerank_run(tmp_path, monkeypatch, prepare)
    assert cli.main(["run", "--retrieval-only", "--rerank", "gte", "--rerank-candidates", "30",
                     "--rerank-sources", "dense_bm25"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    m = runner.read_manifest(run_dir)
    assert m["rerank"]["candidates"] == 30 and m["retrieval_runtime"]["torch_threads"] == 4 and prepared == [1]
    assert cli.main(["run", "--resume", run_dir.name, "--rerank", "none"]) == 0         # the manifest wins (D26)
    levers = ("rerank", "rerank_candidates", "rerank_sources")
    assert [seen[1][k] for k in levers] == ["gte", 30, "dense_bm25"]
    assert "rerank_order" not in seen[1]
    assert prepared == [1, 1] and runner.read_manifest(run_dir)["retrieval_runtime_resumed"][-1]["rerank_load_s"] == 1.5


def test_cli_refuses_before_a_run_when_the_model_cannot_load(tmp_path, monkeypatch, capsys):
    def prepare():
        raise ValueError("cannot load re-ranker x@y (OSError: no network)")
    _cli_rerank_run(tmp_path, monkeypatch, prepare)
    assert cli.main(["run", "--retrieval-only", "--rerank", "gte"]) == 1
    assert "cannot load re-ranker" in capsys.readouterr().err and not paths.RUNS_DIR.exists()


def test_chart_legend_tags_a_reranked_run():
    from hypeonoath.evaluation import charts

    assert charts.experiment_tags(manifest(rerank=RERANK_ON)) != charts.experiment_tags(manifest())


def test_chart_tags_name_model_and_sources_from_manifest_and_csv_row():
    from hypeonoath.evaluation import charts

    assert charts.experiment_tags(manifest(rerank=RERANK_ON))[-1] == "rerank-gte"
    union = dict(RERANK_ON, sources="dense_bm25")
    assert charts.experiment_tags(manifest(rerank=union))[-1] == "rerank-gte-bm25"
    row = {"rerank": "gte", "rerank_sources": "dense"}                                      # runs.csv strings
    assert charts.experiment_tags(row) == ("rerank-gte",)
    assert charts.experiment_tags({"rerank": "none"}) == () and charts.experiment_tags({"rerank": ""}) == ()


def test_resume_refuses_when_the_reranker_build_changed(tmp_path, monkeypatch, capsys):
    seen = _cli_rerank_run(tmp_path, monkeypatch, lambda: {"retrieval_warmup_s": 0.1})
    assert cli.main(["run", "--retrieval-only", "--rerank", "gte"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    m = runner.read_manifest(run_dir)
    m["rerank"] = dict(m["rerank"], revision="older")                        # config changed since the run
    runner.write_manifest(run_dir, m)
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    assert "re-ranker settings" in capsys.readouterr().err and len(seen) == 2


class WarmRetriever(RerankRetriever):
    def __init__(self):
        super().__init__()
        self.embedded, self.lexical_built = [], 0

    def embed_query(self, q):
        self.embedded.append(q)
        return [1.0, 0.0]

    def _lexical(self):
        self.lexical_built += 1


@pytest.mark.parametrize("kwargs, lexical", [({}, 0), ({"rerank": "gte", "reranker": FakeReranker()}, 0),
                                             ({"rerank": "gte", "reranker": FakeReranker(),
                                               "rerank_sources": "dense_bm25"}, 1), ({"hybrid": "rrf"}, 1)])
def test_prepare_warms_the_embedder_and_bm25_for_every_run(kwargs, lexical):
    r = WarmRetriever()
    out = BaselineAnswerer(r, **kwargs).prepare()
    assert r.embedded == ["warm-up"] and r.lexical_built == lexical and "retrieval_warmup_s" in out
    assert ("rerank_load_s" in out) == ("rerank" in kwargs)

