"""Eval side of F-R5 hybrid search (D109-D111): answerer lever, records, settings, D13, report, CLI (T18-T29)."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import charts, cli, paths, runner, tracking
from hypeonoath.evaluation.answerer import BaselineAnswerer, Retrieved
from hypeonoath.evaluation.report import SETTINGS_ROWS
from tests.evaluation.test_cli_gold import frozen_set
from tests.evaluation.test_runner import FakeAnswerer, cv_chunk
from tests.evaluation.test_tracking_report import manifest

HYBRID_ON = {"mode": "rrf", "candidates": 50, "rrf_k": 60, "bm25_weight": 1.0, "k1": 1.5, "b": 0.75, "epsilon": 0.25}


class HybridRetriever:
    """Fake retriever: hybrid_search returns a fused order where a BM25-only hit sits at rank 2."""

    def __init__(self):
        self.calls = []
        self.collection = type("C", (), {"metadata": {}})()

    def hybrid_search(self, q, depth):
        self.calls.append(("hybrid", q, depth))
        hits = [RetrievedItem(cv_chunk("dense1"), 0.9, fused=2 / 61, dense_rank=1, bm25_rank=2),
                RetrievedItem(cv_chunk("bm25only"), 0.1, fused=1 / 61, bm25_rank=1),
                RetrievedItem(cv_chunk("dense2"), 0.85, fused=1 / 62, dense_rank=2)]
        return hits, [[1.0, 0.0], [0.0, 1.0], [0.99, 0.14]]

    def search(self, q, k):
        self.calls.append(("search", q, k))
        return [RetrievedItem(cv_chunk("dense1"), 0.9)]

    def search_with_embeddings(self, q, k):
        self.calls.append(("search_with_embeddings", q, k))
        return [RetrievedItem(cv_chunk("dense1"), 0.9)], [[1.0, 0.0]]

    def assemble(self, hits):
        return list(hits)


# --- answerer (T18-T21) -------------------------------------------------------------------------

def test_hybrid_off_uses_todays_path():
    r = HybridRetriever()
    BaselineAnswerer(r).retrieve("q")
    BaselineAnswerer(r, mmr_lambda=0.7).retrieve("q")
    assert [c[0] for c in r.calls] == ["search", "search_with_embeddings"]


def test_hybrid_on_uses_the_fused_ranking_and_the_rewritten_query(monkeypatch):
    monkeypatch.setattr("hypeonoath.evaluation.answerer.rewrite_query", lambda q, mode: "REWRITTEN")
    r = HybridRetriever()
    got = BaselineAnswerer(r, hybrid="rrf", query_rewrite="third_person", top_k=2).retrieve("q")
    assert r.calls == [("hybrid", "REWRITTEN", paths.EVAL_DEPTH)]                # one query for both legs (OD-5)
    assert [h.chunk.chunk_id for h in got.ranking] == ["dense1", "bm25only", "dense2"]
    assert [h.chunk.chunk_id for h in got.context] == ["dense1", "bm25only"]


def test_hybrid_mmr_keeps_a_bm25_only_hit_near_the_top():
    got = BaselineAnswerer(HybridRetriever(), hybrid="rrf", mmr_lambda=0.7, fetch_k=3).retrieve("q")
    ids = [h.chunk.chunk_id for h in got.ranking]
    # relevance = fused / max fused (D111): bm25only has rel .5 and is unlike dense1 -> rank 2, while
    # dense2 (cosine .85 but nearly the same vector as dense1) drops; with cosine relevance it would sink.
    assert ids == ["dense1", "bm25only", "dense2"]
    assert got.ranking[0].score == 0.9                                            # score stays cosine (D110)


def test_unknown_mode_and_missing_package_refuse_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="unknown hybrid mode"):
        BaselineAnswerer(HybridRetriever(), hybrid="weighted")

    def missing():
        raise ValueError("--hybrid rrf needs rank_bm25")
    monkeypatch.setattr("hypeonoath.evaluation.answerer.lexical.check_available", missing)
    with pytest.raises(ValueError, match="needs rank_bm25"):
        BaselineAnswerer(HybridRetriever(), hybrid="rrf")
    BaselineAnswerer(HybridRetriever())                                           # none: never checked


def test_settings_record_hybrid():
    assert BaselineAnswerer(HybridRetriever()).settings()["hybrid"] == {"mode": "none"}
    on = BaselineAnswerer(HybridRetriever(), hybrid="rrf").settings()["hybrid"]
    assert on == {"mode": "rrf", "candidates": config.HYBRID_CANDIDATES, "rrf_k": config.HYBRID_RRF_K,
                  "bm25_weight": config.HYBRID_BM25_WEIGHT, "k1": config.BM25_K1, "b": config.BM25_B,
                  "epsilon": config.BM25_EPSILON}


# --- runner (T22) -------------------------------------------------------------------------------

class FusedAnswerer(FakeAnswerer):
    def retrieve(self, question):
        got = super().retrieve(question)
        for rank, item in enumerate(got.ranking, 1):
            item.fused, item.dense_rank, item.bm25_rank = 1 / (60 + rank), rank, None
        return Retrieved(got.ranking, got.context, got.query)


def test_runner_writes_leg_ranks_only_for_hybrid(tmp_path):
    from tests.evaluation.gold_fixtures import VALUES, parse, valid_raw, write_corpus

    corpus, questions = write_corpus(tmp_path), parse(valid_raw())[:2]
    runner.run_questions(questions, FusedAnswerer(), tmp_path / "h", "retrieval", VALUES, corpus, log=lambda m: None)
    runner.run_questions(questions, FakeAnswerer(), tmp_path / "p", "retrieval", VALUES, corpus, log=lambda m: None)
    hybrid, plain = runner.read_answers(tmp_path / "h")[0]["ranking"][0], runner.read_answers(tmp_path / "p")[0]["ranking"][0]
    assert hybrid["fused"] == round(1 / 61, 6) and hybrid["dense_rank"] == 1 and hybrid["bm25_rank"] is None
    assert not {"fused", "dense_rank", "bm25_rank"} & set(plain)                  # non-hybrid record unchanged
    assert set(plain) == {"chunk_id", "document", "line_start", "line_end", "score", "groups"}


# --- tracking / metrics / report / charts (T24-T27) ---------------------------------------------

def _base():
    return {"settings": {k: tracking.setting(manifest(), k) for k in tracking.SETTING_KEYS}}


def test_old_manifest_reads_hybrid_none_and_csv_column():
    assert tracking.setting(manifest(), "hybrid") == {"mode": "none"}
    assert tracking.changed_settings(manifest(hybrid={"mode": "none"}), _base()["settings"]) == []
    assert tracking.csv_row(manifest(), {})["hybrid"] == "none"
    assert tracking.csv_row(manifest(hybrid=HYBRID_ON), {})["hybrid"] == "rrf"


def test_hybrid_is_one_changed_setting_and_index_fp_untouched():
    on = manifest(hybrid=HYBRID_ON)
    assert tracking.changed_settings(on, _base()["settings"]) == ["hybrid"]
    warnings = tracking.comparability_warnings(on, {}, _base())
    assert not any("more than one setting" in w for w in warnings)
    assert "different index than the baseline" not in warnings
    both = manifest(hybrid=HYBRID_ON, mmr={"enabled": True, "lambda": 0.7, "fetch_k": 20})
    assert any("more than one setting" in w for w in tracking.comparability_warnings(both, {}, _base()))
    other_k = manifest(hybrid={**HYBRID_ON, "rrf_k": 20})
    assert tracking.changed_settings(other_k, {k: tracking.setting(on, k) for k in tracking.SETTING_KEYS}) == ["hybrid"]


def test_csv_column_position():
    from hypeonoath.evaluation.metrics import METRICS, RUNS_CSV_COLUMNS
    assert RUNS_CSV_COLUMNS.index("hybrid") == RUNS_CSV_COLUMNS.index("query_rewrite") + 1
    assert len(RUNS_CSV_COLUMNS) == 81 and len(METRICS) == 86   # + F-R6: 3 settings, 5 latency / truncation


def test_report_settings_rows():
    rows = dict(SETTINGS_ROWS)
    on, off = manifest(hybrid=HYBRID_ON), manifest()
    assert (rows["hybrid"](on), rows["hybrid candidates"](on), rows["hybrid rrf_k"](on)) == ("rrf", 50, 60)
    assert (rows["hybrid"](off), rows["hybrid candidates"](off), rows["hybrid rrf_k"](off)) == ("none", None, None)


def test_chart_legend_tag():
    assert charts.experiment_tags(manifest(hybrid=HYBRID_ON)) == ("hybrid-rrf",)
    assert charts.experiment_tags({"hybrid": "rrf"}) == ("hybrid-rrf",)          # runs.csv row
    assert charts.experiment_tags({"hybrid": "none"}) == () and charts.experiment_tags(manifest()) == ()


# --- CLI (T28, T29) -----------------------------------------------------------------------------

def test_cli_passes_hybrid_and_resume_reuses_it(tmp_path, monkeypatch):
    seen = []
    frozen_set(tmp_path, monkeypatch)
    original = cli.make_answerer

    def make(name, **kwargs):
        seen.append(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k != "hybrid"}           # the fake index has no BM25 leg
        fake = original(name, **kwargs)
        settings = fake.settings
        fake.settings = lambda: {**settings(), "hybrid": {**HYBRID_ON}}
        return fake
    monkeypatch.setattr(cli, "make_answerer", make)
    assert cli.main(["run", "--retrieval-only", "--hybrid", "rrf"]) == 0
    assert seen[0]["hybrid"] == "rrf"
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    assert runner.read_manifest(run_dir)["hybrid"]["mode"] == "rrf"
    assert cli.main(["run", "--resume", run_dir.name, "--hybrid", "none"]) == 0      # the manifest wins (D26)
    assert seen[1]["hybrid"] == "rrf"


def test_cli_rejects_an_unknown_hybrid_mode():
    with pytest.raises(SystemExit):
        cli.main(["run", "--retrieval-only", "--hybrid", "weighted"])


def test_metrics_command_prints_hybrid_and_terms(capsys):
    assert cli.main(["metrics", "hybrid"]) == 0
    assert "Reciprocal Rank Fusion" in capsys.readouterr().out
    assert cli.main(["metrics", "rrf"]) == 0 and cli.main(["metrics", "bm25"]) == 0
