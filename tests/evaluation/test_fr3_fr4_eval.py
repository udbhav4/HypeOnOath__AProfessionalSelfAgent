"""Eval side of F-R4 / F-R3 (eval plan v29, D94, D95, D98): flags, settings, retrieval_query, D13 count, legends."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import charts, cli, paths, runner, tracking
from hypeonoath.evaluation.answerer import BaselineAnswerer, Retrieved, index_label
from tests.evaluation.test_cli_gold import frozen_set
from tests.evaluation.test_runner import FakeAnswerer, cv_chunk
from tests.evaluation.test_tracking_report import manifest


class RecordingRetriever:
    """Records the query it was searched with; carries collection metadata like a Chroma collection."""

    def __init__(self, metadata=None):
        self.queries = []
        self.collection = type("C", (), {"metadata": metadata or {}})()

    def search(self, q, k):
        self.queries.append(q)
        return [RetrievedItem(cv_chunk("c1"), 0.9)]

    def search_with_embeddings(self, q, k):
        self.queries.append(q)
        return [RetrievedItem(cv_chunk("c1"), 0.9)], [[1.0]]

    def assemble(self, hits):
        return list(hits)


# --- answerer -----------------------------------------------------------------------------------------

def test_rewrite_changes_only_the_retrieval_query(monkeypatch):
    monkeypatch.setattr("hypeonoath.evaluation.answerer.rewrite_query",
                        lambda q, mode: q.replace("you", "he") if mode == "third_person" else q)
    r = RecordingRetriever()
    got = BaselineAnswerer(r, query_rewrite="third_person").retrieve("Are you good?")
    assert r.queries == ["Are he good?"] and got.query == "Are he good?"
    r2 = RecordingRetriever()
    got = BaselineAnswerer(r2).retrieve("Are you good?")                    # default none (D98)
    assert r2.queries == ["Are you good?"] and got.query is None


def test_rewrite_also_feeds_mmr(monkeypatch):
    monkeypatch.setattr("hypeonoath.evaluation.answerer.rewrite_query", lambda q, mode: "REWRITTEN")
    r = RecordingRetriever()
    BaselineAnswerer(r, mmr_lambda=0.7, fetch_k=1, query_rewrite="third_person").retrieve("q")
    assert r.queries == ["REWRITTEN"]


def test_settings_record_rewrite_index_dir_and_index_flags(tmp_path):
    a = BaselineAnswerer(RecordingRetriever({"small_sections_merged": True}), query_rewrite="third_person",
                         index_dir=config.PROJECT_ROOT / "index" / "chroma_fr3_merge")
    s = a.settings()
    assert s["query_rewrite"] == "third_person" and s["index_dir"] == "index/chroma_fr3_merge"
    assert s["small_sections_merged"] is True and s["doc_description_in_chunks"] is False
    plain = BaselineAnswerer(RecordingRetriever()).settings()
    assert plain["query_rewrite"] == "none" and plain["index_dir"] == "index/chroma"
    assert plain["small_sections_merged"] is False and plain["doc_description_in_chunks"] is False
    assert index_label(tmp_path) == tmp_path.resolve().as_posix()              # outside the project: absolute


def test_unknown_rewrite_and_missing_index_folder(tmp_path):
    with pytest.raises(ValueError, match="unknown query rewrite"):
        BaselineAnswerer(RecordingRetriever(), query_rewrite="name")
    a = BaselineAnswerer(index_dir=tmp_path / "missing")                     # lazy: fine until the index is read
    with pytest.raises(ValueError, match="no index in"):
        a.retriever
    assert not (tmp_path / "missing").exists()                               # no empty folder created


# --- runner -------------------------------------------------------------------------------------------

class RewritingAnswerer(FakeAnswerer):
    def retrieve(self, question):
        got = super().retrieve(question)
        return Retrieved(got.ranking, got.context, "rewritten " + question)


def test_runner_stores_retrieval_query_only_when_rewritten(tmp_path):
    from tests.evaluation.gold_fixtures import VALUES, parse, valid_raw, write_corpus

    corpus, questions = write_corpus(tmp_path), parse(valid_raw())[:2]
    runner.run_questions(questions, RewritingAnswerer(), tmp_path / "a", "retrieval", VALUES, corpus, log=lambda m: None)
    runner.run_questions(questions, FakeAnswerer(), tmp_path / "b", "retrieval", VALUES, corpus, log=lambda m: None)
    a, b = runner.read_answers(tmp_path / "a"), runner.read_answers(tmp_path / "b")
    assert a[0]["retrieval_query"] == "rewritten " + questions[0].question
    assert "retrieval_query" not in b[0]


# --- tracking: settings, defaults, D13 count ----------------------------------------------------------

def test_old_manifests_read_the_defaults():
    old = manifest()
    assert tracking.setting(old, "query_rewrite") == "none" and tracking.setting(old, "small_sections_merged") is False
    assert tracking.changed_settings(manifest(query_rewrite="none", small_sections_merged=False), old) == []
    row = tracking.csv_row(old, {})
    assert row["query_rewrite"] == "none" and row["small_sections_merged"] == "false"


def test_rewrite_run_is_one_changed_setting():
    base = {"settings": {k: tracking.setting(manifest(), k) for k in tracking.SETTING_KEYS}}
    new = manifest(query_rewrite="third_person")
    assert tracking.changed_settings(new, base["settings"]) == ["query_rewrite"]
    assert not any("more than one setting" in w for w in tracking.comparability_warnings(new, {}, base))


def test_index_flag_change_does_not_count_index_fp_twice():
    base = {"settings": {k: tracking.setting(manifest(), k) for k in tracking.SETTING_KEYS}}
    merged = manifest(small_sections_merged=True, index_fp="9:y")
    warnings = tracking.comparability_warnings(merged, {}, base)
    assert "different index than the baseline" in warnings
    assert not any("more than one setting" in w for w in warnings)
    both = manifest(small_sections_merged=True, query_rewrite="third_person", index_fp="9:y")
    assert any("more than one setting" in w for w in tracking.comparability_warnings(both, {}, base))
    other_index = manifest(index_fp="9:y", top_k=12)                         # index change without a flag: counted
    assert any("more than one setting" in w for w in tracking.comparability_warnings(other_index, {}, base))


# --- charts legend ------------------------------------------------------------------------------------

def test_experiment_tags_from_manifest_and_csv_row():
    assert charts.experiment_tags(manifest()) == ()
    assert charts.experiment_tags(manifest(query_rewrite="third_person", small_sections_merged=True,
                                           doc_description_in_chunks=True)) == ("3rd-person", "merged", "desc")
    row = {"query_rewrite": "none", "small_sections_merged": "true", "doc_description_in_chunks": "false"}
    assert charts.experiment_tags(row) == ("merged",)
    assert charts.settings_label(12, None, ("3rd-person",)) == "k12 3rd-person"
    assert charts.settings_label(12, "0.7") == "k12 λ0.7"


# --- CLI ----------------------------------------------------------------------------------------------

def test_cli_passes_rewrite_and_index_dir_and_resume_reuses_them(tmp_path, monkeypatch):
    seen = []
    made = frozen_set(tmp_path, monkeypatch)
    original = cli.make_answerer

    def make(name, **kwargs):
        seen.append(kwargs)
        fake = original(name, **kwargs)
        settings = fake.settings
        fake.settings = lambda: {**settings(), "query_rewrite": kwargs["query_rewrite"], "index_dir": "index/v"}
        return fake
    monkeypatch.setattr(cli, "make_answerer", make)
    assert cli.main(["run", "--retrieval-only", "--query-rewrite", "third_person", "--index-dir", "index/v"]) == 0
    assert seen[0]["query_rewrite"] == "third_person" and str(seen[0]["index_dir"]).replace("\\", "/") == "index/v"
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    assert runner.read_manifest(run_dir)["query_rewrite"] == "third_person"
    assert cli.main(["run", "--resume", run_dir.name]) == 0
    assert seen[1]["query_rewrite"] == "third_person"
    assert seen[1]["index_dir"] == config.PROJECT_ROOT / "index/v"
    assert made                                                             # frozen_set's factory was used


def test_cli_rejects_an_unknown_rewrite_mode():
    with pytest.raises(SystemExit):
        cli.main(["run", "--retrieval-only", "--query-rewrite", "owner_name"])


def test_cli_reports_a_missing_index_folder(tmp_path, monkeypatch, capsys):
    frozen_set(tmp_path, monkeypatch)

    def missing(answerer):
        raise ValueError("index folder x does not exist; build it first")
    monkeypatch.setattr(cli, "_index_ids", missing)
    assert cli.main(["run", "--retrieval-only"]) == 1
    assert "does not exist" in capsys.readouterr().err and not paths.RUNS_DIR.exists()


def test_rewrite_without_parser_refuses_at_construction(monkeypatch):
    def missing():
        raise ValueError("needs spaCy")
    monkeypatch.setattr("hypeonoath.evaluation.answerer.check_available", missing)
    with pytest.raises(ValueError, match="needs spaCy"):
        BaselineAnswerer(RecordingRetriever(), query_rewrite="third_person")
    BaselineAnswerer(RecordingRetriever())                                   # none: no parser needed


def test_folder_without_an_index_is_refused_and_left_untouched(tmp_path):
    empty = tmp_path / "not_an_index"
    empty.mkdir()
    with pytest.raises(ValueError, match="no index in"):
        BaselineAnswerer(index_dir=empty).retriever
    assert list(empty.iterdir()) == []                                       # no chroma.sqlite3 created
