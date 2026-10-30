"""gold.py (review-gold, freeze) + cli.py (validate, run, score, compare, promote, metrics);
eval plan v15, 3.3, 9; D49, D71; owner decision B-2 (full stage refused until Stage D)."""
from __future__ import annotations

import csv
import json

import pytest
import yaml

from hypeonoath.evaluation import cli, gold, paths, runner
from hypeonoath.evaluation.questions import load_file
from tests.evaluation.gold_fixtures import VALUES, context, valid_raw
from tests.evaluation.test_runner import FakeAnswerer

HEADER = "# drafts header comment\n# second line\n\n"


def write_drafts(raw, status="draft"):
    for r in raw:
        r["status"] = status
    paths.DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"questions": raw}, sort_keys=False, allow_unicode=True)
    paths.DRAFTS_PATH.write_text(HEADER + body.replace("\n- id:", "\n\n# entry\n- id:"), encoding="utf-8")


def statuses():
    return [q.status for q in load_file(paths.DRAFTS_PATH)[1]]


# --- review-gold ----------------------------------------------------------------------------

def test_review_writes_each_answer_keeps_comments_and_resumes():
    write_drafts(valid_raw(4))
    answers, shown, edited = iter(["y", "?", "e", "n", "q"]), [], []
    counts = gold.review(ask=lambda _: next(answers), say=shown.append, edit=edited.append)
    assert statuses() == ["approved", "rejected", "draft", "draft"]
    assert counts == {"approved": 1, "rejected": 1, "left": 2} and edited == [paths.DRAFTS_PATH]
    text = paths.DRAFTS_PATH.read_text(encoding="utf-8")
    assert text.startswith(HEADER) and text.count("# entry") == 4
    assert any("Checklist:" in s for s in shown) and any("Please answer" in s for s in shown)
    answers = iter(["y", "y"])
    assert gold.review(ask=lambda _: next(answers), say=lambda s: None, edit=lambda p: None)["left"] == 0
    assert statuses() == ["approved", "rejected", "approved", "approved"]


def test_set_status_only_touches_that_entry():
    text = "- id: q1\n  status: draft\n  parts:\n    - id: a\n      status: x\n- id: q2\n  status: draft\n"
    out = gold.set_status(text, "q2", "approved")
    assert out == text.replace("q2\n  status: draft", "q2\n  status: approved")
    with pytest.raises(KeyError):
        gold.set_status(text, "q9", "approved")
    with pytest.raises(KeyError):
        gold.set_status("- id: q1\n  parts: []\n", "q1", "approved")


# --- freeze (G4, V14) -----------------------------------------------------------------------

def test_freeze_refuses_unreviewed_drafts(tmp_path):
    write_drafts(valid_raw())
    assert "70 drafts not reviewed" in gold.freeze(context(tmp_path), privacy_fp="fp")[0]
    assert not paths.QUESTIONS_PATH.exists()


def test_freeze_refuses_an_invalid_approved_set(tmp_path):
    raw = valid_raw()
    write_drafts(raw, "approved")
    text = paths.DRAFTS_PATH.read_text(encoding="utf-8")
    for qid in ("q070", "q069"):
        text = gold.set_status(text, qid, "rejected")
    paths.DRAFTS_PATH.write_text(text, encoding="utf-8")
    assert any("V1 68 approved" in p for p in gold.freeze(context(tmp_path), privacy_fp="fp"))


def test_freeze_writes_meta_and_approved_only_and_archives_the_old_file(tmp_path):
    raw = valid_raw(71)
    raw[-1]["parts"] = raw[-2]["parts"]
    write_drafts(raw, "approved")
    paths.DRAFTS_PATH.write_text(gold.set_status(paths.DRAFTS_PATH.read_text(encoding="utf-8"), "q071", "rejected"),
                                 encoding="utf-8")
    paths.QUESTIONS_PATH.write_text("- question: old\n  expected_answerable: true\n", encoding="utf-8")
    assert gold.freeze(context(tmp_path), privacy_fp="fp123") == []
    meta, qs, errors = load_file(paths.QUESTIONS_PATH)
    assert errors == [] and len(qs) == 70 and all(q.status == "approved" for q in qs)
    assert meta["privacy_fp"] == "fp123" and len(meta["question_set_sha256"]) == 64
    assert "old" in paths.QUESTIONS_V0_PATH.read_text(encoding="utf-8")


# --- run (B-2, D49, D71) ----------------------------------------------------------------------

def frozen_set(tmp_path, monkeypatch, fp="fp123"):
    ctx = context(tmp_path)
    write_drafts(valid_raw(), "approved")
    assert gold.freeze(ctx, privacy_fp="fp123") == []
    monkeypatch.setattr(cli, "default_context", lambda: ctx)
    monkeypatch.setattr(cli, "current_privacy_fp", lambda: fp)
    monkeypatch.setattr(cli, "load_private_values", lambda: VALUES)
    monkeypatch.setattr(cli, "_index_ids", lambda answerer: ["c1", "c2"])
    monkeypatch.setattr(cli, "_count_tokens", lambda: (lambda text: len(text.split())))
    made = []

    def make(name, **kwargs):
        fake = FakeAnswerer()
        fake.settings = lambda: {"prompt_version": kwargs["prompt_version"], "prompt_sha256": "p",
                                 "ollama_model": "m", "top_k": kwargs["top_k"], "eval_depth": 20,
                                 "mmr": {"enabled": False, "lambda": None, "fetch_k": 20}}
        made.append(fake)
        return fake
    monkeypatch.setattr(cli, "make_answerer", make)
    return made


def test_full_stage_is_refused_until_stage_d(capsys):
    assert cli.main(["run"]) == 2
    assert "Stage D" in capsys.readouterr().err
    assert cli.main(["run", "--retrieval-only", "--pin-contexts", "r0"]) == 2


def test_retrieval_run_end_to_end_records_the_answerer(tmp_path, monkeypatch, capsys):
    made = frozen_set(tmp_path, monkeypatch)
    assert cli.main(["run", "--retrieval-only", "--note", "first", "--k", "5"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    manifest = runner.read_manifest(run_dir)
    assert manifest["answerer"] == "fake" and manifest["stage"] == "retrieval" and manifest["top_k"] == 5
    assert manifest["generation_status"] == "retrieval_only" and manifest["privacy_fp"] == "fp123"
    assert manifest["index_fp"] == runner.index_fingerprint(["c1", "c2"])
    assert made[0].generated == [] and len(made[0].retrieved) == 70
    with paths.RUNS_CSV_PATH.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["answerer"] == "fake" and row["note"] == "first" and row["answer_rate"] == "" and row["n"] == "70"
    assert (run_dir / "report.md").exists() and json.loads((run_dir / "scores.json").read_text())["stage"] == "retrieval"
    assert "hit@8" in capsys.readouterr().out


def test_run_refuses_on_a_changed_privacy_fingerprint_before_anything_runs(tmp_path, monkeypatch, capsys):
    made = frozen_set(tmp_path, monkeypatch, fp="changed")
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--retrieval-only"])
    assert exc.value.code == 1 and "privacy decisions changed" in capsys.readouterr().err
    assert made == [] and not paths.RUNS_DIR.exists()


def test_resume_continues_and_refuses_on_a_changed_fingerprint(tmp_path, monkeypatch, capsys):
    frozen_set(tmp_path, monkeypatch)
    assert cli.main(["run", "--retrieval-only"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    answers = (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (run_dir / "answers.jsonl").write_text("\n".join(answers[:30]) + "\n", encoding="utf-8")
    assert cli.main(["run", "--resume", run_dir.name]) == 0
    assert len(runner.read_answers(run_dir)) == 70
    manifest = runner.read_manifest(run_dir)
    assert manifest["resumed_from"] == "retrieval_only" and manifest["generation_status"] == "retrieval_only"
    monkeypatch.setattr(cli, "_index_ids", lambda answerer: ["c1", "c2", "c3"])
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    assert "index_fp changed" in capsys.readouterr().err


def test_score_compare_and_promote(tmp_path, monkeypatch, capsys):
    frozen_set(tmp_path, monkeypatch)
    assert cli.main(["run", "--retrieval-only"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    (run_dir / "report.md").unlink()
    assert cli.main(["score", run_dir.name]) == 0 and (run_dir / "report.md").exists()
    assert cli.main(["compare", run_dir.name, run_dir.name]) == 0
    assert "hit@8" in capsys.readouterr().out
    assert cli.main(["promote", run_dir.name, "--note", "x"]) == 1
    assert "full run" in capsys.readouterr().err and not paths.BASELINE_PATH.exists()
    with pytest.raises(SystemExit):
        cli.main(["score", "missing-run"])


# --- validate / metrics ------------------------------------------------------------------------

def test_validate_drafts_and_frozen_set(tmp_path, monkeypatch, capsys):
    ctx = context(tmp_path)
    monkeypatch.setattr(cli, "default_context", lambda: ctx)
    monkeypatch.setattr(cli, "current_privacy_fp", lambda: "fp123")
    write_drafts(valid_raw())
    assert cli.main(["validate"]) == 0 and "drafts.yaml: 70 entries, OK" in capsys.readouterr().out
    raw = valid_raw()
    raw[0]["category"] = "trivia"
    write_drafts(raw)
    assert cli.main(["validate"]) == 1 and "V3 unknown category" in capsys.readouterr().out

    write_drafts(valid_raw(), "approved")
    assert cli.main(["validate", "--freeze"]) == 0
    assert cli.main(["validate", str(paths.QUESTIONS_PATH)]) == 0
    monkeypatch.setattr(cli, "current_privacy_fp", lambda: "other")
    assert cli.main(["validate", str(paths.QUESTIONS_PATH)]) == 1
    assert "V14 privacy decisions changed" in capsys.readouterr().out


def test_metrics_command_prints_and_regenerates(capsys):
    assert cli.main(["metrics", "mrr"]) == 0
    assert "mrr" in capsys.readouterr().out and paths.METRICS_MD_PATH.exists()
    with pytest.raises(SystemExit):
        cli.main(["metrics", "not-a-metric"])


def test_resume_refuses_on_a_changed_privacy_fingerprint(tmp_path, monkeypatch, capsys):
    """Section 11 'privacy gate': --resume refuses (before anything runs) when the private decisions changed."""
    made = frozen_set(tmp_path, monkeypatch)
    assert cli.main(["run", "--retrieval-only"]) == 0
    (run_dir,) = list(paths.RUNS_DIR.iterdir())
    answers = (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (run_dir / "answers.jsonl").write_text("\n".join(answers[:10]) + "\n", encoding="utf-8")
    monkeypatch.setattr(cli, "current_privacy_fp", lambda: "changed")
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--resume", run_dir.name])
    assert exc.value.code == 1 and "privacy decisions changed" in capsys.readouterr().err
    assert len(made) == 1 and len(runner.read_answers(run_dir)) == 10       # no answerer built, nothing appended
