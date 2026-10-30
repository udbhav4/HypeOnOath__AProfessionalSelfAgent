"""answerer.py + runner.py (eval plan v15, D71, D52, D25, D26; Section 11 'runner')."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.evaluation import runner
from hypeonoath.evaluation.answerer import BaselineAnswerer, Generated, Retrieved, make_answerer
from hypeonoath.serving.llm_client import GenerationResult, LLMUnavailableError
from tests.evaluation.gold_fixtures import APPLICANT, CV, PHONE, VALUES, parse, valid_raw, write_corpus

SECRETS = (PHONE, APPLICANT, "491512345678", "AB7788991", "2345678", "7788991")


def cv_chunk(cid="c1", text="CV\nMSc in AI at Example University", start=1, end=2):
    return Chunk(cid, text, CV, "prose", line_start=start, line_end=end)


class FakeAnswerer:
    name = "fake"

    def __init__(self, replies=None, fail=()):
        self.replies, self.fail = replies or {}, set(fail)
        self.retrieved, self.generated = [], []

    def retrieve(self, question):
        self.retrieved.append(question)
        hits = [RetrievedItem(cv_chunk("c1"), 0.9), RetrievedItem(cv_chunk("c2", "Skills: Python and SQL", 3, 3), 0.4)]
        return Retrieved(hits, hits[:1])

    def generate(self, question, context):
        self.generated.append((question, [i.chunk.chunk_id for i in context]))
        if question in self.fail:
            raise LLMUnavailableError(f"boom while answering; body had {PHONE}")
        return Generated(self.replies.get(question, "I did an MSc in AI [cv]."), {"total_duration": 2_000_000_000,
                         "load_duration": 500_000_000, "prompt_eval_duration": 1_000_000_000,
                         "prompt_eval_count": 100, "eval_duration": 400_000_000, "eval_count": 20}, "PROMPT TEXT")

    def settings(self):
        return {}


@pytest.fixture
def setup(tmp_path):
    corpus = write_corpus(tmp_path)
    return corpus, parse(valid_raw()), tmp_path / "runs" / "r1"


def lines(run_dir):
    return runner.read_answers(run_dir)


def test_retrieval_only_never_calls_generate_and_stores_groups(setup):
    corpus, qs, run_dir = setup
    fake = FakeAnswerer()
    status = runner.run_questions(qs[:3], fake, run_dir, "retrieval", VALUES, corpus, log=lambda m: None,
                                  count_tokens=lambda t: len(t.split()))
    assert status == "retrieval_only" and fake.generated == []
    rec = lines(run_dir)[0]
    assert rec["top1"] == 0.9 and "reply" not in rec and rec["context_tokens"] == 7
    assert rec["ranking"][0]["groups"] == ["a:0"] and rec["ranking"][1]["groups"] == []
    assert rec["context"][0]["chunk"]["chunk_id"] == "c1" and rec["context"][0]["groups"] == ["a:0"]


def test_full_stage_guards_every_reply_and_stores_labels_only(setup):
    corpus, qs, run_dir = setup
    q0, q1 = qs[0].question, qs[1].question
    fake = FakeAnswerer({q0: f"I studied AI. My number is {PHONE}.", q1: f"{APPLICANT}"})
    runner.run_questions(qs[:3], fake, run_dir, "full", VALUES, corpus, log=lambda m: None)
    a, b, c = lines(run_dir)
    assert a["private_hits_pre"] == ["phone number"] and not a["guard_declined"]
    assert "I studied AI." in a["reply"] and a["reply"].endswith("in person.")
    assert b["private_hits_pre"] == ["applicant number"] and b["guard_declined"]
    assert c["private_hits_pre"] == [] and c["reply"] == "I did an MSc in AI [cv]." and c["error"] is None
    raw = (run_dir / "answers.jsonl").read_text(encoding="utf-8")
    assert not any(s in raw for s in SECRETS)                              # no fixture value anywhere


def test_warmup_first_and_untimed(setup):
    corpus, qs, run_dir = setup
    fake = FakeAnswerer()
    runner.run_questions(qs[:2], fake, run_dir, "full", VALUES, corpus, log=lambda m: None)
    assert fake.generated[0] == (runner.WARMUP_QUESTION, [])
    assert [r["id"] for r in lines(run_dir)] == [qs[0].id, qs[1].id]
    t = lines(run_dir)[0]["timings"]
    assert t["ttft_s"] == pytest.approx(1.5) and t["decode_tok_per_s"] == pytest.approx(50.0)


def test_warmup_failure_aborts_before_any_question(setup):
    corpus, qs, run_dir = setup
    fake = FakeAnswerer(fail={runner.WARMUP_QUESTION})
    with pytest.raises(runner.RunAborted) as exc:
        runner.run_questions(qs[:2], fake, run_dir, "full", VALUES, corpus, log=lambda m: None)
    assert lines(run_dir) == [] and PHONE not in str(exc.value)


def test_abort_after_three_errors_in_a_row_then_resume(setup):
    corpus, qs, run_dir = setup
    fake = FakeAnswerer(fail={q.question for q in qs[1:4]})
    with pytest.raises(runner.RunAborted, match="3 LLM errors in a row"):
        runner.run_questions(qs[:6], fake, run_dir, "full", VALUES, corpus, log=lambda m: None)
    done = lines(run_dir)
    assert len(done) == 4 and [d["error"] is None for d in done] == [True, False, False, False]
    assert all(PHONE not in d["error"] for d in done[1:]) and "boom" in done[1]["error"]

    fixed = FakeAnswerer()
    runner.run_questions(qs[:6], fixed, run_dir, "full", VALUES, corpus, log=lambda m: None)
    assert sorted(r["id"] for r in lines(run_dir)) == sorted(q.id for q in qs[:6])
    assert all(r["error"] is None for r in lines(run_dir))                 # errored questions retried (D26)
    assert fixed.retrieved == [q.question for q in qs[1:6]]                # only the clean line is skipped


def test_torn_last_line_is_skipped_but_a_torn_middle_line_fails(setup):
    corpus, qs, run_dir = setup
    runner.run_questions(qs[:2], FakeAnswerer(), run_dir, "retrieval", VALUES, corpus, log=lambda m: None)
    path = run_dir / "answers.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"id": "q00', encoding="utf-8")
    assert len(lines(run_dir)) == 2
    path.write_text('{"id": "q00\n' + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        lines(run_dir)


def test_baseline_rejects_top_k_and_lambda_out_of_range():
    for kwargs in ({"top_k": 0}, {"top_k": 21}, {"mmr_lambda": 1.5}, {"mmr_lambda": -0.1}):
        with pytest.raises(ValueError):
            BaselineAnswerer(FakeRetriever(), **kwargs)


def test_errors_not_in_a_row_do_not_abort(setup):
    corpus, qs, run_dir = setup
    fake = FakeAnswerer(fail={qs[0].question, qs[1].question, qs[3].question, qs[4].question})
    runner.run_questions(qs[:6], fake, run_dir, "full", VALUES, corpus, log=lambda m: None)
    assert len(lines(run_dir)) == 6


def test_pinned_contexts_never_call_retrieve(setup):
    corpus, qs, run_dir = setup
    runner.run_questions(qs[:2], FakeAnswerer(), run_dir, "retrieval", VALUES, corpus, log=lambda m: None)
    pinned = {r["id"]: r for r in lines(run_dir)}
    fake = FakeAnswerer()
    out = run_dir.parent / "r2"
    runner.run_questions(qs[:2], fake, out, "full", VALUES, corpus, pinned=pinned, log=lambda m: None)
    assert fake.retrieved == [] and fake.generated[1] == (qs[0].question, ["c1"])
    rec = lines(out)[0]
    assert rec["pinned"] and rec["context"] == pinned[qs[0].id]["context"]


def test_pinned_run_missing_a_question_aborts(setup):
    corpus, qs, run_dir = setup
    with pytest.raises(runner.RunAborted, match="missing in the pinned run"):
        runner.run_questions(qs[:1], FakeAnswerer(), run_dir, "full", VALUES, corpus, pinned={}, log=lambda m: None)


def test_item_record_round_trip():
    parts = [Chunk("p1", "a", "corpus/code/a.ts", "code", part=1, total_parts=2, group_id="g", symbols=["f"]),
             Chunk("p2", "b", "corpus/code/a.ts", "code", part=2, total_parts=2, group_id="g")]
    item = RetrievedItem(parts[0], 0.5, parts)
    back = runner.item_from_record(json.loads(json.dumps(runner.context_record(item, {}, None))))
    assert back.chunk == item.chunk and back.parts == parts and back.score == 0.5


def test_index_fingerprint_and_resume_problem():
    assert runner.index_fingerprint(["b", "a"]) == runner.index_fingerprint(["a", "b"])
    assert runner.index_fingerprint(["a"]).startswith("1:") and runner.index_fingerprint(["a"]) != runner.index_fingerprint(["b"])
    manifest = {"index_fp": "1:x", "question_set_sha256": "q", "privacy_fp": "p"}
    assert runner.resume_problem(manifest, dict(manifest)) is None
    assert "privacy_fp" in runner.resume_problem(manifest, dict(manifest, privacy_fp="other"))


def test_unknown_stage():
    with pytest.raises(ValueError):
        runner.run_questions([], FakeAnswerer(), None, "fast", VALUES)


# --- BaselineAnswerer -----------------------------------------------------------------------

class FakeRetriever:
    def __init__(self):
        self.hits = [RetrievedItem(cv_chunk(f"c{i}", f"text {i}", i, i), 1.0 - i / 10) for i in range(5)]
        self.vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.1]]

    def search(self, q, k):
        return self.hits[:k]

    def search_with_embeddings(self, q, k):
        return self.hits[:k], self.vectors[:k]

    def assemble(self, hits):
        return list(hits)


def test_baseline_plain_ranking_and_top_k():
    a = BaselineAnswerer(FakeRetriever(), top_k=2)
    got = a.retrieve("q")
    assert [i.chunk.chunk_id for i in got.ranking] == ["c0", "c1", "c2", "c3", "c4"]
    assert [i.chunk.chunk_id for i in got.context] == ["c0", "c1"]


def test_baseline_mmr_lambda_one_equals_plain_and_lower_lambda_diversifies():
    plain = BaselineAnswerer(FakeRetriever(), top_k=3).retrieve("q").ranking
    one = BaselineAnswerer(FakeRetriever(), top_k=3, mmr_lambda=1.0, fetch_k=5).retrieve("q").ranking
    assert [i.chunk.chunk_id for i in one] == [i.chunk.chunk_id for i in plain]
    diverse = BaselineAnswerer(FakeRetriever(), top_k=3, mmr_lambda=0.5, fetch_k=5).retrieve("q").ranking
    assert [i.chunk.chunk_id for i in diverse][:2] == ["c0", "c2"]           # c1 duplicates c0's vector


def test_baseline_generate_uses_prompt_version_and_returns_raw_text():
    seen = []

    def gen(prompt):
        seen.append(prompt)
        return GenerationResult("raw reply", {"eval_count": 3})

    out = BaselineAnswerer(FakeRetriever(), generate=gen, prompt_version="v1").generate("Q?", [])
    assert out.text == "raw reply" and out.stats == {"eval_count": 3} and "QUESTION: Q?" in out.prompt
    assert seen[0].system.startswith("WHO YOU ARE")


def test_baseline_settings_and_bad_arguments():
    s = BaselineAnswerer(FakeRetriever(), top_k=5, mmr_lambda=0.7).settings()
    assert s["top_k"] == 5 and s["eval_depth"] == 20 and s["mmr"] == {"enabled": True, "lambda": 0.7, "fetch_k": 20}
    assert len(s["prompt_sha256"]) == 64 and s["prompt_version"] == "v1"
    with pytest.raises(ValueError, match="unknown PROMPT_VERSION"):
        BaselineAnswerer(FakeRetriever(), prompt_version="v9")
    with pytest.raises(ValueError, match="eval depth"):
        BaselineAnswerer(FakeRetriever(), fetch_k=21)
    with pytest.raises(ValueError, match="unknown answerer"):
        make_answerer("nope")
    assert make_answerer("baseline", retriever=FakeRetriever()).name == "baseline"


def test_pinned_variant_is_guarded_too_and_stores_no_value(setup):
    """Section 11 'runner': the guard runs on every variant's raw text, the pinned-context one included."""
    corpus, qs, run_dir = setup
    runner.run_questions(qs[:2], FakeAnswerer(), run_dir, "retrieval", VALUES, corpus, log=lambda m: None)
    pinned = {r["id"]: r for r in lines(run_dir)}
    out = run_dir.parent / "r2"
    fake = FakeAnswerer({qs[0].question: f"Reach me on {PHONE}."})
    runner.run_questions(qs[:2], fake, out, "full", VALUES, corpus, pinned=pinned, log=lambda m: None)
    rec = lines(out)[0]
    assert rec["pinned"] and rec["private_hits_pre"] == ["phone number"] and rec["guard_declined"]
    assert not any(s in (out / "answers.jsonl").read_text(encoding="utf-8") for s in SECRETS)


def test_logged_progress_and_abort_message_hold_no_value(setup):
    """Privacy: the per-question log line and the abort message carry the redacted error only."""
    corpus, qs, run_dir = setup
    logged = []
    fake = FakeAnswerer(fail={q.question for q in qs[:3]})
    with pytest.raises(runner.RunAborted) as exc:
        runner.run_questions(qs[:4], fake, run_dir, "full", VALUES, corpus, log=logged.append)
    assert len(logged) == 3 and all("boom" in m for m in logged)
    assert not any(s in m for m in logged + [str(exc.value)] for s in SECRETS)
    assert all("[REDACTED: phone number]" in r["error"] for r in lines(run_dir))


def test_wall_s_times_the_generate_call_only(setup, monkeypatch):
    corpus, qs, run_dir = setup
    clock = iter([10.0, 12.5, 99.0])                                       # start, after generate, (never used)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    runner.run_questions(qs[:1], FakeAnswerer(), run_dir, "full", VALUES, corpus, warmup=False, log=lambda m: None)
    assert lines(run_dir)[0]["timings"]["wall_s"] == 2.5
