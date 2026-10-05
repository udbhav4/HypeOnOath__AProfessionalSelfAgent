"""QA (tester-swarm): retriever on a REAL Chroma collection (where filters,
k > count, sibling parts), run_eval.run with fakes, chat.format_sources.
No network / Ollama: vectors are hand-made, the generator is a fake."""
from __future__ import annotations

import importlib.util

from pathlib import Path

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.core.store import get_client
from hypeonoath.indexing import indexer
from hypeonoath.serving import llm_client
from hypeonoath.serving.chat import format_sources
from hypeonoath.serving.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]


def _vec(i: int) -> list[float]:
    v = [0.0] * 8
    v[i % 8] = 1.0
    return v


def _chunks() -> list[Chunk]:
    common = dict(source_document="corpus/code/a.ts", content_type="code", project_name="P", project_key="p",
                  project_description="Desc", language="typescript", chunk_method="ast")
    parts = [Chunk(f"part{n}", f"// [P] a.ts > f (part {n}/3)  [lines {n}0-{n}9]\ncode {n}", line_start=n * 10,
                   line_end=n * 10 + 9, part=n, total_parts=3, group_id="g1", symbols=["f"],
                   symbol_spans=[{"name": "f", "line_start": 10, "line_end": 39, "part": n, "total_parts": 3}],
                   start_byte=n * 100, end_byte=n * 100 + 99, **common) for n in (1, 2, 3)]
    prose = Chunk("prose1", "CV > Skills\nPython, SQL", "corpus/converted/cv.md", "prose",
                  heading_path=["CV", "Skills"], line_start=5, line_end=6, project_name=None)
    summary = Chunk("sum1", "Project: P\nDescription: Desc\nCode files: a.ts", "corpus/code/_projects/p",
                    "project_summary", project_key="p", project_name="P", project_description="Desc",
                    line_start=1, line_end=3)
    return [*parts, prose, summary]


@pytest.fixture
def collection(tmp_path):
    chunks = _chunks()
    indexer.rebuild_index(chunks, [_vec(i) for i in range(len(chunks))], chroma_dir=tmp_path / "chroma")
    return get_client(tmp_path / "chroma").get_collection(config.COLLECTION_NAME)


def _retriever(collection, query_index: int):
    return Retriever(collection, embed_query=lambda q: _vec(query_index), count_tokens=lambda t: len(t.split()))


def test_real_chroma_k_larger_than_collection_and_scores(collection):
    items = _retriever(collection, 3).retrieve("skills?", k=50)
    assert items[0].chunk.chunk_id == "prose1"
    assert items[0].score == pytest.approx(1.0, abs=1e-4)
    assert items[0].chunk.heading_path == ["CV", "Skills"]
    assert items[0].chunk.project_name is None                   # None omitted, then restored as default
    assert all(-1.0 <= i.score <= 1.0 for i in items)


def test_real_chroma_where_filters(collection):
    r = _retriever(collection, 0)
    code_items = r.retrieve("q", where={"content_type": "code"})
    assert {i.chunk.content_type for i in code_items} == {"code"}
    assert len(code_items) == 1                                   # 3 parts of one group -> one item
    assert [c.part for c in code_items[0].ordered_chunks] == [1, 2, 3]
    assert code_items[0].ordered_chunks[1].symbol_spans[0]["part"] == 2
    by_project = r.retrieve("q", where={"project_key": "p"})
    assert {i.chunk.content_type for i in by_project} == {"code", "project_summary"}
    lang = r.retrieve("q", where={"$and": [{"content_type": "code"}, {"language": "typescript"}]})
    assert len(lang) == 1


def test_run_eval_with_fake_retriever_and_generator():
    spec = importlib.util.spec_from_file_location("qa_run_eval", ROOT / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    chunk = Chunk("c1", "// hdr\ncode", "corpus/code/a.ts", "code", line_start=3, line_end=9, chunk_method="fallback")

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, question, k):
            self.calls.append((question, k))
            return [RetrievedItem(chunk, 0.81234)] if "code" in question else []

    def generate(prompt):
        if "boom" in prompt.user:
            raise llm_client.LLMUnavailableError("down")
        return "answer"

    questions = [
        {"question": "code q", "expected_answerable": True, "notes": "n"},
        {"question": "boom q", "expected_answerable": False},
    ]
    fake = FakeRetriever()
    results = run_eval.run(fake, questions, generate=generate, k=3)
    assert fake.calls == [("code q", 3), ("boom q", 3)]
    first, second = results
    assert first["answer"] == "answer" and first["error"] is None and first["top_score"] == 0.8123
    assert (first["sources"][0]["line_start"], first["sources"][0]["line_end"]) == (3, 9)
    assert first["sources"][0]["chunk_method"] == "fallback"
    assert second["answer"] is None and second["error"] == "down" and second["top_score"] is None
    assert second["notes"] == ""


def test_real_questions_file_composition():
    spec = importlib.util.spec_from_file_location("qa_run_eval2", ROOT / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    questions = run_eval.load_questions()
    assert 20 <= len(questions) <= 30                                        # Section 8
    assert sum(not q["expected_answerable"] for q in questions) >= 3
    assert sum("agent.ts" in q["question"] or "code" in q["question"].lower() for q in questions) >= 2
    assert any("Gliimr" in q["question"] for q in questions)                 # project-level


def test_load_questions_rejects_bad_entry(tmp_path):
    spec = importlib.util.spec_from_file_location("qa_run_eval3", ROOT / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    bad = tmp_path / "q.yaml"
    bad.write_text("- question: hi\n")
    with pytest.raises(ValueError):
        run_eval.load_questions(bad)


def test_format_sources_variants():
    parts = _chunks()[:3]
    prose = Chunk("p", "x", "corpus/converted/cv.md", "prose")                 # no line numbers
    code = Chunk("c", "x", "corpus/code/a.ts", "code", line_start=4, line_end=None, chunk_method="ast")
    lines = format_sources([
        RetrievedItem(parts[1], 0.5, parts=parts),
        RetrievedItem(prose, 0.25),
        RetrievedItem(code, 0.125),
    ])
    assert lines[0] == "  [1] corpus/code/a.ts:10-39 (code, ast, parts 1-3/3) score=0.500 id=part2"
    assert lines[1] == "  [2] corpus/converted/cv.md (prose) score=0.250 id=p"
    assert lines[2] == "  [3] corpus/code/a.ts:4-4 (code, ast) score=0.125 id=c"
    assert format_sources([]) == []


def test_cli_main_reports_missing_index(monkeypatch, capsys):
    from hypeonoath.serving import chat

    def broken(*a, **k):
        raise RuntimeError("no collection")

    monkeypatch.setattr(chat, "Retriever", broken)
    assert chat.main([]) == 1
    assert "build_index" in capsys.readouterr().err


def test_fence_and_code_file_duplicate_suppressed():
    from hypeonoath.serving.retriever import _code_hash

    code = Chunk("c", "// [P] a.ts > f  [lines 1-3]\nfunction f() {\n  return 1;\n}\n", "corpus/code/a.ts", "code")
    fence = Chunk("d", "README > Usage\n```ts\nfunction f() {\n  return 1;\n}\n```", "corpus/documents/README.md",
                  "code_fence")
    assert _code_hash(code) == _code_hash(fence)
