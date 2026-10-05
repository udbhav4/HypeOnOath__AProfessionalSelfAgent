"""retriever.py, prompt.py, llm_client.py with a fake collection / fake Ollama."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, RetrievedItem
from hypeonoath.core.store import flatten_metadata
from hypeonoath.serving import llm_client
from hypeonoath.serving.prompt import build_prompt
from hypeonoath.serving.retriever import Retriever


def code(cid, part=None, total=None, group=None, text=None, key="p"):
    return Chunk(cid, text or f"// [P] a.ts > f{' (part %s/%s)' % (part, total) if part else ''}  [lines 1-2]\nbody {cid}",
                 "corpus/code/a.ts", "code", project_name="P", project_key=key, project_description="Desc of P",
                 line_start=10 * (part or 1), line_end=10 * (part or 1) + 5, part=part, total_parts=total, group_id=group)


def prose(cid, text="CV\nfact"):
    return Chunk(cid, text, "corpus/converted/cv.md", "prose")


class FakeCollection:
    def __init__(self, hits, all_chunks):
        self.hits, self.all = hits, all_chunks

    def query(self, query_embeddings, n_results, where, include):
        hits = self.hits[:n_results]
        return {"ids": [[c.chunk_id for c, _ in hits]], "documents": [[c.text for c, _ in hits]],
                "metadatas": [[flatten_metadata(c) for c, _ in hits]],
                "distances": [[1 - s for _, s in hits]]}

    def get(self, where, include):
        found = [c for c in self.all if c.group_id == where["group_id"]]
        return {"ids": [c.chunk_id for c in found], "documents": [c.text for c in found],
                "metadatas": [flatten_metadata(c) for c in found]}


def retriever(hits, all_chunks=(), count=lambda t: len(t.split())):
    return Retriever(FakeCollection(hits, list(all_chunks)), embed_query=lambda q: [0.0], count_tokens=count)


def test_scores_and_order():
    items = retriever([(prose("a"), 0.9), (prose("b", "CV\nother"), 0.5)]).retrieve("q")
    assert [(i.chunk.chunk_id, round(i.score, 3)) for i in items] == [("a", 0.9), ("b", 0.5)]


def test_sibling_parts_fetched_in_order_once():
    parts = [code(f"p{n}", n, 3, "g") for n in (3, 1, 2)]
    items = retriever([(parts[0], 0.9), (parts[2], 0.8)], parts).retrieve("q")
    assert len(items) == 1                                       # group fetched once
    assert [c.part for c in items[0].ordered_chunks] == [1, 2, 3]


def test_sibling_fetch_capped_to_neighbours(monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 10)
    parts = [code(f"p{n}", n, 5, "g") for n in range(1, 6)]
    items = retriever([(parts[2], 0.9)], parts).retrieve("q")
    assert [c.part for c in items[0].ordered_chunks] == [2, 3, 4]


def test_duplicate_code_suppressed():
    a = code("a", text="// [P] a.ts > f  [lines 1-2]\nsame body")
    b = code("b", text="// [P] b.ts > f  [lines 5-6]\nsame body")
    assert [i.chunk.chunk_id for i in retriever([(a, 0.9), (b, 0.8)]).retrieve("q")] == ["a"]


def test_prompt_layout_projects_once_and_citations():
    items = [RetrievedItem(code("c1"), 0.9), RetrievedItem(code("c2"), 0.8), RetrievedItem(prose("d"), 0.7)]
    prompt = build_prompt("What does f do?", items)
    assert prompt.user.count("Desc of P") == 1
    assert prompt.user.index("Projects referenced:") < prompt.user.index("[1] source:")
    assert "[1] source: corpus/code/a.ts:10-15 (code)" in prompt.user
    assert "[3] source: corpus/converted/cv.md (prose)" in prompt.user
    assert prompt.user.rstrip().endswith("QUESTION: What does f do?")
    assert config.REFUSAL_PHRASE in prompt.system and "part i/n" in prompt.system


def test_prompt_without_code_has_no_projects_block():
    summary = Chunk("s", "Project: P\nDescription: D", "corpus/code/_projects/p", "project_summary", project_key="p")
    prompt = build_prompt("q", [RetrievedItem(summary, 0.9)])
    assert "Projects referenced" not in prompt.user


def test_prompt_split_group_label():
    parts = [code(f"p{n}", n, 2, "g") for n in (1, 2)]
    prompt = build_prompt("q", [RetrievedItem(parts[0], 0.9, parts)])
    assert "corpus/code/a.ts:10-25 (code, parts 1-2 of 2)" in prompt.user


# --- llm_client against a fake Ollama ---------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    status = 200
    seen: dict = {}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen = body
        self.send_response(self.status)
        self.end_headers()
        if self.status == 200:
            self.wfile.write(json.dumps({"message": {"content": "  answer [x]  "}}).encode())
        else:
            self.wfile.write(b'{"error": "model not found"}')

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_generate_sends_system_user_and_options(fake_ollama):
    _Handler.status = 200
    prompt = build_prompt("q", [])
    assert llm_client.generate(prompt, model="m", base_url=fake_ollama) == "answer [x]"
    sent = _Handler.seen
    assert sent["model"] == "m" and sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["options"] == {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX}


def test_missing_model_message(fake_ollama):
    _Handler.status = 404
    with pytest.raises(llm_client.LLMUnavailableError, match="ollama pull m"):
        llm_client.generate(build_prompt("q", []), model="m", base_url=fake_ollama)


def test_unreachable_ollama():
    with pytest.raises(llm_client.LLMUnavailableError, match="Cannot reach Ollama"):
        llm_client.generate(build_prompt("q", []), base_url="http://127.0.0.1:9", timeout=2)
