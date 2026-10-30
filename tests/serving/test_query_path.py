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
    from hypeonoath.serving import prompt as prompt_module

    items = [RetrievedItem(code("c1"), 0.9), RetrievedItem(code("c2"), 0.8), RetrievedItem(prose("d"), 0.7)]
    prompt = build_prompt("What does f do?", items)
    assert prompt.user.count("Desc of P") == 1
    assert prompt.user.index("Projects referenced:") < prompt.user.index("[1] source:")
    assert "[1] source: corpus/code/a.ts:10-15 (code; project: P)" in prompt.user
    assert "[3] source: corpus/converted/cv.md (prose)" in prompt.user
    assert prompt.user.endswith("QUESTION: What does f do?\n\n" + prompt_module._PART_REMINDER)
    assert config.OWNER_NAME in prompt.system and "part i/n" in prompt.system


def test_prompt_without_code_has_no_projects_block():
    summary = Chunk("s", "Project: P\nDescription: D", "corpus/code/_projects/p", "project_summary", project_key="p")
    prompt = build_prompt("q", [RetrievedItem(summary, 0.9)])
    assert "Projects referenced" not in prompt.user


def test_prompt_private_note_after_question():
    """F3 mechanism, P0 wording (eval D56): labels once, first-appearance order, ask-in-person guidance."""
    from hypeonoath.serving import prompt as prompt_module

    items = [RetrievedItem(prose("a", "Admitted. Applicant no.: [REDACTED: applicant number]"), 0.9),
             RetrievedItem(prose("b", "Born [REDACTED: date of birth], ref [REDACTED: applicant number]"), 0.8)]
    user = build_prompt("Admitted? Application number?", items).user
    note = user.split("QUESTION: Admitted? Application number?\n\n", 1)[1]
    assert note.startswith("NOTE: the context contains private items: applicant number, date of birth.")
    assert note.count("applicant number") == 1 and note.count("date of birth") == 1
    assert f'"{prompt_module._ASK_IN_PERSON_SAMPLE}"' in note and "answer the other parts normally" in note
    for retired in ("prefers not", "rather not", "write exactly"):
        assert retired not in note
    assert note.endswith("\n\n" + prompt_module._PART_REMINDER)
    assert note.count(prompt_module._PART_REMINDER) == 1


def test_prompt_private_note_holds_labels_only():
    from hypeonoath.core import private_values as pv

    value = pv.values_from_entries([{"id": "x", "decision": "private", "label": "phone number",
                                     "entity_type": "PHONE_NUMBER", "match_kind": "id", "value": "+49 151 2345678",
                                     "normalized": "491512345678", "aliases": [], "surface_forms": ["+49 151 2345678"]}])
    prompt = build_prompt("q", [RetrievedItem(prose("a", "Tel: [REDACTED: phone number]"), 0.9)])
    assert "NOTE: the context contains private items: phone number." in prompt.user
    assert pv.find_private_spans(prompt.as_text(), value) == []        # labels only, never a value


def test_prompt_without_placeholder_has_no_note():
    from hypeonoath.serving.prompt import _PART_REMINDER

    user = build_prompt("q", [RetrievedItem(prose("a"), 0.9)]).user
    assert "NOTE:" not in user and user.endswith("QUESTION: q\n\n" + _PART_REMINDER)


def test_part_reminder_is_always_the_last_line():
    """Fix round FR-2: the every-part reminder ends every user message, once (no context / no NOTE / NOTE)."""
    from hypeonoath.serving.prompt import _PART_REMINDER

    for items in ([], [RetrievedItem(prose("a"), 0.9)],
                  [RetrievedItem(prose("b", "Tel: [REDACTED: phone number]"), 0.9)]):
        user = build_prompt("q", items).user
        assert user.split("\n\n")[-1] == _PART_REMINDER
        assert user.count(_PART_REMINDER) == 1


def test_labels_carry_the_project_and_keep_the_citable_label(doc_registry):
    """Document-registry plan v2, L1 / T-8."""
    from hypeonoath.serving.prompt import _cite_label

    design = Chunk("d1", "Design > ADRs\nfact", "corpus/documents/design.md", "record")
    cv = prose("c1")                                                   # corpus/converted/cv.md, no project
    items = [RetrievedItem(design, 0.9), RetrievedItem(code("k1"), 0.8), RetrievedItem(cv, 0.7)]
    user = build_prompt("q", items).user
    assert "[1] source: corpus/documents/design.md (record; project: Chatbot)" in user
    assert "[2] source: corpus/code/a.ts:10-15 (code; project: P)" in user
    assert "[3] source: corpus/converted/cv.md (prose)\n" in user          # no project -> label as before
    assert [_cite_label(i) for i in items] == ["corpus/documents/design.md", "corpus/code/a.ts:10-15",
                                               "corpus/converted/cv.md"]


def test_projects_and_documents_blocks(doc_registry):
    """L2 + L3 / T-9: document projects listed once, each document described once, blocks before [1]."""
    from hypeonoath.serving import prompt as prompt_module

    design_a = Chunk("d1", "Design\nA", "corpus/documents/design.md", "record")
    design_b = Chunk("d2", "Design\nB", "corpus/documents/design.md", "prose")
    cv = Chunk("c1", "CV\nPhone [REDACTED: phone number]", "corpus/converted/cv.md", "prose")
    items = [RetrievedItem(c, s) for c, s in ((design_a, 0.9), (code("k1"), 0.8), (design_b, 0.7), (cv, 0.6))]
    user = build_prompt("q", items).user
    context = user.split("\n\nQUESTION:")[0]
    projects, documents = context.split("\n\n")[0], context.split("\n\n")[1]
    assert projects == "CONTEXT:\nProjects referenced:\n- Chatbot: A chatbot project.\n- P: Desc of P"
    assert documents == ("Documents referenced:\n- corpus/documents/design.md: Design document of the chatbot.\n"
                         "- corpus/converted/cv.md: My CV.")
    assert user.index("Documents referenced:") < user.index("[1] source:")
    assert user.index("QUESTION: q") < user.index("NOTE:")
    assert user.split("\n\n")[-1] == prompt_module._PART_REMINDER


def test_no_blocks_without_registry_entries_or_for_code_only(doc_registry, caplog):
    unknown = Chunk("u1", "X\nfact", "corpus/documents/unknown.md", "prose")
    summary = Chunk("s", "Project: P\nDescription: D", "corpus/code/_projects/p", "project_summary", project_key="p")
    user = build_prompt("q", [RetrievedItem(unknown, 0.9), RetrievedItem(summary, 0.8)]).user
    assert "Documents referenced" not in user and "Projects referenced" not in user
    assert "(prose)\n" in user                                         # unknown document: label unchanged
    assert "corpus/documents/unknown.md" in caplog.text               # warned once, path only


def test_prompt_split_group_label():
    parts = [code(f"p{n}", n, 2, "g") for n in (1, 2)]
    prompt = build_prompt("q", [RetrievedItem(parts[0], 0.9, parts)])
    assert "corpus/code/a.ts:10-25 (code, parts 1-2 of 2; project: P)" in prompt.user


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
            self.wfile.write(json.dumps({"message": {"content": "  answer [x]  "}, "total_duration": 5,
                                         "eval_count": 3, "done": True}).encode())
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
    result = llm_client.generate(prompt, model="m", base_url=fake_ollama)
    assert result.text == "answer [x]"
    assert result.stats == {"total_duration": 5, "eval_count": 3}          # D25: only the known int stats
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


# --- eval plan v15, Section 8: prompt version parameter, chat --prompt, mmr -------------------------

def test_build_prompt_version_parameter(monkeypatch):
    from hypeonoath.serving import prompt as prompt_module

    monkeypatch.setitem(prompt_module.PROMPTS, "vtest", "TEST PROMPT for {owner_name}")
    assert build_prompt("q", [], "vtest").system == f"TEST PROMPT for {config.OWNER_NAME}"
    assert build_prompt("q", []).system == build_prompt("q", [], config.PROMPT_VERSION).system
    with pytest.raises(ValueError, match="unknown PROMPT_VERSION"):
        build_prompt("q", [], "v404")


def test_chat_prompt_flag_passes_the_version(monkeypatch, capsys):
    import io

    from hypeonoath.serving import chat

    seen = []
    monkeypatch.setattr(chat, "Retriever", lambda: type("R", (), {"retrieve": lambda self, q, k: []})())
    monkeypatch.setattr(chat, "build_prompt", lambda q, items, version: seen.append(version) or build_prompt(q, items))
    monkeypatch.setattr(llm_client, "generate", lambda p: llm_client.GenerationResult("fine"))
    monkeypatch.setattr("sys.stdin", io.StringIO("hi\n"))
    assert chat.main(["--prompt", "v1"]) == 0 and seen == ["v1"]
    with pytest.raises(SystemExit):
        chat.main(["--prompt", "v404"])


def test_mmr_rejects_bad_input_and_keeps_k():
    from hypeonoath.serving.retriever import mmr

    hits = [RetrievedItem(prose("a"), 0.9), RetrievedItem(prose("b"), 0.8), RetrievedItem(prose("c"), 0.1)]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert [i.chunk.chunk_id for i in mmr(hits, vectors, 2, 0.3)] == ["a", "c"]       # b is a duplicate of a
    assert [i.chunk.chunk_id for i in mmr(hits, vectors, 3, 1.0)] == ["a", "b", "c"]
    assert mmr([], [], 3, 0.5) == []
    with pytest.raises(ValueError):
        mmr(hits, vectors, 2, 1.5)
    with pytest.raises(ValueError):
        mmr(hits, vectors[:2], 2, 0.5)


def test_unreadable_ollama_body_is_llm_unavailable(monkeypatch):
    class Broken:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>not json"

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda *a, **k: Broken())
    with pytest.raises(llm_client.LLMUnavailableError, match="unreadable response"):
        llm_client.generate(build_prompt("q", []))
