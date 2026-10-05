"""block_loader.py + chunker.py: per-block typing and per-type rules (Sections 2, 3)."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import Block
from hypeonoath.indexing import block_loader
from hypeonoath.indexing.block_loader import LoadedDocument, parse_markdown, split_front_matter
from hypeonoath.indexing.chunker import Chunker, make_chunks

MD = """---
date: 01.01.2026
project_name: Demo
---

# Title

Intro paragraph.

## Section A

- item one
- item two

| Name | Role |
|------|------|
| Ada  | Eng  |
| Bob  | PM   |

```python
def f():
    return 1
```

## Links

- https://github.com/example
"""


def test_front_matter_split_and_line_offset():
    meta, body, offset = split_front_matter(MD)
    assert meta == {"date": "01.01.2026", "project_name": "Demo"} and offset == 4
    assert body.startswith("\n# Title")


def test_markdown_blocks_types_headings_and_lines():
    meta, body, offset = split_front_matter(MD)
    blocks = parse_markdown(body, "doc.md", offset)
    types = [b.block_type for b in blocks]
    assert types == ["heading", "prose", "heading", "list", "table", "code_fence", "heading", "list"]
    table = blocks[4]
    assert table.heading_path == ["Title", "Section A"]
    assert table.meta["rows"] == [["Name", "Role"], ["Ada", "Eng"], ["Bob", "PM"]] and table.meta["has_header"]
    lines = MD.split("\n")
    assert lines[blocks[1].meta["line_start"] - 1] == "Intro paragraph."
    assert lines[table.meta["line_start"] - 1].startswith("| Name")
    assert blocks[5].text.startswith("```python") and blocks[5].text.endswith("```")
    assert blocks[7].heading_path == ["Title", "Links"]


@pytest.fixture
def chunker(count_tokens):
    return Chunker(count_tokens)


def doc_from(md):
    meta, body, offset = split_front_matter(md)
    return LoadedDocument("doc.md", parse_markdown(body, "doc.md", offset), meta.get("project_name"), meta.get("date"))


def test_chunk_types_and_heading_prefix(chunker):
    chunks = chunker.chunk_document(doc_from(MD))
    kinds = [c.content_type for c in chunks]
    assert kinds == ["prose", "prose", "record", "record", "code_fence", "prose"]
    assert chunks[0].text == "Title\nIntro paragraph."
    record = chunks[2]
    assert record.text == "Title > Section A\nName: Ada\nRole: Eng"
    assert record.line_start == MD.split("\n").index("| Ada  | Eng  |") + 1
    assert all(c.project_name == "Demo" and c.date == "01.01.2026" for c in chunks)
    assert chunks[4].text.endswith("```") and "def f():" in chunks[4].text      # fence intact


def test_prose_windows_within_budget_with_overlap(chunker, count_tokens):
    sentences = " ".join(f"Sentence number {i} talks about topic {i} in some detail." for i in range(200))
    chunks = chunker.chunk_document(doc_from(f"# H\n\n{sentences}\n"))
    assert len(chunks) > 2
    prefix = count_tokens(config.DOC_PREFIX)
    assert all(count_tokens(c.text) + prefix <= config.PROSE_MAX_TOKENS for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_b = b.text.split("\n", 1)[1].split(". ")[0]
        assert first_sentence_b in a.text            # ~10-15% overlap carried over
    assert all(c.text.startswith("H\n") for c in chunks)


def test_no_overlap_across_headings(chunker):
    chunks = chunker.chunk_document(doc_from("# A\n\nAlpha text.\n\n# B\n\nBeta text.\n"))
    assert [c.text for c in chunks] == ["A\nAlpha text.", "B\nBeta text."]


def test_non_record_table_single_chunk(chunker):
    block = Block("table", "| a long first cell that is clearly not a header label at all, it is data | 2020 |\n|---|---|\n| x | y |",
                  "doc.md", ["T"], {"rows": [["a long first cell that is clearly not a header label at all, it is data", "2020"], ["x", "y"]]})
    chunks = chunker.chunk_document(LoadedDocument("doc.md", [block]))
    assert [c.content_type for c in chunks] == ["table"]


@pytest.mark.parametrize("rows, expected", [
    ([["Name", "Role"], ["a", "b"]], True),
    ([["Name", "Role"], ["a"]], False),              # ragged
    ([["2020", "2021"], ["a", "b"]], False),         # numeric header
    ([["Name"]], False),                             # no data rows
])
def test_record_list_heuristic(rows, expected):
    assert Chunker._is_record_list(rows) is expected


def test_oversized_code_fence_kept_and_logged(chunker, monkeypatch, isolated_logs):
    monkeypatch.setattr(config, "EMBED_MAX_TOKENS", 10)
    chunks = chunker.chunk_document(doc_from("# H\n\n```\n" + "x = 1\n" * 50 + "```\n"))
    assert len(chunks) == 1 and chunks[0].content_type == "code_fence"
    assert "oversize_code_fence" in (isolated_logs / "document_loading_log_path.jsonl").read_text()


def test_chunk_ids_stable_and_unique(chunker):
    a = [c.chunk_id for c in chunker.chunk_document(doc_from(MD))]
    b = [c.chunk_id for c in chunker.chunk_document(doc_from(MD))]
    assert a == b and len(set(a)) == len(a)


def test_code_blocks_pass_through(count_tokens):
    meta = {"chunk_id": "abc", "language": "python", "line_start": 1, "line_end": 3, "symbols": ["f"],
            "symbol_spans": [], "chunk_method": "ast", "start_byte": 0, "end_byte": 9, "project_key": "p",
            "project_name": "P", "project_description": "d", "date": None}
    chunks = make_chunks([], [Block("code_chunk", "# h\ncode", "corpus/code/a.py", [], meta)], count_tokens)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.chunk_id, c.content_type, c.text, c.project_key) == ("abc", "code", "# h\ncode", "p")


# --- real corpus (docling branch + Links re-attachment + fallback) -----------------------

def test_real_converted_cv_uses_docling_and_keeps_links():
    md = config.CONVERTED_DIR / "CV_Udbhav.md"
    if not md.exists():
        pytest.skip("converted CV not present")
    doc = block_loader.load_converted_file(md)
    assert doc.project_name == "AI specific CV" and doc.date
    assert any(b.block_type == "table" for b in doc.blocks)
    links = [b for b in doc.blocks if b.heading_path == ["Links"]]
    assert links and "github.com" in links[0].text
    assert all(b.meta.get("line_start") for b in doc.blocks if b.block_type != "heading")


def test_docling_failure_falls_back_to_markdown(tmp_path, monkeypatch, isolated_logs):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    md = tmp_path / "x.md"
    md.write_text("---\ndate: 01.01.2026\n---\n\n## Head\n\nBody text.\n")
    (tmp_path / "x.docling.json").write_text("{not json")
    doc = block_loader.load_converted_file(md)
    assert [b.block_type for b in doc.blocks] == ["heading", "prose"]
    assert "docling_fallback" in (isolated_logs / "document_loading_log_path.jsonl").read_text()
