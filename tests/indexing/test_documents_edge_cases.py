"""QA (tester-swarm) edge cases for block_loader.py / chunker.py / secret_scan.py.

xfail(strict=True) = confirmed bug; drop the marker once fixed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypeonoath.core import config
from hypeonoath.core.records import Chunk
from hypeonoath.core.store import flatten_metadata, restore_chunk
from hypeonoath.indexing import block_loader, secret_scan
from hypeonoath.indexing.block_loader import LoadedDocument, load_markdown_file, parse_markdown
from hypeonoath.indexing.chunker import Chunker


@pytest.fixture(scope="module")
def chunker(count_tokens):
    return Chunker(count_tokens)


@pytest.fixture
def in_project(tmp_path, monkeypatch):
    """Let rel_to_root() / _source_lines() resolve files under tmp_path."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    secret_scan._source_lines.cache_clear()
    yield tmp_path
    secret_scan._source_lines.cache_clear()


TRICKY = """Setext Title
============

> Quote with a list:
> - a
> - b

### Deep heading (skips h2)

1. first
   - nested item
2. second

## Back to h2

<!-- html comment -->

    indented code block

| Key | Value |
|-----|-------|
| a \\| b | 1 |
"""


def test_tricky_markdown_types_and_heading_paths():
    blocks = parse_markdown(TRICKY, "doc.md")
    types = [(b.block_type, b.heading_path) for b in blocks]
    assert ("heading", ["Setext Title"]) in types
    assert ("prose", ["Setext Title"]) in types                         # blockquote -> one prose block
    assert ("list", ["Setext Title", "Deep heading (skips h2)"]) in types
    # h2 after h3 pops the h3 (level >= 2), keeps the h1
    assert ("table", ["Setext Title", "Back to h2"]) in types
    assert ("code_fence", ["Setext Title", "Back to h2"]) in types       # indented code = atomic block
    nested = next(b for b in blocks if b.block_type == "list")
    assert "nested item" in nested.text                                  # nested list stays with its parent
    table = next(b for b in blocks if b.block_type == "table")
    assert table.meta["rows"][1][0] == "a | b"                           # escaped pipe


def test_crlf_markdown_with_front_matter_lines(in_project):
    path = in_project / "crlf.md"
    path.write_bytes(b"---\r\ndate: 01.01.2026\r\nproject_name: X\r\n---\r\n\r\n# H\r\n\r\nPara one.\r\n")
    doc = load_markdown_file(path)
    assert (doc.project_name, doc.date) == ("X", "01.01.2026")
    prose = next(b for b in doc.blocks if b.block_type == "prose")
    lines = path.read_bytes().decode().split("\r\n")
    assert lines[prose.meta["line_start"] - 1] == "Para one."


@pytest.mark.parametrize("text", ["", "---\ndate: 01.01.2026\n---\n", "# Only a heading\n", "\n\n   \n"])
def test_empty_or_heading_only_documents(chunker, text):
    meta, body, offset = block_loader.split_front_matter(text)
    doc = LoadedDocument("doc.md", parse_markdown(body, "doc.md", offset), meta.get("project_name"), meta.get("date"))
    assert chunker.chunk_document(doc) == []


def test_record_rows_point_at_their_own_file_line(chunker):
    md = "# T\n\nIntro.\n\n| Name | Role |\n|------|------|\n| Ada | Eng |\n| Bob | PM |\n"
    doc = LoadedDocument("doc.md", parse_markdown(md, "doc.md"))
    records = [c for c in chunker.chunk_document(doc) if c.content_type == "record"]
    lines = md.split("\n")
    assert [lines[c.line_start - 1] for c in records] == ["| Ada | Eng |", "| Bob | PM |"]
    assert all(c.line_start == c.line_end for c in records)


def test_long_prose_and_list_respect_budget_and_progress(chunker, count_tokens):
    sentences = " ".join(f"Sentence number {i} talks about retrieval quality and chunking." for i in range(200))
    items = "\n".join(f"- bullet {i} about embedding models and vector stores" for i in range(200))
    md = f"# A\n\n{sentences}\n\n## B\n\n{items}\n"
    doc = LoadedDocument("doc.md", parse_markdown(md, "doc.md"))
    chunks = chunker.chunk_document(doc)
    prefix = count_tokens(config.DOC_PREFIX)
    assert all(count_tokens(c.text) + prefix <= config.PROSE_MAX_TOKENS for c in chunks)
    for i in (0, 199):                                   # nothing dropped
        assert any(f"Sentence number {i} " in c.text for c in chunks)
        assert any(f"bullet {i} " in c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_unsplittable_token_run_still_within_budget(chunker, count_tokens):
    # NB: one huge alphanumeric word is a single [UNK] for nomic's WordPiece
    # (max 100 chars/word), so use punctuation-separated pieces instead.
    blob = "/".join(f"seg{i}" for i in range(800))
    doc = LoadedDocument("doc.md", parse_markdown(f"# A\n\nSee https://example.org/{blob} here.\n", "doc.md"))
    prefix = count_tokens(config.DOC_PREFIX)
    assert all(count_tokens(c.text) + prefix <= config.PROSE_MAX_TOKENS for c in chunker.chunk_document(doc))


def test_record_chunk_within_budget(chunker, count_tokens):
    long_cell = " ".join(f"word{i}" for i in range(400))
    md = f"| Name | Notes |\n|------|-------|\n| Ada | {long_cell} |\n"
    doc = LoadedDocument("doc.md", parse_markdown(md, "doc.md"))
    prefix = count_tokens(config.DOC_PREFIX)
    assert all(count_tokens(c.text) + prefix <= config.PROSE_MAX_TOKENS for c in chunker.chunk_document(doc))


def test_heading_with_separator_round_trips():
    chunk = Chunk("d", "x", "doc.md", "prose", heading_path=["Inputs > Outputs", "Detail"])
    flat = flatten_metadata(chunk)
    assert restore_chunk("d", "x", flat).heading_path == ["Inputs > Outputs", "Detail"]


# --- scan ------------------------------------------------------------------------------

def test_finding_in_hard_wrapped_paragraph_reports_its_own_line(chunker, in_project):
    md = ("# A\n\nThis paragraph is hard wrapped across\nseveral physical lines of the file\n"
          "and only here mentions fake.person@example.org as contact.\n")
    path = in_project / "doc.md"
    path.write_text(md, encoding="utf-8", newline="\n")
    doc = load_markdown_file(path)
    findings = secret_scan.scan_chunks(chunker.chunk_document(doc))
    assert [f.line for f in findings if f.finding_type == "email_address"] == [5]


def test_allowlist_edge_cases(tmp_path):
    f = secret_scan.Finding("corpus/documents/cv.md", 12, "email_address", "pii-regex")
    base = {"file": "corpus/documents/cv.md", "finding_type": "EMAIL_ADDRESS", "reason": "public"}
    assert secret_scan.build_report([f], [base]).allowlisted == 1                       # type case-insensitive
    assert secret_scan.build_report([f], [{**base, "line": 12}]).allowlisted == 1
    assert secret_scan.build_report([f], [{**base, "line": 13}]).allowlisted == 0
    assert secret_scan.build_report([f], [{**base, "file": "corpus/documents/other.md"}]).allowlisted == 0
    empty = tmp_path / "allow.yaml"
    empty.write_text("")
    assert secret_scan.load_allowlist(empty) == []
    partial = tmp_path / "partial.yaml"
    partial.write_text("- {file: a.md}\n- {finding_type: email_address}\n")
    assert secret_scan.load_allowlist(partial) == []                                    # incomplete entries ignored


def test_crlf_text_scan_line_numbers():
    text = "x = 1\r\ncontact = 'someone.fake@example.org'\r\ny = 2\r\n"
    findings = secret_scan.scan_text(text, "corpus/code/a.py")
    assert [(f.line, f.finding_type) for f in findings] == [(2, "email_address")]
