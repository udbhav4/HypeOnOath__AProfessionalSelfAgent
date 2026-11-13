"""QA review of F-R3 (eval plan v29, D95 / D98): edge cases of the sibling rule, the description line and the
index-folder guard. """
from __future__ import annotations

import pytest

from hypeonoath.core import config, registry
from hypeonoath.core.records import Block
from hypeonoath.indexing import build_index, indexer
from hypeonoath.indexing.block_loader import LoadedDocument
from hypeonoath.indexing.chunker import Chunker, code_block_to_chunk

DOC = "corpus/converted/notes.md"
OFF = {"small_sections_merged": False, "doc_description_in_chunks": False}
MERGE = {**OFF, "small_sections_merged": True}


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(autouse=True)
def small_bounds(monkeypatch):
    monkeypatch.setattr(config, "PROSE_MIN_TOKENS", 30)
    monkeypatch.setattr(config, "PROSE_MAX_TOKENS", 50)
    monkeypatch.setattr(registry, "DOCUMENTS", {"notes.pdf": {"description": "My notes on things.", "project": None}})


def heading(path, title, line):
    return Block("heading", title, DOC, list(path), {"line_start": line, "line_end": line})


def prose(path, n, line):
    return Block("prose", " ".join(["w"] * n) + ".", DOC, list(path), {"line_start": line, "line_end": line})


def table(path, rows, line):
    text = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return Block("table", text, DOC, list(path), {"rows": rows, "has_header": True, "line_start": line,
                                                  "line_end": line + len(rows)})


def run(*blocks, merge=True, desc=False):
    return Chunker(words, merge_small=merge, descriptions=desc).chunk_document(LoadedDocument(DOC, list(blocks)))


# --- sibling rule ---------------------------------------------------------------------------------------

def test_same_named_parent_heading_reopened_is_a_boundary():
    out = run(heading([], "Cert", 1), heading(["Cert"], "x", 2), prose(["Cert", "x"], 3, 3),
              heading([], "Cert", 4), heading(["Cert"], "y", 5), prose(["Cert", "y"], 3, 6),
              prose(["OTHER"], 40, 7))                                    # not a small document
    assert len([c for c in out if c.heading_path[0] == "Cert"]) == 2


def test_merged_rows_with_description_stay_within_max_and_carry_the_line_once():
    rows = [["K", "V"], *[[f"k{i}", "v v v"] for i in range(12)]]
    out = run(table(["T"], rows, 10), desc=True)
    assert len(out) > 1
    for c in out:
        assert words(c.text) + 1 <= config.PROSE_MAX_TOKENS
        assert c.text.count("notes — My notes on things.") == 1 and c.text.split("\n")[1] == "T"
    assert all(c.line_start <= c.line_end for c in out)
    assert [c.line_start for c in out] == sorted(c.line_start for c in out)


def test_merged_section_heading_path_is_the_first_units_and_lines_span_all():
    out = run(prose(["P", "a"], 3, 5), prose(["P", "b"], 3, 9), prose(["P", "c"], 3, 14))
    assert len(out) == 1 and out[0].heading_path == ["P", "a"] and (out[0].line_start, out[0].line_end) == (5, 14)


def test_nested_subsection_between_siblings_breaks_the_run():
    out = run(prose(["P", "a"], 3, 1), prose(["P", "a", "deep"], 3, 2), prose(["P", "b"], 3, 3))
    assert len(out) == 3


def test_flags_off_record_rows_unchanged_with_description_off():
    rows = [["K", "V"], ["a", "1"], ["b", "2"]]
    assert len(run(table(["T"], rows, 1), merge=False)) == 2


def test_code_project_line_comes_on_top_of_the_code_chunker_sizing(monkeypatch):
    """Owner decision 2026-11-11: accepted exception -- code chunks keep the code chunker's boundaries and ids;
    the F-R3 project line comes on top of CODE_TARGET_TOKENS (far below EMBED_MAX_TOKENS)."""
    monkeypatch.setattr(config, "CODE_TARGET_TOKENS", 10)
    body = "[Demo] a.py > f [lines 1-2]\n" + " ".join(["x"] * 8)               # 9 words + prefix = at target
    block = Block("code_chunk", body, "corpus/code/a.py", [],
                  {"chunk_id": "abc", "project_name": "Demo", "project_description": "A demo project."})
    chunk = code_block_to_chunk(block, True)
    assert chunk.chunk_id == "abc" and chunk.text.endswith(body)
    assert words(chunk.text) + 1 > config.CODE_TARGET_TOKENS


# --- metadata + guard -------------------------------------------------------------------------------------

def test_collection_metadata_flags_off_is_byte_identical_to_before():
    assert indexer._collection_metadata("fp", OFF) == indexer._collection_metadata("fp") == {
        "hnsw:space": "cosine", "embed_model": config.EMBED_MODEL_NAME,
        "embed_revision": config.EMBED_MODEL_REVISION or "unpinned", "privacy_fp": "fp"}
    assert indexer._collection_metadata("fp", MERGE)["small_sections_merged"] is True


@pytest.mark.parametrize("spelling", ["upper", "trailing", "dotdot", "relative"])
def test_guard_sees_the_production_folder_through_any_spelling(spelling, tmp_path, monkeypatch):
    prod = tmp_path / "index" / "chroma"
    prod.mkdir(parents=True)
    (prod / "chroma.sqlite3").write_bytes(b"")
    monkeypatch.setattr(config, "CHROMA_DIR", prod)
    monkeypatch.chdir(tmp_path)
    target = {"upper": str(prod).upper(), "trailing": str(prod) + "\\", "dotdot": str(prod / ".." / "chroma"),
              "relative": "index/chroma"}[spelling]
    from pathlib import Path
    assert "never goes into" in build_index.target_problem(Path(target), MERGE)
    assert build_index.target_problem(Path(target), OFF) is None


def test_guard_refuses_parents_and_non_index_folders(tmp_path, monkeypatch):
    prod = tmp_path / "index" / "chroma"
    prod.mkdir(parents=True)
    monkeypatch.setattr(config, "CHROMA_DIR", prod)
    assert build_index.target_problem(tmp_path, MERGE)                       # contains production
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    assert build_index.target_problem(tmp_path / "file.txt", MERGE)          # a file
    assert build_index.target_problem(tmp_path / "index" / "fresh", MERGE) is None


def test_guard_refuses_the_production_swap_folder_names(tmp_path, monkeypatch):
    prod = tmp_path / "index" / "chroma"
    prod.mkdir(parents=True)
    monkeypatch.setattr(config, "CHROMA_DIR", prod)
    assert build_index.target_problem(prod.with_name("chroma" + indexer.OLD_SUFFIX), MERGE)


def test_boolean_optional_flags_default_to_config_and_accept_no_form(monkeypatch):
    seen = {}
    monkeypatch.setattr(build_index, "target_problem", lambda d, c: seen.update(c) or "stop")
    assert build_index.main(["--no-merge-small-sections", "--no-doc-descriptions"]) == 1
    assert seen == OFF
    seen.clear()
    monkeypatch.setattr(config, "SMALL_SECTIONS_MERGED", True)              # default follows config at parse time
    build_index.main([])
    assert seen["small_sections_merged"] is True
