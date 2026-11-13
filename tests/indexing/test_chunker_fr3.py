"""chunker.py F-R3 (eval plan v29, D95, D98): the sibling rule and the description line, both off by default."""
from __future__ import annotations

import pytest

from hypeonoath.core import config, registry
from hypeonoath.core.records import Block
from hypeonoath.indexing import chunker
from hypeonoath.indexing.block_loader import LoadedDocument
from hypeonoath.indexing.chunker import Chunker, code_block_to_chunk, description_line, make_chunks

DOC = "corpus/converted/notes.md"


def words(text: str) -> int:
    """Fake tokenizer: one token per word ("search_document: " = 1)."""
    return len(text.split())


@pytest.fixture(autouse=True)
def small_bounds(monkeypatch):
    # 30 / 50 "tokens" instead of 300 / 500 keeps the fixtures readable; the rule is the same.
    monkeypatch.setattr(config, "PROSE_MIN_TOKENS", 30)
    monkeypatch.setattr(config, "PROSE_MAX_TOKENS", 50)
    monkeypatch.setattr(registry, "DOCUMENTS", {"notes.pdf": {"description": "My notes on things.", "project": None}})


def prose(heading, n_words, line, word="w"):
    return Block("prose", " ".join([word] * n_words) + ".", DOC, list(heading),
                 {"line_start": line, "line_end": line})


def table(heading, rows, line=50):
    text = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return Block("table", text, DOC, list(heading), {"rows": rows, "has_header": True, "line_start": line,
                                                     "line_end": line + len(rows)})


def doc(*blocks):
    return LoadedDocument(DOC, list(blocks))


def chunks(document, merge=True, desc=False):
    return Chunker(words, merge_small=merge, descriptions=desc).chunk_document(document)


# --- defaults and "flag off -> unchanged" (D98) ---------------------------------------------------

def test_flags_default_off_in_config():
    assert config.SMALL_SECTIONS_MERGED is False and config.DOC_DESCRIPTION_IN_CHUNKS is False


def test_explicit_off_equals_config_default_and_one_chunk_per_section():
    d = doc(prose(["A", "x"], 5, 1), prose(["A", "y"], 5, 2), table(["A"], [["K", "V"], ["a", "1"], ["b", "2"]]))
    default = [c.to_dict() for c in Chunker(words).chunk_document(d)]
    off = [c.to_dict() for c in chunks(d, merge=False, desc=False)]
    assert default == off and len(off) == 4                        # 2 sections + 2 record rows


def test_config_default_is_read_at_call_time(monkeypatch):
    d = doc(prose(["A", "x"], 5, 1), prose(["A", "y"], 5, 2))
    monkeypatch.setattr(config, "SMALL_SECTIONS_MERGED", True)
    assert len(Chunker(words).chunk_document(d)) == 1
    assert len(make_chunks([d], [], words)) == 1


# --- sibling rule: sections -------------------------------------------------------------------------

def test_small_sibling_sections_merge_and_keep_their_heading_lines():
    out = chunks(doc(prose(["P", "x"], 5, 1), prose(["P", "y"], 5, 3), prose(["P", "z"], 5, 7)))
    assert len(out) == 1
    assert out[0].text == "P > x\nw w w w w.\n\nP > y\nw w w w w.\n\nP > z\nw w w w w."
    assert (out[0].line_start, out[0].line_end) == (1, 7) and out[0].heading_path == ["P", "x"]


def test_different_parent_headings_never_merge():
    out = chunks(doc(prose(["P", "x"], 5, 1), prose(["Q", "y"], 5, 2), prose(["P", "x", "deep"], 5, 3)))
    assert len(out) == 3


def test_unit_at_min_stands_alone_and_result_stays_under_max():
    # 25 + 5 heading/line words: "P > x" adds 3 -> 28 + 1 prefix = 29 < 30 absorbs; 2nd unit 35 >= 30 stands alone
    out = chunks(doc(prose(["P", "x"], 25, 1), prose(["P", "y"], 32, 2), prose(["P", "z"], 5, 3)))
    assert len(out) == 3
    # two small units whose sum would pass 50 are not merged
    out = chunks(doc(prose(["P", "x"], 24, 1), prose(["P", "y"], 24, 2)))
    assert len(out) == 2
    assert all(words(c.text) + 1 <= config.PROSE_MAX_TOKENS for c in out)


def test_greedy_group_stops_once_it_reaches_min():
    out = chunks(doc(prose(["P", "a"], 12, 1), prose(["P", "b"], 12, 2), prose(["P", "c"], 5, 3)))
    # a+b = 2*(12+3) + 1 = 31 >= 30 -> c starts a new chunk
    assert [c.heading_path[-1] for c in out] == ["a", "c"]


def test_split_section_stands_alone():
    out = chunks(doc(prose(["P", "x"], 5, 1), prose(["P", "big"], 80, 2), prose(["P", "y"], 5, 3)))
    texts = [c.text.split("\n")[0] for c in out]
    assert texts[0] == "P > x" and texts[-1] == "P > y" and len(out) >= 4   # big split, nothing merged into it


def test_top_level_sections_merge_only_in_a_small_document():
    small = chunks(doc(prose([], 3, 1), prose(["NAME"], 4, 2), prose(["CERT"], 4, 3)))
    assert len(small) == 1                                                  # certificate-like document
    big = chunks(doc(prose([], 3, 1), prose(["NAME"], 4, 2), prose(["CERT"], 4, 3), prose(["OTHER"], 40, 4)))
    assert len(big) == 4


def test_table_or_fence_between_sections_breaks_the_run():
    fence = Block("code_fence", "```\nx = 1\n```", DOC, ["P"], {"line_start": 9, "line_end": 11})
    out = chunks(doc(prose(["P", "x"], 5, 1), fence, prose(["P", "y"], 5, 12)))
    assert [c.content_type for c in out] == ["prose", "code_fence", "prose"]


# --- sibling rule: table rows -------------------------------------------------------------------------

def test_rows_of_one_table_pack_with_their_labels():
    rows = [["Problem", "Fix"], ["slow", "cache"], ["crash", "retry"], ["leak", "close"]]
    out = chunks(doc(table(["Design"], rows)))
    assert len(out) == 1 and out[0].content_type == "record"
    assert out[0].text == "Design\nProblem: slow\nFix: cache\n\nProblem: crash\nFix: retry\n\nProblem: leak\nFix: close"
    assert (out[0].line_start, out[0].line_end) == (52, 54)


def test_rows_never_merge_across_tables_or_with_sections():
    rows = [["K", "V"], ["a", "1"], ["b", "2"]]
    out = chunks(doc(table(["T"], rows, 10), table(["T"], rows, 20), prose(["T"], 3, 30)))
    assert [c.content_type for c in out] == ["record", "record", "prose"]


def test_a_row_split_at_field_boundaries_stands_alone():
    rows = [["A", "B"], [" ".join(["x"] * 30), " ".join(["y"] * 30)], ["s", "t"]]
    out = chunks(doc(table(["T"], rows)))
    assert len(out) == 3                                                    # 2 pieces of row 1 + row 2 alone


def test_merged_chunk_ids_are_unique_and_stable():
    d = doc(prose(["P", "x"], 5, 1), prose(["P", "y"], 5, 2), table(["P"], [["K", "V"], ["a", "1"], ["b", "2"]]))
    first, second = chunks(d), chunks(d)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


# --- description line ---------------------------------------------------------------------------------

def test_description_line_is_first_on_every_document_chunk():
    out = chunks(doc(prose(["P", "x"], 5, 1), table(["P"], [["K", "V"], ["a", "1"], ["b", "2"]])), merge=False,
                 desc=True)
    assert all(c.text.startswith("notes — My notes on things.\n") for c in out)
    assert out[0].text.split("\n")[1] == "P > x"                           # heading line follows


def test_description_line_counts_inside_the_budget():
    out = chunks(doc(prose(["P", "x"], 120, 1)), merge=False, desc=True)
    assert all(words(c.text) + 1 <= config.PROSE_MAX_TOKENS for c in out)


def test_description_line_lookup():
    assert description_line(DOC) == "notes — My notes on things."
    assert description_line("corpus/converted/unknown.md") is None
    projects = {"demo": {"name": "Demo App", "description": "A demo.\nTwo lines."}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "CODE_PROJECTS", projects)
        assert description_line("corpus/code/nb.ipynb", "Demo App") == "Demo App — A demo. Two lines."


def test_code_chunk_gets_the_project_line_and_keeps_its_id():
    block = Block("code_chunk", "[Demo] a.py > f [lines 1-2]\ndef f(): pass", "corpus/code/a.py", [],
                  {"chunk_id": "abc", "project_name": "Demo", "project_description": "A demo\nproject."})
    off, on = code_block_to_chunk(block, False), code_block_to_chunk(block, True)
    assert off.text == block.text and on.chunk_id == off.chunk_id == "abc"
    assert on.text == "Demo — A demo project.\n" + block.text


def test_doc_prefix_constant_is_an_em_dash():
    assert chunker.DESCRIPTION_SEPARATOR == " — "
