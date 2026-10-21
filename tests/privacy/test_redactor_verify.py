"""privacy.redactor + privacy.verify (privacy plan v10, 6.7, 6.8, P-D46, P-D49; tests 14.6, 14.10)."""
from __future__ import annotations

import hashlib
import json

import pytest

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.core.records import Block, Chunk
from hypeonoath.indexing import chunker
from hypeonoath.indexing.block_loader import LoadedDocument, parse_docling
from hypeonoath.indexing.project_registry import ProjectInfo
from hypeonoath.privacy import redactor, verify

PHONE = pv.PrivateValue("pv_p", "phone number", "id", "491512345678", ("+49 151 2345678",))
NAME = pv.PrivateValue("pv_n", "name", "text", "max mustermann", ("Max Mustermann",))
VALUES = [PHONE, NAME]
P_PHONE, P_NAME = "[REDACTED: phone number]", "[REDACTED: name]"


def doc(*blocks):
    return LoadedDocument("corpus/converted/cv.md", list(blocks), project_name="CV of Max Mustermann")


def test_every_block_kind_and_project_string_is_redacted():
    heading = Block("heading", "Max Mustermann", "corpus/converted/cv.md", [], {"line_start": 1})
    prose = Block("prose", "Call +49 151 2345678 now.", "corpus/converted/cv.md", ["Max Mustermann"],
                  {"line_start": 3})
    table = Block("table", "| Name | Phone |\n|---|---|\n| Max Mustermann | +49 151 2345678 |",
                  "corpus/converted/cv.md", ["Max Mustermann"],
                  {"rows": [["Name", "Phone"], ["Max Mustermann", "+49 151 2345678"]], "has_header": True})
    code = Block("code_chunk", "# [Proj by Max Mustermann] a.py\nPHONE = '+49 151 2345678'", "corpus/code/a.py", [],
                 {"project_name": "Proj by Max Mustermann", "project_description": "Max Mustermann's app",
                  "symbols": ["call_max_mustermann"], "chunk_id": "c"})
    document = doc(heading, prose, table)
    stats = redactor.redact_blocks([document], [code], VALUES)

    assert heading.text == P_NAME and prose.heading_path == [P_NAME] and table.heading_path == [P_NAME]
    assert prose.text == f"Call {P_PHONE} now."
    assert table.meta["rows"] == [["Name", "Phone"], [P_NAME, P_PHONE]]
    assert P_NAME in table.text and P_PHONE in table.text
    assert code.text.endswith(f"PHONE = '{P_PHONE}'") and P_NAME in code.text
    assert code.meta["project_name"] == f"Proj by {P_NAME}" and code.meta["project_description"].startswith(P_NAME)
    assert document.project_name == f"CV of {P_NAME}"
    assert stats.blocks_changed == 4 and stats.labels["phone number"] == 4 and stats.labels["name"] >= 7

    projects = {"a.py": ProjectInfo("demo", "Max Mustermann tools", "Call +49 151 2345678")}
    redactor.redact_projects(projects, VALUES)
    assert projects["a.py"] == ProjectInfo("demo", f"{P_NAME} tools", f"Call {P_PHONE}")


def test_redaction_records_where_each_value_sat():
    """P-D49: lines as detection numbers them (documents: line_start + offset; code: header = line 0)."""
    prose = Block("prose", "intro\nCall +49 151 2345678 now.", "corpus/converted/cv.md", [], {"line_start": 3})
    code = Block("code_chunk", "# header\nx = 1\nPHONE = '+49 151 2345678'", "corpus/code/a.py", [],
                 {"line_start": 10})
    stats = redactor.redact_blocks([doc(prose)], [code], VALUES)
    assert stats.locations["pv_p"] == {"corpus/converted/cv.md": {4}, "corpus/code/a.py": {11}}


def test_no_values_changes_nothing():
    block = Block("prose", "Max Mustermann", "x.md")
    assert redactor.redact_blocks([doc(block)], [], []).total == 0 and block.text == "Max Mustermann"


def test_redacted_record_chunks_never_hold_the_value(count_tokens):
    table = Block("table", "| Name | Phone |\n|---|---|\n| Max Mustermann | +49 151 2345678 |", "cv.md", [],
                  {"rows": [["Name", "Phone"], ["Max Mustermann", "+49 151 2345678"]], "has_header": True,
                   "line_start": 1})
    document = LoadedDocument("cv.md", [table])
    redactor.redact_blocks([document], [], VALUES)
    chunks = chunker.make_chunks([document], [], count_tokens)
    assert chunks and not verify.fix_chunks(chunks, VALUES)


# --- files on disk --------------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path
    for name, path in {"PROJECT_ROOT": root, "CORPUS_DIR": root / "corpus",
                       "CONVERTED_DIR": root / "corpus" / "converted",
                       "DOCUMENTS_DIR": root / "corpus" / "documents"}.items():
        monkeypatch.setattr(config, name, path)
        path.mkdir(parents=True, exist_ok=True)
    (root / "corpus" / "raw").mkdir()
    return root


def _docling_fixture(path):
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    document = DoclingDocument(name="cv")
    document.add_heading(text="Contact")
    document.add_text(label=DocItemLabel.TEXT, text="Max Mustermann, +49 151 2345678", orig="Max Mustermann, +49 151 2345678")
    document.save_as_json(path)


def test_disk_files_redacted_raw_untouched_and_idempotent(corpus):
    converted, documents, raw = corpus / "corpus" / "converted", corpus / "corpus" / "documents", corpus / "corpus" / "raw"
    md = converted / "cv.md"
    md.write_text("---\ndate: 01.01.2026\nproject_name: null\n---\n\n## Contact\n\nMax Mustermann &amp; co, "
                  "+49 151 2345678\n", encoding="utf-8")
    _docling_fixture(converted / "cv.docling.json")
    (documents / "notes.txt").write_text("phone +49 151 2345678", encoding="utf-8")
    (documents / "clean.md").write_text("nothing private here", encoding="utf-8")
    raw_file = raw / "cv.md"
    raw_file.write_text("Max Mustermann +49 151 2345678", encoding="utf-8")
    raw_before = (raw_file.stat().st_mtime_ns, hashlib.sha256(raw_file.read_bytes()).hexdigest())

    changed = redactor.redact_files_on_disk(VALUES)
    assert changed == ["corpus/converted/cv.docling.json", "corpus/converted/cv.md", "corpus/documents/notes.txt"]
    text = md.read_text(encoding="utf-8")
    assert text.startswith("---\ndate: 01.01.2026\nproject_name: null\n---\n")       # front matter untouched
    assert f"{P_NAME} &amp; co, {P_PHONE}" in text
    data = json.loads((converted / "cv.docling.json").read_text(encoding="utf-8"))
    assert data["texts"][1]["text"] == f"{P_NAME}, {P_PHONE}" and data["texts"][1]["orig"] == data["texts"][1]["text"]
    assert data["name"] == "cv"                                                           # only text/orig changed
    assert (raw_file.stat().st_mtime_ns, hashlib.sha256(raw_file.read_bytes()).hexdigest()) == raw_before
    assert redactor.redact_files_on_disk(VALUES) == []                                    # idempotent

    # The redacted JSON still loads with docling and yields the placeholders as blocks.
    blocks = parse_docling(converted / "cv.docling.json", text.split("---\n", 2)[2], "corpus/converted/cv.md", 4)
    assert any(P_PHONE in b.text for b in blocks)


def test_front_matter_with_a_value_stays_valid_yaml(corpus):
    import yaml

    md = corpus / "corpus" / "converted" / "cv.md"
    md.write_text("---\ntitle: CV of Max Mustermann\ndate: 01.01.2026\n---\nbody\n", encoding="utf-8")
    redactor.redact_files_on_disk(VALUES)
    front = md.read_text(encoding="utf-8").split("---\n")[1]
    assert yaml.safe_load(front) == {"title": f"CV of {P_NAME}", "date": "01.01.2026"}


def test_no_values_touches_no_file(corpus):
    (corpus / "corpus" / "converted" / "cv.md").write_text("Max Mustermann", encoding="utf-8")
    assert redactor.redact_files_on_disk([]) == []


# --- verify -----------------------------------------------------------------------------------------

def test_value_split_across_blocks_is_fixed_after_chunking(count_tokens):
    """P-D46: the chunker re-joins the value; verify redacts it in the chunk and gives it a new id."""
    first = Block("prose", "My name is Max", "cv.md", ["About"], {"line_start": 3, "line_end": 3})
    second = Block("prose", "Mustermann and I build things.", "cv.md", ["About"], {"line_start": 5, "line_end": 5})
    document = LoadedDocument("cv.md", [first, second])
    stats = redactor.redact_blocks([document], [], [NAME])
    assert stats.total == 0                                    # neither block alone holds the value
    chunks = chunker.make_chunks([document], [], count_tokens)
    old_ids = [c.chunk_id for c in chunks]
    assert verify.fix_chunks(chunks, [NAME]) == {"name": 1}
    assert all("Mustermann" not in c.text for c in chunks) and any(P_NAME in c.text for c in chunks)
    assert [c.chunk_id for c in chunks] != old_ids
    assert not verify.fix_chunks(chunks, [NAME])               # second pass: nothing left


def test_verify_fixes_metadata_strings_keeping_the_id_when_text_is_clean():
    chunk = Chunk("c", "clean", "corpus/code/a.py", "code", project_name="P", project_description="by Max Mustermann",
                  heading_path=["Max Mustermann"], symbols=["f"])
    assert verify.fix_chunks([chunk], VALUES) == {"name": 2}
    assert chunk.project_description == f"by {P_NAME}" and chunk.heading_path == [P_NAME] and chunk.chunk_id == "c"
    assert not verify.fix_chunks([Chunk("d", "fine", "x.md", "prose")], VALUES)
    assert not verify.fix_chunks([chunk], [])
