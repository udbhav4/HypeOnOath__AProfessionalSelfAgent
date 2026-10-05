"""
code_chunker.py against the guideline v4 Section 8A fixtures + rules R1-R6.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import tree_sitter_language_pack as tslp

from hypeonoath.core import config
from hypeonoath.indexing import code_chunker as cc

from tests.conftest import FIXTURES


def chunk(chunker, project, name, tmp_path=None, src=FIXTURES):
    path = src / name
    return chunker.chunk_file(path, name, project, "01.01.2026")


def ranges(blocks):
    return [(b.meta["start_byte"], b.meta["end_byte"]) for b in blocks]


def code_of(block):
    return block.text.split("\n", 1)[1]


def read_log(isolated_logs, name="code_chunking_log_path"):
    path = isolated_logs / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# --- R1 / R2 / R5: small whole functions -------------------------------------------

def test_small_functions_whole_packed_and_covered(code_chunker_instance, project):
    result = chunk(code_chunker_instance, project, "small_functions.py")
    blocks = result.blocks
    data = (FIXTURES / "small_functions.py").read_bytes()
    tree = tslp.get_parser("python").parse(data)
    symbols = cc.extract_symbols(tree, data)
    assert {s.name for s in symbols} >= {"add", "sub", "mul", "div", "hypot", "env_name"}
    # R1: every function whole in exactly one chunk
    for sym in symbols:
        holders = [b for b in blocks if b.meta["start_byte"] <= sym.start_byte and sym.end_byte <= b.meta["end_byte"]]
        assert len(holders) == 1, sym.name
    # R2: tiny file -> packed into few chunks, several functions per chunk
    assert len(blocks) < len(symbols)
    assert max(len(b.meta["symbols"]) for b in blocks) >= 2
    # R5: imports + constants covered
    assert code_chunker_instance._covers(data, [cc._Range(a, b) for a, b in ranges(blocks)])
    assert any("import math" in code_of(b) and "MAX_RETRIES" in code_of(b) for b in blocks)


def test_header_format_and_metadata(code_chunker_instance, project):
    block = chunk(code_chunker_instance, project, "small_functions.py").blocks[0]
    header = block.text.split("\n", 1)[0]
    assert header.startswith("# [Demo Project] small_functions.py")
    assert f"[lines {block.meta['line_start']}-{block.meta['line_end']}]" in header
    meta = block.meta
    assert meta["project_key"] == "demo" and meta["project_description"]
    assert meta["language"] == "python" and meta["chunk_method"] == "ast"
    assert meta["date"] == "01.01.2026"
    assert len(meta["chunk_id"]) == 16


# --- qualified symbols + R4 --------------------------------------------------------------

def test_class_methods_qualified_and_not_merged_with_top_level(code_chunker_instance, project):
    blocks = chunk(code_chunker_instance, project, "class_methods.ts").blocks
    all_symbols = {s for b in blocks for s in b.meta["symbols"]}
    assert {"Cache", "Cache.get", "Cache.set", "Cache.clear", "makeCache"} <= all_symbols
    assert blocks[0].text.startswith("// [Demo Project] class_methods.ts")  # TS comment style


def test_r4_merge_requires_same_parent(code_chunker_instance):
    data = (FIXTURES / "class_methods.ts").read_bytes()
    symbols = cc.extract_symbols(tslp.get_parser("typescript").parse(data), data)
    cache = next(s for s in symbols if s.name == "Cache")
    get = next(s for s in symbols if s.name == "Cache.get")
    make = next(s for s in symbols if s.name == "makeCache")
    method_piece = cc._Range(get.start_byte, get.end_byte)
    top_level = cc._Range(make.start_byte, make.end_byte)
    assert cc._parent(data, method_piece, symbols) == symbols.index(cache)
    assert cc._parent(data, top_level, symbols) is None
    merged = code_chunker_instance._merge(data, [method_piece, top_level], symbols)
    assert len(merged) == 2   # different parents -> never merged, even though they'd fit


# --- R3 / R6: oversized function ------------------------------------------------------------

def test_oversized_function_split_into_numbered_parts(code_chunker_instance, project, count_tokens):
    blocks = chunk(code_chunker_instance, project, "oversized_function.py").blocks
    parts = [b for b in blocks if b.meta.get("total_parts")]
    assert len(parts) >= 2
    assert [b.meta["part"] for b in parts] == list(range(1, len(parts) + 1))
    assert {b.meta["total_parts"] for b in parts} == {len(parts)}
    assert len({b.meta["group_id"] for b in parts}) == 1
    for b in parts:
        assert f"(part {b.meta['part']}/{len(parts)})" in b.text.split("\n", 1)[0]
    # R6: whole text (header + code) + task prefix within the target budget
    for b in blocks:
        assert count_tokens(b.text) + count_tokens(config.DOC_PREFIX) <= config.CODE_TARGET_TOKENS


# --- Step 7: broken syntax -> fallback ----------------------------------------------------------

def test_broken_syntax_within_budget_and_logged_if_fallback(code_chunker_instance, project, count_tokens, isolated_logs):
    blocks = chunk(code_chunker_instance, project, "broken_syntax.py").blocks
    assert blocks
    for b in blocks:
        assert count_tokens(b.text) + count_tokens(config.DOC_PREFIX) <= config.CODE_TARGET_TOKENS
    data = (FIXTURES / "broken_syntax.py").read_bytes()
    assert code_chunker_instance._covers(data, [cc._Range(a, b) for a, b in ranges(blocks)])
    if any(b.meta["chunk_method"] == "fallback" for b in blocks):
        assert any(r["status"] == "code_chunk_fallback" for r in read_log(isolated_logs))


def test_forced_fallback_uses_line_windows_with_overlap(code_chunker_instance, project, isolated_logs):
    data = (FIXTURES / "oversized_function.py").read_bytes()
    found, _, _ = code_chunker_instance.chunk_ranges(data, None, "oversized_function.py")  # unsupported language
    assert all(r.method == "fallback" for r in found)
    assert any(b.start < a.end for a, b in zip(found, found[1:]))         # ~10% overlap
    assert code_chunker_instance._covers(data, found)
    assert read_log(isolated_logs)[-1]["reason"] == "unsupported_language"


def test_whole_file_single_chunk_triggers_fallback(code_chunker_instance, monkeypatch):
    data = (FIXTURES / "oversized_function.py").read_bytes()
    monkeypatch.setattr(code_chunker_instance, "_ast_ranges", lambda text, lang: [cc._Range(0, len(data))])
    found, _, _ = code_chunker_instance.chunk_ranges(data, "python", "x.py")
    assert len(found) > 1 and all(r.method == "fallback" for r in found)


def test_chonkie_value_error_triggers_fallback(code_chunker_instance, monkeypatch):
    data = (FIXTURES / "small_functions.py").read_bytes()

    def boom(text, lang):
        raise ValueError("Unsupported language")

    monkeypatch.setattr(code_chunker_instance, "_ast_ranges", boom)
    found, _, _ = code_chunker_instance.chunk_ranges(data, "python", "x.py")
    assert found and all(r.method == "fallback" for r in found)


def test_hard_cut_of_single_overlong_line(code_chunker_instance, count_tokens):
    line = ("token_" * 1500).encode()
    pieces = code_chunker_instance._hard_cut(line, 0, len(line))
    assert len(pieces) > 1
    assert pieces[0][0] == 0 and pieces[-1][1] == len(line)
    assert all(count_tokens(line[a:b].decode()) <= code_chunker_instance.budget + 2 for a, b in pieces)


# --- Steps 1-2: exclusion, empty, encoding --------------------------------------------------------

def test_minified_bundle_excluded():
    assert cc.exclusion_reason(FIXTURES / "bundle.min.js", "bundle.min.js") == "excluded_name"


@pytest.mark.parametrize("rel, expected", [
    ("node_modules/lib/x.js", "excluded_directory"),
    ("vendor/x.py", "excluded_directory"),
])
def test_excluded_directories(tmp_path, rel, expected):
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n")
    assert cc.exclusion_reason(path, rel) == expected


def test_generated_and_long_line_and_size(tmp_path, monkeypatch):
    generated = tmp_path / "gen.go"
    generated.write_text("// Code generated by protoc. DO NOT EDIT.\npackage x\n")
    assert cc.exclusion_reason(generated, "gen.go") == "generated_file"
    long_line = tmp_path / "long.js"
    long_line.write_text("var x = '" + "a" * 1200 + "';\n")
    assert cc.exclusion_reason(long_line, "long.js") == "line_too_long"
    monkeypatch.setattr(config, "MAX_CODE_FILE_BYTES", 10)
    assert cc.exclusion_reason(generated, "gen.go") == "too_large"
    ok = tmp_path / "ok.py"
    ok.write_text("x = 1\n")
    monkeypatch.setattr(config, "MAX_CODE_FILE_BYTES", 200_000)
    assert cc.exclusion_reason(ok, "ok.py") is None


def test_list_code_files_skips_metadata_and_reports_exclusions(tmp_path, monkeypatch):
    for name in ("a.py", "bundle.min.js", "_metadata.yaml"):
        shutil.copy(FIXTURES / ("small_functions.py" if name != "bundle.min.js" else name), tmp_path / name)
    monkeypatch.setattr(config, "CODE_METADATA_PATH", tmp_path / "_metadata.yaml")
    included, excluded = cc.list_code_files(tmp_path)
    assert included == ["a.py"]
    assert excluded == {"bundle.min.js": "excluded_name"}


def test_empty_file_skipped_and_logged(code_chunker_instance, project, isolated_logs):
    result = chunk(code_chunker_instance, project, "empty.py")
    assert result.blocks == []
    assert read_log(isolated_logs)[-1] == {**read_log(isolated_logs)[-1], "status": "empty_file_skipped", "file": "empty.py"}


def test_latin1_does_not_crash_and_logs_non_utf8(code_chunker_instance, project, isolated_logs):
    result = chunk(code_chunker_instance, project, "latin1.c")
    assert result.blocks
    assert any(r["status"] == "non_utf8" for r in read_log(isolated_logs))


# --- Step 8: byte offsets (developer-owned item 3) ---------------------------------------------------

def test_unicode_offsets_are_bytes_and_lines_correct(code_chunker_instance, project):
    """
    Acceptance criteria 3A.11 item 3: lines match, stored offsets re-slice
    exactly. Chonkie 1.7.0 offsets are BYTES (start_index = start_byte).
    """
    data = (FIXTURES / "unicode.py").read_bytes()
    text = data.decode("utf-8")
    raw = code_chunker_instance._ast_ranges(text, "python")
    assert len(data) != len(text)       # the fixture really is multi-byte
    assert max(r.end for r in raw) == len(data), \
        "offsets behaved like str indices -- Chonkie changed; re-check the byte assumption"
    assert all(data[r.start:r.end].decode("utf-8") for r in raw)   # byte slices decode cleanly
    blocks = chunk(code_chunker_instance, project, "unicode.py").blocks
    lines = text.split("\n")
    for b in blocks:
        code = data[b.meta["start_byte"]:b.meta["end_byte"]].decode("utf-8")
        assert code == code_of(b)
        expected = "\n".join(lines[b.meta["line_start"] - 1:b.meta["line_end"]])
        assert code.strip() == expected.strip()
    grusse = next(b for b in blocks if "grüße" in b.meta["symbols"])
    assert lines[grusse.meta["symbol_spans"][[s["name"] for s in grusse.meta["symbol_spans"]].index("grüße")]["line_start"] - 1].startswith("def grüße")


# --- Step 4: notebooks (developer-owned item 5) -------------------------------------------------------

def test_notebook_outputs_dropped_and_cells_routed(code_chunker_instance, project):
    result = chunk(code_chunker_instance, project, "notebook.ipynb")
    code_text = "\n".join(b.text for b in result.blocks)
    prose_text = "\n".join(bl.text for d in result.notebook_docs for bl in d.blocks)
    for text in (code_text, prose_text):
        assert "SECRET_OUTPUT_MARKER" not in text and "iVBORw0KGgo" not in text
    assert "def add(a, b)" in code_text and "# %% [cell 1]" in code_text
    assert "Analysis notebook" in prose_text
    assert all(b.block_type == "code_chunk" for b in result.blocks)
    assert {bl.block_type for d in result.notebook_docs for bl in d.blocks} <= {"heading", "prose"}
    assert any(s["name"] == "cell 1" for b in result.blocks for s in b.meta["symbol_spans"])


def test_notebook_without_kernel_metadata_defaults_to_python(code_chunker_instance, project, tmp_path):
    nb = json.loads((FIXTURES / "notebook.ipynb").read_text())
    nb["metadata"] = {}
    (tmp_path / "bare.ipynb").write_text(json.dumps(nb))
    result = code_chunker_instance.chunk_file(tmp_path / "bare.ipynb", "bare.ipynb", project, None)
    assert result.blocks and result.blocks[0].meta["language"] == "python"


# --- SQL: no symbols ------------------------------------------------------------------------------------

def test_sql_chunks_with_empty_symbols(code_chunker_instance, project):
    blocks = chunk(code_chunker_instance, project, "query.sql").blocks
    assert blocks and all(b.meta["symbols"] == [] for b in blocks)
    assert blocks[0].text.startswith("-- [Demo Project] query.sql")


# --- determinism (Section 9.1 idempotency) ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["small_functions.py", "oversized_function.py", "class_methods.ts", "notebook.ipynb"])
def test_chunk_ids_stable_across_runs(code_chunker_instance, project, name):
    first = [b.meta["chunk_id"] for b in chunk(code_chunker_instance, project, name).blocks]
    second = [b.meta["chunk_id"] for b in chunk(code_chunker_instance, project, name).blocks]
    assert first == second and len(set(first)) == len(first)


# --- raw-text scan happens inside the chunker (Step 3) -----------------------------------------------------

def test_raw_scan_findings_returned_without_values(code_chunker_instance, project):
    result = chunk(code_chunker_instance, project, "fake_secret.py")
    assert any(f.finding_type == "AWS Access Key" and f.line == 2 for f in result.findings)
    assert "AKIA" not in repr(result.findings)


# --- the real code file (R1, R5, R6 on agent.ts) ----------------------------------------------------------

def test_real_agent_ts_contract(code_chunker_instance, project, count_tokens):
    path = config.CODE_DIR / "agent.ts"
    if not path.exists():
        pytest.skip("corpus/code/agent.ts not present")
    blocks = code_chunker_instance.chunk_file(path, "agent.ts", project, None).blocks
    data = path.read_bytes()
    symbols = cc.extract_symbols(tslp.get_parser("typescript").parse(data), data)
    budget = code_chunker_instance.budget
    for sym in symbols:     # R1
        if code_chunker_instance._tokens(data, sym.start_byte, sym.end_byte) <= budget:
            assert any(b.meta["start_byte"] <= sym.start_byte and sym.end_byte <= b.meta["end_byte"] for b in blocks), sym.name
    assert code_chunker_instance._covers(data, [cc._Range(a, b) for a, b in ranges(blocks)])     # R5
    prefix = count_tokens(config.DOC_PREFIX)
    assert all(count_tokens(b.text) + prefix <= config.CODE_TARGET_TOKENS for b in blocks)  # R6
    groups = {}
    for b in blocks:
        if b.meta.get("group_id"):
            groups.setdefault(b.meta["group_id"], []).append(b.meta["part"])
    for parts in groups.values():
        assert sorted(parts) == list(range(1, len(parts) + 1))


# --- boundary snapping + notebook scan (post-QA regressions) ----------------------------

def test_boundaries_snap_to_line_starts_and_keep_doc_comments(code_chunker_instance):
    """Chonkie can cut mid-line (`export | async function`) and leave a doc comment behind."""
    data = b"const a = 1;\n/** Doc for f. */\nexport async function f() {\n  return 1;\n}\n"
    export_at = data.index(b"export")
    fragments = [cc._Range(0, export_at), cc._Range(export_at, export_at + 7), cc._Range(export_at + 7, len(data))]
    snapped = code_chunker_instance._snap_boundaries(data, fragments)
    assert [data[r.start:r.end].split(b"\n")[0] for r in snapped] == [b"const a = 1;", b"/** Doc for f. */"]
    assert snapped[0].start == 0 and snapped[-1].end == len(data)


def test_notebook_scan_ignores_outputs_and_dedupes_with_chunks(code_chunker_instance, project, tmp_path):
    from hypeonoath.indexing import secret_scan

    nb = json.loads((FIXTURES / "notebook.ipynb").read_text())
    nb["cells"][1]["source"] = ["key = 'AKIAIOSFODNN7EXAMPLE'\n"]
    nb["cells"][2]["outputs"][0]["data"]["text/plain"] = ["contact: only.in.output@example.com"]
    (tmp_path / "nb.ipynb").write_text(json.dumps(nb))
    result = code_chunker_instance.chunk_file(tmp_path / "nb.ipynb", "nb.ipynb", project, None)
    assert not any(f.finding_type == "email_address" for f in result.findings)     # output never scanned
    from hypeonoath.indexing.chunker import code_block_to_chunk
    chunk_findings = secret_scan.scan_chunks([code_block_to_chunk(b) for b in result.blocks])
    report = secret_scan.build_report(result.findings + chunk_findings, [])
    aws = [f for f in report.findings if f.finding_type == "AWS Access Key"]
    assert len(aws) == 1                                                             # raw + chunk merged
