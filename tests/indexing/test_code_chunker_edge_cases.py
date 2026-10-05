"""QA (tester-swarm) edge cases for code_chunker.py -- guideline v4 3A.4/3A.5.

Tests marked xfail(strict=True) document confirmed bugs; they will start
XPASS-failing once the bug is fixed, which is the signal to drop the marker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypeonoath.core import config
from hypeonoath.indexing.code_chunker import exclusion_reason, list_code_files

from tests.conftest import FIXTURES

pytestmark = pytest.mark.model


def _chunk(chunker, tmp_path: Path, name: str, content: str | bytes, project):
    path = tmp_path / name
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return chunker.chunk_file(path, name, project, "01.01.2026"), path.read_bytes()


def _code_of(block) -> str:
    return block.text.split("\n", 1)[1]


def _assert_offsets_and_lines(blocks, data: bytes):
    for b in blocks:
        m = b.meta
        assert data[m["start_byte"]:m["end_byte"]].decode("utf-8") == _code_of(b)
        assert data.count(b"\n", 0, m["start_byte"]) + 1 == m["line_start"]


# --- CRLF -------------------------------------------------------------------------

CRLF_SRC = "import os\r\n\r\nLIMIT = 3\r\n\r\n\r\ndef a():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"


def test_crlf_file_offsets_lines_and_symbol_spans(code_chunker_instance, project, tmp_path):
    result, data = _chunk(code_chunker_instance, tmp_path, "crlf.py", CRLF_SRC, project)
    assert result.blocks
    _assert_offsets_and_lines(result.blocks, data)
    spans = {s["name"]: (s["line_start"], s["line_end"]) for b in result.blocks for s in b.meta["symbol_spans"]}
    assert spans == {"a": (6, 7), "b": (10, 11)}
    # header line itself carries no stray carriage return
    assert all("\r" not in b.text.split("\n", 1)[0] for b in result.blocks)


# --- .h with C++ content -------------------------------------------------------------

CPP_HEADER = """#pragma once
#include <string>
namespace demo {
template <typename T>
class Box {
 public:
  explicit Box(T v) : v_(v) {}
  T get() const { return v_; }
  void set(const T& v) { v_ = v; }
 private:
  T v_;
};
}  // namespace demo
"""


def test_h_file_with_cpp_content_uses_cpp_grammar(code_chunker_instance, project, tmp_path):
    result, data = _chunk(code_chunker_instance, tmp_path, "box.h", CPP_HEADER, project)
    assert {b.meta["language"] for b in result.blocks} == {"cpp"}
    assert all(b.meta["chunk_method"] == "ast" for b in result.blocks)
    symbols = {s for b in result.blocks for s in b.meta["symbols"]}
    assert {"demo.Box", "demo.Box.get", "demo.Box.set"} <= symbols
    assert result.blocks[0].text.startswith("// [Demo Project] box.h")


def test_plain_c_header_stays_c(code_chunker_instance, project, tmp_path):
    src = "#ifndef X_H\n#define X_H\nint add(int a, int b);\nstatic int twice(int x) { return 2 * x; }\n#endif\n"
    result, _ = _chunk(code_chunker_instance, tmp_path, "x.h", src, project)
    assert {b.meta["language"] for b in result.blocks} == {"c"}


# --- Python class with a split method + nested function -------------------------------

def _class_src() -> str:
    body = "\n".join(f"        v{i} = compute_value_{i}(alpha, beta, gamma) + {i}" for i in range(120))
    return f'''class Engine:
    """Engine doc."""

    def small(self):
        return 1

    def big(self, alpha, beta, gamma):
        def helper(x):
            return x * 2
{body}
        return helper(v0)

    def after(self):
        return 3


def top():
    return 4
'''


@pytest.fixture
def class_result(code_chunker_instance, project, tmp_path):
    return _chunk(code_chunker_instance, tmp_path, "engine.py", _class_src(), project)


def test_split_method_parts_contiguous_and_budgeted(class_result, count_tokens, code_chunker_instance):
    result, data = class_result
    _assert_offsets_and_lines(result.blocks, data)
    parts = [b for b in result.blocks if b.meta.get("total_parts")]
    assert parts, "Engine.big must be split"
    assert [b.meta["part"] for b in parts] == list(range(1, len(parts) + 1))
    assert {b.meta["total_parts"] for b in parts} == {len(parts)}
    assert len({b.meta["group_id"] for b in parts}) == 1
    for b in result.blocks:                                       # R6
        assert count_tokens(b.text) + code_chunker_instance.prefix_tokens <= config.CODE_TARGET_TOKENS
    # the container class is never given part numbers, only the function
    assert all("Engine (part" not in b.text.split("\n", 1)[0] for b in result.blocks)
    assert any("Engine.big (part 1/" in b.text.split("\n", 1)[0] for b in parts)


def test_small_siblings_of_split_method_stay_whole(class_result):
    result, data = class_result
    text = data.decode("utf-8")
    for snippet in ("    def small(self):\n        return 1", "    def after(self):\n        return 3",
                    "def top():\n    return 4"):
        snippet = snippet.replace("\n", "\r\n") if "\r\n" in text else snippet
        holders = [b for b in result.blocks if snippet.strip() in _code_of(b)]
        assert len(holders) == 1, snippet                         # R1


def test_nested_function_gets_qualified_name(class_result):
    result, _ = class_result
    assert any("Engine.big.helper" in b.meta["symbols"] for b in result.blocks)


def test_first_part_is_not_a_lone_signature(class_result):
    # Fixed (was QA bug 3). Part 1 holds the signature PLUS following code; in
    # this fixture that is the nested `helper`, kept whole per R1 rather than
    # padded further with line-cut body.
    result, _ = class_result
    first = next(b for b in result.blocks if b.meta.get("part") == 1)
    code_lines = [line for line in _code_of(first).splitlines() if line.strip()]
    assert len(code_lines) > 1 and "def helper" in _code_of(first)


def test_adjacent_ast_chunks_do_not_share_lines(class_result):
    result, _ = class_result
    ast = [b for b in result.blocks if b.meta["chunk_method"] == "ast"]
    for prev, nxt in zip(ast, ast[1:]):
        assert prev.meta["line_end"] < nxt.meta["line_start"], (prev.meta["line_end"], nxt.meta["line_start"])


# --- misc contract checks --------------------------------------------------------------

def test_chunk_ids_unique_across_fixture_files(code_chunker_instance, project):
    ids = []
    for name in ("small_functions.py", "class_methods.ts", "oversized_function.py", "unicode.py", "query.sql"):
        ids += [b.meta["chunk_id"] for b in code_chunker_instance.chunk_file(FIXTURES / name, name, project, None).blocks]
    assert len(ids) == len(set(ids))


def test_whitespace_only_file_skipped(code_chunker_instance, project, tmp_path):
    result, _ = _chunk(code_chunker_instance, tmp_path, "blank.py", "  \r\n\t\n", project)
    assert result.blocks == [] and result.findings == []


@pytest.mark.parametrize("rel", ["vendor\\lib.js", "a/node_modules/x.js", "build/out.py"])
def test_exclusion_with_windows_and_posix_rel_paths(tmp_path, rel):
    path = tmp_path / "f.js"
    path.write_text("x = 1\n")
    if "\\" in rel and Path("a\\b").parts == ("a\\b",):
        pytest.skip("backslash is not a separator on this OS")
    assert exclusion_reason(path, rel) == "excluded_directory"


def test_underscore_code_files_not_silently_dropped(tmp_path, monkeypatch):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    monkeypatch.setattr(config, "CODE_METADATA_PATH", code_dir / "_metadata.yaml")
    (code_dir / "_metadata.yaml").write_text("{}\n")
    (code_dir / "_utils.py").write_text("def f():\n    return 1\n")
    (code_dir / "__init__.py").write_text("from .x import y\n")
    included, excluded = list_code_files(code_dir)
    assert {"_utils.py", "__init__.py"} <= set(included) | set(excluded)


def test_header_with_all_names_dropped_is_well_formed(code_chunker_instance, project):
    header = code_chunker_instance._header(project, "x.py", ["a", "b", "c"], (1, 2), "python", 0)
    assert " > , " not in header
