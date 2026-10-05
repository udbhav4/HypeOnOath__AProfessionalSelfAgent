"""
Step 1-code -- code file -> finished `code_chunk` Blocks (guideline v4,
Section 3A; design doc 5.1.1, ADR-6).

Chonkie's CodeChunker (AST-aware size-based split, on tree-sitter-language-
pack) does the core splitting; this wrapper closes the gaps it leaves:

  0  pinned versions (config.PINNED_VERSIONS / pyproject.toml)
  1  exclusion filter (vendored / minified / generated / huge files)
  2  safe UTF-8 decoding (non_utf8 logged, empty files skipped)
  3  secret + personal-info scan of the raw text (warnings only)
  4  notebooks: code cells -> code; markdown cells -> prose; outputs DROPPED
  5  explicit language from the extension map (no auto-detection)
  6  Chonkie with the embedding model's tokenizer and a token budget of
     CODE_TARGET_TOKENS - HEADER_TOKEN_RESERVE - PREFIX_TOKENS
  7  fallback: unsupported language / parse failure / mostly-ERROR tree /
     whole-file-as-one-chunk / coverage gap -> line windows with ~10% overlap
  8  byte offsets -> 1-based lines (Chonkie offsets are BYTES: its source
     sets start_index = code_chunk.start_byte; verified for chonkie 1.7.0 by
     tests/test_code_chunker.py::test_unicode_offsets_are_bytes)
  9  symbols (qualified names) + context header, re-parsed with tree-sitter
  10 split-function parts: part i/n + group_id
  11 assemble Blocks (project fields from the registry, date from _metadata)

Contract fixes on top of Chonkie's raw output (all observed on the real
agent.ts with chonkie 1.7.0), applied in this order by chunk_ranges():
  1 snap: Chonkie cuts at AST node boundaries, sometimes mid-line
    (`export | async function`, inside a template literal) and away from a
    function's doc comment. Boundaries move to line starts, up over a
    preceding comment block; one-line fragments merge into the next chunk.
  2 fit (R6/R3): Chonkie turns the token budget into a BYTE budget using the
    file's average bytes-per-token, so some chunks overshoot. Only an
    oversized chunk is re-split -- at whole-symbol boundaries first (R1),
    lines only as the last resort; the rest of the file keeps AST chunks.
  3 merge (R2/R4): adjacent chunks are packed while they fit the budget AND
    share the same parent (same enclosing class/function, or module level).
  4 repack: a function known to be split is re-cut as a whole so its parts
    are filled to the budget (part 1 is never a lone signature line).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Block, make_chunk_id, make_group_id
from hypeonoath.indexing.block_loader import LoadedDocument, parse_markdown
from hypeonoath.indexing.project_registry import ProjectInfo
from hypeonoath.indexing.secret_scan import Finding, scan_text

logger = logging.getLogger(__name__)

# Definition-like node types, used ONLY for labels (symbols / header / parts).
# A language missing from these sets still chunks fine -- just without names.
FUNCTION_NODE_TYPES = frozenset({
    "function_definition", "function_declaration", "function_item", "method_definition",
    "method_declaration", "constructor_declaration", "function", "method", "singleton_method",
    "arrow_function_declarator", "generator_function_declaration", "local_function",
    "function_signature",
})
CONTAINER_NODE_TYPES = frozenset({
    "class_definition", "class_declaration", "class_specifier", "struct_specifier",
    "interface_declaration", "enum_declaration", "struct_item", "impl_item", "trait_item",
    "enum_item", "module", "namespace_definition", "object_declaration", "type_alias_declaration",
    "abstract_class_declaration", "class", "trait_declaration",
})
_NAME_FIELDS = ("name", "declarator")

# Comment syntax for the context header line (3A.7): "# ..." if unknown.
_COMMENT_STYLE: dict[str, tuple[str, str]] = {
    **{lang: ("//", "") for lang in (
        "javascript", "typescript", "tsx", "java", "c", "cpp", "csharp", "go", "rust", "php",
        "swift", "kotlin", "scala", "dart", "scss")},
    **{lang: ("--", "") for lang in ("sql", "lua")},
    "css": ("/*", " */"),
    "html": ("<!--", " -->"),
}


@dataclass(frozen=True)
class Symbol:
    name: str          # qualified, e.g. "Cache.get"
    kind: str          # "function" | "container"
    start_byte: int
    end_byte: int
    line_start: int
    line_end: int


@dataclass
class _Range:
    start: int
    end: int
    method: str = "ast"


@dataclass
class CodeFileResult:
    blocks: list[Block] = field(default_factory=list)
    notebook_docs: list[LoadedDocument] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def _log(**fields) -> None:
    log_json_line(config.CODE_CHUNKING_LOG_PATH, **fields)


# --- Step 1: exclusion ------------------------------------------------------------

def exclusion_reason(path: Path, rel_path: str) -> str | None:
    """Why this code file must not be indexed, or None to index it."""
    parts = set(Path(rel_path).parts[:-1])
    if parts & config.EXCLUDED_DIR_PARTS:
        return "excluded_directory"
    name = path.name.lower()
    if name.endswith(config.EXCLUDED_NAME_SUFFIXES):
        return "excluded_name"
    if path.stat().st_size > config.MAX_CODE_FILE_BYTES:
        return "too_large"
    lines = path.read_bytes().decode("utf-8", errors="replace").split("\n")
    if any(marker in line.lower() for line in lines[: config.GENERATED_MARKER_LINES]
           for marker in config.GENERATED_MARKERS):
        return "generated_file"
    if path.suffix.lower() != config.NOTEBOOK_EXTENSION:   # notebooks are one-line JSON blobs
        if any(len(line) > config.MAX_LINE_CHARS for line in lines):
            return "line_too_long"
    return None


# Virtual folder used for project-summary sources; never real code.
_RESERVED_PREFIX = config.PROJECT_SUMMARY_SOURCE_PREFIX.removeprefix("corpus/code/")


def list_code_files(code_dir: Path | None = None) -> tuple[list[str], dict[str, str]]:
    """(included rel paths, {excluded rel path: reason}) under corpus/code/."""
    code_dir = config.CODE_DIR if code_dir is None else code_dir
    included, excluded = [], {}
    for path in sorted(p for p in code_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(code_dir).as_posix()
        if path == config.CODE_METADATA_PATH or rel.startswith(_RESERVED_PREFIX):
            continue
        reason = exclusion_reason(path, rel)
        if reason:
            excluded[rel] = reason
        else:
            included.append(rel)
    return included, excluded


# --- Step 5: languages ---------------------------------------------------------------

def self_check() -> list[str]:
    """
    Make sure every mapped grammar is available up front (downloads missing
    ones once), so a missing grammar shows at startup, not mid-run. Having
    >19 grammars downloaded also stops Chonkie from calling download_all()
    on first use. Returns the names that could not be loaded.
    """
    import tree_sitter_language_pack as tslp

    failed = []
    for name in sorted(set(config.EXT_TO_LANGUAGE.values())):
        try:
            if not tslp.has_language(name):
                tslp.download([name])
            tslp.get_parser(name)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            _log(status="grammar_unavailable", language=name, reason=str(exc))
    return failed


# --- Steps 8-10: helpers ----------------------------------------------------------------

def _line_of(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def _line_span(data: bytes, start: int, end: int) -> tuple[int, int]:
    """
    1-based (first, last) line of the NON-whitespace content in [start, end).
    Chonkie ranges can begin/end inside a neighbour's indentation; counting
    from the trimmed content stops adjacent chunks claiming the same line.
    """
    chunk = data[start:end]
    first = start + len(chunk) - len(chunk.lstrip())
    last = start + len(chunk.rstrip()) - 1
    if last < first:                      # whitespace-only range
        first = last = start
    return _line_of(data, first), _line_of(data, last)


def _node_name(node) -> str | None:
    for field_name in _NAME_FIELDS:
        child = node.child_by_field_name(field_name)
        while child is not None and child.child_by_field_name("declarator") is not None:
            child = child.child_by_field_name("declarator")     # C/C++ declarators nest
        if child is not None and child.text:
            text = child.text.decode("utf-8", errors="replace")
            return text.split("(")[0].strip() or None
    return None


def extract_symbols(tree, data: bytes) -> list[Symbol]:
    """Definition-like nodes with qualified names (Class.method)."""
    symbols: list[Symbol] = []

    def visit(node, prefix: list[str]) -> None:
        kind = ("function" if node.type in FUNCTION_NODE_TYPES
                else "container" if node.type in CONTAINER_NODE_TYPES else None)
        name = _node_name(node) if kind else None
        # `const foo = () => {}` / `const foo = function () {}`
        if kind is None and node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                kind, name = "function", _node_name(node)
        child_prefix = prefix
        if kind and name:
            qualified = ".".join([*prefix, name])
            start, end = node.start_byte, node.end_byte
            symbols.append(Symbol(qualified, kind, start, end, *_line_span(data, start, end)))
            child_prefix = [*prefix, name]
        for child in node.children:
            visit(child, child_prefix)

    visit(tree.root_node, [])
    return sorted(symbols, key=lambda s: (s.start_byte, -s.end_byte))


def error_ratio(tree, size: int) -> float:
    """Share of bytes covered by top-level ERROR nodes."""
    if size == 0:
        return 0.0
    error_bytes = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.is_error:
            error_bytes += node.end_byte - node.start_byte
            continue
        if node.has_error:
            stack.extend(node.children)
    return error_bytes / size


def _core(data: bytes, rng: _Range) -> tuple[int, int]:
    """Range with surrounding whitespace trimmed (for containment tests)."""
    start, end = rng.start, rng.end
    while start < end and data[start:start + 1].isspace():
        start += 1
    while end > start and data[end - 1:end].isspace():
        end -= 1
    return start, end


def _parent(data: bytes, rng: _Range, symbols: list[Symbol]) -> int | None:
    """
    Innermost symbol strictly bigger than the chunk that contains it (R4's
    "parent"); None = module level. A chunk that IS a whole class is
    module-level, so it may still be packed with its top-level siblings.
    """
    start, end = _core(data, rng)
    best = None
    for index, sym in enumerate(symbols):
        contains = sym.start_byte <= start and end <= sym.end_byte
        strictly_bigger = sym.start_byte < start or end < sym.end_byte
        if contains and strictly_bigger:
            if best is None or (sym.end_byte - sym.start_byte) < (symbols[best].end_byte - symbols[best].start_byte):
                best = index
    return best


# --- the chunker ---------------------------------------------------------------------------

class CodeChunker:
    def __init__(self, count_tokens, tokenizer):
        self.count = count_tokens
        self.tokenizer = tokenizer
        self.prefix_tokens = count_tokens(config.DOC_PREFIX)
        self.budget = config.CODE_TARGET_TOKENS - config.HEADER_TOKEN_RESERVE - self.prefix_tokens
        if config.CODE_TARGET_TOKENS > config.EMBED_MAX_TOKENS:
            raise ValueError("CODE_TARGET_TOKENS must not exceed EMBED_MAX_TOKENS")
        if self.budget <= 0:
            raise ValueError("Code budget <= 0: raise CODE_TARGET_TOKENS or lower HEADER_TOKEN_RESERVE")

    # -- Step 7: line windows -----------------------------------------------------------

    def _line_windows(self, data: bytes, start: int, end: int, overlap_ratio: float) -> list[_Range]:
        """Windows of whole lines within [start, end), each <= budget tokens."""
        lines: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            nl = data.find(b"\n", pos, end)
            stop = end if nl == -1 else nl + 1
            lines.extend(self._hard_cut(data, pos, stop))
            pos = stop
        windows: list[_Range] = []
        i = 0
        while i < len(lines):
            j = i
            while j + 1 < len(lines) and self._tokens(data, lines[i][0], lines[j + 1][1]) <= self.budget:
                j += 1
            windows.append(_Range(lines[i][0], lines[j][1], "fallback"))
            if j + 1 >= len(lines):
                break
            # ~overlap_ratio of the window's lines repeat at the start of the next one
            back = int((j - i + 1) * overlap_ratio)
            i = max(i + 1, j + 1 - back)
        return windows

    def _hard_cut(self, data: bytes, start: int, end: int) -> list[tuple[int, int]]:
        """A single line longer than the budget is cut on token boundaries."""
        if self._tokens(data, start, end) <= self.budget:
            return [(start, end)]
        text = data[start:end].decode("utf-8", errors="replace")
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        pieces, char_start = [], 0
        for k in range(self.budget, len(offsets), self.budget):
            char_cut = offsets[k][0]
            pieces.append((char_start, char_cut))
            char_start = char_cut
        pieces.append((char_start, len(text)))
        result = []
        for a, b in pieces:
            byte_a = start + len(text[:a].encode("utf-8"))
            byte_b = start + len(text[:b].encode("utf-8"))
            if byte_b > byte_a:
                result.append((byte_a, byte_b))
        return result

    def _tokens(self, data: bytes, start: int, end: int) -> int:
        return self.count(data[start:end].decode("utf-8", errors="replace"))

    # -- Steps 6-7: ranges --------------------------------------------------------------

    def _ast_ranges(self, text: str, language: str) -> list[_Range]:
        from chonkie import CodeChunker as ChonkieCodeChunker

        chunker = ChonkieCodeChunker(language=language, tokenizer=self.tokenizer, chunk_size=self.budget)
        # Chonkie offsets are byte offsets into text.encode("utf-8").
        return [_Range(c.start_index, c.end_index) for c in chunker.chunk(text)]

    def _snap_boundaries(self, data: bytes, ranges: list[_Range]) -> list[_Range]:
        """
        Move each boundary between adjacent chunks to the start of its line,
        then up over a directly preceding comment block. Chonkie cuts at AST
        node boundaries, which can fall mid-line (`export | async function`,
        inside a template literal) and leave a function's doc comment in the
        previous chunk; whole lines keep line numbers unambiguous.
        """
        if not ranges:
            return ranges
        bounds = [ranges[0].start]
        for nxt in ranges[1:]:
            cut = self._line_start(data, nxt.start)
            while cut > bounds[-1]:
                above = self._line_start(data, cut - 1)
                line = data[above:cut].strip()
                if not line or not line.startswith(self._COMMENT_STARTS):
                    break
                cut = above
            # A fragment that lies wholly inside one line (Chonkie emits e.g. a
            # lone "export " or "`") collapses here and merges into the next chunk.
            if cut > bounds[-1] and data[bounds[-1]:cut].strip():
                bounds.append(cut)
        bounds.append(ranges[-1].end)
        return [_Range(a, b) for a, b in zip(bounds, bounds[1:])]

    def _fit(self, data: bytes, ranges: list[_Range], symbols: list[Symbol]) -> list[_Range]:
        """Re-split only the chunks over budget (R3 last resort, R6)."""
        out: list[_Range] = []
        for rng in ranges:
            if self._tokens(data, rng.start, rng.end) <= self.budget:
                out.append(rng)
            else:
                out += self._resplit(data, rng, symbols)
        return out

    def _resplit(self, data: bytes, rng: _Range, symbols: list[Symbol], exclude: Symbol | None = None) -> list[_Range]:
        """
        Cut an over-budget range at whole-symbol boundaries first (keeps every
        function that fits whole -- R1), recursing into symbols that are
        still too big; plain line windows only where no symbol boundary is
        left (R3: "line cut as a last resort").
        """
        segments = self._split_at_symbols(data, rng, symbols, exclude)
        out: list[_Range] = []
        for seg in segments:
            if self._tokens(data, seg.start, seg.end) <= self.budget:
                out.append(seg)
                continue
            whole = next((s for s in symbols if s is not exclude and s.end_byte <= seg.end
                          and self._cut_before(data, s, seg.start) == seg.start), None)
            if whole is not None and len(segments) > 1:
                out += self._resplit(data, seg, symbols, exclude=whole)
            else:
                pieces = self._line_windows(data, seg.start, seg.end, overlap_ratio=0.0)
                out += [_Range(p.start, p.end, "ast") for p in pieces]
                _log(status="code_chunk_line_split", start_byte=seg.start, end_byte=seg.end, pieces=len(pieces))
        return self._merge(data, out, symbols)

    @staticmethod
    def _line_start(data: bytes, offset: int) -> int:
        return data.rfind(b"\n", 0, offset) + 1

    _COMMENT_STARTS = (b"//", b"/*", b"*", b"#", b"@", b'"""', b"'''", b"--")

    def _cut_before(self, data: bytes, sym: Symbol, floor: int) -> int:
        """Start of the symbol's line, moved up over its doc comment / decorators."""
        cut = max(floor, self._line_start(data, sym.start_byte))
        while cut > floor:
            prev = max(floor, self._line_start(data, cut - 1))
            line = data[prev:cut].strip()
            if not line or not line.startswith(self._COMMENT_STARTS):
                break
            cut = prev
        return cut

    def _split_at_symbols(self, data: bytes, rng: _Range, symbols: list[Symbol],
                          exclude: Symbol | None = None) -> list[_Range]:
        """Cut a range before/after each outermost symbol inside it (except `exclude`)."""
        inner = [s for s in symbols if rng.start <= s.start_byte and s.end_byte <= rng.end and s is not exclude]
        outermost = [s for s in inner if not any(o is not s and o.start_byte <= s.start_byte
                                                 and s.end_byte <= o.end_byte for o in inner)]
        cuts = set()
        for sym in outermost:
            cuts.add(self._cut_before(data, sym, rng.start))
            nl = data.find(b"\n", sym.end_byte, rng.end)
            cuts.add(rng.end if nl == -1 else nl + 1)
        points = sorted(c for c in cuts if rng.start < c < rng.end)
        bounds = [rng.start, *points, rng.end]
        return [_Range(a, b) for a, b in zip(bounds, bounds[1:]) if data[a:b].strip()] or [rng]

    def _repack_split_functions(self, data: bytes, ranges: list[_Range], symbols: list[Symbol]) -> list[_Range]:
        """
        Once a function is known to be split, re-cut its WHOLE range so its
        parts are filled to the budget -- otherwise Chonkie's fragments can
        leave part 1 as a lone signature line.
        """
        for sym in sorted((s for s in symbols if s.kind == "function"), key=lambda s: s.start_byte - s.end_byte):
            idx = [i for i, r in enumerate(ranges) if r.start < sym.end_byte and sym.start_byte < r.end]
            if len(idx) < 2 or any(ranges[i].start <= sym.start_byte and sym.end_byte <= ranges[i].end for i in idx):
                continue
            lo, hi = ranges[idx[0]].start, ranges[idx[-1]].end
            fn_start = self._cut_before(data, sym, lo)
            nl = data.find(b"\n", sym.end_byte, hi)
            fn_end = hi if nl == -1 else nl + 1
            region: list[_Range] = []
            if data[lo:fn_start].strip():
                region += self._fit(data, [_Range(lo, fn_start)], symbols)
            if self._tokens(data, fn_start, fn_end) > self.budget:
                region += self._resplit(data, _Range(fn_start, fn_end), symbols, exclude=sym)
            else:
                region.append(_Range(fn_start, fn_end))
            if data[fn_end:hi].strip():
                region += self._fit(data, [_Range(fn_end, hi)], symbols)
            ranges = ranges[:idx[0]] + self._merge(data, region, symbols) + ranges[idx[-1] + 1:]
        return ranges

    def _merge(self, data: bytes, ranges: list[_Range], symbols: list[Symbol]) -> list[_Range]:
        """Pack adjacent chunks with the same parent while they fit (R2 + R4)."""
        merged: list[_Range] = []
        for rng in ranges:
            if merged:
                prev = merged[-1]
                gap = data[prev.end:rng.start]
                if (
                    prev.method == rng.method == "ast"
                    and not gap.strip()
                    and _parent(data, prev, symbols) == _parent(data, rng, symbols)
                    and self._tokens(data, prev.start, rng.end) <= self.budget
                ):
                    merged[-1] = _Range(prev.start, rng.end, "ast")
                    continue
            merged.append(rng)
        return merged

    @staticmethod
    def _covers(data: bytes, ranges: list[_Range]) -> bool:
        """R5: every non-whitespace byte is in at least one chunk."""
        covered_to = 0
        for rng in sorted(ranges, key=lambda r: r.start):
            if data[covered_to:rng.start].strip():
                return False
            covered_to = max(covered_to, rng.end)
        return not data[covered_to:].strip()

    def chunk_ranges(self, data: bytes, language: str | None, rel: str) -> tuple[list[_Range], list[Symbol], str | None]:
        """Final chunk byte ranges + symbols (+ the language actually used)."""
        text = data.decode("utf-8")
        tree, symbols, reason = None, [], None
        if language:
            import tree_sitter_language_pack as tslp

            try:
                tree = tslp.get_parser(language).parse(data)
                if language == "c" and rel.lower().endswith(".h") and error_ratio(tree, len(data)) > config.ERROR_NODE_FALLBACK_RATIO:
                    cpp_tree = tslp.get_parser("cpp").parse(data)
                    if error_ratio(cpp_tree, len(data)) < error_ratio(tree, len(data)):
                        language, tree = "cpp", cpp_tree
                symbols = extract_symbols(tree, data)
            except Exception as exc:  # noqa: BLE001
                reason = f"parser_error: {exc}"

        ranges: list[_Range] = []
        if language is None:
            reason = "unsupported_language"
        elif reason is None and tree is not None and error_ratio(tree, len(data)) > config.ERROR_NODE_FALLBACK_RATIO:
            reason = "too_many_parse_errors"
        if reason is None:
            try:
                ranges = self._ast_ranges(text, language)
            except ValueError as exc:                 # unsupported language / grammar failure
                reason = f"chonkie_error: {exc}"
        if reason is None and len(ranges) == 1 and self._tokens(data, ranges[0].start, ranges[0].end) > self.budget:
            reason = "whole_file_single_chunk"         # Chonkie's silent parse-failure mode
        if reason is None:
            ranges = self._snap_boundaries(data, ranges)
            ranges = self._merge(data, self._fit(data, ranges, symbols), symbols)
            ranges = self._repack_split_functions(data, ranges, symbols)
            if not self._covers(data, ranges):
                reason = "coverage_gap"
        if reason is not None:
            ranges = self._line_windows(data, 0, len(data), config.FALLBACK_OVERLAP_RATIO)
            _log(status="code_chunk_fallback", file=rel, reason=reason)
            logger.warning("code chunk fallback for %s: %s", rel, reason)
        return ranges, symbols, language

    # -- Steps 9-11: assembly ---------------------------------------------------------------

    def _header(self, project: ProjectInfo, rel: str, names: list[str], lines: tuple[int, int],
                language: str | None, max_names: int | None = None) -> str:
        prefix, suffix = _COMMENT_STYLE.get(language or "", ("#", ""))
        shown = names if max_names is None else names[:max_names]
        listing = ", ".join(shown)
        if max_names is not None and len(names) > max_names:
            listing = f"{listing}, … (+{len(names) - max_names} more)" if shown else f"… ({len(names)} symbols)"
        body = f"[{project.name}] {Path(rel).name}" + (f" > {listing}" if listing else "")
        return f"{prefix} {body}  [lines {lines[0]}-{lines[1]}]{suffix}"

    def build_blocks(self, data: bytes, ranges: list[_Range], symbols: list[Symbol], rel: str,
                     language: str | None, project: ProjectInfo, date: str | None,
                     extra_spans: list[dict] | None = None) -> list[Block]:
        source = f"corpus/code/{rel}"
        # Step 10: which function-like symbols got split, and their part order.
        part_info: dict[Symbol, list[int]] = {}
        for sym in symbols:
            overlapping = [i for i, r in enumerate(ranges) if r.start < sym.end_byte and sym.start_byte < r.end]
            whole = any(r.start <= sym.start_byte and sym.end_byte <= r.end for r in ranges)
            if sym.kind == "function" and not whole and len(overlapping) > 1:
                part_info[sym] = overlapping

        blocks: list[Block] = []
        for index, rng in enumerate(ranges):
            code = data[rng.start:rng.end].decode("utf-8")
            line_span = _line_span(data, rng.start, rng.end)
            inside = [s for s in symbols if rng.start < s.end_byte and s.start_byte < rng.end]
            spans, names = [], []
            for sym in inside:
                span = {"name": sym.name, "line_start": sym.line_start, "line_end": sym.line_end}
                label = sym.name
                if sym in part_info:
                    order = part_info[sym]
                    span.update(part=order.index(index) + 1, total_parts=len(order))
                    label += f" (part {span['part']}/{span['total_parts']})"
                spans.append(span)
                names.append(label)
            spans += [s for s in (extra_spans or []) if s["line_start"] <= line_span[1] and s["line_end"] >= line_span[0]]

            meta = {
                "language": language or "unknown", "line_start": line_span[0], "line_end": line_span[1],
                "symbols": [s.name for s in inside], "symbol_spans": spans, "chunk_method": rng.method,
                "start_byte": rng.start, "end_byte": rng.end, "project_key": project.key,
                "project_name": project.name, "project_description": project.description, "date": date,
            }
            split = [s for s in inside if s in part_info]
            if split:
                # Top-level part fields describe the OUTERMOST split function
                # (fetching its group also brings back any nested split code).
                outer = min(split, key=lambda s: (s.start_byte, -s.end_byte))
                order = part_info[outer]
                meta.update(part=order.index(index) + 1, total_parts=len(order),
                            group_id=make_group_id(source, outer.name, outer.start_byte))
                unrelated = [s for s in split if s is not outer and not (outer.start_byte <= s.start_byte and s.end_byte <= outer.end_byte)]
                if unrelated:
                    _log(status="multi_split_chunk", file=rel, line_start=line_span[0],
                         symbols=[outer.name, *[s.name for s in unrelated]])

            header = self._header(project, rel, names, line_span, language)
            code_tokens = self.count(code)          # counted once; header re-counted per try
            max_names = len(names)
            while self.count(header) + code_tokens + self.prefix_tokens > config.CODE_TARGET_TOKENS and max_names > 0:
                max_names -= 1          # list fewer names in the HEADER only; metadata keeps all
                header = self._header(project, rel, names, line_span, language, max_names)
            text = header + "\n" + code
            if self.count(text) + self.prefix_tokens > config.CODE_TARGET_TOKENS:
                _log(status="header_over_reserve", file=rel, line_start=line_span[0])
            meta["chunk_id"] = make_chunk_id(source, rng.start, rng.end, text)
            blocks.append(Block("code_chunk", text, source, [], meta))
        return blocks

    # -- Steps 2-4 + entry point ---------------------------------------------------------------

    def chunk_file(self, path: Path, rel: str, project: ProjectInfo, date: str | None) -> CodeFileResult:
        result = CodeFileResult()
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            _log(status="non_utf8", file=rel)
        if not text.strip():
            _log(status="empty_file_skipped", file=rel)
            return result

        extra_spans: list[dict] = []
        if path.suffix.lower() == config.NOTEBOOK_EXTENSION:
            text, language, extra_spans, result.notebook_docs = self._notebook(text, rel, project, date)
        else:
            language = config.EXT_TO_LANGUAGE.get(path.suffix.lower())
        # Scan the text that gets indexed: for notebooks that is the extracted
        # cells (outputs are never indexed), with the same line numbers as
        # the code chunks, so raw and chunk findings deduplicate.
        result.findings = scan_text(text, f"corpus/code/{rel}")
        if not text.strip():
            return result

        data = text.encode("utf-8")   # decoded-with-replace text: offsets refer to this
        ranges, symbols, language = self.chunk_ranges(data, language, rel)
        result.blocks = self.build_blocks(data, ranges, symbols, rel, language, project, date, extra_spans)
        return result

    def _notebook(self, text: str, rel: str, project: ProjectInfo, date: str | None):
        """
        Step 4: code cells joined with "# %% [cell N]" separators (cell
        index kept in symbol_spans); markdown cells -> prose; outputs dropped.
        Line numbers of notebook chunks refer to this joined code text.
        """
        import nbformat

        nb = nbformat.reads(text, as_version=4)
        meta = nb.get("metadata", {})
        language = (meta.get("kernelspec", {}).get("language")
                    or meta.get("language_info", {}).get("name")
                    or config.NOTEBOOK_DEFAULT_LANGUAGE).lower()
        language = language if language in set(config.EXT_TO_LANGUAGE.values()) else config.NOTEBOOK_DEFAULT_LANGUAGE
        prefix = _COMMENT_STYLE.get(language, ("#", ""))[0]

        code_parts, spans, line = [], [], 1
        markdown_cells = []
        for index, cell in enumerate(nb.cells):
            source = cell.get("source", "")
            if cell.cell_type == "code" and source.strip():
                piece = f"{prefix} %% [cell {index}]\n{source.rstrip()}\n"
                n_lines = piece.count("\n")
                spans.append({"name": f"cell {index}", "line_start": line, "line_end": line + n_lines - 1})
                code_parts.append(piece)
                line += n_lines + 1          # +1 for the blank separator line
            elif cell.cell_type == "markdown" and source.strip():
                markdown_cells.append(source)
            # cell outputs are never read: they can hold images, data dumps or secrets

        docs = []
        if markdown_cells:
            source_doc = f"corpus/code/{rel}"
            blocks = parse_markdown("\n\n".join(markdown_cells), source_doc)
            for block in blocks:          # notebook markdown has no stable file lines
                block.meta.pop("line_start", None)
                block.meta.pop("line_end", None)
            docs.append(LoadedDocument(source_doc, blocks, project.name, date))
        return "\n".join(code_parts), language, spans, docs


def chunk_code_files(resolved: dict[str, ProjectInfo], code_metadata: dict, count_tokens, tokenizer) -> CodeFileResult:
    """All (already validated) code files -> one combined result."""
    chunker = CodeChunker(count_tokens, tokenizer)
    failed = self_check()
    if failed:
        logger.warning("Grammars unavailable (files fall back to line windows): %s", failed)
    combined = CodeFileResult()
    for rel in sorted(resolved):
        entry = code_metadata.get(rel) or {}
        result = chunker.chunk_file(config.CODE_DIR / rel, rel, resolved[rel], entry.get("date"))
        combined.blocks += result.blocks
        combined.notebook_docs += result.notebook_docs
        combined.findings += result.findings
    return combined
