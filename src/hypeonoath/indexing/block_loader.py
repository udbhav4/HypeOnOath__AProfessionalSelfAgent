"""
Step 1 -- document -> typed blocks (guideline v4, Section 2; design doc 5.1).

Two document paths, one output contract (records.Block):

  * PDF/DOCX-origin (corpus/converted/<f>.md with a sibling <f>.docling.json):
    load docling's own typed structure (SectionHeaderItem, TextItem,
    ListItem, TableItem, ...) instead of re-parsing the flattened Markdown.
    If the JSON is missing or fails to load, fall back to the Markdown path
    on the .md sibling and log it (same "degrade, log, keep going" approach
    as Phase 0's converter.py).
  * Native .md/.txt (corpus/documents/): walk markdown-it-py's block-token
    stream. Fenced code is one atomic `code_fence` block, never split.

Code files are NOT handled here -- code_chunker.py emits finished
`code_chunk` blocks for those.

Line numbers (meta line_start/line_end) always refer to the .md/.txt file
that was read, which is also the chunk's source_document -- so citations and
scan warnings point at real lines. For docling items they are located by a
best-effort text search in the .md sibling (docling keeps page provenance,
not Markdown lines).

Note: Phase 0's link_extractor appends a "## Links" section to the .md only;
it is not part of the docling JSON. It is re-parsed from the .md and appended
so profile links (LinkedIn/GitHub) stay retrievable.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Block
from hypeonoath.ingestion import config as ingestion_config
from hypeonoath.ingestion.cleaner import resolve_document_target
from hypeonoath.ingestion.manifest import load_manifest

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_LINKS_HEADING_RE = re.compile(r"(?m)^## Links[ \t]*$")
_WS_RE = re.compile(r"\s+")

# docling text labels that are page furniture, not content.
_SKIPPED_DOCLING_LABELS = frozenset({"page_header", "page_footer"})


@dataclass
class LoadedDocument:
    """One document's blocks plus the document-level metadata from Phase 0."""

    source_document: str                 # relative to project root, posix
    blocks: list[Block] = field(default_factory=list)
    project_name: str | None = None
    date: str | None = None


def rel_to_root(path: Path) -> str:
    return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()


def split_front_matter(text: str) -> tuple[dict, str, int]:
    """
    Return (front-matter dict, body, number of lines the front-matter took),
    so body line numbers can be shifted back to file line numbers.
    """
    text = text.replace("\r\n", "\n")
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text, 0
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[match.end():], match.group(0).count("\n")


# --- Markdown path -----------------------------------------------------------

def _markdown_parser() -> MarkdownIt:
    return MarkdownIt("commonmark").enable("table")


def _close_index(tokens: list, start: int) -> int:
    """Index of the *_close token matching the *_open token at `start`."""
    close_type = tokens[start].type.replace("_open", "_close")
    level = tokens[start].level
    for j in range(start + 1, len(tokens)):
        if tokens[j].type == close_type and tokens[j].level == level:
            return j
    return len(tokens) - 1


def _table_rows(tokens: list, start: int, end: int) -> tuple[list[list[str]], bool]:
    rows: list[list[str]] = []
    has_header = False
    current: list[str] | None = None
    for tok in tokens[start:end]:
        if tok.type == "thead_open":
            has_header = True
        elif tok.type == "tr_open":
            current = []
        elif tok.type == "inline" and current is not None:
            current.append(tok.content.strip())
        elif tok.type == "tr_close" and current is not None:
            rows.append(current)
            current = None
    return rows, has_header


def parse_markdown(
    body: str,
    source_document: str,
    line_offset: int = 0,
    base_heading_path: list[str] | None = None,
) -> list[Block]:
    """Walk markdown-it-py's block tokens into typed Blocks."""
    tokens = _markdown_parser().parse(body)
    lines = body.split("\n")
    stack: list[tuple[int, str]] = []          # (heading level, text)
    base = list(base_heading_path or [])
    blocks: list[Block] = []

    def heading_path() -> list[str]:
        return base + [text for _, text in stack]

    def make(block_type: str, token, text: str, extra: dict | None = None) -> None:
        start, end = token.map if token.map else (0, 0)
        meta = {"line_start": start + 1 + line_offset, "line_end": end + line_offset}
        meta.update(extra or {})
        if text.strip():
            blocks.append(Block(block_type, text.rstrip("\n"), source_document, heading_path(), meta))

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.level != 0:
            i += 1
            continue
        src = "\n".join(lines[tok.map[0]:tok.map[1]]) if tok.map else ""

        if tok.type == "heading_open":
            level = int(tok.tag[1])
            title = tokens[i + 1].content.strip() if i + 1 < len(tokens) else ""
            while stack and stack[-1][0] >= level:
                stack.pop()
            make("heading", tok, title)
            stack.append((level, title))
            i = _close_index(tokens, i) + 1
        elif tok.type in ("paragraph_open", "blockquote_open", "html_block"):
            make("prose", tok, src)
            i = _close_index(tokens, i) + 1 if tok.nesting == 1 else i + 1
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            make("list", tok, src)
            i = _close_index(tokens, i) + 1
        elif tok.type == "table_open":
            end = _close_index(tokens, i)
            rows, has_header = _table_rows(tokens, i, end)
            make("table", tok, src, {"rows": rows, "has_header": has_header})
            i = end + 1
        elif tok.type in ("fence", "code_block"):
            make("code_fence", tok, src, {"info": tok.info.strip()})
            i += 1
        else:  # hr and anything else carries no retrievable content
            i = _close_index(tokens, i) + 1 if tok.nesting == 1 else i + 1
    return blocks


# --- docling path --------------------------------------------------------------

def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(text)).strip().lower()


class _LineLocator:
    """Best-effort mapping of docling item text to a line in the .md sibling."""

    def __init__(self, body: str, line_offset: int):
        self.lines = [_normalize(line) for line in body.split("\n")]
        self.offset = line_offset
        self.cursor = 0  # items come in reading order -> search forward first

    def locate(self, text: str) -> tuple[int | None, int | None]:
        needle = _normalize(text)[:40]
        if not needle:
            return None, None
        order = list(range(self.cursor, len(self.lines))) + list(range(0, self.cursor))
        for idx in order:
            if needle in self.lines[idx]:
                self.cursor = idx
                span = max(1, text.count("\n") + 1)
                return idx + 1 + self.offset, idx + span + self.offset
        return None, None


def parse_docling(json_path: Path, body: str, source_document: str, line_offset: int) -> list[Block]:
    """Docling's typed items -> Blocks. Raises on load failure (caller falls back)."""
    from docling_core.types.doc import (  # heavy import, only when needed
        CodeItem,
        DoclingDocument,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    doc = DoclingDocument.load_from_json(json_path)
    locator = _LineLocator(body, line_offset)
    stack: list[tuple[int, str]] = []
    blocks: list[Block] = []

    def add(block_type: str, text: str, locate_text: str | None = None, extra: dict | None = None) -> None:
        if not text.strip():
            return
        line_start, line_end = locator.locate(locate_text or text)
        meta = {"line_start": line_start, "line_end": line_end, **(extra or {})}
        path = [t for _, t in stack]
        # Merge consecutive list items under the same heading into one list block.
        if block_type == "list" and blocks and blocks[-1].block_type == "list" and blocks[-1].heading_path == path:
            blocks[-1].text += "\n" + text
            if line_end is not None:
                blocks[-1].meta["line_end"] = line_end
            return
        blocks.append(Block(block_type, text, source_document, path, meta))

    for item, _level in doc.iterate_items():
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            level = getattr(item, "level", 1) if isinstance(item, SectionHeaderItem) else 0
            title = item.text.strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            add("heading", title)
            stack.append((level, title))
        elif isinstance(item, TableItem):
            grid = item.data.grid
            rows = [[cell.text.strip() for cell in row] for row in grid]
            has_header = bool(grid) and all(getattr(cell, "column_header", False) for cell in grid[0])
            first_cell = rows[0][0] if rows and rows[0] else ""
            add("table", item.export_to_markdown(doc=doc), first_cell, {"rows": rows, "has_header": has_header})
        elif isinstance(item, CodeItem):
            add("code_fence", f"```\n{item.text}\n```", item.text)
        elif isinstance(item, ListItem):
            add("list", f"- {item.text.strip()}", item.text)
        elif isinstance(item, PictureItem):
            continue  # images carry no text; captions arrive as TextItems
        elif isinstance(item, TextItem):
            if str(item.label) in _SKIPPED_DOCLING_LABELS:
                continue
            add("prose", item.text.strip())
    return blocks


def _links_section(body: str, source_document: str, line_offset: int) -> list[Block]:
    """Re-parse Phase 0's appended '## Links' section from the .md sibling."""
    match = _LINKS_HEADING_RE.search(body)
    if not match:
        return []
    offset = line_offset + body.count("\n", 0, match.start())
    return parse_markdown(body[match.start():], source_document, offset)


# --- Entry points --------------------------------------------------------------

def load_markdown_file(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body, offset = split_front_matter(raw)
    source = rel_to_root(path)
    return LoadedDocument(source, parse_markdown(body, source, offset), meta.get("project_name"), meta.get("date"))


def load_converted_file(md_path: Path) -> LoadedDocument:
    """PDF/DOCX-origin document: docling JSON first, Markdown fallback."""
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    meta, body, offset = split_front_matter(raw)
    source = rel_to_root(md_path)
    json_path = md_path.with_name(md_path.name[: -len(md_path.suffix)] + config.DOCLING_JSON_SUFFIX)

    blocks: list[Block] | None = None
    if json_path.exists():
        try:
            blocks = parse_docling(json_path, body, source, offset)
            if not any(b.block_type == "heading" and b.text == "Links" for b in blocks):
                blocks += _links_section(body, source, offset)
        except Exception as exc:  # noqa: BLE001 -- any load failure degrades to Markdown
            logger.warning("docling load failed for %s (%s); using Markdown", json_path, exc)
            log_json_line(config.DOCUMENT_LOADING_LOG_PATH, file=source, status="docling_fallback", reason=str(exc))
    else:
        log_json_line(config.DOCUMENT_LOADING_LOG_PATH, file=source, status="docling_missing")

    if blocks is None:
        blocks = parse_markdown(body, source, offset)
    return LoadedDocument(source, blocks, meta.get("project_name"), meta.get("date"))


def load_documents(manifest: dict | None = None) -> list[LoadedDocument]:
    """
    Load every document Phase 0 produced, using Phase 0's own target
    resolution (converted .md for PDF/DOCX, the original for .md/.txt), so
    the two phases can't disagree about which file holds a document's text.
    """
    manifest = load_manifest(ingestion_config.MANIFEST_PATH) if manifest is None else manifest
    documents: list[LoadedDocument] = []
    for rel_path in sorted(manifest):
        target = resolve_document_target(rel_path, manifest[rel_path])
        if target is None:
            continue
        path, is_converted = target
        if not path.exists():
            log_json_line(config.DOCUMENT_LOADING_LOG_PATH, file=rel_path, status="missing_target")
            continue
        documents.append(load_converted_file(path) if is_converted else load_markdown_file(path))
    return documents
