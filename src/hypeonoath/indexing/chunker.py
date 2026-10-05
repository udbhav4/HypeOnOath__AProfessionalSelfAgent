"""
Step 2 -- typed blocks -> Chunk records (guideline v4, Section 3).

Per-block-type rules (design doc 5.1, v1.7):
  prose / list  grouped under their heading path; ~300-500 tokens per chunk,
                ~10-15% overlap between consecutive chunks of the SAME group
                (never across headings -- headings are free semantic
                boundaries). Split at sentence / list-item boundaries.
  table         one chunk per table, or one `record` chunk per row when the
                table is a real record list (uniform rows + header-like
                first row) -- so a fact is never separated from its label.
  code_fence    kept intact, never split; over the model limit -> logged,
                not silently truncated.
  code_chunk    already sized by code_chunker.py -> passed through 1:1.

Every prose/table/fence chunk's text starts with its heading path line
("ABOUT ME" / "PROJECTS > Gliimr"): it is embedded and shown to the LLM, so
a chunk keeps its context ("which project is this bullet about?").

Token counts come from the embedding model's own tokenizer (count_tokens is
injected), and the "search_document: " prefix is reserved inside every
budget, because it is embedded too.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Block, Chunk, make_chunk_id
from hypeonoath.indexing.block_loader import LoadedDocument

logger = logging.getLogger(__name__)

CountTokens = Callable[[str], int]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_NUMERIC_RE = re.compile(r"^[\d\s.,:%/+-]*$")
_HEADER_CELL_MAX_CHARS = 60


@dataclass
class _Unit:
    text: str
    sep: str           # joiner placed before this unit when it isn't first
    block_index: int
    line_start: int | None
    line_end: int | None


def _heading_line(heading_path: list[str]) -> str:
    return " > ".join(heading_path)


def _with_heading(heading_path: list[str], body: str) -> str:
    line = _heading_line(heading_path)
    return f"{line}\n{body}" if line else body


def _units_for_block(block: Block, index: int) -> list[_Unit]:
    """Split a prose/list block at sentence / item boundaries."""
    ls, le = block.meta.get("line_start"), block.meta.get("line_end")
    if block.block_type == "list":
        pieces, sep = [p for p in block.text.split("\n") if p.strip()], "\n"
    else:
        pieces, sep = [p for p in _SENTENCE_SPLIT_RE.split(block.text.strip()) if p.strip()], " "
    return [_Unit(p, sep, index, ls, le) for p in pieces]


def _split_oversized(text: str, budget: int, count_tokens: CountTokens) -> list[str]:
    """
    Last resort for one unit bigger than the budget: split on words, then
    halve any single "word" that is still too big (a long URL or path with
    no spaces), so no piece can exceed the budget.
    """
    words, pieces, current = text.split(), [], []
    for word in words:
        candidate = " ".join([*current, word])
        if current and count_tokens(candidate) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))

    def halve(piece: str) -> list[str]:
        if count_tokens(piece) <= budget or len(piece) < 2:
            return [piece]
        mid = len(piece) // 2
        return halve(piece[:mid]) + halve(piece[mid:])

    return [part for piece in pieces for part in halve(piece)]


def _pack_lines(lines: list[str], budget: int, count_tokens: CountTokens) -> list[str]:
    """Greedily pack whole lines into pieces <= budget (long lines word-split)."""
    pieces: list[str] = []
    current: list[str] = []
    for line in lines:
        units = [line] if count_tokens(line) <= budget else _split_oversized(line, budget, count_tokens)
        for unit in units:
            if current and count_tokens("\n".join([*current, unit])) > budget:
                pieces.append("\n".join(current))
                current = []
            current.append(unit)
    if current:
        pieces.append("\n".join(current))
    return pieces


class Chunker:
    def __init__(self, count_tokens: CountTokens):
        self.count = count_tokens
        self.prefix_tokens = count_tokens(config.DOC_PREFIX)

    # -- prose groups -------------------------------------------------------

    def _join(self, units: list[_Unit]) -> str:
        out = ""
        for n, unit in enumerate(units):
            if n == 0:
                out = unit.text
            else:
                prev = units[n - 1]
                out += ("\n\n" if unit.block_index != prev.block_index else unit.sep) + unit.text
        return out

    def _chunk_group(self, doc: LoadedDocument, blocks: list[tuple[int, Block]]) -> list[Chunk]:
        heading_path = blocks[0][1].heading_path
        heading_tokens = self.count(_heading_line(heading_path)) if heading_path else 0
        budget = config.PROSE_MAX_TOKENS - self.prefix_tokens - heading_tokens - 1
        overlap_budget = int(config.PROSE_MAX_TOKENS * config.PROSE_OVERLAP_RATIO)

        units: list[_Unit] = []
        for index, block in blocks:
            for unit in _units_for_block(block, index):
                if self.count(unit.text) > budget:
                    units += [_Unit(p, " ", unit.block_index, unit.line_start, unit.line_end)
                              for p in _split_oversized(unit.text, budget, self.count)]
                else:
                    units.append(unit)

        windows: list[list[_Unit]] = []
        current: list[_Unit] = []
        for unit in units:
            if current and self.count(self._join([*current, unit])) > budget:
                windows.append(current)
                # Overlap: carry trailing units of the previous window, but
                # never the whole window (guarantees forward progress).
                carry: list[_Unit] = []
                for prev in reversed(current[1:]):
                    if self.count(self._join([prev, *carry])) > overlap_budget:
                        break
                    carry.insert(0, prev)
                current = carry if self.count(self._join([*carry, unit])) <= budget else []
            current.append(unit)
        if current:
            windows.append(current)

        chunks = []
        for sub_index, window in enumerate(windows):
            body = self._join(window)
            starts = [u.line_start for u in window if u.line_start is not None]
            ends = [u.line_end for u in window if u.line_end is not None]
            chunks.append(self._doc_chunk(doc, "prose", body, heading_path, window[0].block_index,
                                          sub_index, min(starts, default=None), max(ends, default=None)))
        return chunks

    # -- tables -------------------------------------------------------------

    @staticmethod
    def _is_record_list(rows: list[list[str]]) -> bool:
        """Uniform column count + a header-like first row (guideline Section 3)."""
        if len(rows) < 2 or len({len(r) for r in rows}) != 1:
            return False
        return all(
            cell and len(cell) <= _HEADER_CELL_MAX_CHARS and not _NUMERIC_RE.match(cell)
            for cell in rows[0]
        )

    def _chunk_table(self, doc: LoadedDocument, index: int, block: Block) -> list[Chunk]:
        rows: list[list[str]] = block.meta.get("rows") or []
        ls, le = block.meta.get("line_start"), block.meta.get("line_end")
        if self._is_record_list(rows):
            header = rows[0]
            heading_tokens = self.count(_heading_line(block.heading_path)) if block.heading_path else 0
            budget = config.PROSE_MAX_TOKENS - self.prefix_tokens - heading_tokens - 1
            chunks = []
            for row_n, row in enumerate(rows[1:], start=1):
                fields = [f"{h}: {v}" for h, v in zip(header, row) if v]
                if not fields:
                    continue
                # Markdown rows sit on line_start + 1 (header) + 1 (separator) + n - 1.
                line = ls + row_n + 1 if ls is not None and block.meta.get("has_header") else None
                # A row over budget is split at field boundaries (each piece
                # keeps its "Column:" labels), never mid-label.
                for k, body in enumerate(_pack_lines(fields, budget, self.count)):
                    chunks.append(self._doc_chunk(doc, "record", body, block.heading_path, index, f"{row_n}.{k}",
                                                  line if line is not None else ls, line if line is not None else le))
            return chunks

        text = block.text
        if self.count(_with_heading(block.heading_path, text)) + self.prefix_tokens <= config.PROSE_MAX_TOKENS:
            return [self._doc_chunk(doc, "table", text, block.heading_path, index, 0, ls, le)]
        # Oversized non-record table: split into row groups, repeating the
        # header lines so every piece keeps its column labels.
        lines = text.split("\n")
        head = lines[:2] if len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ") else []
        body_lines = lines[len(head):]
        budget = config.PROSE_MAX_TOKENS
        chunks, group = [], []
        for line in body_lines:
            candidate = "\n".join([*head, *group, line])
            if group and self.count(_with_heading(block.heading_path, candidate)) + self.prefix_tokens > budget:
                chunks.append("\n".join([*head, *group]))
                group = []
            group.append(line)
        if group:
            chunks.append("\n".join([*head, *group]))
        return [self._doc_chunk(doc, "table", t, block.heading_path, index, n, ls, le) for n, t in enumerate(chunks)]

    # -- fences ---------------------------------------------------------------

    def _chunk_fence(self, doc: LoadedDocument, index: int, block: Block) -> Chunk:
        chunk = self._doc_chunk(doc, "code_fence", block.text, block.heading_path, index, 0,
                                block.meta.get("line_start"), block.meta.get("line_end"))
        if self.count(chunk.text) + self.prefix_tokens > config.EMBED_MAX_TOKENS:
            # Kept intact by rule; make the embed-time truncation visible.
            logger.warning("code fence over %d tokens: %s line %s", config.EMBED_MAX_TOKENS,
                           doc.source_document, chunk.line_start)
            log_json_line(config.DOCUMENT_LOADING_LOG_PATH, file=doc.source_document,
                          line=chunk.line_start, status="oversize_code_fence")
        return chunk

    # -- shared ---------------------------------------------------------------

    def _doc_chunk(self, doc: LoadedDocument, content_type: str, body: str, heading_path: list[str],
                   block_index: int, sub_index: int | str, line_start: int | None, line_end: int | None) -> Chunk:
        text = _with_heading(heading_path, body)
        return Chunk(
            chunk_id=make_chunk_id(doc.source_document, block_index, sub_index, text),
            text=text,
            source_document=doc.source_document,
            content_type=content_type,
            project_name=doc.project_name,
            date=doc.date,
            heading_path=list(heading_path),
            line_start=line_start,
            line_end=line_end,
        )

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        """Apply the per-block-type rules to one document, in reading order."""
        chunks: list[Chunk] = []
        group: list[tuple[int, Block]] = []

        def flush() -> None:
            if group:
                chunks.extend(self._chunk_group(doc, group))
                group.clear()

        for index, block in enumerate(doc.blocks):
            if block.block_type in ("prose", "list"):
                if group and group[-1][1].heading_path != block.heading_path:
                    flush()
                group.append((index, block))
                continue
            flush()
            if block.block_type == "table":
                chunks.extend(self._chunk_table(doc, index, block))
            elif block.block_type == "code_fence":
                chunks.append(self._chunk_fence(doc, index, block))
            elif block.block_type == "code_chunk":
                chunks.append(code_block_to_chunk(block))
            # "heading" blocks only define boundaries / heading_path.
        flush()
        return chunks


def code_block_to_chunk(block: Block) -> Chunk:
    """`code_chunk` blocks are finished chunks: convert 1:1, no re-sizing."""
    meta = block.meta
    return Chunk(
        chunk_id=meta["chunk_id"],
        text=block.text,
        source_document=block.source_document,
        content_type="code",
        project_name=meta.get("project_name"),
        date=meta.get("date"),
        heading_path=[],
        line_start=meta.get("line_start"),
        line_end=meta.get("line_end"),
        project_key=meta.get("project_key"),
        project_description=meta.get("project_description"),
        language=meta.get("language"),
        symbols=list(meta.get("symbols") or []),
        symbol_spans=list(meta.get("symbol_spans") or []),
        part=meta.get("part"),
        total_parts=meta.get("total_parts"),
        group_id=meta.get("group_id"),
        chunk_method=meta.get("chunk_method"),
        start_byte=meta.get("start_byte"),
        end_byte=meta.get("end_byte"),
    )


def make_chunks(documents: list[LoadedDocument], code_blocks: list[Block], count_tokens: CountTokens) -> list[Chunk]:
    """All documents + all code_chunk blocks -> Chunk records."""
    chunker = Chunker(count_tokens)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))
    chunks.extend(code_block_to_chunk(b) for b in code_blocks)
    return chunks
