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

Retrieval experiment F-R3 (eval plan v29, D95), two flags, both OFF by default
(config.SMALL_SECTIONS_MERGED / DOC_DESCRIPTION_IN_CHUNKS, D98); with both off
the chunks are byte-identical to the rules above:
  merge_small   the general sibling rule. Every unit the chunker splits -- a
                section (heading group) under a parent heading, or a row of a
                record-list table -- that is under PROSE_MIN_TOKENS absorbs the
                next sibling unit (next section under the SAME parent heading,
                next row of the SAME table) while the result stays <=
                PROSE_MAX_TOKENS. A unit of PROSE_MIN_TOKENS or more stands
                alone (a section split into several windows is such a unit, so
                is a row split at field boundaries). Sections keep their heading
                line, rows their "Header: value" labels. Top-level sections (no
                common parent heading) merge only when the whole document is
                under PROSE_MIN_TOKENS. Never merged: across documents, parent
                headings or tables; tables with sections; code / fences.
  descriptions  one first line "document title -- description" on every chunk
                (title = the document's registry file name without extension);
                code chunks get "project name -- project description". The line
                counts inside every document-chunk budget; code chunks were sized
                by code_chunker before, so there it comes on top of
                CODE_TARGET_TOKENS (still far below EMBED_MAX_TOKENS). Project summaries already are
                the project's name + description and get no line.

Token counts come from the embedding model's own tokenizer (count_tokens is
injected), and the "search_document: " prefix is reserved inside every
budget, because it is embedded too.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from hypeonoath.core import config, document_registry, registry
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Block, Chunk, make_chunk_id
from hypeonoath.indexing.block_loader import LoadedDocument

logger = logging.getLogger(__name__)

CountTokens = Callable[[str], int]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_NUMERIC_RE = re.compile(r"^[\d\s.,:%/+-]*$")
_HEADER_CELL_MAX_CHARS = 60


DESCRIPTION_SEPARATOR = " — "        # "title — description" (D95)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def description_line(source_document: str, project_name: str | None = None) -> str | None:
    """
    F-R3 prefix line for a document's chunks: "<file stem> — <registry description>"; a notebook (indexed as a
    document under corpus/code/) gets its project's "name — description". None when nothing is registered.
    """
    key = document_registry.document_key(source_document)
    if key is not None:
        info = document_registry.document_info(source_document)
        if info and info.description.strip():
            return f"{PurePosixPath(key).stem}{DESCRIPTION_SEPARATOR}{_one_line(info.description)}"
        return None
    project = next((p for p in registry.CODE_PROJECTS.values() if p.get("name") == project_name), None)
    return project_line(project.get("name"), project.get("description")) if project else None


def project_line(name: str | None, description: str | None) -> str | None:
    """F-R3 prefix line of a code chunk: "<project name> — <project description>" (one line)."""
    if not name or not (description or "").strip():
        return None
    return f"{_one_line(name)}{DESCRIPTION_SEPARATOR}{_one_line(description)}"


@dataclass
class _Draft:
    """A chunk before its id exists: the sibling rule packs drafts, then each becomes one Chunk."""
    content_type: str
    text: str                 # full chunk text, heading line(s) included, description line not yet
    heading_path: list[str]
    block_index: int
    sub_index: int | str
    line_start: int | None
    line_end: int | None
    # Drafts with equal keys, adjacent in reading order, are siblings; None = never merged (tables, fences,
    # code, and every piece of a split section / row: a split unit stands alone).
    sibling_key: tuple | None = None
    chunk: Chunk | None = None          # a finished chunk (code_chunk block): passed through, never merged


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
    def __init__(self, count_tokens: CountTokens, *, merge_small: bool | None = None,
                 descriptions: bool | None = None):
        """F-R3 flags: None = the config default when the Chunker is made (off until a keep decision, D98)."""
        self.count = count_tokens
        self.prefix_tokens = count_tokens(config.DOC_PREFIX)
        self.merge_small = config.SMALL_SECTIONS_MERGED if merge_small is None else merge_small
        self.descriptions = config.DOC_DESCRIPTION_IN_CHUNKS if descriptions is None else descriptions
        # Per document (set in chunk_document): the F-R3 description line and the tokens it takes
        # from every budget (line + newline). None / 0 while descriptions are off.
        self._description: str | None = None
        self._reserve = 0

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

    def _chunk_group(self, blocks: list[tuple[int, Block]], parent_opener: int | None = None) -> list[_Draft]:
        heading_path = blocks[0][1].heading_path
        heading_tokens = self.count(_heading_line(heading_path)) if heading_path else 0
        budget = config.PROSE_MAX_TOKENS - self.prefix_tokens - heading_tokens - 1 - self._reserve
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

        # Sibling rule (D95): a section is one unit only when it fits one window; a split section stands alone.
        # The key names the parent heading by text AND by the block that opened it, so two parents with the
        # same text ("# Cert" twice) stay different parents.
        key = ("section", tuple(heading_path[:-1]), parent_opener) if len(windows) == 1 else None
        drafts = []
        for sub_index, window in enumerate(windows):
            starts = [u.line_start for u in window if u.line_start is not None]
            ends = [u.line_end for u in window if u.line_end is not None]
            drafts.append(_Draft("prose", _with_heading(heading_path, self._join(window)), list(heading_path),
                                 window[0].block_index, sub_index, min(starts, default=None),
                                 max(ends, default=None), key))
        return drafts

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

    def _chunk_table(self, index: int, block: Block) -> list[_Draft]:
        rows: list[list[str]] = block.meta.get("rows") or []
        ls, le = block.meta.get("line_start"), block.meta.get("line_end")
        heading = block.heading_path
        if self._is_record_list(rows):
            header = rows[0]
            heading_tokens = self.count(_heading_line(heading)) if heading else 0
            budget = config.PROSE_MAX_TOKENS - self.prefix_tokens - heading_tokens - 1 - self._reserve
            drafts = []
            for row_n, row in enumerate(rows[1:], start=1):
                fields = [f"{h}: {v}" for h, v in zip(header, row) if v]
                if not fields:
                    continue
                # Markdown rows sit on line_start + 1 (header) + 1 (separator) + n - 1.
                line = ls + row_n + 1 if ls is not None and block.meta.get("has_header") else None
                # A row over budget is split at field boundaries (each piece
                # keeps its "Column:" labels), never mid-label.
                pieces = _pack_lines(fields, budget, self.count)
                for k, body in enumerate(pieces):
                    drafts.append(_Draft("record", _with_heading(heading, body), list(heading), index,
                                         f"{row_n}.{k}", line if line is not None else ls,
                                         line if line is not None else le,
                                         ("rows", index) if len(pieces) == 1 else None))
            return drafts

        text = block.text
        if (self.count(_with_heading(heading, text)) + self.prefix_tokens + self._reserve
                <= config.PROSE_MAX_TOKENS):
            return [_Draft("table", _with_heading(heading, text), list(heading), index, 0, ls, le)]
        # Oversized non-record table: split into row groups, repeating the
        # header lines so every piece keeps its column labels.
        lines = text.split("\n")
        head = lines[:2] if len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ") else []
        body_lines = lines[len(head):]
        budget = config.PROSE_MAX_TOKENS - self._reserve
        pieces, group = [], []
        for line in body_lines:
            candidate = "\n".join([*head, *group, line])
            if group and self.count(_with_heading(heading, candidate)) + self.prefix_tokens > budget:
                pieces.append("\n".join([*head, *group]))
                group = []
            group.append(line)
        if group:
            pieces.append("\n".join([*head, *group]))
        return [_Draft("table", _with_heading(heading, t), list(heading), index, n, ls, le)
                for n, t in enumerate(pieces)]

    # -- fences ---------------------------------------------------------------

    def _chunk_fence(self, doc: LoadedDocument, index: int, block: Block) -> _Draft:
        draft = _Draft("code_fence", _with_heading(block.heading_path, block.text), list(block.heading_path),
                       index, 0, block.meta.get("line_start"), block.meta.get("line_end"))
        if self._size(draft.text) > config.EMBED_MAX_TOKENS:
            # Kept intact by rule; make the embed-time truncation visible.
            logger.warning("code fence over %d tokens: %s line %s", config.EMBED_MAX_TOKENS,
                           doc.source_document, draft.line_start)
            log_json_line(config.DOCUMENT_LOADING_LOG_PATH, file=doc.source_document,
                          line=draft.line_start, status="oversize_code_fence")
        return draft

    # -- sibling rule (F-R3, D95) ----------------------------------------------

    def _size(self, text: str) -> int:
        """Tokens of a chunk as embedded: description line (if on) + text + the "search_document: " prefix."""
        return self.count(self._final_text(text)) + self.prefix_tokens

    @staticmethod
    def _merged_text(drafts: list[_Draft]) -> str:
        if len(drafts) == 1:
            return drafts[0].text
        if drafts[0].content_type == "record":
            # Rows of one table: its heading line once, then the rows, each with its "Header: value" labels.
            heading_path = drafts[0].heading_path
            skip = len(_heading_line(heading_path)) + 1 if heading_path else 0
            return _with_heading(heading_path, "\n\n".join(d.text[skip:] for d in drafts))
        return "\n\n".join(d.text for d in drafts)          # sections: every heading line kept

    def _absorbs(self, group: list[_Draft], nxt: _Draft, small_document: bool) -> bool:
        """Can the group (last draft = its latest sibling) take `nxt` under the sibling rule?"""
        last = group[-1]
        if nxt.sibling_key is None or nxt.sibling_key != last.sibling_key:
            return False
        if nxt.sibling_key[:2] == ("section", ()) and not small_document:
            return False                                     # top-level sections: only in a small document
        return (self._size(self._merged_text(group)) < config.PROSE_MIN_TOKENS
                and self._size(nxt.text) < config.PROSE_MIN_TOKENS          # a unit >= MIN stands alone
                and self._size(self._merged_text([*group, nxt])) <= config.PROSE_MAX_TOKENS)

    def _pack_siblings(self, drafts: list[_Draft], small_document: bool) -> list[_Draft]:
        """Greedy, in reading order: a group under PROSE_MIN_TOKENS takes the next sibling while it fits."""
        packed: list[_Draft] = []
        group: list[_Draft] = []

        def flush() -> None:
            if len(group) == 1:
                packed.append(group[0])
            elif group:
                starts = [d.line_start for d in group if d.line_start is not None]
                ends = [d.line_end for d in group if d.line_end is not None]
                packed.append(replace(group[0], text=self._merged_text(group),
                                      line_start=min(starts, default=None), line_end=max(ends, default=None)))
            group.clear()

        for draft in drafts:
            if group and self._absorbs(group, draft, small_document):
                group.append(draft)
                continue
            flush()
            group.append(draft)
        flush()
        return packed

    # -- shared ---------------------------------------------------------------

    def _final_text(self, text: str) -> str:
        return f"{self._description}\n{text}" if self._description else text

    def _to_chunk(self, doc: LoadedDocument, draft: _Draft) -> Chunk:
        if draft.chunk is not None:
            return draft.chunk
        text = self._final_text(draft.text)
        return Chunk(
            chunk_id=make_chunk_id(doc.source_document, draft.block_index, draft.sub_index, text),
            text=text,
            source_document=doc.source_document,
            content_type=draft.content_type,
            project_name=doc.project_name,
            date=doc.date,
            heading_path=list(draft.heading_path),
            line_start=draft.line_start,
            line_end=draft.line_end,
        )

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        """Apply the per-block-type rules to one document, in reading order."""
        self._description = description_line(doc.source_document, doc.project_name) if self.descriptions else None
        self._reserve = self.count(self._description) + 1 if self._description else 0
        drafts: list[_Draft] = []
        group: list[tuple[int, Block]] = []
        # depth -> block index of the latest heading at that depth (a heading block's heading_path is its parent
        # path), so a section's parent heading is identified by the block that opened it (D95).
        openers: dict[int, int] = {}

        def flush() -> None:
            if group:
                depth = len(group[0][1].heading_path) - 2            # depth of the parent's own heading block
                drafts.extend(self._chunk_group(group, openers.get(depth) if depth >= 0 else None))
                group.clear()

        for index, block in enumerate(doc.blocks):
            if block.block_type in ("prose", "list"):
                if group and group[-1][1].heading_path != block.heading_path:
                    flush()
                group.append((index, block))
                continue
            flush()
            if block.block_type == "table":
                drafts.extend(self._chunk_table(index, block))
            elif block.block_type == "code_fence":
                drafts.append(self._chunk_fence(doc, index, block))
            elif block.block_type == "code_chunk":
                chunk = code_block_to_chunk(block, self.descriptions)      # finished; never a sibling
                drafts.append(_Draft("code", chunk.text, [], index, 0, None, None, chunk=chunk))
            elif block.block_type == "heading":                  # boundaries / heading_path only
                depth = len(block.heading_path)
                openers[depth] = index
                for deeper in [d for d in openers if d > depth]:
                    del openers[deeper]
        flush()

        if self.merge_small:
            small = self.count("\n\n".join(b.text for b in doc.blocks)) < config.PROSE_MIN_TOKENS
            drafts = self._pack_siblings(drafts, small)
        return [self._to_chunk(doc, d) for d in drafts]


def code_block_to_chunk(block: Block, descriptions: bool | None = None) -> Chunk:
    """
    `code_chunk` blocks are finished chunks: convert 1:1, no re-sizing. With F-R3 descriptions on (D95),
    the text gets the project's "name — description" line first; the chunk id stays the code chunker's.
    """
    meta = block.meta
    on = config.DOC_DESCRIPTION_IN_CHUNKS if descriptions is None else descriptions
    line = project_line(meta.get("project_name"), meta.get("project_description")) if on else None
    return Chunk(
        chunk_id=meta["chunk_id"],
        text=f"{line}\n{block.text}" if line else block.text,
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


def make_chunks(documents: list[LoadedDocument], code_blocks: list[Block], count_tokens: CountTokens, *,
                merge_small: bool | None = None, descriptions: bool | None = None) -> list[Chunk]:
    """All documents + all code_chunk blocks -> Chunk records (F-R3 flags: None = config default, off; D95, D98)."""
    chunker = Chunker(count_tokens, merge_small=merge_small, descriptions=descriptions)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))
    chunks.extend(code_block_to_chunk(b, chunker.descriptions) for b in code_blocks)
    return chunks
