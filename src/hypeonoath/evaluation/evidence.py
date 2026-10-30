"""
Evidence matching (eval plan v15, 4.1; D20, T1).

A chunk MATCHES an alternative {document, quote} when it comes from the same document and
contains the normalised quote, or (fallback) its line range overlaps a line range where the
quote occurs in that document. A group is FOUND in a list of chunks when any alternative
matches any chunk. Placeholder groups ([REDACTED: <label>]) use the same matching.

Normalisation: whitespace runs -> one space, casefold. Nothing else (quotes are copied
from the final redacted corpus text, so both sides hold the same characters).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from hypeonoath.core import paths
from hypeonoath.core.records import Chunk


class AlternativeLike(Protocol):
    document: str
    quote: str


class GroupLike(Protocol):
    any_of: list


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _normalized_with_lines(text: str) -> tuple[str, list[int]]:
    """normalize(text) plus, per output character, its 1-based line in `text`."""
    out: list[str] = []
    lines: list[int] = []
    line, pending_space = 1, False
    for ch in text:
        if ch.isspace():
            pending_space = bool(out)
            if ch == "\n":
                line += 1
            continue
        if pending_space:
            out.append(" ")
            lines.append(line)
            pending_space = False
        for c in ch.casefold():
            out.append(c)
            lines.append(line)
    return "".join(out), lines


class Corpus:
    """Read-only access to corpus files (paths relative to the project root), cached."""

    def __init__(self, root: Path | None = None):
        self.root = paths.PROJECT_ROOT if root is None else root
        self._cache: dict[str, tuple[str, list[int]] | None] = {}

    def _load(self, document: str) -> tuple[str, list[int]] | None:
        if document not in self._cache:
            path = self.root / document
            try:
                self._cache[document] = _normalized_with_lines(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                self._cache[document] = None
        return self._cache[document]

    def exists(self, document: str) -> bool:
        return self._load(document) is not None

    def contains(self, document: str, quote: str) -> bool:
        loaded = self._load(document)
        needle = normalize(quote)
        return bool(loaded and needle and needle in loaded[0])

    def quote_lines(self, document: str, quote: str) -> list[tuple[int, int]]:
        """(first line, last line) of every occurrence of the quote in the document."""
        loaded = self._load(document)
        needle = normalize(quote)
        if not loaded or not needle:
            return []
        text, lines = loaded
        found, start = [], text.find(needle)
        while start != -1:
            found.append((lines[start], lines[start + len(needle) - 1]))
            start = text.find(needle, start + 1)
        return found


def chunk_matches(alt: AlternativeLike, chunk: Chunk, corpus: Corpus) -> bool:
    if chunk.source_document != alt.document:
        return False
    if normalize(alt.quote) in normalize(chunk.text):
        return True
    if chunk.line_start is None:
        return False
    end = chunk.line_end if chunk.line_end is not None else chunk.line_start
    return any(first <= end and chunk.line_start <= last for first, last in corpus.quote_lines(alt.document, alt.quote))


def group_found(group: GroupLike, chunks: Iterable[Chunk], corpus: Corpus) -> bool:
    chunks = list(chunks)
    return any(chunk_matches(alt, chunk, corpus) for alt in group.any_of for chunk in chunks)
