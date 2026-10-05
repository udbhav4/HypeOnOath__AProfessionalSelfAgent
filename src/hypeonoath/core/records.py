"""
Shared record types for the Phase 1 pipeline: Block (loader output) and
Chunk (chunker output), plus the content-derived ID helpers.

Kept in one tiny module so block_loader.py, code_chunker.py, chunker.py,
project_registry.py and indexer.py all agree on one contract without
importing each other's heavy dependencies (docling_core, chonkie, ...).
See plans/phase1-core-pipeline-code-implementation-guideline-v4.md,
Sections 2 (Block) and 3 (Chunk, chunk_id rule).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Block types every loader may emit (guideline Section 2).
BLOCK_TYPES = frozenset({"heading", "prose", "table", "code_fence", "list", "code_chunk"})


@dataclass
class Block:
    """One typed block of a source file, before chunking."""

    block_type: str
    text: str
    source_document: str            # path relative to the project root (posix)
    heading_path: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_type not in BLOCK_TYPES:
            raise ValueError(f"Unknown block_type {self.block_type!r}")


@dataclass
class Chunk:
    """
    One retrievable, embeddable unit. `text` is exactly what gets embedded
    (minus the task prefix, which embedder.py adds) and exactly what the LLM
    sees. Every other field becomes Chroma metadata (indexer.flatten_metadata).
    """

    chunk_id: str
    text: str
    source_document: str
    content_type: str                # prose|table|record|code|code_fence|project_summary
    project_name: str | None = None
    date: str | None = None
    heading_path: list[str] = field(default_factory=list)
    line_start: int | None = None    # 1-based, in source_document (when known)
    line_end: int | None = None
    # Code / project-summary only (compulsory for those, guideline 3A.12)
    project_key: str | None = None
    project_description: str | None = None
    # Code only (guideline 3A.6)
    language: str | None = None
    symbols: list[str] = field(default_factory=list)
    symbol_spans: list[dict[str, Any]] = field(default_factory=list)
    part: int | None = None
    total_parts: int | None = None
    group_id: str | None = None
    chunk_method: str | None = None  # "ast" | "fallback"
    start_byte: int | None = None
    end_byte: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(source_document: str, start: int | str, end: int | str, text: str) -> str:
    """
    Content-derived, stable chunk ID (guideline Section 3 / ADR-13):
    sha256(source_document:start:end:sha256(text))[:16]. Code passes byte
    offsets; non-code blocks pass their position within the source. Same
    text at the same location -> same ID on every run, so re-indexing
    unchanged input produces no diff in the committed Chroma folder.
    """
    return sha256_hex(f"{source_document}:{start}:{end}:{sha256_hex(text)}")[:16]


def make_group_id(source_document: str, qualified_symbol: str, symbol_start_byte: int) -> str:
    """ID shared by every part of one split function (guideline 3A.4 Step 10)."""
    return sha256_hex(f"{source_document}{qualified_symbol}{symbol_start_byte}")[:16]


def json_dumps_stable(value: Any) -> str:
    """Deterministic JSON (sorted keys) so flattened metadata never churns."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@dataclass
class RetrievedItem:
    """
    One retrieval result: the hit chunk, its similarity score, and -- for a
    piece of a split function -- the sibling parts fetched to complete it
    (guideline Section 6). `parts` is empty for ordinary hits.
    """

    chunk: Chunk
    score: float
    parts: list[Chunk] = field(default_factory=list)

    @property
    def ordered_chunks(self) -> list[Chunk]:
        """The chunks to show the LLM, in part order 1 -> n (or just the hit)."""
        return self.parts or [self.chunk]
