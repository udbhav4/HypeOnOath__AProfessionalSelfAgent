"""
Redaction of private values (privacy plan v3, 6.7; P-D10, P-D11).

  (a) in memory, BEFORE chunking -- authoritative for the index:
      redact_blocks()    every Block's text, table cells (meta["rows"], used
                         for record chunks), heading paths (a heading's text
                         is copied into the heading_path of later blocks),
                         code-chunk metadata strings, document project names;
      redact_projects()  ProjectInfo name/description (project-summary chunks
                         and code-chunk headers are built from them).
      Chunk ids are hashes of chunk text and token budgets are computed while
      chunking, so redacting first keeps both consistent.
  (b) on disk -- the derived corpus files (gold quotes are taken from them and
      they are committed): corpus/converted/*.md + *.docling.json and
      corpus/documents/*.md|*.txt.  corpus/raw/ is NEVER touched (owner
      originals, P-D11).

Every marked value becomes "[REDACTED: <label>]". Placeholders hold no value,
so running twice changes nothing (idempotent).
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypeonoath.core import config, paths
from hypeonoath.core import private_values as pv
from hypeonoath.core.records import Block

if TYPE_CHECKING:
    from hypeonoath.indexing.block_loader import LoadedDocument
    from hypeonoath.indexing.project_registry import ProjectInfo

_FRONT_MATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_DOCLING_TEXT_KEYS = frozenset({"text", "orig"})
_CODE_META_STRINGS = ("project_name", "project_description")


@dataclass
class RedactionStats:
    labels: Counter = field(default_factory=Counter)      # label -> spans replaced
    blocks_changed: int = 0
    # P-D49: decision id -> file -> lines where block text held the value (paths and lines only)
    locations: dict[str, dict[str, set[int | None]]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.labels.values())

    def add(self, spans: list[pv.Span]) -> None:
        self.labels.update(s.label for s in spans)


def _redact(text: str, values: list[pv.PrivateValue], stats: RedactionStats) -> str:
    redacted, spans = pv.redact(text, values)
    stats.add(spans)
    return redacted


def _record_lines(block: Block, spans: list[pv.Span], first_line: int | None, stats: RedactionStats) -> None:
    """Where each value sat, as detection would report it (line of the span inside the block)."""
    for span in spans:
        line = None if first_line is None else first_line + block.text.count("\n", 0, span.start)
        stats.locations.setdefault(span.value_id, {}).setdefault(block.source_document, set()).add(line)


def _redact_block(block: Block, values: list[pv.PrivateValue], stats: RedactionStats,
                  first_line: int | None = None) -> bool:
    before = (block.text, list(block.heading_path), repr(block.meta))
    redacted, spans = pv.redact(block.text, values)
    stats.add(spans)
    _record_lines(block, spans, first_line, stats)
    block.text = redacted
    block.heading_path = [_redact(h, values, stats) for h in block.heading_path]
    rows = block.meta.get("rows")
    if rows:
        block.meta["rows"] = [[_redact(cell, values, stats) for cell in row] for row in rows]
    for key in _CODE_META_STRINGS:
        if isinstance(block.meta.get(key), str):
            block.meta[key] = _redact(block.meta[key], values, stats)
    if block.meta.get("symbols"):
        block.meta["symbols"] = [_redact(s, values, stats) for s in block.meta["symbols"]]
    return before != (block.text, block.heading_path, repr(block.meta))


def redact_blocks(documents: list["LoadedDocument"], code_blocks: list[Block],
                  values: list[pv.PrivateValue]) -> RedactionStats:
    """Redact every loaded Block in place (documents incl. notebook docs, code blocks)."""
    stats = RedactionStats()
    if not values:
        return stats
    for doc in documents:
        if isinstance(doc.project_name, str):
            doc.project_name = _redact(doc.project_name, values, stats)
        for block in doc.blocks:
            stats.blocks_changed += _redact_block(block, values, stats, block.meta.get("line_start"))
    for block in code_blocks:
        # Line 0 of a code chunk is its header (same numbering as privacy.detector).
        line_start = block.meta.get("line_start")
        stats.blocks_changed += _redact_block(block, values, stats, line_start - 1 if line_start is not None else None)
    return stats


def redact_projects(projects: dict[str, "ProjectInfo"], values: list[pv.PrivateValue],
                    stats: RedactionStats | None = None) -> None:
    """Redact ProjectInfo name/description in place (the dict's values are replaced)."""
    stats = stats if stats is not None else RedactionStats()
    if not values:
        return
    for key, info in list(projects.items()):
        name = _redact(info.name, values, stats)
        description = _redact(info.description, values, stats)
        if (name, description) != (info.name, info.description):
            projects[key] = replace(info, name=name, description=description)


# --- Files on disk --------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _with_html_forms(values: list[pv.PrivateValue]) -> list[pv.PrivateValue]:
    """Converted .md files are HTML-escaped ("&amp;"): also match escaped spellings."""
    extra = []
    for value in values:
        escaped = tuple(f for f in (html.escape(s, quote=False) for s in value.surface_forms)
                        if f not in value.surface_forms)
        if escaped:
            extra.append(replace(value, surface_forms=escaped))
    return values + extra


def _redact_yaml_strings(node: Any, values: list[pv.PrivateValue]) -> Any:
    if isinstance(node, str):
        return pv.redact(node, values)[0]
    if isinstance(node, list):
        return [_redact_yaml_strings(item, values) for item in node]
    if isinstance(node, dict):
        return {k: _redact_yaml_strings(v, values) for k, v in node.items()}
    return node


def redact_markdown_text(text: str, values: list[pv.PrivateValue]) -> str:
    """
    Body: plain redaction. Front matter: left byte-for-byte unless it holds a
    value; then it is re-serialised from parsed YAML, so the placeholder
    (which contains ": ") can't break the YAML.
    """
    values = _with_html_forms(values)
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return pv.redact(text, values)[0]
    front, body = match.group(0), text[match.end():]
    body = pv.redact(body, values)[0]
    if pv.find_private_spans(match.group(1), values):
        import yaml

        meta = yaml.safe_load(match.group(1)) or {}
        dumped = yaml.safe_dump(_redact_yaml_strings(meta, values), sort_keys=False, allow_unicode=True)
        front = f"---\n{dumped.rstrip()}\n---\n"
    return front + body


def _redact_docling(node: Any, values: list[pv.PrivateValue]) -> Any:
    """Only "text" / "orig" string fields change -- the JSON stays loadable by docling."""
    if isinstance(node, dict):
        return {k: (pv.redact(v, values)[0] if k in _DOCLING_TEXT_KEYS and isinstance(v, str)
                    else _redact_docling(v, values)) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_docling(item, values) for item in node]
    return node


def _disk_targets() -> list[Path]:
    converted, documents = config.CONVERTED_DIR, config.DOCUMENTS_DIR
    targets = [p for p in converted.glob("*.md")] if converted.is_dir() else []
    if converted.is_dir():
        targets += converted.glob(f"*{config.DOCLING_JSON_SUFFIX}")
    if documents.is_dir():
        targets += [p for p in documents.rglob("*") if p.suffix.lower() in (".md", ".txt") and p.is_file()]
    return sorted(set(targets))


def _is_raw(path: Path) -> bool:
    try:
        path.resolve().relative_to((config.CORPUS_DIR / paths.RAW_DIR.name).resolve())
        return True
    except ValueError:
        return False


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def redact_files_on_disk(values: list[pv.PrivateValue]) -> list[str]:
    """Redact the derived corpus files in place (atomic per file). Returns changed paths."""
    if not values:
        return []
    changed: list[str] = []
    for path in _disk_targets():
        if _is_raw(path):                     # defence in depth: originals are never modified
            continue
        original = path.read_text(encoding="utf-8")
        if path.name.endswith(config.DOCLING_JSON_SUFFIX):
            data = json.loads(original)
            new = json.dumps(_redact_docling(data, values), ensure_ascii=False, indent=2)
            # Compare parsed content: re-dumping alone must not count as a change.
            if json.loads(new) == data:
                continue
            new += "\n" if original.endswith("\n") else ""
        else:
            new = redact_markdown_text(original, values)
            if new == original:
                continue
        _atomic_write(path, new)
        changed.append(_rel(path))
    return changed
