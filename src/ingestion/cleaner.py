"""
Step 3.5: clean document body text.

Normalizes whitespace/line-ending artifacts in every document that landed in
corpus/documents/ (.md/.txt originals) or was produced in corpus/converted/
(docling's PDF/DOCX output), and -- for converted files only -- verifies a
Markdown heading survived conversion, logging a warning (not a failure) if
none is found. See plans/phase0-steps-3.5-3.8-code-implementation-guideline-v2.md,
Section 5.

Front-matter handling: deliberately does NOT use frontmatter_utils/python-
frontmatter here. Tested directly (2026-09-20): frontmatter.dumps() on a Post
with an empty metadata dict emits a stray '---\\n{}\\n---\\n' block into files
that never had front-matter -- and at this point in the pipeline (Step 3.5
always runs before Step 3.6), no file has front-matter yet on a first pass.
Round-tripping through the library here would inject that empty block into
every document. Instead, any existing front-matter block (relevant on a
re-run where metadata_tagger.py already ran) is detected with a simple regex,
left byte-for-byte untouched, and only the body after it is cleaned.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from . import config
from .logging_utils import log_json_line
from .manifest import save_manifest

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```")
_LEADING_WHITESPACE_RE = re.compile(r"^(\s*)(.*)$", re.DOTALL)
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s")

# Text files considered "document" but never routed through Step 3.3's
# conversion (already Markdown/plain text) -- kept in sync with
# config.CONVERTIBLE_DOCUMENT_EXTENSIONS' complement within DOCUMENT_EXTENSIONS.
_ALREADY_TEXT_EXTENSIONS = frozenset({".md", ".txt"})


def _split_front_matter(text: str) -> tuple[str, str]:
    """
    Split off a leading '---\\n...\\n---\\n' YAML front-matter block, if
    present, leaving it untouched. Returns (front_matter_or_empty, body).
    """
    match = _FRONT_MATTER_RE.match(text)
    if match:
        return match.group(0), text[match.end():]
    return "", text


def _collapse_internal_spaces(line: str) -> str:
    """
    Collapse runs of 2+ spaces to one, but only within a line's content --
    leading indentation is preserved untouched, since Markdown nested lists
    and indented code rely on exact leading-space counts.
    """
    leading, rest = _LEADING_WHITESPACE_RE.match(line).groups()
    return leading + _MULTI_SPACE_RE.sub(" ", rest)


def clean_text(body: str) -> str:
    """
    Apply Step 3.5's cleaning operations to a document body (front-matter,
    if any, must already be split off by the caller):
      1. Normalize line endings to '\\n'.
      2. Strip non-printable/control characters outside standard whitespace.
      3. Per line: strip trailing whitespace; collapse repeated internal
         spaces (2+ -> 1) -- skipped entirely inside fenced code blocks,
         where whitespace can be semantically significant.
      4. Collapse 3+ consecutive blank lines down to one.
    """
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHAR_RE.sub("", text)

    cleaned_lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            cleaned_lines.append(line.rstrip())
            continue
        line = line.rstrip()
        if not in_fence:
            line = _collapse_internal_spaces(line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return _MULTI_BLANK_LINE_RE.sub("\n\n", text)


def verify_headings(body: str) -> bool:
    """True if at least one Markdown heading (# through ######) is present."""
    return _HEADING_RE.search(body) is not None


def resolve_document_target(rel_path: str, entry: dict) -> tuple[Path, bool] | None:
    """
    Given a manifest entry for a 'document'-category file, return
    (path_to_clean_or_tag, is_converted_output), or None if there is nothing
    to process for this entry (e.g. a PDF/DOCX whose conversion failed, so
    no Markdown output exists to touch). Shared with metadata_tagger.py,
    which needs the identical target-resolution logic for Step 3.6.
    """
    if entry.get("destination_category") != "document":
        return None

    converted_output = entry.get("converted_output")
    if converted_output:
        return config.CONVERTED_DIR / converted_output, True

    doc_path = config.DOCUMENTS_DIR / rel_path
    if doc_path.suffix.lower() in _ALREADY_TEXT_EXTENSIONS and doc_path.exists():
        return doc_path, False

    return None  # e.g. a PDF/DOCX with no converted_output -- conversion failed


def clean_file(path: Path, check_headings: bool) -> None:
    """Clean a single file in place and log the result."""
    raw_text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw_text)
    cleaned_body = clean_text(body)
    path.write_text(front_matter + cleaned_body, encoding="utf-8")

    rel_for_log = path.relative_to(config.CORPUS_DIR).as_posix()
    if not check_headings:
        log_json_line(config.CLEANING_LOG_PATH, file=rel_for_log, status="cleaned")
        return

    if verify_headings(cleaned_body):
        log_json_line(config.CLEANING_LOG_PATH, file=rel_for_log, status="cleaned", heading_found=True)
    else:
        log_json_line(
            config.CLEANING_LOG_PATH, file=rel_for_log, status="cleaned_no_heading_warning", heading_found=False
        )
        logger.warning("No Markdown heading found in %s after cleaning", rel_for_log)


def run_cleaning(manifest: dict, raw_rel_paths: list[str]) -> dict:
    """
    Clean exactly the documents corresponding to raw_rel_paths (the files
    router.py/converter.py just processed this run). Scoping to this list,
    rather than re-scanning all of corpus/documents/ + corpus/converted/, is
    what makes cleaning re-run-safe -- unchanged files are never re-cleaned.
    """
    for rel_path_str in raw_rel_paths:
        entry = manifest.get(rel_path_str)
        if entry is None:
            continue

        target = resolve_document_target(rel_path_str, entry)
        if target is None:
            continue

        path, is_converted = target
        clean_file(path, check_headings=is_converted)
        entry["cleaned"] = True

    save_manifest(config.MANIFEST_PATH, manifest)
    return manifest
