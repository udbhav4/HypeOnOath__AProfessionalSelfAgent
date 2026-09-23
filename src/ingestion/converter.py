"""
Step 3.3: PDF/DOCX -> Markdown conversion, via docling.

Converts simple text-based PDF and DOCX files that landed in
corpus/documents/ (via router.py) into Markdown, written to corpus/converted/
-- never in-place. Any single file that fails conversion is skipped and
logged; the run always continues, per explicit design (see the guideline's
Section 8 and Section 1A). Scanned/image-only PDFs are explicitly out of
scope and, if encountered, are expected to surface as a conversion failure
here rather than being pre-detected (confirmed with user 2026-09-15).

Library changed 2026-09-15 from pymupdf4llm (PDF) + mammoth/markdownify
(DOCX) to a single library, `docling`, for both formats -- see
plans/phase0-steps-3.1-3.3-code-implementation-guideline.md Section 1B.
Reason: pymupdf4llm misjudged a text region in the real file CV_Udbhav.pdf
as a table, swallowing a heading and concatenating words; docling's
dedicated table-structure model was chosen to target that failure
specifically, and was confirmed (2026-09-15) to fix it against the real
file -- "WORK EXPERIENCE" now renders as its own heading, not a table cell.

Hyperlink extraction added 2026-09-19 -- see Section 8A. docling's own
hyperlink tracking is unreliable (confirmed: it dropped two real PDF links
by grouping them into one coarse text item), so links are independently
re-extracted straight from the source file via link_extractor.py and
appended to the Markdown output, regardless of what docling itself captured.

Structured-object persistence added 2026-09-22 -- see
plans/phase0-step-3.3-docling-structure-persistence-guideline.md and
plans/system-design-rag-chatbot.md Section 5.4. Alongside the flattened
".md" output, docling's own typed DoclingDocument (headings, TextItem,
TableItem, etc.) is now also persisted as "<file>.docling.json" via its
built-in save_as_json()/load_from_json() (confirmed present and
round-trip-faithful against a real corpus file, docling 2.127.0, spiked
2026-09-22 -- see the guideline's Section 2). This lets Phase 1's
block-level chunker load typed blocks directly instead of re-parsing the
flattened Markdown text to re-derive structure docling already computed
once. This write is independent of, and never fails, the Markdown write --
see convert_file()'s docstring.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError as DoclingConversionError
from docling_core.types.doc import DoclingDocument

from . import config
from .link_extractor import extract_links, format_links_section
from .logging_utils import log_json_line
from .manifest import save_manifest

logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """Raised when a single document cannot be converted to Markdown."""


@lru_cache(maxsize=1)
def _get_converter() -> DocumentConverter:
    """
    docling's DocumentConverter loads ML model weights on first use, which
    is expensive -- cached so one run shares a single instance across every
    file instead of reloading models per file.
    """
    return DocumentConverter()


def _convert_document(path: Path) -> DoclingDocument:
    """
    Convert a single PDF or DOCX file via docling and return its typed
    DoclingDocument. Handles both formats identically -- docling dispatches
    internally based on the file's actual content, not just its extension.

    Returns the structured document rather than already-flattened Markdown
    so convert_file() can both export Markdown *and* persist the structured
    object itself (see module docstring) from a single conversion call.
    """
    try:
        result = _get_converter().convert(str(path))
    except DoclingConversionError as exc:
        raise ConversionError(f"docling conversion failed: {exc}") from exc

    if result.status.name != "SUCCESS":
        raise ConversionError(
            f"docling reported status={result.status.name}, errors={result.errors}"
        )

    return result.document


_CONVERTERS = {
    ".pdf": _convert_document,
    ".docx": _convert_document,
}


def convert_file(doc_path: Path) -> tuple[Path | None, Path | None]:
    """
    Convert a single file in corpus/documents/ to Markdown in
    corpus/converted/, preserving its relative path with the extension
    swapped to '.md'. Also persists docling's typed structured document as a
    sibling '<file>.docling.json' (see module docstring).

    Returns (markdown_output_path, structured_output_path). Either or both
    may be None: both are None if the file isn't a convertible type or
    conversion failed outright (failures are logged here, never raised
    further up); markdown_output_path is set but structured_output_path is
    None if only the structured-object write failed -- that failure is
    independent of, and never invalidates, a successful Markdown export.
    """
    ext = doc_path.suffix.lower()
    converter_fn = _CONVERTERS.get(ext)
    if converter_fn is None:
        return None, None  # not a convertible type (e.g. already-Markdown/.txt) -- nothing to do

    rel_path = doc_path.relative_to(config.DOCUMENTS_DIR)
    output_path = (config.CONVERTED_DIR / rel_path).with_suffix(".md")
    structured_output_path = output_path.with_name(output_path.stem + config.DOCLING_JSON_SUFFIX)

    try:
        structured_document = converter_fn(doc_path)
    except ConversionError as exc:
        log_json_line(
            config.CONVERSION_LOG_PATH,
            file=rel_path.as_posix(),
            status="failed",
            error=str(exc),
        )
        logger.warning("Conversion failed for %s: %s", rel_path, exc)
        return None, None

    markdown_text = structured_document.export_to_markdown()

    # Always re-extract links directly from the source file -- independent
    # of docling, which is known to silently drop some real links (Section 8A).
    links = extract_links(doc_path)
    markdown_text += format_links_section(links)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")

    log_json_line(
        config.CONVERSION_LOG_PATH,
        file=rel_path.as_posix(),
        status="converted",
        output=output_path.relative_to(config.CORPUS_DIR).as_posix(),
        links_extracted=len(links),
    )

    # Structured-object persistence is independent of the Markdown write
    # above: a failure here is logged and degrades to structured_output=None
    # (Phase 1 falls back to re-parsing the Markdown for this one file), but
    # never fails convert_file() or discards the already-written Markdown.
    try:
        structured_document.save_as_json(structured_output_path)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        log_json_line(
            config.CONVERSION_LOG_PATH,
            file=rel_path.as_posix(),
            status="structured_export_failed",
            error=str(exc),
        )
        logger.warning("Structured-object export failed for %s: %s", rel_path, exc)
        return output_path, None

    return output_path, structured_output_path


def run_conversion(manifest: dict, raw_rel_paths: list[str]) -> dict:
    """
    Convert exactly the documents corresponding to raw_rel_paths (the files
    router.py just routed this run -- new or modified) that both landed in
    corpus/documents/ and are a convertible type. Updates each file's
    manifest entry with its converted_output path (or None on failure) and
    persists the manifest. This scoping -- rather than re-scanning all of
    corpus/documents/ -- is what makes conversion re-run-safe: unchanged
    files are never reconverted.
    """
    config.CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path_str in raw_rel_paths:
        entry = manifest.get(rel_path_str)
        if entry is None or entry.get("destination_category") != "document":
            continue

        doc_path = config.DOCUMENTS_DIR / rel_path_str
        if doc_path.suffix.lower() not in config.CONVERTIBLE_DOCUMENT_EXTENSIONS:
            continue  # e.g. .md/.txt -- already Markdown/plain text, nothing to convert

        output_path, structured_output_path = convert_file(doc_path)
        entry["converted_output"] = (
            output_path.relative_to(config.CONVERTED_DIR).as_posix() if output_path else None
        )
        entry["structured_output"] = (
            structured_output_path.relative_to(config.CONVERTED_DIR).as_posix()
            if structured_output_path
            else None
        )

    save_manifest(config.MANIFEST_PATH, manifest)
    return manifest
