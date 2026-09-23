"""
Independent hyperlink extraction for Step 3.3 conversion.

docling's own hyperlink tracking is unreliable: confirmed (2026-09-17) that
it grouped a multi-link contact-info line into one coarse TextItem and left
its `hyperlink` field None, silently dropping two real PDF links. Rather than
rely on docling, this module reads hyperlinks directly from each format's
own raw link data -- see
plans/phase0-steps-3.1-3.3-code-implementation-guideline.md Section 8A.

  - PDF: link annotations, via PyMuPDF's `page.get_links()` -- reads the
    PDF's /Annots objects directly, no layout/text-grouping heuristic
    involved, so this is not subject to docling's failure mode.
  - DOCX: hyperlink relationships, via python-docx's `document.part.rels` --
    same reasoning, raw XML relationship data, not layout-dependent.

Always run for every converted document, independent of what docling itself
managed to capture.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT


def _dedupe_preserve_order(urls: list[str]) -> list[str]:
    """
    Remove duplicate URLs, keeping the first occurrence's position.
    Decided 2026-09-19 (resolving the guideline's Section 9, open item #10):
    listing the same link twice adds no value to a citation-style list.
    """
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _extract_pdf_links(path: Path) -> list[str]:
    """Return every URI link target found in a PDF, in document order."""
    urls: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri")
                if link.get("kind") == fitz.LINK_URI and uri:
                    # source PDFs have been observed with stray trailing
                    # whitespace baked into the link's URI itself (a data
                    # artifact in the source document, not an extraction bug)
                    urls.append(uri.strip())
    return _dedupe_preserve_order(urls)


def _extract_docx_links(path: Path) -> list[str]:
    """Return every external hyperlink target found in a DOCX."""
    document = Document(str(path))
    urls = [
        rel.target_ref.strip()
        for rel in document.part.rels.values()
        if rel.reltype == RT.HYPERLINK and rel.is_external
    ]
    return _dedupe_preserve_order(urls)


_EXTRACTORS = {
    ".pdf": _extract_pdf_links,
    ".docx": _extract_docx_links,
}


def extract_links(path: Path) -> list[str]:
    """
    Extract every hyperlink from a source document, independent of any
    Markdown conversion library. Returns an empty list for unsupported
    extensions or documents with no links.
    """
    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        return []
    return extractor(path)


def format_links_section(links: list[str]) -> str:
    """
    Format extracted links as a Markdown section to append to a converted
    document's text. Returns an empty string for no links -- an empty
    "## Links" section would be noise for documents with none.
    """
    if not links:
        return ""
    bullet_lines = "\n".join(f"- {url}" for url in links)
    return f"\n\n## Links\n\n{bullet_lines}\n"
