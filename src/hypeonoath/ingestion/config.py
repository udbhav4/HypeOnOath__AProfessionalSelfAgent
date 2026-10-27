"""
Central configuration for the Phase 0 ingestion pipeline.

Single source of truth for folder paths and the code/document extension
lists, per the guideline's instruction to centralize constants here rather
than scatter them across router.py / converter.py.
"""
from __future__ import annotations

# --- Project layout ---------------------------------------------------------
# Folder paths are defined once in hypeonoath.core.paths (shared with the
# indexing and serving packages) and re-exported here, so Phase 0 code keeps
# reading them as config.<NAME>.
from hypeonoath.core.paths import (  # noqa: F401 -- re-exported names
    CODE_DIR,
    CODE_METADATA_PATH,
    CONVERTED_DIR,
    CORPUS_DIR,
    DOCLING_JSON_SUFFIX,
    DOCUMENTS_DIR,
    LOGS_DIR,
    MANIFEST_PATH,
    PROJECT_ROOT,
    QUARANTINE_DIR,
    RAW_DIR,
)

ROUTING_LOG_PATH = LOGS_DIR / "routing.jsonl"
CONVERSION_LOG_PATH = LOGS_DIR / "conversion.jsonl"
CLEANING_LOG_PATH = LOGS_DIR / "cleaning.jsonl"
TAGGING_LOG_PATH = LOGS_DIR / "tagging.jsonl"


# --- Step 3.2: extension classification -------------------------------------
# Broad starter list, per plans/phase0-document-prep-subplan-v4.md Section 4.3.
# Anything not in CODE_EXTENSIONS or DOCUMENT_EXTENSIONS falls through to
# quarantine -- there is deliberately no implicit "everything else is a
# document" default (see the guideline's Section 6 open-item discussion).
CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h",
        ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
        ".kt", ".kts", ".scala", ".sh", ".bash", ".sql", ".html", ".css",
        ".scss", ".r", ".pl", ".lua", ".dart",
    }
)

# Confirmed with user 2026-09-15 (guideline Section 6 / Section 1A).
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".md", ".txt"})

# Extensions within DOCUMENT_EXTENSIONS that Step 3.3 must actually convert
# to Markdown (".md" and ".txt" are already Markdown/plain text -- nothing
# to convert).
CONVERTIBLE_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})

# DOCLING_JSON_SUFFIX (docling's persisted structured DoclingDocument, written
# alongside each converted ".md" file with the same base name, per
# plans/phase0-step-3.3-docling-structure-persistence-guideline.md) is imported
# from hypeonoath.core.paths above.

# --- Step 3.6: metadata tagging ---------------------------------------------
# Date format is fixed as DD.MM.YYYY throughout the pipeline (front-matter,
# _metadata.yaml, and tagging.jsonl alike) -- confirmed with user 2026-09-20
# (guideline v2/v3, Section 6A). Never emit ISO (YYYY-MM-DD) or a
# locale-dependent format.
DATE_FORMAT = "%d.%m.%Y"

# Per-file date override, keyed by path relative to corpus/raw/. Wins over
# the file's mtime whenever an entry exists for it. Populated 2026-09-20 per
# explicit user instruction ("include the file dates given by me and keep
# them as is") -- values are exactly as given, not reinterpreted.
CUSTOM_DATE_OVERRIDES: dict[str, str] = {
    "agent.ts": "10.05.2026",
    "Cover_Letter_Udbhav.pdf": "18.09.2026",
    "CV_Udbhav.pdf": "20.09.2026",
    "CV_Udbhav2.pdf": "20.09.2026",
    "README.md": "18.05.2026",
    "system-design-rag-chatbot.md": "20.09.2026",
}

# --- Project + document registry --------------------------------------------
# Defined in hypeonoath.core.registry (serving reads it too, invariant I5) and
# re-exported here for the ingestion and indexing code.
from hypeonoath.core.registry import (  # noqa: E402, F401 -- re-exported names
    CODE_PROJECT_MAP,
    CODE_PROJECTS,
    DOCUMENTS,
)
