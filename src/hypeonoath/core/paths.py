"""
Project folder layout -- the one place that knows where corpus/ and index/ live.

Shared by every subpackage (ingestion, indexing, serving), so they can never
disagree about where files are. Both config modules re-export these names
(hypeonoath.ingestion.config, hypeonoath.core.config); code reads them from
there, which keeps test monkeypatching on a single, obvious target.
"""
from __future__ import annotations

import os
from pathlib import Path

# This file lives at <project_root>/src/hypeonoath/core/paths.py -> parents[3].
# HYPEONOATH_ROOT overrides it for a non-editable install (e.g. a deployed
# build, where __file__ points into site-packages instead of the repo).
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ["HYPEONOATH_ROOT"]).resolve() if os.environ.get("HYPEONOATH_ROOT") else _DEFAULT_ROOT

CORPUS_DIR = PROJECT_ROOT / "corpus"
RAW_DIR = CORPUS_DIR / "raw"                  # immutable source of truth (Step 3.1)
CODE_DIR = CORPUS_DIR / "code"                # Step 3.2 output
DOCUMENTS_DIR = CORPUS_DIR / "documents"      # Step 3.2 output
QUARANTINE_DIR = CORPUS_DIR / "quarantine"    # Step 3.2 output
CONVERTED_DIR = CORPUS_DIR / "converted"      # Step 3.3 output

MANIFEST_PATH = CORPUS_DIR / ".manifest.json"
LOGS_DIR = CORPUS_DIR / ".logs"
CODE_METADATA_PATH = CODE_DIR / "_metadata.yaml"

# Sibling file docling's structured document is saved to, next to its .md
# (Step 3.3 follow-up, design doc ADR-12).
DOCLING_JSON_SUFFIX = ".docling.json"

INDEX_DIR = PROJECT_ROOT / "index"
