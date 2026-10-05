"""
Corpus preparation entry point (roadmap Phase 0, Steps 3.1-3.6): raw files ->
routed, converted, cleaned and tagged corpus.

Run with:  python -m hypeonoath.ingestion.prepare_corpus   (or: hypeonoath-prepare)
from the project root, with the package installed: `pip install -e ".[ingestion]"`.

Runs routing (3.2) -> conversion (3.3) -> cleaning (3.5) -> tagging (3.6) in
sequence. Steps 3.4 (removed, see phase0-document-prep-subplan-v5.md Section
3C), 3.7 (manual sensitive-content review) and 3.8 (git commit -- performed
by the document owner directly, per explicit user instruction) are not run
here. Every step is individually re-run-safe (manifest-driven): running this
repeatedly with no changes to corpus/raw/ does no redundant work.
"""
from __future__ import annotations

import logging
import sys

from hypeonoath.ingestion.cleaner import run_cleaning
from hypeonoath.ingestion.converter import run_conversion
from hypeonoath.ingestion.metadata_tagger import run_tagging
from hypeonoath.ingestion.router import run_routing


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Corpus preparation starting")

    manifest, processed_rel_paths = run_routing()
    manifest = run_conversion(manifest, processed_rel_paths)
    manifest = run_cleaning(manifest, processed_rel_paths)
    manifest = run_tagging(manifest, processed_rel_paths)

    logger.info("Corpus preparation finished (%d file(s) processed this run)", len(processed_rel_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
