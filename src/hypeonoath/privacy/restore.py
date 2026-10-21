"""
Rebuild from raw when a value goes public (privacy plan v8, 6.7.1; P-D33).

Derived corpus files (corpus/converted/*, corpus/documents/*) are redacted on
disk, so a value switched private -> public can't be put back by text
replacement (several values can share one label, the exact spelling at that
spot is unknown, .docling.json holds the text twice). corpus/raw/ is never
modified, so each affected document is rebuilt from it with the existing
Phase 0 functions -- for that file only -- and every value that is still
private is redacted again.

  * raw_sources_for()   derived file paths (decision locations) -> raw rel paths
  * restore_pending()   rebuild every queued document, then clear the queue

run_conversion / run_cleaning / run_tagging process exactly the rel paths they
are given (no hash check -- verified 2026-10-16); only run_routing skips
unchanged files, so routing is done per file with route_file(). Code files are
never redacted on disk, so they never need this.
"""
from __future__ import annotations

import logging

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line

logger = logging.getLogger(__name__)


def _rel_to_root(path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def raw_sources_for(files: list[str]) -> list[str]:
    """Raw rel paths (manifest keys) whose derived document is one of `files`."""
    from hypeonoath.ingestion import config as ingestion_config
    from hypeonoath.ingestion.cleaner import resolve_document_target
    from hypeonoath.ingestion.manifest import load_manifest

    wanted = set(files)
    if not wanted:
        return []
    manifest = load_manifest(ingestion_config.MANIFEST_PATH)
    sources = []
    for rel_path, entry in sorted(manifest.items()):
        target = resolve_document_target(rel_path, entry)
        if target is not None and _rel_to_root(target[0]) in wanted:
            sources.append(rel_path)
    return sources


def restore_pending(data) -> list[str]:
    """
    Rebuild every document queued in data.pending_restore from corpus/raw/,
    re-apply the current private decisions to the derived files, and remove
    each rebuilt path from the queue (the caller saves). A raw file that no
    longer exists is dropped from the queue (nothing to rebuild). Any other
    failure propagates with the rest of the queue intact. Returns the rebuilt
    raw rel paths (paths only, no values).
    """
    if not data.pending_restore:
        return []
    from hypeonoath.ingestion import config as ingestion_config
    from hypeonoath.ingestion.cleaner import run_cleaning
    from hypeonoath.ingestion.converter import run_conversion
    from hypeonoath.ingestion.manifest import load_manifest, save_manifest
    from hypeonoath.ingestion.metadata_tagger import run_tagging
    from hypeonoath.ingestion.router import route_file
    from hypeonoath.privacy import decisions as dec
    from hypeonoath.privacy.redactor import redact_files_on_disk

    raw_dir = ingestion_config.RAW_DIR
    rebuilt: list[str] = []
    for rel_path in list(data.pending_restore):
        raw_file = raw_dir / rel_path
        if not raw_file.is_file():
            logger.warning("Rebuild skipped: raw file %s no longer exists", rel_path)
            data.pending_restore.remove(rel_path)
            continue
        manifest = load_manifest(ingestion_config.MANIFEST_PATH)
        manifest[rel_path] = route_file(raw_file, raw_dir)
        save_manifest(ingestion_config.MANIFEST_PATH, manifest)
        manifest = run_conversion(manifest, [rel_path])
        manifest = run_cleaning(manifest, [rel_path])
        run_tagging(manifest, [rel_path])
        rebuilt.append(rel_path)
        data.pending_restore.remove(rel_path)

    if rebuilt:
        redact_files_on_disk(dec.private_values(data.decisions))
        log_json_line(config.PRIVACY_LOG_PATH, event="restored_from_raw", files=rebuilt)
    return rebuilt
