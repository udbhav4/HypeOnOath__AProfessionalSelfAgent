"""
Step 3.2: automatic code/document/quarantine routing.

Copies (never moves) every new or modified file under corpus/raw/ into
exactly one of corpus/code/, corpus/documents/, or corpus/quarantine/,
based on file extension, and logs every decision as a JSON Line. Unchanged
files (per the manifest) are skipped entirely. Deleted files have their
routed/converted outputs cleaned up. corpus/raw/ itself is never written to
or deleted from -- see __init__.py for the immutability decision.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import config
from .logging_utils import log_json_line
from .manifest import DiffResult, compute_hash, diff, load_manifest, save_manifest

logger = logging.getLogger(__name__)

_CATEGORY_DIRS = {
    "code": config.CODE_DIR,
    "document": config.DOCUMENTS_DIR,
    "quarantine": config.QUARANTINE_DIR,
}


def _iter_raw_files(raw_dir: Path) -> list[Path]:
    """Return every regular file under raw_dir, recursively."""
    if not raw_dir.exists():
        return []
    return [p for p in raw_dir.rglob("*") if p.is_file()]


def classify_extension(path: Path) -> str:
    """
    Return 'code', 'document', or 'quarantine' for a given file, based on
    its extension against config.CODE_EXTENSIONS / config.DOCUMENT_EXTENSIONS.
    There is deliberately no default that assumes an unrecognized extension
    is a document -- see the guideline's Section 6 discussion.
    """
    ext = path.suffix.lower()
    if ext in config.CODE_EXTENSIONS:
        return "code"
    if ext in config.DOCUMENT_EXTENSIONS:
        return "document"
    return "quarantine"


def route_file(raw_file: Path, raw_root: Path) -> dict:
    """
    Copy a single file from corpus/raw/ into its destination folder and log
    the decision. Returns the manifest entry to store for this file
    (converted_output is left None here -- converter.py fills it in later,
    if applicable).
    """
    rel_path = raw_file.relative_to(raw_root)
    category = classify_extension(raw_file)
    dest_root = _CATEGORY_DIRS[category]
    dest_path = dest_root / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # copy2 preserves metadata (mtime) but never touches the source -- raw_file
    # is only ever opened here in a read capacity, by shutil internally.
    shutil.copy2(raw_file, dest_path)

    ext = raw_file.suffix.lower()
    if category == "quarantine":
        action, reason = "quarantined", f"extension '{ext}' not in code or document list"
    else:
        action, reason = "routed", f"extension '{ext}' matched {category} list"

    log_json_line(
        config.ROUTING_LOG_PATH,
        file=rel_path.as_posix(),
        action=action,
        destination=dest_path.relative_to(config.CORPUS_DIR).as_posix(),
        reason=reason,
    )

    return {
        "hash": compute_hash(raw_file),
        "destination_category": category,
        "converted_output": None,
        "structured_output": None,
    }


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_deleted(rel_path_str: str, manifest: dict) -> None:
    """
    A file that was previously routed has been removed from corpus/raw/:
    delete its routed copy (and converted output, if any) and drop its
    manifest entry.
    """
    entry = manifest.get(rel_path_str)
    if entry is None:
        return

    dest_root = _CATEGORY_DIRS.get(entry.get("destination_category"))
    if dest_root is not None:
        _remove_if_exists(dest_root / rel_path_str)

    converted_output = entry.get("converted_output")
    if converted_output:
        _remove_if_exists(config.CONVERTED_DIR / converted_output)

    structured_output = entry.get("structured_output")
    if structured_output:
        _remove_if_exists(config.CONVERTED_DIR / structured_output)

    del manifest[rel_path_str]

    log_json_line(
        config.ROUTING_LOG_PATH,
        file=rel_path_str,
        action="deleted",
        destination=None,
        reason="source file removed from corpus/raw/; outputs cleaned up",
    )


def run_routing() -> tuple[dict, list[str]]:
    """
    Route every new/modified file in corpus/raw/, skip unchanged files, and
    clean up outputs for deleted files. Returns the updated (and already
    persisted) manifest, plus the list of raw-relative paths processed this
    run (new + modified) -- converter.py uses this list to know exactly
    which documents need (re-)conversion.
    """
    for directory in (
        config.RAW_DIR,
        config.CODE_DIR,
        config.DOCUMENTS_DIR,
        config.QUARANTINE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config.MANIFEST_PATH)
    raw_files = _iter_raw_files(config.RAW_DIR)
    diff_result: DiffResult = diff(raw_files, config.RAW_DIR, manifest)

    processed_rel_paths: list[str] = []
    for raw_file in diff_result.new + diff_result.modified:
        rel_path_str = raw_file.relative_to(config.RAW_DIR).as_posix()
        manifest[rel_path_str] = route_file(raw_file, config.RAW_DIR)
        processed_rel_paths.append(rel_path_str)

    for rel_path_str in diff_result.deleted:
        _cleanup_deleted(rel_path_str, manifest)

    save_manifest(config.MANIFEST_PATH, manifest)

    logger.info(
        "Routing complete: %d new, %d modified, %d unchanged, %d deleted",
        len(diff_result.new),
        len(diff_result.modified),
        len(diff_result.unchanged),
        len(diff_result.deleted),
    )
    return manifest, processed_rel_paths
