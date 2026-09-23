"""
Content-hash manifest for safe re-runs of the Phase 0 pipeline.

Implements the design in plans/phase0-document-prep-subplan-v4.md, Section 3A:
on every run, each file under corpus/raw/ is classified as new, modified,
unchanged, or deleted by comparing its SHA-256 hash against the manifest
recorded on the previous run. Unchanged files are skipped entirely; deleted
files trigger cleanup of their routed/converted outputs (handled by
router.py, which owns the richer per-file manifest entries -- this module
only owns hashing, manifest I/O, and the new/modified/unchanged/deleted diff).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_HASH_CHUNK_SIZE = 65536  # 64 KiB, a reasonable read-buffer size for hashing


def compute_hash(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    """
    Load the manifest JSON file. Returns an empty dict if it doesn't exist
    yet (e.g., the very first run).
    """
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, data: dict) -> None:
    """
    Atomically write the manifest: write to a temp file in the same
    directory, then rename over the real path. Confirmed with user
    2026-09-15 (guideline Section 7 / Section 1A) -- this prevents a crash
    mid-write from leaving a corrupted/partial .manifest.json.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=".manifest_", suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, manifest_path)  # atomic on both POSIX and Windows
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class DiffResult:
    """Classification of every file under corpus/raw/ against the manifest."""

    new: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # manifest keys no longer present on disk


def diff(current_files: list[Path], raw_root: Path, manifest: dict) -> DiffResult:
    """
    Compare current_files (absolute paths under raw_root) against the
    manifest's recorded hashes to classify each as new, modified, or
    unchanged, and detect any manifest entries whose source file no longer
    exists (deleted).
    """
    result = DiffResult()
    current_rel_paths: set[str] = set()

    for file_path in current_files:
        rel_path = file_path.relative_to(raw_root).as_posix()
        current_rel_paths.add(rel_path)
        current_hash = compute_hash(file_path)
        entry = manifest.get(rel_path)

        if entry is None:
            result.new.append(file_path)
        elif entry.get("hash") != current_hash:
            result.modified.append(file_path)
        else:
            result.unchanged.append(file_path)

    for rel_path in manifest:
        if rel_path not in current_rel_paths:
            result.deleted.append(rel_path)

    return result
