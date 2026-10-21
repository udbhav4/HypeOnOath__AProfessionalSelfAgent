"""
Scan stage of the privacy pipeline (privacy plan v10: P-D37, P-D38, P-D39, P-D48).

Detection (Presidio + spaCy + detect-secrets) runs HERE, once, on the files
that are new or changed since their last scan -- the index build never
detects. Every value found is written to decisions.json at once with its
default state (secrets private, P-D31; everything else public, P-D27), and
each file holding a value decisions.json did not have before is marked
*pending review* until the owner's Done (P-D40, P-D41).

The scan record corpus/.privacy/scan_state.json holds paths, hashes and
pending flags only -- never values:
    {"setup": <detector-setup hash>, "files": {path: {"hash": ..., "pending": bool}}}
A file is re-scanned when its hash changes; a new detector setup (Presidio /
model versions, recognizer config, score threshold, routing switches)
re-scans every file (P-D38). A file that is gone loses its record and
locations; public decisions left without any location are dropped, private
ones always stay (P-D48).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy import detector

if TYPE_CHECKING:
    from hypeonoath.indexing.build_index import LoadedCorpus

_SETUP_PACKAGES = ("presidio-analyzer", "spacy", "detect-secrets")
_SETUP_CONF_FILES = ("recognizers.yaml", "nlp.yaml")


# --- the record --------------------------------------------------------------------------------------

@dataclass
class ScanState:
    setup: str = ""
    files: dict[str, dict[str, Any]] = field(default_factory=dict)     # path -> {"hash", "pending"}

    def pending(self) -> list[str]:
        return sorted(f for f, record in self.files.items() if record.get("pending"))


def load_state(path: Path | None = None) -> ScanState:
    path = config.PRIVACY_SCAN_STATE_PATH if path is None else path
    if not path.exists():
        return ScanState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ScanState(raw.get("setup", ""), dict(raw.get("files") or {}))


def save_state(state: ScanState, path: Path | None = None) -> None:
    """Atomic, like decisions.save."""
    path = config.PRIVACY_SCAN_STATE_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".scan_state.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"setup": state.setup, "files": dict(sorted(state.files.items()))}, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def setup_hash() -> str:
    """Everything that changes WHAT detection finds (P-D38)."""
    parts = []
    for dist in (*_SETUP_PACKAGES, *config.PRIVACY_SPACY_MODELS.values()):
        try:
            parts.append(f"{dist}={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{dist}=missing")
    for name in _SETUP_CONF_FILES:
        conf = detector.CONF_DIR / name
        parts.append(f"{name}={hashlib.sha256(conf.read_bytes()).hexdigest() if conf.exists() else 'missing'}")
    parts += [repr(config.PRIVACY_MIN_SCORE), repr(config.PRIVACY_LANGUAGE_ROUTING),
              repr(config.PRIVACY_CODE_DROPPED_ENTITIES), repr(config.PRIVACY_NER_RECOGNIZERS)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def unit_files() -> list[str]:
    """Every file detection reads, named as Candidate.file names it (posix, relative to the project root)."""
    from hypeonoath.indexing import code_chunker
    from hypeonoath.indexing.block_loader import document_targets, rel_to_root

    files = [rel_to_root(path) for path, _converted in document_targets()]
    code, _excluded = code_chunker.list_code_files()
    files += [f"corpus/code/{rel}" for rel in code]
    if config.CODE_METADATA_PATH.exists():
        files.append(detector.project_strings_file())
    return sorted(set(files))


def file_hash(rel: str) -> str:
    """Bytes of the file (+ its .docling.json, which the loader reads first)."""
    path = config.PROJECT_ROOT / rel
    digest = hashlib.sha256(path.read_bytes() if path.exists() else b"")
    sibling = path.with_name(path.name[: -len(path.suffix)] + config.DOCLING_JSON_SUFFIX) if path.suffix else None
    if sibling is not None and sibling.exists():
        digest.update(sibling.read_bytes())
    return digest.hexdigest()[:16]


def stale_files(state: ScanState, files: list[str] | None = None) -> list[str]:
    """Files whose last scan is out of date: new, changed, gone, or all of them after a setup change."""
    files = unit_files() if files is None else files
    if state.setup != setup_hash():
        return files
    gone = [f for f in state.files if f not in set(files)]
    return sorted(gone + [f for f in files if state.files.get(f, {}).get("hash") != file_hash(f)])


def blockers() -> tuple[list[str], list[str]]:
    """(pending review, not scanned) -- the build refuses while either is non-empty (P-D42)."""
    state = load_state()
    return state.pending(), stale_files(state)


def mark_reviewed(files: list[str] | None = None) -> None:
    """Done (P-D41): the state shown is the owner's; those files (default: all) are no longer pending."""
    state = load_state()
    for rel, record in state.files.items():
        if files is None or rel in files:
            record["pending"] = False
    save_state(state)


def refresh_hashes() -> None:
    """After a build: its disk redaction / restore changed scanned files, which hold no new values."""
    state = load_state()
    for rel in state.files:
        if (config.PROJECT_ROOT / rel).exists():
            state.files[rel]["hash"] = file_hash(rel)
    save_state(state)


# --- the scan -----------------------------------------------------------------------------------------

def _detect(documents, code_blocks, projects):
    """Detection seam (tests swap in a fake detector without spaCy models)."""
    return detector.detect_blocks(documents, code_blocks, projects)


@dataclass
class ScanResult:
    scanned: list[str]                # files detected on this run
    pending: list[str]                # every file pending review after this run
    new_values: int                   # decisions created (values decisions.json did not hold)
    deleted: list[str]                # files gone since the last scan

    def summary(self) -> str:
        return (f"Privacy scan: {len(self.scanned)} file(s) scanned, {self.new_values} new value(s); "
                f"{len(self.pending)} file(s) pending review" + (f"; {len(self.deleted)} file(s) gone" if self.deleted else "")
                + ".")


def _drop_vanished(data: dec.DecisionFile, groups: list[dec.CandidateGroup], scanned: set[str]) -> None:
    """A public value no longer found in a re-scanned file loses that location (private ones may be redacted there)."""
    index = dec.DecisionIndex(data.decisions)
    found: set[tuple[str, str]] = set()
    for group in groups:
        decision = index.by_decision_id(group.id) or next(iter(index.by_value(group.value_keys())), None)
        if decision is not None:
            found.update((decision.id, file) for file in group.files)
    for decision in data.decisions:
        if not decision.is_private:
            decision.locations = [loc for loc in decision.locations
                                  if loc["file"] not in scanned or (decision.id, loc["file"]) in found]


def run(corpus: "LoadedCorpus", *, full: bool = False) -> ScanResult:
    """Detect on new / changed files (all with full=True), store every value, mark pending files."""
    state = load_state()
    setup = setup_hash()
    files = unit_files()
    if full or state.setup != setup:
        scanned = files
    else:
        scanned = [f for f in files if state.files.get(f, {}).get("hash") != file_hash(f)]
    deleted = sorted(f for f in state.files if f not in set(files))
    todo = set(scanned)

    documents = [d for d in corpus.all_documents if d.source_document in todo]
    code_blocks = [b for b in corpus.code.blocks if b.source_document in todo]
    projects = corpus.resolved if detector.project_strings_file() in todo else {}
    candidates = _detect(documents, code_blocks, projects) if (documents or code_blocks or projects) else []

    reviewable, _allowlisted = dec.import_scan_allowlist().filter(candidates)
    data = dec.load()
    before = {d.id for d in data.decisions}
    groups = dec.group_candidates(reviewable)
    dec.resolve_all(groups, data)                     # R1/R5/R2/R3/RS, prune retired, refresh locations
    _drop_vanished(data, groups, todo)
    dec.set_keep_for_undecided(groups, data, source="detected")     # P-D39: every value stored
    data.decisions = [d for d in data.decisions if d.is_private or d.locations]   # P-D48
    new = [d for d in data.decisions if d.id not in before]
    newly_pending = {f for d in new for f in d.files() if f in todo}
    dec.save(data)

    for rel in deleted:
        state.files.pop(rel, None)
    for rel in scanned:
        was_pending = bool(state.files.get(rel, {}).get("pending"))
        state.files[rel] = {"hash": file_hash(rel), "pending": was_pending or rel in newly_pending}
    state.setup = setup
    save_state(state)
    log_json_line(config.PRIVACY_LOG_PATH, event="scan", files=len(scanned), new_values=len(new),
                  pending_files=len(state.pending()), deleted_files=len(deleted))
    return ScanResult(scanned, state.pending(), len(new), deleted)
