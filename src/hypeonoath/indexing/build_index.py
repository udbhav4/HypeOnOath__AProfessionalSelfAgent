"""
Index build (offline half of the RAG pipeline): reviewed corpus -> privacy ->
chunks -> verification -> scan -> embeddings -> Chroma (guideline v4,
Section 5 run order; design doc 3.4; privacy plan v10, 4.2 + 7.1).

Run from the project root (package installed: `pip install -e ".[ingestion,indexing]"`):
    python -m hypeonoath.indexing.build_index      (or: hypeonoath-index)
Normally started by leaving `hypeonoath-privacy review` (= Done, P-D43).

Strict order -- the old index is replaced only after everything before it
succeeded, so any failure leaves the previous good index working. One build
at a time: a second build waits for the first (P-D47).
   1. project check   -> STOPS (exit 2) listing every code file without a
                         valid project; nothing else runs
  1a. review check    -> STOPS (exit 3) while files are pending review or not
                         scanned yet (P-D42); --force (development) scans the
                         unscanned files with the defaults and builds anyway
  1b. REBUILD FROM RAW the documents queued because a value went public
      (decisions.json pending_restore, P-D33); the queue is saved even on failure
   2. load all documents, code files (code blocks) and notebook documents
   3. NO DETECTION (P-D37): the scan (hypeonoath-privacy review) already
      stored every value with its decision in decisions.json
   4. REDACT every private value: blocks + project strings in memory (what
      gets indexed) and the derived corpus files on disk (never corpus/raw/);
      the matches refresh the decisions' locations (P-D49)
   5. chunk + one project_summary chunk per code project
   6. VERIFY = find-and-fix every chunk with the same matcher (P-D46): a value
      the chunker re-joined is redacted in the chunk; never fails
   7. secret scan of the chunks (detect-secrets) -> report of what remains
   8. embed every chunk ("search_document: " prefix)
   9. only now replace the index: same-folder swap when the privacy
      fingerprint is unchanged, a FRESH folder (old one deleted) when it
      changed or there is no index yet (P-D12)
  10. write the scan report; print the scan + privacy summaries LAST
      (counts and labels only -- never a value)

Exit codes: 0 success, 1 any other failure (previous index untouched),
2 project check failed, 3 privacy review needed (P-D42). (4 is retired:
verification fixes instead of failing, P-D46.)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from hypeonoath.core import config, embedder
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.indexing import chunker, code_chunker, indexer, project_registry, secret_scan
from hypeonoath.indexing.block_loader import LoadedDocument, load_documents
from hypeonoath.core import document_registry

logger = logging.getLogger(__name__)

EXIT_PROJECT_CHECK = 2
EXIT_NOT_REVIEWED = 3


@dataclass
class LoadedCorpus:
    """Steps 1-2 output, shared with `hypeonoath-privacy review` (same loading, same text)."""

    resolved: dict[str, project_registry.ProjectInfo]
    documents: list[LoadedDocument]
    code: code_chunker.CodeFileResult

    @property
    def all_documents(self) -> list[LoadedDocument]:
        return self.documents + self.code.notebook_docs


def load_corpus(after_check: Callable[[], None] | None = None) -> LoadedCorpus:
    """
    Steps 1-2. Raises project_registry.ProjectValidationError before reading
    anything else. `after_check` runs between the project check and loading
    (the build's rebuild-from-raw step 1b).
    """
    code_files, excluded = code_chunker.list_code_files()
    for rel, reason in excluded.items():
        logger.info("Excluded code file %s (%s)", rel, reason)
        code_chunker._log(status="excluded", file=rel, reason=reason)
    code_metadata = project_registry.load_code_metadata()
    resolved = project_registry.validate_code_projects(code_files, code_metadata)
    if after_check is not None:
        after_check()
    tokenizer = embedder.get_tokenizer()          # shared tokenizer = the embedding model's
    documents = load_documents()
    code = code_chunker.chunk_code_files(resolved, code_metadata, embedder.count_tokens, tokenizer)
    return LoadedCorpus(resolved, documents, code)


def _restore_from_raw(restored: list[str]) -> None:
    """Step 1b: rebuild queued documents from corpus/raw/ (P-D33); the queue is saved even on failure."""
    from hypeonoath.privacy import decisions as dec
    from hypeonoath.privacy import restore

    data = dec.load()
    if not data.pending_restore:
        return
    try:
        restored += restore.restore_pending(data)
    finally:
        dec.save(data)


class _NotReviewed(Exception):
    """Step 1a: files are pending review or not scanned yet (P-D42)."""

    def __init__(self, pending: list[str], stale: list[str]):
        super().__init__("privacy review needed")
        self.pending, self.stale = pending, stale

    def format(self) -> str:
        lines = [f"Privacy review needed before indexing (P-D42): {len(self.pending)} file(s) pending review, "
                 f"{len(self.stale)} file(s) not scanned yet."]
        lines += [f"  pending:     {f}" for f in self.pending] + [f"  not scanned: {f}" for f in self.stale]
        lines.append("Run: hypeonoath-privacy review   (leaving the review = Done, then the index is built)")
        lines.append("Development only: hypeonoath-index --force (unscanned files get the defaults: secrets "
                     "private, everything else public).")
        return "\n".join(lines)


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def build_lock(poll_seconds: float = 1.0):
    """
    P-D47: one build at a time; a second build WAITS, then runs. An OS file
    lock, so a crashed build never leaves it held.
    """
    path = config.PRIVACY_BUILD_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        told = False
        while not _try_lock(handle):
            if not told:
                print("Another index build is running -- waiting for it to finish.", file=sys.stderr)
                told = True
            time.sleep(poll_seconds)
        try:
            yield
        finally:
            _unlock(handle)


def _update_locations(decisions, stats) -> bool:
    """P-D49: files the scan did not re-read learn a newly private value's lines from the redaction matches."""
    changed = False
    for decision in decisions:
        for file, lines in stats.locations.get(decision.id, {}).items():
            changed |= decision.set_file_locations(file, list(lines))
    return changed


def _unreviewed_raw_findings(raw: list[secret_scan.Finding], decisions) -> list[secret_scan.Finding]:
    """
    Raw-code secret findings the privacy scan never stored (e.g. split across
    chunk boundaries). Stored ones are either redacted (private) or still in a
    chunk (kept), and then the chunk scan reports them.
    """
    from hypeonoath.privacy.detector import SECRET_PREFIX

    seen = {(loc["file"], loc.get("line"), etype.removeprefix(SECRET_PREFIX))
            for d in decisions for etype in d.all_types() if etype.startswith(SECRET_PREFIX) for loc in d.locations}
    return [f for f in raw if (f.file, f.line, f.finding_type) not in seen]


def _labels(counter: Counter) -> str:
    return ", ".join(f"{label} ×{n}" for label, n in sorted(counter.items())) or "none"


def _privacy_summary(redacted: Counter, fingerprint: str, fixed: Counter, files: list[str],
                     restored: list[str], pending: list[str]) -> str:
    lines = [f"Privacy: {sum(redacted.values())} private value occurrence(s) redacted "
             f"(labels: {_labels(redacted)}); index fingerprint {fingerprint}."]
    if fixed:
        lines.append(f"Privacy: verify fixed {sum(fixed.values())} occurrence(s) the chunker re-joined "
                     f"(labels: {_labels(fixed)}).")
    if restored:
        lines.append(f"Privacy: restored {len(restored)} document(s) from raw.")
    if files:
        lines.append(f"Privacy: redacted {len(files)} corpus file(s) on disk.")
    if pending:
        lines.append(f"Privacy: --force: {len(pending)} file(s) still pending review (hypeonoath-privacy review).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Chroma index from reviewed files (privacy decisions applied, no detection).")
    parser.add_argument("--force", action="store_true",
                        help="development only: build although files are pending review or not scanned")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with build_lock():
        return _build(force=args.force)


def _build(*, force: bool) -> int:
    logger.info("Index build starting")
    from hypeonoath.privacy import decisions as dec
    from hypeonoath.privacy import redactor, scan, verify

    # 1 + 1a + 1b + 2. Project check (first, before anything is read or written), review check
    # (P-D42), rebuild queued documents from raw, then load everything.
    restored: list[str] = []

    def after_check() -> None:
        pending, stale = scan.blockers()
        if (pending or stale) and not force:
            raise _NotReviewed(pending, stale)
        _restore_from_raw(restored)

    try:
        corpus = load_corpus(after_check=after_check)
    except project_registry.ProjectValidationError as exc:
        print(exc.format(), file=sys.stderr)
        return EXIT_PROJECT_CHECK
    except _NotReviewed as exc:
        print(exc.format(), file=sys.stderr)
        return EXIT_NOT_REVIEWED
    resolved, documents, code = corpus.resolved, corpus.documents, corpus.code
    # Document registry check (document-registry plan v2, DR-9): here, not in load_corpus, so a
    # missing description stops only the build, not the privacy review that also loads the corpus.
    doc_failures = document_registry.validate_documents([d.source_document for d in documents])
    if doc_failures:
        print("\n".join([
            f"Indexing stopped: {len(doc_failures)} document-registry problem(s). "
            "Every document needs a description (document-registry plan v2, DR-9).",
            *[f"  - {failure}" for failure in doc_failures],
            "Fix DOCUMENTS / CODE_PROJECTS in src/hypeonoath/core/registry.py, then re-run this indexer. "
            "The existing index was not touched.",
        ]), file=sys.stderr)
        return EXIT_PROJECT_CHECK
    all_documents = corpus.all_documents
    if force:
        scan.run(corpus)                  # --force: unscanned files still get the defaults (secrets private)

    # 3. No detection here (P-D37): every value and its decision is already in decisions.json.
    data = dec.load()
    decisions = data.decisions

    # 4. Redact (memory first -- authoritative for the index -- then disk); locations from the matches.
    values = dec.private_values(decisions)
    fingerprint = dec.privacy_fingerprint(decisions)
    stats = redactor.redact_blocks(all_documents, code.blocks, values)
    redactor.redact_projects(resolved, values, stats)
    changed_files = redactor.redact_files_on_disk(values)
    if _update_locations(decisions, stats):
        dec.save(data)

    # 5. Chunk.
    chunks = chunker.make_chunks(all_documents, code.blocks, embedder.count_tokens)
    chunks += project_registry.build_project_summary_chunks(resolved)
    logger.info("Chunked %d document(s) + %d code file(s) into %d chunks: %s",
                len(documents), len(resolved), len(chunks), dict(Counter(c.content_type for c in chunks)))

    # 6. Verify = find-and-fix (P-D46): values the chunker re-joined are redacted in the chunk.
    fixed = verify.fix_chunks(chunks, values)

    # 7. Secret scan: what is still in the index (kept / allowlisted) + raw-code hits never stored.
    report = secret_scan.build_report(_unreviewed_raw_findings(code.findings, decisions)
                                      + secret_scan.scan_chunks(chunks))

    # 8. Embed.
    vectors = embedder.embed_documents([c.text for c in chunks])

    # 9. Only now replace the index (fresh folder when the privacy decisions changed).
    fresh = indexer.current_privacy_fp() != fingerprint
    indexer.rebuild_index(chunks, vectors, privacy_fp=fingerprint, fresh_dir=fresh)
    scan.refresh_hashes()                 # disk redaction / restore changed scanned files, nothing new in them

    # 10. Report + summaries printed LAST (no values anywhere).
    pending = scan.load_state().pending()
    secret_scan.write_report(report)
    log_json_line(config.PRIVACY_LOG_PATH, event="build", redacted=dict(stats.labels),
                  files_redacted=len(changed_files), privacy_fp=fingerprint, fresh_dir=fresh,
                  verify_fixed=dict(fixed), restored=len(restored), forced=force, pending_files=len(pending))
    print(f"\nIndexed {len(chunks)} chunks into {config.CHROMA_DIR} (collection '{config.COLLECTION_NAME}')"
          f"{' -- fresh folder, old index deleted' if fresh else ''}.")
    print(secret_scan.format_summary(report))
    print(_privacy_summary(stats.labels, fingerprint, fixed, changed_files, restored, pending))
    return 0


if __name__ == "__main__":
    sys.exit(main())
