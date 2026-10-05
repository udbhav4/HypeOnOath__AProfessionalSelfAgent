"""
Index build (offline half of the RAG pipeline): corpus -> chunks -> scan ->
embeddings -> Chroma (guideline v4, Section 5 run order; design doc 3.4).

Run from the project root (package installed: `pip install -e ".[ingestion,indexing]"`):
    python -m hypeonoath.indexing.build_index      (or: hypeonoath-index)

Strict order -- the old index is deleted only after everything before it
succeeded, so any failure leaves the previous good index working:
  1. project check  -> STOPS (exit 2) listing every code file without a
                       valid project; nothing else runs
  2. load + chunk all documents and code files, + one project_summary
     chunk per code project
  3. secret + personal-info scan of every chunk -> warnings, run CONTINUES
  4. embed every chunk ("search_document: " prefix)
  5. only now replace the Chroma collection (full rebuild)
  6. write .logs/scan_warnings.jsonl and print the warning summary LAST

Exit codes: 0 success (even with scan warnings), 2 project check failed,
1 any other failure (previous index untouched).
"""
from __future__ import annotations

import logging
import sys
from collections import Counter

from hypeonoath.core import config, embedder
from hypeonoath.indexing import chunker, code_chunker, indexer, project_registry, secret_scan
from hypeonoath.indexing.block_loader import load_documents

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Index build starting")

    # 1. Project check (first, before anything is read or written).
    code_files, excluded = code_chunker.list_code_files()
    for rel, reason in excluded.items():
        logger.info("Excluded code file %s (%s)", rel, reason)
        code_chunker._log(status="excluded", file=rel, reason=reason)
    code_metadata = project_registry.load_code_metadata()
    try:
        resolved = project_registry.validate_code_projects(code_files, code_metadata)
    except project_registry.ProjectValidationError as exc:
        print(exc.format(), file=sys.stderr)
        return 2

    # 2. Load + chunk everything (shared tokenizer = the embedding model's).
    tokenizer = embedder.get_tokenizer()
    documents = load_documents()
    code = code_chunker.chunk_code_files(resolved, code_metadata, embedder.count_tokens, tokenizer)
    chunks = chunker.make_chunks(documents + code.notebook_docs, code.blocks, embedder.count_tokens)
    chunks += project_registry.build_project_summary_chunks(resolved)
    logger.info("Chunked %d document(s) + %d code file(s) into %d chunks: %s",
                len(documents), len(resolved), len(chunks), dict(Counter(c.content_type for c in chunks)))

    # 3. Scan every chunk -- collect warnings, never stop.
    report = secret_scan.build_report(code.findings + secret_scan.scan_chunks(chunks))

    # 4. Embed.
    vectors = embedder.embed_documents([c.text for c in chunks])

    # 5. Only now replace the index.
    indexer.rebuild_index(chunks, vectors)

    # 6. Report + summary printed LAST.
    secret_scan.write_report(report)
    print(f"\nIndexed {len(chunks)} chunks into {config.CHROMA_DIR} (collection '{config.COLLECTION_NAME}').")
    print(secret_scan.format_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
