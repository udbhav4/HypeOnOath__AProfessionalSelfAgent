"""
hypeonoath -- a recruiter-facing, grounded RAG chatbot over the owner's own
documents and code.

Subpackages (dependency direction: everything may import core; serving
imports ONLY core, never indexing or ingestion -- see tests/test_entrypoints.py):

  core       shared building blocks: paths, config, records, embedder, store
  ingestion  corpus preparation (Phase 0): raw files -> cleaned, tagged corpus
  indexing   Pipeline A (offline): corpus -> chunks -> scan -> vectors -> Chroma
  serving    Pipeline B (online): question -> retrieve -> prompt -> answer
"""
