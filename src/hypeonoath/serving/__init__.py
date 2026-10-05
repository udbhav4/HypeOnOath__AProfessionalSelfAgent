"""Pipeline B (online): question -> retrieve -> prompt -> LLM -> answer + citations.
Entry point: python -m hypeonoath.serving.chat (the FastAPI app joins in Phase 2).
Imports only hypeonoath.core -- never indexing or ingestion."""
