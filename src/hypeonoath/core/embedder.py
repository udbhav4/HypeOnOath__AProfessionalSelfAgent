"""
Step 3 -- text -> vectors, the ONLY embedding code in the project (guideline
v4, Section 4; design doc Section 2.2 invariant + ADR-3).

  * nomic-ai/nomic-embed-text-v1.5, in-process via sentence-transformers,
    pinned to a Hub revision (config.EMBED_MODEL_REVISION). Ollama is never
    used for embedding: its build and the HF weights are not guaranteed to
    produce identical vectors.
  * One loaded model object, two role-specific entry points. nomic requires
    a task prefix on every input and missing/swapped prefixes degrade
    retrieval silently, so callers never write prefixes themselves:
        embed_documents(texts) -> "search_document: " + text   (indexer.py)
        embed_query(text)      -> "search_query: " + text      (retriever.py)
  * get_tokenizer()/count_tokens() expose the SAME model's tokenizer, so
    prose chunks, code chunks (Chonkie) and the model count tokens alike.
  * Vectors are L2-normalized; the Chroma collection uses cosine space.

The model is loaded lazily on first use (once per process), so importing
this module stays cheap for code paths that never embed.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load the pinned embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.EMBED_MODEL_NAME,
        revision=config.EMBED_MODEL_REVISION or None,
        trust_remote_code=config.EMBED_TRUST_REMOTE_CODE,
    )
    model.max_seq_length = config.EMBED_MAX_TOKENS
    return model


def get_tokenizer():
    """The embedding model's own tokenizer (shared with both chunkers)."""
    return get_model().tokenizer


def count_tokens(text: str) -> int:
    """Token count as the model sees it (no special tokens)."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def _encode(texts: list[str]) -> list[list[float]]:
    for text in texts:
        n = count_tokens(text)
        if n > config.EMBED_MAX_TOKENS:
            # Chunkers should make this impossible; never let it be silent.
            logger.warning("Text of %d tokens exceeds %d and will be truncated", n, config.EMBED_MAX_TOKENS)
            log_json_line(config.INDEXING_LOG_PATH, status="embed_truncation", tokens=n, preview=text[:80])
    vectors = get_model().encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Index-time embedding: adds the "search_document: " prefix."""
    if not texts:
        return []
    return _encode([config.DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    """Query-time embedding: adds the "search_query: " prefix."""
    return _encode([config.QUERY_PREFIX + text])[0]
