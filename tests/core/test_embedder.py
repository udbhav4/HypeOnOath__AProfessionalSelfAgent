"""embedder.py: prefixes, determinism, dimensions (guideline 8A, Section 2.2)."""
from __future__ import annotations

import numpy as np

from hypeonoath.core import config


def test_document_prefix_really_added(embedder_module):
    manual = embedder_module.get_model().encode([config.DOC_PREFIX + "x"], normalize_embeddings=True)[0]
    assert np.allclose(embedder_module.embed_documents(["x"])[0], manual, atol=1e-5)


def test_query_and_document_vectors_differ(embedder_module):
    assert not np.allclose(embedder_module.embed_query("x"), embedder_module.embed_documents(["x"])[0], atol=1e-3)


def test_repeat_embedding_identical(embedder_module):
    a = embedder_module.embed_documents(["same text", "other"])
    b = embedder_module.embed_documents(["same text", "other"])
    assert np.allclose(a, b, atol=1e-6)


def test_dimension_and_normalization(embedder_module):
    vector = embedder_module.embed_query("hello")
    assert len(vector) == config.EMBED_DIM == 768
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-4


def test_empty_input(embedder_module):
    assert embedder_module.embed_documents([]) == []


def test_tokenizer_is_the_models_own(embedder_module):
    assert embedder_module.get_tokenizer() is embedder_module.get_model().tokenizer
    assert embedder_module.count_tokens(config.DOC_PREFIX) == 4


def test_pinned_revision_and_no_remote_code():
    assert len(config.EMBED_MODEL_REVISION) == 40 and not config.EMBED_TRUST_REMOTE_CODE
