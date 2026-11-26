"""
Cross-encoder re-ranking for retrieval experiment F-R6 (plans/fr6-reranking-design-plan-v2.md).

  Reranker(alias)                 -> lazy sentence_transformers.CrossEncoder at the pinned revision
  Reranker.score(query, texts)    -> one relevance score per text (the model's default activation)
  Reranker.count_truncated(...)   -> how many (query, text) inputs exceed max_length (they get cut, FR6-6)
  rerank(hits, scores)            -> hits ordered by score, ties by their incoming position (deterministic)
  rescale_to_cosine(ce, cosines)  -> CE scores mapped into the pool's cosine range, for MMR's relevance (FR6-3)

The cross-encoder reads the query and the chunk together, so it judges relevance from both texts at once;
`score` on a RetrievedItem stays the cosine similarity (the gate reads cosine, FR6-2). sentence-transformers
is a pinned base dependency, but it is imported only when a model is loaded, so importing this module costs
nothing (serving import test). Query and chunk texts are never logged.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem


def check_available(alias: str) -> None:
    """ValueError when the alias is unknown or sentence-transformers' CrossEncoder cannot be imported, so a
    re-rank run refuses before it starts (as the F-R4 / F-R5 checks). The model itself loads in prepare()."""
    if alias not in config.RERANK_MODELS:
        raise ValueError(f"unknown re-ranker {alias!r}; known: {sorted(config.RERANK_MODELS)} (config.RERANK_MODELS)")
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError as exc:
        raise ValueError(f"--rerank needs sentence-transformers' CrossEncoder ({type(exc).__name__}: {exc})") from None


class Reranker:
    """One cross-encoder model, loaded on first use. `model` = an injected scorer (tests: no download)."""

    def __init__(self, alias: str, max_length: int = config.RERANK_MAX_LENGTH,
                 batch_size: int = config.RERANK_BATCH_SIZE, model: Any = None):
        if alias not in config.RERANK_MODELS:
            raise ValueError(f"unknown re-ranker {alias!r}; known: {sorted(config.RERANK_MODELS)}")
        if max_length < 1 or batch_size < 1:
            raise ValueError(f"max_length and batch_size must be >= 1, got {max_length}, {batch_size}")
        self.alias = alias
        self.model_id, self.revision = config.RERANK_MODELS[alias]
        self.max_length, self.batch_size = max_length, batch_size
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            def load(local_only: bool):
                return CrossEncoder(self.model_id, revision=self.revision, max_length=self.max_length,
                                    local_files_only=local_only)
            try:
                try:
                    self._model = load(True)           # cached: no Hub request, eval runs stay offline
                except Exception:  # noqa: BLE001 -- not cached yet (first use): download once
                    self._model = load(False)
            except Exception as exc:  # noqa: BLE001 -- download / cache / config errors differ per version
                raise ValueError(f"cannot load re-ranker {self.model_id}@{self.revision[:12]} "
                                 f"({type(exc).__name__}: {exc})") from None
        return self._model

    @property
    def activation(self) -> str:
        """Name of the activation the model applies to its logits (recorded in the manifest)."""
        fn = getattr(self.model, "activation_fn", None)
        return type(fn).__name__ if fn is not None else "none"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in the given order; [] for no texts."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                    convert_to_numpy=True)
        return [float(s) for s in scores]

    def count_truncated(self, query: str, texts: Sequence[str]) -> int:
        """Inputs whose tokenized (query, text) pair is longer than max_length (counted before truncation)."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not texts:
            return 0
        # verbose=False: counting long inputs is the point here, the "longer than max length" warning is not
        encoded = tokenizer([query] * len(texts), list(texts), truncation=False, verbose=False)["input_ids"]
        return sum(1 for ids in encoded if len(ids) > self.max_length)


def rerank(hits: Sequence[RetrievedItem], scores: Sequence[float]) -> list[RetrievedItem]:
    """
    Copies of `hits` with rerank_score and pre_rerank_rank (1-based incoming position) set, best score first;
    ties keep the incoming order (the cosine / candidate order), so the result is deterministic.
    """
    if len(scores) != len(hits):
        raise ValueError(f"rerank needs one score per hit ({len(hits)} hits, {len(scores)} scores)")
    scored = [dataclasses.replace(h, rerank_score=float(s), pre_rerank_rank=i)
              for i, (h, s) in enumerate(zip(hits, scores), 1)]
    return sorted(scored, key=lambda h: (-h.rerank_score, h.pre_rerank_rank))


def rescale_to_cosine(ce_scores: Sequence[float], cosines: Sequence[float]) -> list[float]:
    """
    FR6-3: rel = cos_min + (ce - ce_min) / (ce_max - ce_min) * (cos_max - cos_min), over the same items.
    Keeps the CE's order and relative gaps but gives them the pool's cosine spread, so MMR's redundancy
    penalty (chunk-to-chunk cosine) still counts. All CE scores equal (it tells nothing apart) -> each
    item's own cosine.
    """
    if len(ce_scores) != len(cosines):
        raise ValueError("rescale_to_cosine needs one cosine per CE score")
    if not ce_scores:
        return []
    ce_min, ce_max = min(ce_scores), max(ce_scores)
    if ce_max == ce_min:
        return [float(c) for c in cosines]
    cos_min, cos_max = min(cosines), max(cosines)
    span = (cos_max - cos_min) / (ce_max - ce_min)
    return [cos_min + (s - ce_min) * span for s in ce_scores]
