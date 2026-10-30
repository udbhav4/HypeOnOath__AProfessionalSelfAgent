"""
The chatbot as the harness sees it (eval plan v15, D71).

The runner talks to the chatbot only through an Answerer with two methods:
  retrieve(question)          -> Retrieved(ranking, context)   raw top-20 ranking + the items put in the prompt
  generate(question, context) -> Generated(text, stats, prompt)  raw, UNGUARDED reply + Ollama stats
Two methods because --retrieval-only must never call the LLM, --pin-contexts feeds a stored
context into generation, and retrieval metrics need the raw ranking. The privacy guard is
NOT part of an answerer: the runner guards every variant identically (D52).

BaselineAnswerer = today's chain. Later option-d variants (V11-15: reranker, question split,
hosted LLM) are new classes registered in ANSWERERS and picked with --answerer NAME.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import paths
from hypeonoath.serving import llm_client
from hypeonoath.serving.prompt import _PART_REMINDER, Prompt, build_prompt, system_rules


@dataclass
class Retrieved:
    ranking: list[RetrievedItem]          # raw ranked hits (after MMR if on), best first, <= EVAL_DEPTH
    context: list[RetrievedItem]          # exactly what goes into the prompt


@dataclass
class Generated:
    text: str                             # raw reply, before the guard
    stats: dict[str, int] = field(default_factory=dict)
    prompt: str = ""                      # full prompt text (system + user)


class Answerer(Protocol):
    name: str

    def retrieve(self, question: str) -> Retrieved: ...

    def generate(self, question: str, context: list[RetrievedItem]) -> Generated: ...

    def settings(self) -> dict[str, Any]: ...

    def index_ids(self) -> list[str]: ...           # chunk ids of the index it reads (index fingerprint)


class BaselineAnswerer:
    """Retriever.search(k=20) -> optional MMR -> assemble(top_k) -> build_prompt(version) -> Ollama."""

    name = "baseline"

    def __init__(self, retriever=None, top_k: int = config.DEFAULT_TOP_K, mmr_lambda: float | None = config.MMR_LAMBDA,
                 fetch_k: int = config.MMR_FETCH_K, prompt_version: str = config.PROMPT_VERSION,
                 generate: Callable[[Prompt], llm_client.GenerationResult] | None = None):
        if not 1 <= top_k <= paths.EVAL_DEPTH:
            raise ValueError(f"top_k must be 1-{paths.EVAL_DEPTH} (the eval depth), got {top_k}")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"MMR lambda must be in [0, 1], got {mmr_lambda}")
        if fetch_k > paths.EVAL_DEPTH:
            raise ValueError(f"fetch_k ({fetch_k}) cannot exceed the eval depth ({paths.EVAL_DEPTH})")
        system_rules(prompt_version)                      # unknown version -> ValueError now, not mid-run
        self._retriever = retriever
        self.top_k, self.mmr_lambda, self.fetch_k = top_k, mmr_lambda, fetch_k
        self.prompt_version = prompt_version
        self._generate = generate or llm_client.generate

    @property
    def retriever(self):
        if self._retriever is None:                       # lazy: --pin-contexts never needs the index
            from hypeonoath.serving.retriever import Retriever
            self._retriever = Retriever()
        return self._retriever

    def retrieve(self, question: str) -> Retrieved:
        from hypeonoath.serving.retriever import mmr

        if self.mmr_lambda is None:
            ranking = self.retriever.search(question, paths.EVAL_DEPTH)
        else:
            hits, vectors = self.retriever.search_with_embeddings(question, paths.EVAL_DEPTH)
            head = mmr(hits[:self.fetch_k], vectors[:self.fetch_k], self.fetch_k, self.mmr_lambda)
            ranking = head + hits[self.fetch_k:]
        return Retrieved(ranking, self.retriever.assemble(ranking[:self.top_k]))

    def generate(self, question: str, context: list[RetrievedItem]) -> Generated:
        prompt = build_prompt(question, context, self.prompt_version)
        result = self._generate(prompt)
        return Generated(result.text, dict(result.stats), prompt.as_text())

    def index_ids(self) -> list[str]:
        return self.retriever.collection.get(include=[])["ids"]

    def settings(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            # system text + the fixed last user line; the NOTE line / labels are code, tracked by git_sha
            "prompt_sha256": hashlib.sha256((system_rules(self.prompt_version) + "\n" + _PART_REMINDER)
                                            .encode("utf-8")).hexdigest(),
            "ollama_model": config.OLLAMA_MODEL,
            "top_k": self.top_k,
            "eval_depth": paths.EVAL_DEPTH,
            "mmr": {"enabled": self.mmr_lambda is not None, "lambda": self.mmr_lambda, "fetch_k": self.fetch_k},
        }


ANSWERERS: dict[str, type] = {"baseline": BaselineAnswerer}


def make_answerer(name: str, **kwargs) -> Answerer:
    try:
        cls = ANSWERERS[name]
    except KeyError:
        raise ValueError(f"unknown answerer {name!r}; known: {sorted(ANSWERERS)}") from None
    return cls(**kwargs)
