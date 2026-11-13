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
from pathlib import Path
from typing import Any, Protocol

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import paths
from hypeonoath.serving import llm_client
from hypeonoath.serving.query_rewrite import check_available, rewrite_query
from hypeonoath.serving.prompt import _PART_REMINDER, Prompt, build_prompt, system_rules


@dataclass
class Retrieved:
    ranking: list[RetrievedItem]          # raw ranked hits (after MMR if on), best first, <= EVAL_DEPTH
    context: list[RetrievedItem]          # exactly what goes into the prompt
    query: str | None = None              # the retrieval query when it differs from the question (F-R4, D94)


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


def index_label(index_dir: Path) -> str:
    """The index folder as recorded in the manifest: relative to the project root when inside it (D95)."""
    try:
        return index_dir.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return index_dir.resolve().as_posix()


class BaselineAnswerer:
    """
    [optional third-person query rewrite, F-R4] -> Retriever.search(k=20) -> optional MMR ->
    assemble(top_k) -> build_prompt(version) -> Ollama. The index is read from `index_dir`
    (default: the production index/chroma; F-R3 variants live in their own folders, D95).
    """

    name = "baseline"

    def __init__(self, retriever=None, top_k: int = config.DEFAULT_TOP_K, mmr_lambda: float | None = config.MMR_LAMBDA,
                 fetch_k: int = config.MMR_FETCH_K, prompt_version: str = config.PROMPT_VERSION,
                 generate: Callable[[Prompt], llm_client.GenerationResult] | None = None,
                 query_rewrite: str = config.QUERY_REWRITE, index_dir: Path | str | None = None):
        if not 1 <= top_k <= paths.EVAL_DEPTH:
            raise ValueError(f"top_k must be 1-{paths.EVAL_DEPTH} (the eval depth), got {top_k}")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"MMR lambda must be in [0, 1], got {mmr_lambda}")
        if fetch_k > paths.EVAL_DEPTH:
            raise ValueError(f"fetch_k ({fetch_k}) cannot exceed the eval depth ({paths.EVAL_DEPTH})")
        system_rules(prompt_version)                      # unknown version -> ValueError now, not mid-run
        if query_rewrite not in config.QUERY_REWRITE_MODES:
            raise ValueError(f"unknown query rewrite {query_rewrite!r}; known: {list(config.QUERY_REWRITE_MODES)}")
        if query_rewrite != "none":
            check_available()                              # missing parser -> refuse now, not 80 silent fallbacks
        self.query_rewrite = query_rewrite
        self.index_dir = Path(index_dir) if index_dir is not None else config.CHROMA_DIR
        self._retriever = retriever
        self.top_k, self.mmr_lambda, self.fetch_k = top_k, mmr_lambda, fetch_k
        self.prompt_version = prompt_version
        self._generate = generate or llm_client.generate

    @property
    def retriever(self):
        if self._retriever is None:                       # lazy: --pin-contexts never needs the index
            from hypeonoath.core.store import get_client
            from hypeonoath.serving.retriever import Retriever
            if not (self.index_dir / "chroma.sqlite3").is_file():   # get_client() would create an empty index
                raise ValueError(f"no index in {self.index_dir}; "
                                 "build it first: hypeonoath-index --index-dir <folder> [flags]")
            try:
                collection = get_client(self.index_dir).get_collection(config.COLLECTION_NAME)
            except Exception as exc:  # noqa: BLE001 -- chromadb raises NotFoundError / ValueError per version
                raise ValueError(f"no collection {config.COLLECTION_NAME!r} in {self.index_dir} ({exc})") from None
            self._retriever = Retriever(collection)
        return self._retriever

    def retrieve(self, question: str) -> Retrieved:
        from hypeonoath.serving.retriever import mmr

        query = rewrite_query(question, self.query_rewrite)      # only the retrieval query changes (D94)
        if self.mmr_lambda is None:
            ranking = self.retriever.search(query, paths.EVAL_DEPTH)
        else:
            hits, vectors = self.retriever.search_with_embeddings(query, paths.EVAL_DEPTH)
            head = mmr(hits[:self.fetch_k], vectors[:self.fetch_k], self.fetch_k, self.mmr_lambda)
            ranking = head + hits[self.fetch_k:]
        return Retrieved(ranking, self.retriever.assemble(ranking[:self.top_k]),
                         query if query != question else None)

    def generate(self, question: str, context: list[RetrievedItem]) -> Generated:
        prompt = build_prompt(question, context, self.prompt_version)
        result = self._generate(prompt)
        return Generated(result.text, dict(result.stats), prompt.as_text())

    def index_ids(self) -> list[str]:
        return self.retriever.collection.get(include=[])["ids"]

    def index_settings(self) -> dict[str, bool]:
        """The F-R3 chunking flags the index was built with (collection metadata; absent = off, D95)."""
        meta = getattr(getattr(self.retriever, "collection", None), "metadata", None) or {}
        return {key: bool(meta.get(key, False)) for key in ("doc_description_in_chunks", "small_sections_merged")}

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
            "query_rewrite": self.query_rewrite,                                  # F-R4 (D94)
            "index_dir": index_label(self.index_dir),                             # F-R3 (D95)
            **self.index_settings(),                                              # D88 / D95: from the index itself
        }


ANSWERERS: dict[str, type] = {"baseline": BaselineAnswerer}


def make_answerer(name: str, **kwargs) -> Answerer:
    try:
        cls = ANSWERERS[name]
    except KeyError:
        raise ValueError(f"unknown answerer {name!r}; known: {sorted(ANSWERERS)}") from None
    return cls(**kwargs)
