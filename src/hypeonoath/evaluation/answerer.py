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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.evaluation import paths
from hypeonoath.serving import lexical, llm_client, reranker as reranker_mod
from hypeonoath.serving.query_rewrite import check_available, rewrite_query
from hypeonoath.serving.prompt import _PART_REMINDER, Prompt, build_prompt, system_rules


@dataclass
class Retrieved:
    ranking: list[RetrievedItem]          # raw ranked hits (after MMR if on), best first, <= EVAL_DEPTH
    context: list[RetrievedItem]          # exactly what goes into the prompt
    query: str | None = None              # the retrieval query when it differs from the question (F-R4, D94)
    rerank_s: float | None = None         # F-R6: wall time of the cross-encoder call (None = re-ranking off)
    rerank_truncated: int | None = None   # F-R6: inputs longer than max_length (cut before scoring, FR6-6)


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
    [optional third-person query rewrite, F-R4] -> Retriever.search(k=20) [or hybrid_search: dense + BM25
    fused by RRF, F-R5] -> optional MMR [or F-R6: cross-encoder re-ranking of a candidate pool, before or
    after MMR] -> assemble(top_k) -> build_prompt(version) -> Ollama. The index is read from `index_dir`
    (default: the production index/chroma; F-R3 variants live in their own folders, D95).
    """

    name = "baseline"

    def __init__(self, retriever=None, top_k: int = config.DEFAULT_TOP_K, mmr_lambda: float | None = config.MMR_LAMBDA,
                 fetch_k: int = config.MMR_FETCH_K, prompt_version: str = config.PROMPT_VERSION,
                 generate: Callable[[Prompt], llm_client.GenerationResult] | None = None,
                 query_rewrite: str = config.QUERY_REWRITE, index_dir: Path | str | None = None,
                 hybrid: str = config.HYBRID, rerank: str = config.RERANK,
                 rerank_candidates: int = config.RERANK_CANDIDATES, rerank_sources: str = config.RERANK_SOURCES,
                 reranker=None):
        if not 1 <= top_k <= paths.EVAL_DEPTH:
            raise ValueError(f"top_k must be 1-{paths.EVAL_DEPTH} (the eval depth), got {top_k}")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"MMR lambda must be in [0, 1], got {mmr_lambda}")
        if fetch_k < 1:          # > EVAL_DEPTH is allowed: MMR then picks EVAL_DEPTH out of fetch_k (D112)
            raise ValueError(f"fetch_k must be >= 1, got {fetch_k}")
        system_rules(prompt_version)                      # unknown version -> ValueError now, not mid-run
        if query_rewrite not in config.QUERY_REWRITE_MODES:
            raise ValueError(f"unknown query rewrite {query_rewrite!r}; known: {list(config.QUERY_REWRITE_MODES)}")
        if query_rewrite != "none":
            check_available()                              # missing parser -> refuse now, not 80 silent fallbacks
        if hybrid not in config.HYBRID_MODES:
            raise ValueError(f"unknown hybrid mode {hybrid!r}; known: {list(config.HYBRID_MODES)}")
        if hybrid != "none":
            lexical.check_available()                      # missing rank_bm25 -> refuse now (F-R5, D109)
        if rerank != "none":                               # F-R6: refuse a bad lever now, not mid-run
            reranker_mod.check_available(rerank)
            if hybrid != "none":
                raise ValueError("--rerank cannot be combined with --hybrid rrf: the cross-encoder orders the "
                                 "candidates itself; for BM25 candidates use --rerank-sources dense_bm25")
            if rerank_candidates < paths.EVAL_DEPTH:
                raise ValueError(f"rerank_candidates must be >= {paths.EVAL_DEPTH} (the eval depth), "
                                 f"got {rerank_candidates}")
            if rerank_sources not in config.RERANK_SOURCE_MODES:
                raise ValueError(f"unknown rerank sources {rerank_sources!r}; known: {list(config.RERANK_SOURCE_MODES)}")
            if rerank_sources == "dense_bm25":
                lexical.check_available()                  # the BM25 leg needs rank_bm25
        self.rerank, self.rerank_candidates = rerank, rerank_candidates
        self.rerank_sources = rerank_sources
        self._reranker = reranker
        self.query_rewrite = query_rewrite
        self.hybrid = hybrid
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

    @property
    def reranker(self) -> reranker_mod.Reranker:
        if self._reranker is None:                        # lazy: the model loads in prepare() / first use
            self._reranker = reranker_mod.Reranker(self.rerank)
        return self._reranker

    def prepare(self) -> dict[str, Any]:
        """
        Warm-up before the question loop (FR6-7), so question 1's retrieve_s carries no load time: one query
        embedding (loads the embedder), the BM25 index when a BM25 leg is used, and -- with re-ranking on --
        the cross-encoder scoring one dummy pair. Returns what the manifest records. A re-ranker that cannot be
        loaded raises ValueError -> the run refuses before it starts.
        """
        retriever = self.retriever
        started = time.perf_counter()
        embed = getattr(retriever, "embed_query", None)
        if embed is not None:
            embed("warm-up")
        if (self.hybrid != "none" or (self.rerank != "none" and self.rerank_sources == "dense_bm25")) \
                and hasattr(retriever, "_lexical"):
            retriever._lexical()                          # the full-collection BM25 build, once per run
        out: dict[str, Any] = {"retrieval_warmup_s": round(time.perf_counter() - started, 3)}
        if self.rerank != "none":
            import torch

            started = time.perf_counter()
            self.reranker.score("warm-up", ["warm-up"])
            out.update(rerank_load_s=round(time.perf_counter() - started, 3), activation=self.reranker.activation,
                       torch_threads=torch.get_num_threads())
        return out

    def _score(self, query: str, hits: list[RetrievedItem]) -> tuple[list[float], float, int]:
        """CE scores for the hits' stored texts, the predict time (s) and the count of truncated inputs."""
        texts = [h.chunk.text for h in hits]
        started = time.perf_counter()
        scores = self.reranker.score(query, texts)
        elapsed = round(time.perf_counter() - started, 4)
        return scores, elapsed, self.reranker.count_truncated(query, texts)

    def _retrieve_reranked(self, query: str) -> tuple[list[RetrievedItem], float, int]:
        """
        F-R6 ranking (<= EVAL_DEPTH): candidates (dense, or dense ∪ BM25) -> CE order -> optional MMR over
        the top fetch_k of that order with relevance = CE score rescaled to the pool's cosine range (FR6-3).
        The CE scores the whole fetched pool: max(EVAL_DEPTH, candidates, fetch_k when MMR is on).
        """
        from hypeonoath.serving.reranker import rerank, rescale_to_cosine
        from hypeonoath.serving.retriever import mmr

        depth = max(paths.EVAL_DEPTH, self.rerank_candidates, self.fetch_k if self.mmr_lambda is not None else 0)
        hits, vectors = self.retriever.candidates(query, depth, self.rerank_sources)
        scores, elapsed, truncated = self._score(query, hits)
        ordered = rerank(hits, scores)
        if self.mmr_lambda is None:
            return ordered[:paths.EVAL_DEPTH], elapsed, truncated
        vectors = [vectors[h.pre_rerank_rank - 1] for h in ordered]         # follow the CE order
        head_hits, head_vectors = ordered[:self.fetch_k], vectors[:self.fetch_k]
        relevance = rescale_to_cosine([h.rerank_score for h in head_hits], [h.score for h in head_hits])
        head = mmr(head_hits, head_vectors, min(self.fetch_k, paths.EVAL_DEPTH), self.mmr_lambda, relevance)
        return (head + ordered[self.fetch_k:])[:paths.EVAL_DEPTH], elapsed, truncated

    def retrieve(self, question: str) -> Retrieved:
        from hypeonoath.serving.retriever import mmr

        query = rewrite_query(question, self.query_rewrite)      # only the retrieval query changes (D94)
        if self.rerank != "none":                                 # F-R6: the CE sees this same query (FR6-4)
            ranking, rerank_s, truncated = self._retrieve_reranked(query)
            return Retrieved(ranking, self.retriever.assemble(ranking[:self.top_k]),
                             query if query != question else None, rerank_s, truncated)
        # D112: with MMR, search max(EVAL_DEPTH, fetch_k) hits; MMR re-orders the top fetch_k and picks at most
        # EVAL_DEPTH of them; hits after fetch_k keep their order. The ranking is always <= EVAL_DEPTH long, so
        # the metrics read the same depth whatever fetch_k is (fetch_k <= EVAL_DEPTH = exactly as before).
        depth = paths.EVAL_DEPTH if self.mmr_lambda is None else max(paths.EVAL_DEPTH, self.fetch_k)
        if self.hybrid == "rrf":                                  # F-R5 (D109): same query for both legs (OD-5)
            hits, vectors = self.retriever.hybrid_search(query, depth)
            relevance = None
            if self.mmr_lambda is not None:                       # D111: MMR relevance = fused / max fused
                top = max((h.fused or 0.0 for h in hits[:self.fetch_k]), default=0.0)
                relevance = [(h.fused or 0.0) / top if top else 0.0 for h in hits[:self.fetch_k]]
        elif self.mmr_lambda is None:
            hits, vectors, relevance = self.retriever.search(query, depth), [], None
        else:
            hits, vectors = self.retriever.search_with_embeddings(query, depth)
            relevance = None
        if self.mmr_lambda is None:
            ranking = hits[:paths.EVAL_DEPTH]
        else:
            head = mmr(hits[:self.fetch_k], vectors[:self.fetch_k], min(self.fetch_k, paths.EVAL_DEPTH),
                       self.mmr_lambda, relevance)
            ranking = (head + hits[self.fetch_k:])[:paths.EVAL_DEPTH]
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

    def hybrid_settings(self) -> dict[str, Any]:
        """The F-R5 levers as recorded in the manifest; parameters only when hybrid is on."""
        if self.hybrid == "none":
            return {"mode": "none"}
        return {"mode": self.hybrid, "candidates": config.HYBRID_CANDIDATES, "rrf_k": config.HYBRID_RRF_K,
                "bm25_weight": config.HYBRID_BM25_WEIGHT, "k1": config.BM25_K1, "b": config.BM25_B,
                "epsilon": config.BM25_EPSILON}

    def rerank_settings(self) -> dict[str, Any]:
        """The F-R6 levers as recorded in the manifest; parameters only when re-ranking is on."""
        if self.rerank == "none":
            return {"mode": "none"}
        model_id, revision = config.RERANK_MODELS[self.rerank]
        return {"mode": self.rerank, "model": model_id, "revision": revision, "candidates": self.rerank_candidates,
                "sources": self.rerank_sources, "max_length": config.RERANK_MAX_LENGTH,
                "batch_size": config.RERANK_BATCH_SIZE}

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
            "hybrid": self.hybrid_settings(),                                     # F-R5 (D109)
            "rerank": self.rerank_settings(),                                     # F-R6
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
