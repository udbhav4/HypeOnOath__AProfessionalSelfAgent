"""
Keyword (lexical) ranking for retrieval experiment F-R5 (eval D109; plan fr5-hybrid-search-code-plan-v2).

  tokenize(text)        -> lower-cased words, numbers (decimals kept) and code identifiers
                           (whole identifier + its camelCase / snake_case parts); stop words dropped
  BM25Index.build(...)  -> Okapi BM25 over the chunk texts of one index (rank_bm25.BM25Okapi, OD-2)
  BM25Index.top(q, n)   -> [(chunk_id, score)] best first, score > 0 only, deterministic ties

`rank_bm25` is an eval-only dependency (`[eval]` extra, D98) and is imported lazily inside
BM25Index.build / check_available, so importing this module never loads it (serving import ban).
Texts and tokens are never logged: the index holds redacted text, but nothing here needs to print it.
"""
from __future__ import annotations

import re

from hypeonoath.core import config

# Words or identifiers (letters first, then letters / digits / underscores), or numbers with an optional
# decimal part ("0.5" stays one token: sampling values and limits are the literals F-R5 targets).
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
# camelCase boundaries: "runReact" -> run|React, "HTTPServer" -> HTTP|Server, "v2Model" -> v2|Model.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Short on purpose (OD-6): IDF already down-weights words common in the corpus; a long list would drop
# meaningful words. Second-person and other pronouns are here; the owner's pronouns come from config.
STOPWORDS: frozenset[str] = frozenset("""
a an the and or but nor so yet if then than as of at by for from in into on onto to with without
over under about after before between through during per via
is am are was were be been being do does did done have has had having can could will would shall
should may might must
what which who whom whose when where why how
this that these those there here
i me my mine myself we us our ours ourselves you your yours yourself yourselves
it its itself they them their theirs themselves
not no
""".split())


def _stopwords() -> frozenset[str]:
    """STOPWORDS + the owner's pronouns, read at call time (D94 amended: never hard-coded)."""
    return STOPWORDS | {p.lower() for p in config.OWNER_PRONOUNS}


def tokenize(text: str) -> list[str]:
    """
    Tokens for BM25, in text order, duplicates kept (term frequency matters).
    An identifier with an underscore or an inner capital yields the whole identifier lower-cased and then
    each part: "runReactHypothesisLoop" -> runreacthypothesisloop, run, react, hypothesis, loop.
    Stop words and one-letter words are dropped; digits are kept ("0", "5"). Non-ASCII letters are skipped.
    """
    stop = _stopwords()
    out: list[str] = []
    for raw in _TOKEN.findall(text or ""):
        if raw[0].isdigit():
            out.append(raw)
            continue
        parts = [p for piece in raw.split("_") for p in _CAMEL.split(piece) if p]
        words = [raw.lower()] + ([p.lower() for p in parts] if len(parts) > 1 else [])
        out.extend(w for w in words if len(w) > 1 and w not in stop)
    return out


def check_available() -> None:
    """ValueError when rank_bm25 is missing, so a hybrid run refuses before it starts (as F-R4's check)."""
    try:
        import rank_bm25  # noqa: F401
    except ImportError as exc:
        raise ValueError(f"--hybrid rrf needs {config.BM25_PACKAGE} ({type(exc).__name__}: {exc}); "
                         "install: pip install -e \".[eval]\"") from None


class BM25Index:
    """Okapi BM25 over one index's chunk texts; the only place that touches rank_bm25 (swap = one class)."""

    def __init__(self, ids: list[str], bm25) -> None:
        self.ids = ids
        self._bm25 = bm25

    @classmethod
    def build(cls, ids: list[str], texts: list[str], k1: float = config.BM25_K1, b: float = config.BM25_B,
              epsilon: float = config.BM25_EPSILON) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError(f"BM25 needs one text per id ({len(ids)} ids, {len(texts)} texts)")
        if not ids:
            raise ValueError("index is empty: nothing to search")
        check_available()
        from rank_bm25 import BM25Okapi

        # rank_bm25 divides by the mean document length: an index whose texts all tokenize to nothing
        # would divide by zero. Single empty documents are fine (they score 0 for every query).
        corpus = [tokenize(t) for t in texts]
        if not any(corpus):
            raise ValueError("no searchable words in the index: every chunk tokenized to nothing")
        return cls(list(ids), BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon))

    def top(self, query_tokens: list[str], n: int) -> list[tuple[str, float]]:
        """Best n (chunk_id, score) with score > 0; ties by chunk id. A repeated query token counts once."""
        terms = list(dict.fromkeys(query_tokens))
        if not terms or n <= 0:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(((cid, float(s)) for cid, s in zip(self.ids, scores) if s > 0),
                        key=lambda pair: (-pair[1], pair[0]))
        return ranked[:n]
