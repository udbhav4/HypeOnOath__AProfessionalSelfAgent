"""F-R5 lexical leg (eval D109): tokenizer, BM25 wrapper, lazy rank_bm25 import (plan fr5 v2, T1-T8, T30, T31)."""
from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

from hypeonoath.core import config
from hypeonoath.serving import lexical
from hypeonoath.serving.lexical import BM25Index, tokenize


# --- tokenize (T1-T4) ---------------------------------------------------------------------------

def test_identifiers_whole_and_parts():
    assert tokenize("runReactHypothesisLoop") == ["runreacthypothesisloop", "run", "react", "hypothesis", "loop"]
    assert tokenize("ACK_ONLY_WORDS") == ["ack_only_words", "ack", "only", "words"]
    assert tokenize("store_chat_message") == ["store_chat_message", "store", "chat", "message"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]
    assert tokenize("selectBeliefWindow(") == ["selectbeliefwindow", "select", "belief", "window"]


def test_plain_words_are_not_duplicated():
    assert tokenize("Glance replies") == ["glance", "replies"]


def test_numbers_and_decimals_kept():
    assert tokenize("n_predict: 400, temperature: 0.5,") == ["n_predict", "predict", "400", "temperature", "0.5"]
    assert tokenize("budget = 15; gate = 0.35;") == ["budget", "15", "gate", "0.35"]
    assert tokenize("7") == ["7"]                                      # a single digit is a number, kept


def test_stopwords_and_one_letter_words_dropped():
    assert tokenize("What is the temperature you use for a reply?") == ["temperature", "use", "reply"]
    assert "tell" in tokenize("Tell me about yourself")               # content verbs are kept


def test_owner_pronouns_come_from_config(monkeypatch):
    assert tokenize("he uses his model") == ["uses", "model"]
    monkeypatch.setattr(config, "OWNER_PRONOUNS", ("she", "her", "her", "herself"))
    assert tokenize("he uses her model") == ["he", "uses", "model"]   # read at call time, not hard-coded


def test_empty_punctuation_and_non_ascii():
    assert tokenize("") == [] and tokenize(None) == [] and tokenize("... --- !!!") == []
    assert tokenize("Grüße café") == ["gr", "caf"]                     # ASCII runs only; documented, no crash


# --- BM25 (T5-T8) -------------------------------------------------------------------------------

IDS = ["a", "b", "c", "d"]
TEXTS = ["temperature 0.5 glance reply model", "doctor report temperature 0.1 model",
         "career summary analytics model", "model model model model model model"]


def test_rare_term_ranks_its_document_first():
    idx = BM25Index.build(IDS, TEXTS)
    assert idx.top(["glance"], 5)[0][0] == "a"
    assert idx.top(["doctor", "report"], 5)[0][0] == "b"


def test_common_term_gets_the_epsilon_floor_not_a_negative_score():
    idx = BM25Index.build(IDS, TEXTS)                    # "model" is in all 4 documents: idf < 0 -> epsilon floor
    scored = dict(idx.top(["model"], 10))
    assert scored and all(s > 0 for s in scored.values())
    assert idx.top(["glance", "model"], 1)[0][0] == "a"   # the rare word decides, the common one barely moves it


def test_negative_mean_idf_floor_never_returns_non_positive_scores():
    # Tester gap: when most terms are in more than half the documents the mean idf is < 0, so the
    # library's epsilon floor is negative; such terms then subtract, and `top` must drop scores <= 0.
    idx = BM25Index.build(["a", "b", "c"], ["alpha beta rare", "alpha beta", "alpha beta"])
    assert idx.top(["alpha"], 5) == []                    # floor < 0: no positive score anywhere
    assert [cid for cid, _ in idx.top(["rare"], 5)] == ["a"]
    assert idx.top(["rare", "alpha", "beta"], 5) == []    # negative floors outweigh the rare term (library behaviour)


def test_parameters_reach_the_library(monkeypatch):
    seen = {}
    import rank_bm25

    class Spy(rank_bm25.BM25Okapi):
        def __init__(self, corpus, **kw):
            seen.update(kw)
            super().__init__(corpus, **kw)
    monkeypatch.setattr(rank_bm25, "BM25Okapi", Spy)
    BM25Index.build(IDS, TEXTS)
    assert seen == {"k1": config.BM25_K1, "b": config.BM25_B, "epsilon": config.BM25_EPSILON}


def test_longer_document_with_same_tf_scores_lower():
    idx = BM25Index.build(["short", "long", "x"], ["glance reply", "glance reply " + "filler words " * 20,
                                                    "other text here"])
    ranked = idx.top(["glance"], 3)
    assert [cid for cid, _ in ranked] == ["short", "long"]


def test_no_known_term_empty_query_and_repeated_tokens():
    idx = BM25Index.build(IDS, TEXTS)
    assert idx.top(["kubernetes"], 5) == [] and idx.top([], 5) == [] and idx.top(["glance"], 0) == []
    assert idx.top(["glance", "glance", "glance"], 5) == idx.top(["glance"], 5)


TIE_IDS = ["z", "y", "x", "w", "v"]
TIE_TEXTS = ["same words", "same words", "other text", "alpha beta", "gamma delta"]


def test_ties_by_id_and_deterministic():
    idx = BM25Index.build(TIE_IDS, TIE_TEXTS)
    assert [cid for cid, _ in idx.top(["same"], 5)] == ["y", "z"]                # equal scores -> id asc
    assert idx.top(["same", "other"], 5) == BM25Index.build(TIE_IDS, TIE_TEXTS).top(["same", "other"], 5)


def test_build_input_errors():
    with pytest.raises(ValueError, match="one text per id"):
        BM25Index.build(["a"], [])
    with pytest.raises(ValueError, match="empty"):
        BM25Index.build([], [])
    with pytest.raises(ValueError, match="no searchable words"):
        BM25Index.build(["a", "b"], ["the of", "!!!"])
    BM25Index.build(["a", "b"], ["glance", "!!!"])          # one empty document is fine


# --- packaging (T30, T31) -----------------------------------------------------------------------

def _hide_rank_bm25(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "rank_bm25" or name.startswith("rank_bm25."):
            raise ImportError("No module named 'rank_bm25'")
        return real_import(name, *args, **kwargs)
    monkeypatch.delitem(sys.modules, "rank_bm25", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_missing_package_refuses_with_install_hint(monkeypatch):
    _hide_rank_bm25(monkeypatch)
    with pytest.raises(ValueError, match=r"rank_bm25.*\.\[eval\]"):
        lexical.check_available()
    with pytest.raises(ValueError, match="rank_bm25"):
        BM25Index.build(["a"], ["glance"])


def test_importing_lexical_does_not_load_rank_bm25():
    probe = ("import sys; import hypeonoath.serving.lexical as l; l.tokenize('a b'); "
             "print('rank_bm25' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
