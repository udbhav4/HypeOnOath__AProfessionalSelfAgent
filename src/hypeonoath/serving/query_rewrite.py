"""
Retrieval-query rewrite (eval plan v29, D94; experiment F-R4, behind a flag that is off by default, D98).

third_person(question) rewrites a recruiter's second-person question into the third person before
retrieval -- "What are you proud of?" -> "What is he proud of?" -- so the query reads as a question
about a person's background instead of a message to an assistant (the "you" wording pulled LLM-prompt
code to the top). Never into the owner's name (name magnet, D94). Only the RETRIEVAL query changes:
the original question still goes into the prompt and the reply.

How (deterministic, no LLM):
  1. contractions with "you" are expanded first (you're -> you are, you've -> you have, ...);
  2. spaCy (`en_core_web_sm`) parses the question; each "you" becomes the subject pronoun when it is
     the subject of a verb (nsubj / nsubjpass), else the object pronoun (dobj, pobj, dative, the
     subject of a noun as in "makes you a good fit", ...); "your" / "yours" -> possessive,
     "yourself" -> reflexive when the clause's subject is "you", else the object pronoun ("Tell me
     about yourself" -> "... about him"). The pronouns come from config.OWNER_PRONOUNS, never hard-coded;
  3. the finite verb that agreed with "you" is conjugated to the third person singular with
     LemmInflect (are -> is, do -> does, have -> has, build -> builds, were -> was).

spaCy and LemmInflect are imported lazily (the serving install does not have them until F-R4 is kept)
and the model is loaded once. Any failure -- missing package, model, odd parse -- returns the
question unchanged (the eval then measures the original query for that question).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from hypeonoath.core import config

logger = logging.getLogger(__name__)

_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE)
# you're / you've / you'll / you'd (straight or curly apostrophe). "you'd" is read as "you would"
# ("you had" is rarer in questions); either way the modal / past form needs no conjugation.
_CONTRACTIONS = {"re": "are", "ve": "have", "ll": "will", "d": "would"}
_CONTRACTION_RE = re.compile(r"\b(you)['’](re|ve|ll|d)\b", re.IGNORECASE)
_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_PLURAL_ADDRESS = {"guys", "all", "folks"}
_VERB_POS = {"VERB", "AUX"}


def expand_contractions(text: str) -> str:
    """you're -> you are, you've -> you have, you'll -> you will, you'd -> you would (case kept on 'you')."""
    return _CONTRACTION_RE.sub(lambda m: f"{m.group(1)} {_CONTRACTIONS[m.group(2).lower()]}", text)


def check_available() -> None:
    """Load spaCy + the model + LemmInflect once; ValueError when missing, so a run refuses before it starts
    instead of silently measuring the unchanged questions under a third_person label."""
    try:
        import lemminflect  # noqa: F401

        _nlp()
    except (ImportError, OSError) as exc:
        raise ValueError(f"--query-rewrite third_person needs spaCy {config.QUERY_REWRITE_SPACY_MODEL} and "
                         f"LemmInflect ({type(exc).__name__}: {exc}); install: pip install -e \".[eval]\" "
                         "&& pip install -r requirements-eval-models.txt") from None


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    # Only the parser and tagger are needed; NER is the slowest component and is unused.
    return spacy.load(config.QUERY_REWRITE_SPACY_MODEL, exclude=["ner"])


def _match_case(word: str, like: str) -> str:
    if like.isupper() and len(like) > 1:
        return word.upper()
    return word[:1].upper() + word[1:] if like[:1].isupper() else word


def _third_person_verb(token) -> str | None:
    """The 3rd-person-singular form of a finite verb that agreed with 'you'; None = leave it as is."""
    from lemminflect import getInflection

    if token.tag_ == "VBP":                                       # are / do / have / build (present)
        forms = getInflection(token.lemma_.lower(), tag="VBZ")
    elif token.tag_ == "VBD" and token.lower_ == "were":          # the only past form that agrees
        return _match_case("was", token.text)
    else:
        return None                                               # modal, past, base form: unchanged
    return _match_case(forms[0], token.text) if forms else None


def _is_subject(token) -> bool:
    """'you' as the subject of a verb; 'you' coordinated with a subject counts as that subject."""
    dep, head = token.dep_, token.head
    if dep == "conj":
        dep, head = head.dep_, head.head
    return dep in _SUBJECT_DEPS and head.pos_ in _VERB_POS


def _clause_subject_is_you(token) -> bool:
    """
    A reflexive stays reflexive only when its clause's subject is 'you' ("Did you teach yourself
    Python?" -> "... he teach himself ..."). Without it -- an imperative such as "Tell me about
    yourself", whose implied subject is the listener -- the third-person form is the object pronoun
    ("... about him"; "about himself" would be wrong).
    """
    verb = token.head
    while verb.pos_ not in _VERB_POS and verb.head is not verb:
        verb = verb.head
    return any(c.dep_ in _SUBJECT_DEPS and c.lower_ == "you" for c in verb.children)


def _rewrite(question: str) -> str:
    subject, obj, possessive, reflexive = config.OWNER_PRONOUNS
    doc = _nlp()(expand_contractions(question))
    replace: dict[int, str] = {}
    for token in doc:
        word = token.lower_
        if word == "you":
            nxt = doc[token.i + 1].lower_ if token.i + 1 < len(doc) else ""
            prev = doc[token.i - 1].lower_ if token.i else ""
            if nxt in _PLURAL_ADDRESS or prev == "thank":
                continue                          # "you guys" / "you all" / "thank you": not the owner
            if _is_subject(token):
                replace[token.i] = subject
                plural = token.dep_ == "conj" or any(c.dep_ == "conj" for c in token.children)
                if plural:
                    continue                      # "you and your team do" -> "he and his team do"
                head = token.head
                # The finite verb: an auxiliary of the head ("Do you have", "Have you built") or the
                # head itself ("What are you proud of", "you know"); only one of them is finite.
                for verb in [*(c for c in head.children if c.dep_ in ("aux", "auxpass")), head]:
                    form = _third_person_verb(verb)
                    if form is not None:
                        replace[verb.i] = form
                        break
            else:
                replace[token.i] = obj
        elif word in ("your", "yours"):
            replace[token.i] = possessive
        elif word in ("yourself", "yourselves"):
            replace[token.i] = reflexive if _clause_subject_is_you(token) else obj
    out = []
    for token in doc:
        new = replace.get(token.i)
        # Pronouns: lower case unless at the start or the original was capitalised ("You" -> "He").
        text = token.text if new is None else (_match_case(new, token.text) if token.text[:1].isupper() else new)
        out.append(text + token.whitespace_)
    return "".join(out)


def third_person(question: str) -> str:
    """The retrieval query in the third person; the question unchanged when it has no 'you' or on any error."""
    if not _SECOND_PERSON_RE.search(question):
        return question                                           # no parse, no model load
    try:
        return _rewrite(question)
    except Exception as exc:  # noqa: BLE001 -- any parser / package / model problem: use the original (D94)
        logger.warning("query rewrite failed (%s: %s); using the original question", type(exc).__name__, exc)
        return question


def rewrite_query(question: str, mode: str | None = None) -> str:
    """The retrieval query for `mode` (None = config.QUERY_REWRITE at call time); unknown mode -> ValueError."""
    mode = config.QUERY_REWRITE if mode is None else mode
    if mode not in config.QUERY_REWRITE_MODES:
        raise ValueError(f"unknown query rewrite {mode!r}; known: {list(config.QUERY_REWRITE_MODES)}")
    return third_person(question) if mode == "third_person" else question
