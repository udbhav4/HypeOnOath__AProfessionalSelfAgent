"""
Detection: Presidio (+ spaCy NER, English + German) and detect-secrets ->
in-memory Candidates (privacy plan v4, 5.2, 6.2, 6.3, 6.3.1; P-D1 - P-D5,
P-D25, P-D26).

  * build_analyzer()  Presidio AnalyzerEngine from privacy/conf/nlp.yaml +
                      recognizers.yaml (every local recognizer, en + de),
                      cached -- loading the spaCy models is the slow part.
  * route_languages() which language(s) a text is analysed in (P-D25):
                      its own language when clear (spaCy stop-word counts),
                      else its document's, else both. Pattern recognizers are
                      registered for both languages, so they always run.
  * detect_text()     one text -> Candidates (union over the routed languages;
                      NER results dropped for code, P-D26; secrets).
  * detect_blocks()   every loaded Block (documents, notebook docs, code
                      blocks) + project name/description strings -- exactly
                      the text that gets chunked and embedded.

A Candidate holds the matched VALUE and its line, in memory only, for the
owner's review screen. It is never serialised, logged or put in an exception
message (a static test greps for that); logs and reports use the value-less
secret_scan.Finding or plain counts.
"""
from __future__ import annotations

import inspect
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.records import Block

if TYPE_CHECKING:  # heavy / indexing-only imports, for type hints only
    from presidio_analyzer import AnalyzerEngine

    from hypeonoath.indexing.block_loader import LoadedDocument
    from hypeonoath.indexing.project_registry import ProjectInfo

logger = logging.getLogger(__name__)

CONF_DIR = Path(__file__).resolve().parent / "conf"
NLP_CONF_PATH = CONF_DIR / "nlp.yaml"
RECOGNIZERS_CONF_PATH = CONF_DIR / "recognizers.yaml"
SECRET_PREFIX = "SECRET:"


@dataclass(frozen=True)
class Candidate:
    """One possible personal value / secret. IN MEMORY ONLY -- never serialised."""

    file: str                    # posix, relative to project root
    line: int | None             # source line (best effort; None when unknown)
    entity_type: str             # "PHONE_NUMBER", "DE_TAX_ID", "SECRET:AWS Access Key", ...
    recognizer: str              # Presidio recognizer name or "detect-secrets"
    score: float                 # Presidio score; 1.0 for detect-secrets
    value: str                   # exact matched text
    line_text: str               # the full line holding the value (display only)
    start: int                   # offset of value inside line_text (for highlighting)
    before: str = ""             # previous line (display context, plan 6.5)
    after: str = ""              # next line

    def __repr__(self) -> str:   # a stray repr()/f-string can never show the value
        return f"Candidate({self.file}:{self.line}, {self.entity_type}, score={self.score:.2f})"


# --- Analyzer ------------------------------------------------------------------------

def _accepts_name(cls: type) -> bool:
    params = inspect.signature(cls.__init__).parameters
    return "name" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _direct_recognizers(entries: list[dict[str, Any]], languages: tuple[str, ...], flags: int) -> list[Any]:
    """
    Instantiate predefined recognizers whose __init__ takes no `name`.

    Presidio 2.2.364's YAML loader always passes `name=`; a few recognizers
    that Presidio ships DISABLED (so the path was never exercised upstream:
    UsMbiRecognizer, KrBrnRecognizer, KrDriverLicenseRecognizer) reject it.
    They are built here with their own defaults + any context from the YAML.
    """
    from presidio_analyzer import PatternRecognizer
    from presidio_analyzer.recognizer_registry.recognizers_loader_utils import RecognizerListLoader

    recognizers = []
    for entry in entries:
        cls = RecognizerListLoader.get_existing_recognizer_cls(entry.get("class_name", entry["name"]))
        for language_conf in RecognizerListLoader._get_recognizer_languages(entry, languages):
            kwargs = {"supported_language": language_conf["supported_language"]}
            if language_conf.get("context"):
                kwargs["context"] = language_conf["context"]
            recognizer = cls(**kwargs)
            if isinstance(recognizer, PatternRecognizer):
                recognizer.global_regex_flags = flags
            recognizers.append(recognizer)
    return recognizers


@lru_cache(maxsize=1)
def build_analyzer() -> "AnalyzerEngine":
    """Presidio AnalyzerEngine with our NLP + recognizer config (cached per process)."""
    import yaml
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider
    from presidio_analyzer.recognizer_registry.recognizers_loader_utils import RecognizerListLoader

    languages = tuple(config.PRIVACY_LANGUAGES)
    registry_conf = yaml.safe_load(RECOGNIZERS_CONF_PATH.read_text(encoding="utf-8"))
    registry_conf["supported_languages"] = list(languages)
    direct, loadable = [], []
    for entry in registry_conf["recognizers"]:
        is_predefined = entry.get("type", "predefined") == "predefined"
        if is_predefined and not _accepts_name(
                RecognizerListLoader.get_existing_recognizer_cls(entry.get("class_name", entry["name"]))):
            direct.append(entry)
        else:
            loadable.append(entry)
    registry_conf["recognizers"] = loadable

    nlp_engine = NlpEngineProvider(conf_file=str(NLP_CONF_PATH)).create_engine()
    registry = RecognizerRegistryProvider(registry_configuration=registry_conf).create_recognizer_registry()
    for recognizer in _direct_recognizers(direct, languages, registry_conf.get("global_regex_flags", 26)):
        registry.add_recognizer(recognizer)
    # The spaCy NER recognizer (PERSON, LOCATION, NRP, DATE_TIME, MONEY) is only
    # added automatically when AnalyzerEngine builds its own registry.
    nlp_engine.load()
    registry.add_nlp_recognizer(nlp_engine=nlp_engine)
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=list(languages))
    logger.info("Privacy analyzer ready: %d recognizers, languages %s", len(registry.recognizers), languages)
    return analyzer


# --- Detection -----------------------------------------------------------------------

@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    entity_type: str
    score: float
    recognizer: str


def _overlaps(a: _Hit, b: _Hit) -> bool:
    return a.start < b.end and b.start < a.end


def _dedupe_hits(hits: Iterable[_Hit]) -> list[_Hit]:
    """
    Union over languages / recognizers: overlapping spans of the SAME type
    keep only the best (highest score, then longest); different types are
    all kept -- the owner sees each and decides once per value.
    """
    kept: list[_Hit] = []
    for hit in sorted(hits, key=lambda h: (-h.score, -(h.end - h.start), h.start)):
        if not any(k.entity_type == hit.entity_type and _overlaps(k, hit) for k in kept):
            kept.append(hit)
    return sorted(kept, key=lambda h: (h.start, h.end, h.entity_type))


def _presidio_hits(text: str, analyzer: Any, languages: Iterable[str], is_code: bool = False,
                   stats: Counter | None = None) -> list[_Hit]:
    """Presidio results in each routed language; NER (P-D26) and URL (P-D34) results dropped on code."""
    hits: list[_Hit] = []
    for language in languages:
        for result in analyzer.analyze(text=text, language=language, score_threshold=config.PRIVACY_MIN_SCORE):
            recognizer = str((result.recognition_metadata or {}).get("recognizer_name", "presidio"))
            if is_code and recognizer in config.PRIVACY_NER_RECOGNIZERS:
                if stats is not None:
                    stats["ner_dropped_on_code"] += 1
                continue
            if is_code and result.entity_type in config.PRIVACY_CODE_DROPPED_ENTITIES:
                if stats is not None:
                    stats["url_dropped_on_code"] += 1
                continue
            hits.append(_Hit(result.start, result.end, result.entity_type, float(result.score), recognizer))
    return hits


# --- Language routing (P-D25, plan 6.3.1) ------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+")
_GERMAN_CHARS = frozenset("äöüß")


@lru_cache(maxsize=1)
def _stop_words() -> dict[str, frozenset[str]]:
    """
    Per-language stop words minus the words both lists share (they carry no
    signal). Importing spacy.lang.* loads word lists only, never a model.
    """
    from spacy.lang.de.stop_words import STOP_WORDS as DE
    from spacy.lang.en.stop_words import STOP_WORDS as EN

    shared = DE & EN
    return {"en": frozenset(EN - shared), "de": frozenset(DE - shared)}


def text_language(text: str) -> str | None:
    """'en' / 'de' when the text clearly is that language, else None (short, mixed or unclear)."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < config.PRIVACY_LANG_MIN_WORDS:
        return None
    stop = _stop_words()
    hits = {lang: sum(w in stop[lang] for w in words) for lang in ("en", "de")}
    hits["de"] += sum(1 for w in words if _GERMAN_CHARS & set(w))      # umlaut / ß tie-breaker
    for lang, other in (("de", "en"), ("en", "de")):
        if (hits[lang] >= config.PRIVACY_LANG_MIN_HITS
                and hits[lang] >= config.PRIVACY_LANG_MARGIN * max(1, hits[other])):
            return lang
    return None


def document_language(doc: "LoadedDocument") -> str | None:
    """Language of a document's non-code text (the fallback for its short blocks)."""
    return text_language("\n".join(b.text for b in doc.blocks
                                   if b.block_type not in config.PRIVACY_CODE_BLOCK_TYPES))


def route_languages(text: str, doc_language: str | None = None) -> tuple[str, ...]:
    """The language(s) to analyse `text` in: its own, else its document's, else all (v3 recall)."""
    languages = tuple(config.PRIVACY_LANGUAGES)
    if not config.PRIVACY_LANGUAGE_ROUTING:
        return languages
    lang = text_language(text) or doc_language
    return (lang,) if lang in languages else languages


def _secret_hits(text: str, file: str) -> list[_Hit]:
    """detect-secrets hits as spans; an unlocatable value -> the whole line (plan 6.3)."""
    from hypeonoath.indexing.secret_scan import _secret_matches

    lines = text.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    hits: list[_Hit] = []
    for line_number, secret_type, value in _secret_matches(lines, file):
        line = lines[line_number - 1]
        offset = line.find(value) if value else -1
        if offset >= 0:
            start, end = line_starts[line_number - 1] + offset, line_starts[line_number - 1] + offset + len(value)
        else:
            stripped = line.strip()
            start = line_starts[line_number - 1] + line.find(stripped)
            end = start + len(stripped)
        if end > start:
            hits.append(_Hit(start, end, SECRET_PREFIX + secret_type, 1.0, "detect-secrets"))
    return hits


def _to_candidates(text: str, hits: list[_Hit], file: str, first_line: int | None) -> list[Candidate]:
    lines = text.split("\n")
    candidates: list[Candidate] = []
    for hit in hits:
        value = text[hit.start:hit.end]
        if not value.strip():
            continue
        index = text.count("\n", 0, hit.start)               # 0-based line inside the text
        line_offset = hit.start - (text.rfind("\n", 0, hit.start) + 1)
        candidates.append(Candidate(
            file=file,
            line=first_line + index if first_line is not None else None,
            entity_type=hit.entity_type,
            recognizer=hit.recognizer,
            score=hit.score,
            value=value,
            line_text=lines[index],
            start=line_offset,
            before=lines[index - 1] if index > 0 else "",
            after=lines[index + 1] if index + 1 < len(lines) else "",
        ))
    return candidates


def detect_text(text: str, file: str, first_line: int | None, *, analyzer: Any = None,
                languages: Iterable[str] | None = None, is_code: bool = False,
                stats: Counter | None = None) -> list[Candidate]:
    """
    All Presidio + detect-secrets candidates in one text (positions mapped to
    lines). `languages`: from route_languages(); None = every PRIVACY_LANGUAGES.
    `is_code`: drop NER results (P-D26); patterns and secrets are kept.
    `stats`: optional counter for the routing log (counts only).
    """
    if not text or not text.strip():
        return []
    analyzer = analyzer or build_analyzer()
    languages = tuple(config.PRIVACY_LANGUAGES) if languages is None else tuple(languages)
    hits = _dedupe_hits(_presidio_hits(text, analyzer, languages, is_code, stats) + _secret_hits(text, file))
    return _to_candidates(text, hits, file, first_line)


@dataclass(frozen=True)
class BlockText:
    """One text to detect on, and how it is routed (plan 6.3.1)."""

    text: str
    source: str
    first_line: int | None
    is_code: bool
    languages: tuple[str, ...]


def _block_texts(documents: list["LoadedDocument"], code_blocks: list[Block]) -> Iterable[BlockText]:
    code_languages = tuple(config.PRIVACY_CODE_LANGUAGES)
    for doc in documents:
        doc_language = document_language(doc) if config.PRIVACY_LANGUAGE_ROUTING else None
        for block in doc.blocks:
            is_code = block.block_type in config.PRIVACY_CODE_BLOCK_TYPES
            languages = code_languages if is_code else route_languages(block.text, doc_language)
            yield BlockText(block.text, block.source_document, block.meta.get("line_start"), is_code, languages)
    for block in code_blocks:
        # Line 0 of a code chunk is its header; code line i sits at line_start + i - 1.
        line_start = block.meta.get("line_start")
        yield BlockText(block.text, block.source_document, line_start - 1 if line_start is not None else None,
                        True, code_languages)


def project_strings_file() -> str:
    """Where project name/description strings come from (shown in review)."""
    try:
        return config.CODE_METADATA_PATH.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return config.CODE_METADATA_PATH.as_posix()


def detect_blocks(documents: list["LoadedDocument"], code_blocks: list[Block],
                  projects: dict[str, "ProjectInfo"], *, analyzer: Any = None) -> list[Candidate]:
    """
    Detect on every Block's text (documents incl. notebook docs, code blocks)
    and every distinct project name/description. Identical (file, line, entity_type, value) candidates are merged.
    """
    analyzer = analyzer or build_analyzer()
    found: list[Candidate] = []
    stats: Counter = Counter()            # NER results dropped on code
    blocks: Counter = Counter()           # texts per route: en / de / both / code
    for item in _block_texts(documents, code_blocks):
        blocks["code" if item.is_code else item.languages[0] if len(item.languages) == 1 else "both"] += 1
        found += detect_text(item.text, item.source, item.first_line, analyzer=analyzer,
                             languages=item.languages, is_code=item.is_code, stats=stats)

    metadata_file = project_strings_file()
    strings = sorted({s for info in projects.values() for s in (info.name, info.description) if s})
    for text in strings:                       # prose: routed, NER kept
        found += detect_text(text, metadata_file, None, analyzer=analyzer, languages=route_languages(text))
    log_json_line(config.PRIVACY_LOG_PATH, event="detect_routing", blocks=dict(blocks),
                  ner_dropped_on_code=stats["ner_dropped_on_code"],
                  url_dropped_on_code=stats["url_dropped_on_code"],
                  routing_enabled=config.PRIVACY_LANGUAGE_ROUTING)

    unique: dict[tuple[str, int | None, str, str], Candidate] = {}
    for candidate in found:
        key = (candidate.file, candidate.line, candidate.entity_type, candidate.value)
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    return list(unique.values())
