"""
The shared private-value matcher (privacy plan v3, Sections 5.3, 6.4.1, 6.6).

One implementation, used in three places so "what was removed" and "what is
checked" can never disagree:
  * indexing -- redaction of blocks / disk files (hypeonoath.privacy.redactor)
    and post-chunk verification (hypeonoath.privacy.verify);
  * serving  -- the answer-time guard (hypeonoath.serving.guard);
  * eval     -- the private_leak metric and gold validation (later).

STDLIB ONLY (+ hypeonoath.core.config): the serving install has no pyyaml,
presidio or spacy (P-D6), which is also why the decisions file is JSON.
A boundary test enforces this.

Contents:
  ENTITY_KIND / DEFAULT_LABEL   entity type -> match kind / suggested label
  normalize / parse_date        per-kind normalisation (id, date, email, text)
  value_keys                    type-independent keys: "same value" = shared key
  load_private_values           decisions.json -> PrivateValue list (private only)
  find_private_spans / redact   matching on any text, positions in the original
  privacy_fingerprint           hash of the private decisions (index metadata)

Nothing here ever logs, prints or raises with a value in the message.
"""
from __future__ import annotations

import datetime
import difflib
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypeonoath.core import config

MATCH_KINDS = ("id", "date", "email", "text")

# --- Entity type -> match kind -> default label (plan 5.3) -------------------

_TEXT_TYPES = frozenset({"PERSON", "LOCATION", "NRP", "URL", "MANUAL", "PLACE_OF_BIRTH", "NATIONALITY"})
# Resolved per value: "date" when the value parses as a date, else "text".
_DATE_TYPES = frozenset({"DATE_TIME", "DATE_OF_BIRTH"})

ENTITY_KIND: dict[str, str] = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "id",
    "MONEY": "id",
    "IBAN_CODE": "id",
    "CREDIT_CARD": "id",
    "CRYPTO": "id",
    "IP_ADDRESS": "id",
    "MEDICAL_LICENSE": "id",
    "STUDENT_ID": "id",                 # keyword recognizers (privacy/conf/custom_recognizers.yaml)
    "APPLICANT_ID": "id",
    **{t: "text" for t in _TEXT_TYPES},
}

DEFAULT_LABEL: dict[str, str] = {
    "EMAIL_ADDRESS": "email address",
    "PHONE_NUMBER": "phone number",
    "DATE_TIME": "date",
    "MONEY": "salary / amount",
    "PERSON": "name",
    "LOCATION": "address / place",
    "NRP": "nationality",
    "URL": "link",
    "MANUAL": "private detail",
    "IBAN_CODE": "bank account number",
    "CREDIT_CARD": "credit card number",
    "CRYPTO": "crypto wallet address",
    "IP_ADDRESS": "IP address",
    "MEDICAL_LICENSE": "medical licence number",
    "US_SSN": "social security number",
    "US_PASSPORT": "passport number",
    "US_DRIVER_LICENSE": "driver's licence number",
    "US_ITIN": "taxpayer ID",
    "US_BANK_NUMBER": "bank account number",
    "UK_NHS": "NHS number",
    "UK_NINO": "national insurance number",
    "IN_AADHAAR": "Aadhaar number",
    "IN_PAN": "PAN number",
    "IN_PASSPORT": "passport number",
    "IN_VOTER": "voter ID",
    "IN_GSTIN": "GST number",
    "IN_VEHICLE_REGISTRATION": "vehicle registration",
    "DATE_OF_BIRTH": "date of birth",
    "PLACE_OF_BIRTH": "place of birth",
    "NATIONALITY": "nationality",
    "STUDENT_ID": "student ID",
    "APPLICANT_ID": "applicant number",
}

# Country-prefixed Presidio types (US_SSN, DE_TAX_ID, IN_PAN, ...) are ID numbers.
_COUNTRY_TYPE_RE = re.compile(r"^[A-Z]{2}_[A-Z0-9_]+$")
_FALLBACK_LABEL = "private detail"


def match_kind_for(entity_type: str, value: str) -> str:
    """Match kind for a finding (plan 5.3); unknown types: id if >= 4 digits, else text."""
    if entity_type in _DATE_TYPES:
        return "date" if parse_date(value) is not None else "text"
    if entity_type in ENTITY_KIND:
        return ENTITY_KIND[entity_type]
    if entity_type.startswith("SECRET:") or _COUNTRY_TYPE_RE.match(entity_type):
        return "id"
    return "id" if _digit_count(value) >= 4 else "text"


def default_label(entity_type: str) -> str:
    """Suggested placeholder label; the owner confirms or edits it in review."""
    if entity_type in DEFAULT_LABEL:
        return DEFAULT_LABEL[entity_type]
    if entity_type.startswith("SECRET:"):
        return "secret"
    if _COUNTRY_TYPE_RE.match(entity_type):
        # DE_TAX_ID -> "tax ID", DE_ID_CARD -> "ID card"
        words = entity_type.split("_")[1:]
        return " ".join("ID" if w == "ID" else w.lower() for w in words) or _FALLBACK_LABEL
    return entity_type.replace("_", " ").lower() if entity_type else _FALLBACK_LABEL


def _digit_count(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


# --- Dates ---------------------------------------------------------------------

_MONTHS: dict[str, int] = {}
for _number, _names in enumerate((
    ("january", "jan", "januar", "jänner", "jaenner"),
    ("february", "feb", "februar"),
    ("march", "mar", "märz", "maerz", "mär", "mrz"),
    ("april", "apr"),
    ("may", "mai"),
    ("june", "jun", "juni"),
    ("july", "jul", "juli"),
    ("august", "aug"),
    ("september", "sep", "sept"),
    ("october", "oct", "oktober", "okt"),
    ("november", "nov"),
    ("december", "dec", "dezember", "dez"),
), start=1):
    for _name in _names:
        _MONTHS[_name] = _number

# Longest names first so "march" wins over "mar" inside the alternation.
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))

# Each pattern: (regex, group order). Years are always 4 digits -- two-digit
# years are too ambiguous and are never parsed (plan 6.6).
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\d)(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)"), "dmy|mdy"),
    (re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)"), "ymd"),
    (re.compile(rf"(?<!\d)(\d{{1,2}})\.?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})(?!\d)", re.IGNORECASE), "dMy"),
    (re.compile(rf"(?<![^\W\d_])({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\.?,?\s+(\d{{4}})(?!\d)",
                re.IGNORECASE), "Mdy"),
)


def _make_date(year: int, month: int, day: int) -> datetime.date | None:
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def _dates_from_match(match: re.Match[str], order: str) -> list[datetime.date]:
    """Every valid reading of one match, preferred reading first (day-first)."""
    a, b, c = match.group(1), match.group(2), match.group(3)
    readings: list[datetime.date | None] = []
    for form in order.split("|"):
        if form == "dmy":
            readings.append(_make_date(int(c), int(b), int(a)))
        elif form == "mdy":
            readings.append(_make_date(int(c), int(a), int(b)))
        elif form == "ymd":
            readings.append(_make_date(int(a), int(b), int(c)))
        elif form == "dMy":
            readings.append(_make_date(int(c), _MONTHS[b.lower()], int(a)))
        elif form == "Mdy":
            readings.append(_make_date(int(c), _MONTHS[a.lower()], int(b)))
    out: list[datetime.date] = []
    for date in readings:
        if date is not None and date not in out:
            out.append(date)
    return out


def parse_date(text: str) -> datetime.date | None:
    """The whole of `text` as a date (DD/MM tried before MM/DD), else None."""
    stripped = text.strip()
    for pattern, order in _DATE_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match:
            dates = _dates_from_match(match, order)
            if dates:
                return dates[0]
    return None


def _find_dates(text: str) -> Iterable[tuple[int, int, list[datetime.date]]]:
    """Every date-looking substring of `text` with all of its valid readings."""
    for pattern, order in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            dates = _dates_from_match(match, order)
            if dates:
                yield match.start(), match.end(), dates


# --- Normalisation (plan 6.6) ----------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")
_UMLAUTS = str.maketrans({"ß": "ss", "ä": "ae", "ö": "oe", "ü": "ue"})
_WORD_ABBREVIATIONS = {"st.": "street", "str.": "strasse", "rd.": "road", "ave.": "avenue"}


def _strip_edges(word: str, keep_trailing_dot: bool = False) -> str:
    start, end = 0, len(word)
    while start < end and not word[start].isalnum():
        start += 1
    while end > start and not word[end - 1].isalnum() and not (keep_trailing_dot and word[end - 1] == "."):
        end -= 1
    return word[start:end]


def _normalize_word(word: str) -> str:
    word = _strip_edges(word, keep_trailing_dot=True)
    if word in _WORD_ABBREVIATIONS:
        word = _WORD_ABBREVIATIONS[word]
    elif word.endswith("str.") and len(word) > 4:      # German compound: "Lauteschlägerstr."
        word = word[:-4] + "strasse"
    return _strip_edges(word)


def normalize(value: str, match_kind: str) -> str:
    """Canonical form of a value for its match kind (plan 6.6 table)."""
    if match_kind == "id":
        return _NON_ALNUM_RE.sub("", value).upper()
    if match_kind == "date":
        date = parse_date(value)
        return date.isoformat() if date is not None else ""
    if match_kind == "email":
        return value.strip().lower()
    if match_kind == "text":
        text = unicodedata.normalize("NFKC", value).lower().translate(_UMLAUTS)
        words = (_normalize_word(w) for w in text.split())
        return " ".join(w for w in words if w)
    raise ValueError(f"Unknown match kind {match_kind!r}")


def value_keys(value: str) -> frozenset[str]:
    """
    Type-independent keys (plan 6.4.1). Two values are "the same value" when
    their key sets intersect -- exact string equality, never fuzzy, so a
    decision can't silently spread to a merely similar value.
    """
    keys: set[str] = set()
    text = normalize(value, "text")
    if text:
        keys.add("text:" + text)
    ident = normalize(value, "id")
    if _digit_count(ident) >= 4:        # short words never become ID keys
        keys.add("id:" + ident)
    date = parse_date(value)
    if date is not None:
        keys.add("date:" + date.isoformat())
    if EMAIL_RE.fullmatch(value.strip()):
        keys.add("email:" + normalize(value, "email"))
    return frozenset(keys)


# --- Private values ------------------------------------------------------------------

@dataclass(frozen=True)
class PrivateValue:
    """One match form of one private decision (a decision with aliases yields several)."""

    id: str
    label: str
    match_kind: str                 # id | date | email | text
    normalized: str
    surface_forms: tuple[str, ...]


@dataclass(frozen=True)
class Span:
    """A located private value. Positions refer to the original text; no value field."""

    start: int
    end: int
    value_id: str
    label: str
    how: str                        # "exact" | "partial_id" | "fuzzy" | "date" | "surface"

    def __len__(self) -> int:
        return self.end - self.start


def placeholder(label: str) -> str:
    return config.REDACTED_PLACEHOLDER.format(label=label)


def values_from_entries(entries: Iterable[dict[str, Any]]) -> list[PrivateValue]:
    """
    decisions.json entries -> PrivateValues: `private` entries only, one per
    distinct (match_kind, normalized) of the decision and of each alias, all
    sharing the decision's id, label and surface forms (plan 6.4.1).
    """
    values: list[PrivateValue] = []
    for entry in entries:
        if entry.get("decision") != "private":
            continue
        label = entry.get("label") or _FALLBACK_LABEL
        surface = tuple(dict.fromkeys(s for s in (entry.get("surface_forms") or [entry.get("value", "")]) if s))
        forms = [(entry.get("match_kind"), entry.get("normalized"))]
        forms += [(a.get("match_kind"), a.get("normalized")) for a in entry.get("aliases") or []]
        seen: set[tuple[str, str]] = set()
        for kind, normalized in forms:
            if kind not in MATCH_KINDS or not normalized or (kind, normalized) in seen:
                continue
            seen.add((kind, normalized))
            values.append(PrivateValue(entry["id"], label, kind, normalized, surface))
    return values


def read_decisions_file(path: Path | None = None) -> dict[str, Any]:
    """Raw decisions document; a missing file is an empty one. Errors never quote content."""
    path = config.PRIVACY_DECISIONS_PATH if path is None else path
    if not path.exists():
        return {"schema_version": 2, "entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Privacy decisions file {path} is unreadable ({type(exc).__name__})") from None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(f"Privacy decisions file {path} has no 'entries' list")
    return data


def load_private_values(path: Path | None = None) -> list[PrivateValue]:
    """Every private match form from decisions.json; a missing file -> []."""
    return values_from_entries(read_decisions_file(path)["entries"])


def privacy_fingerprint(values: Iterable[PrivateValue]) -> str:
    """
    sha256 over every private decision's id, label and sorted normalised forms
    (incl. aliases) -> 16 hex chars; "none" when nothing is private. Stored as
    `privacy_fp` in the Chroma collection metadata (plan 5.1, 7.3).
    """
    by_id: dict[str, tuple[str, set[str]]] = {}
    for value in values:
        _label, forms = by_id.setdefault(value.id, (value.label, set()))
        forms.add(value.normalized)
    if not by_id:
        return "none"
    lines = sorted(f"{vid}:{label}:{','.join(sorted(forms))}" for vid, (label, forms) in by_id.items())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


# --- Matching (plan 6.6) ---------------------------------------------------------------

_ID_RUN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9 ./\-]{2,}[A-Za-z0-9])?")
_WORD_RE = re.compile(r"\S+")
_MIN_SURFACE_CHARS = 3        # shorter literal forms would redact random substrings


def _surface_spans(text: str, value: PrivateValue) -> list[Span]:
    spans: list[Span] = []
    for form in value.surface_forms:
        if len(form.strip()) < _MIN_SURFACE_CHARS:
            continue
        left = r"(?<!\w)" if form[0].isalnum() else ""
        right = r"(?!\w)" if form[-1].isalnum() else ""
        for match in re.finditer(left + re.escape(form) + right, text, re.IGNORECASE):
            spans.append(Span(match.start(), match.end(), value.id, value.label, "surface"))
    return spans


def _alnum_map(text: str, start: int, end: int) -> tuple[str, list[int]]:
    """Uppercase alphanumerics of text[start:end] + each char's offset in `text`."""
    chars, offsets = [], []
    for i in range(start, end):
        if text[i].isascii() and text[i].isalnum():
            chars.append(text[i].upper())
            offsets.append(i)
    return "".join(chars), offsets


def _partial_pieces(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """
    A run holding > 1 separate digit group longer than 4 is split at
    whitespace for the partial check, so unrelated numbers aren't glued.
    """
    run = text[start:end]
    if len([g for g in re.findall(r"\d+", run) if len(g) > 4]) <= 1:
        return [(start, end)]
    return [(start + m.start(), start + m.end()) for m in re.finditer(r"\S+", run)]


def _aligned_occurrence(text: str, norm: str, offsets: list[int], target: str) -> tuple[int, int] | None:
    """
    First occurrence of `target` in a run's normalised form that starts and
    ends on token boundaries of the original text ("call +49 151 2345678 now"
    yes; "12345 67890" for a stored "123456789" no -- that would glue two
    unrelated numbers together).
    """
    position = norm.find(target)
    while position >= 0:
        start, end = offsets[position], offsets[position + len(target) - 1] + 1
        if (start == 0 or not text[start - 1].isalnum()) and (end == len(text) or not text[end].isalnum()):
            return start, end
        position = norm.find(target, position + 1)
    return None


def _id_spans(text: str, value: PrivateValue) -> list[Span]:
    target = value.normalized
    if not target:
        return []
    spans: list[Span] = []
    min_len = config.GUARD_PARTIAL_ID_MIN
    for run in _ID_RUN_RE.finditer(text):
        if _digit_count(run.group(0)) < 4:
            continue
        norm, offsets = _alnum_map(text, run.start(), run.end())
        exact = _aligned_occurrence(text, norm, offsets, target)
        if exact is not None:
            spans.append(Span(*exact, value.id, value.label, "exact"))
            continue
        if len(target) < min_len:
            continue
        for p_start, p_end in _partial_pieces(text, run.start(), run.end()):
            piece, piece_offsets = _alnum_map(text, p_start, p_end)
            match = difflib.SequenceMatcher(None, piece, target, autojunk=False).find_longest_match(
                0, len(piece), 0, len(target))
            # >= GUARD_PARTIAL_ID_MIN chars of the stored ID, at least 4 of them
            # digits -- a shared run of letters is not an ID leak.
            matched = piece[match.a:match.a + match.size]
            if match.size >= min_len and _digit_count(matched) >= 4:
                start, end = piece_offsets[match.a], piece_offsets[match.a + match.size - 1] + 1
                # Widen to whole tokens: "0[private]" would still hint at the number.
                while start > p_start and text[start - 1].isalnum():
                    start -= 1
                while end < p_end and text[end].isalnum():
                    end += 1
                spans.append(Span(start, end, value.id, value.label, "partial_id"))
    return spans


def _date_spans(text: str, value: PrivateValue) -> list[Span]:
    return [Span(s, e, value.id, value.label, "date")
            for s, e, dates in _find_dates(text) if any(d.isoformat() == value.normalized for d in dates)]


def _email_spans(text: str, value: PrivateValue) -> list[Span]:
    return [Span(m.start(), m.end(), value.id, value.label, "exact")
            for m in EMAIL_RE.finditer(text) if m.group(0).lower() == value.normalized]


def _trim(text: str, start: int, end: int, normalized: str) -> tuple[int, int]:
    """Drop edge punctuation from a window when it doesn't change the normalised form."""
    s, e = start, end
    while s < e and not text[s].isalnum():
        s += 1
    while e > s and not text[e - 1].isalnum():
        e -= 1
    return (s, e) if s < e and normalize(text[s:e], "text") == normalized else (start, end)


class _Words:
    """Word tokens of one text, each normalised once (shared by every text value)."""

    def __init__(self, text: str):
        self.matches = list(_WORD_RE.finditer(text))
        self.normalized = [normalize(m.group(0), "text") for m in self.matches]

    def window(self, i: int, size: int) -> str:
        return " ".join(w for w in self.normalized[i:i + size] if w)


def _text_spans(text: str, value: PrivateValue, words: _Words | None = None) -> list[Span]:
    target = value.normalized
    if not target:
        return []
    words = words or _Words(text)
    n = len(target.split())
    fuzzy_ok = len(target) >= config.GUARD_FUZZY_MIN_CHARS
    exact: list[Span] = []
    fuzzy: list[tuple[float, Span]] = []
    for size in range(max(1, n - 1), n + 2):
        for i in range(0, len(words.matches) - size + 1):
            start, end = words.matches[i].start(), words.matches[i + size - 1].end()
            window = words.window(i, size)
            if not window:
                continue
            if window == target:
                s, e = _trim(text, start, end, target)
                exact.append(Span(s, e, value.id, value.label, "exact"))
            elif fuzzy_ok:
                matcher = difflib.SequenceMatcher(None, window, target, autojunk=False)
                # real_quick_ratio / quick_ratio are cheap upper bounds of ratio().
                if (matcher.real_quick_ratio() >= config.GUARD_FUZZY_RATIO
                        and matcher.quick_ratio() >= config.GUARD_FUZZY_RATIO):
                    ratio = matcher.ratio()
                    if ratio >= config.GUARD_FUZZY_RATIO:
                        fuzzy.append((ratio, Span(start, end, value.id, value.label, "fuzzy")))
    # Neighbouring windows (n-1 .. n+1 words) overlap: keep exact hits, then
    # the best-scoring fuzzy window per region, so the span is not widened by
    # an extra word that merely still clears the ratio.
    chosen = list(exact)
    for _ratio, span in sorted(fuzzy, key=lambda rs: (-rs[0], len(rs[1]))):
        if not any(span.start < c.end and c.start < span.end for c in chosen):
            chosen.append(span)
    return chosen


_KIND_MATCHERS: dict[str, Callable[[str, PrivateValue], list[Span]]] = {
    "id": _id_spans, "date": _date_spans, "email": _email_spans,
}


def _merge(spans: list[Span]) -> list[Span]:
    """Union of overlapping ranges; id/label/how come from the longest member span."""
    merged: list[Span] = []
    group: list[Span] = []
    end = -1
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if group and span.start < end:
            group.append(span)
            end = max(end, span.end)
            continue
        if group:
            merged.append(_collapse(group, end))
        group, end = [span], span.end
    if group:
        merged.append(_collapse(group, end))
    return merged


def _collapse(group: list[Span], end: int) -> Span:
    best = max(group, key=len)
    return Span(group[0].start, end, best.value_id, best.label, best.how)


def find_private_spans(text: str, values: list[PrivateValue]) -> list[Span]:
    """Every occurrence of every private value in `text`: merged, non-overlapping, sorted."""
    if not text or not values:
        return []
    spans: list[Span] = []
    words: _Words | None = None
    for value in values:
        spans += _surface_spans(text, value)          # cheapest + most precise, first
        if value.match_kind == "text":
            words = words or _Words(text)
            spans += _text_spans(text, value, words)
        else:
            spans += _KIND_MATCHERS[value.match_kind](text, value)
    return _merge([s for s in spans if s.end > s.start])


def redact(text: str, values: list[PrivateValue],
           replacement: Callable[[Span], str] | None = None) -> tuple[str, list[Span]]:
    """Replace every private span (default: its [REDACTED: <label>] placeholder)."""
    replacement = replacement or (lambda span: placeholder(span.label))
    spans = find_private_spans(text, values)
    for span in reversed(spans):
        text = text[:span.start] + replacement(span) + text[span.end:]
    return text, spans
