"""
Answer-time privacy guard (privacy plan v3, 6.9; behaviour per the P0 guideline, eval D63).

The safety net behind the real protection (values are redacted before
indexing, so the model never sees them). Checks the final answer ONLY for
values the owner marked private -- no grounding check, no detection of
unknown PII (owner decision P-D14) -- using the shared matcher
(core.private_values: exact after normalisation, dates in any format,
partial IDs >= 6 chars, fuzzy text >= 0.9; no embeddings, P-D15).

  * no hit                      -> the answer unchanged
  * hit                         -> every sentence / line holding a hit is dropped
                                   and config.GUARD_LINE is appended
  * hits >= GUARD_FULL_DECLINE_SHARE of the answer, or < GUARD_MIN_WORDS_LEFT
    words left after the drop   -> the reply is GUARD_LINE alone

The line never names a label and never says that anything is private (eval D56).
Pure Python (stdlib + core): part of the small serving install (P-D6).
Logs labels and the declined flag only -- never a value or the answer text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.private_values import PrivateValue, Span, find_private_spans

# A segment ends after . ! ? (optionally followed by a closing quote / bracket) plus
# spaces, or at any newline (bullet lines are their own segment). Splitting needn't
# be perfect: every segment overlapping a hit is dropped, so a mis-split only keeps
# or drops a little more public text, never the value (T-P0-3).
_BOUNDARY_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))[ \t]+|\n+")
_CITATION_RE = re.compile(r"\[[^\]]*\]")
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class GuardResult:
    text: str                 # what the user sees
    hits: tuple[str, ...]     # labels of private values found in the raw answer (no values)
    declined: bool            # the reply is GUARD_LINE alone


def _segments(text: str) -> list[tuple[int, int, str]]:
    """(start, end, separator after it) for every sentence / line of text."""
    out, start = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        out.append((start, m.start(), m.group()))
        start = m.end()
    out.append((start, len(text), ""))
    return out


def _drop_hit_segments(text: str, spans: list[Span]) -> str:
    """The text without every segment that overlaps a span (half-open intervals)."""
    kept = [text[s:e] + sep for s, e, sep in _segments(text)
            if not any(span.start < e and s < span.end for span in spans)]
    return "".join(kept).strip()


def _word_count(text: str) -> int:
    """Words outside [citations], so a bare "[cv.md]" does not count as content."""
    return len(_WORD_RE.findall(_CITATION_RE.sub(" ", text)))


def guard_answer(answer: str, values: list[PrivateValue]) -> GuardResult:
    spans = find_private_spans(answer, values)
    if not spans:
        return GuardResult(answer, (), False)

    labels = list(dict.fromkeys(s.label for s in spans))
    share = sum(len(s) for s in spans) / max(1, len(answer.strip()))
    kept = "" if share >= config.GUARD_FULL_DECLINE_SHARE else _drop_hit_segments(answer, spans)
    declined = (_word_count(kept) < config.GUARD_MIN_WORDS_LEFT
                or bool(find_private_spans(kept, values)))          # re-check (T-P0-4)
    text = config.GUARD_LINE if declined else f"{kept}\n\n{config.GUARD_LINE}"
    log_json_line(config.PRIVACY_LOG_PATH, event="guard", labels=labels, declined=declined)
    return GuardResult(text, tuple(labels), declined)
