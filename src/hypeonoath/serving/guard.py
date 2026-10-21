"""
Answer-time privacy guard (privacy plan v3, 6.9; P-D14 - P-D17).

The safety net behind the real protection (values are redacted before
indexing, so the model never sees them). Checks the final answer ONLY for
values the owner marked private -- no grounding check, no detection of
unknown PII (owner decision P-D14) -- using the shared matcher
(core.private_values: exact after normalisation, dates in any format,
partial IDs >= 6 chars, fuzzy text >= 0.9; no embeddings, P-D15).

  * no hit                    -> the answer unchanged
  * hit, small share          -> only the leaked span becomes "[private]" + one note line
  * hit making up > 50% of the answer, or the question asks for that
    label                     -> templated decline ("... prefers not to share their <label> here")

Pure Python (stdlib + core): part of the small serving install (P-D6).
Logs labels and the decline flag only -- never a value or the answer text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.core.private_values import PrivateValue, find_private_spans

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GuardResult:
    text: str                 # what the user sees
    hits: tuple[str, ...]     # labels of private values found in the raw answer (no values)
    declined: bool            # full decline used


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _asked_for(label: str, question: str) -> bool:
    """Every word of the label occurs in the question ("date of birth" in "What's his date of birth?")."""
    label_words = _words(label)
    return bool(label_words) and label_words <= _words(question)


def _join_labels(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def guard_answer(answer: str, question: str, values: list[PrivateValue], person: str = "third") -> GuardResult:
    spans = find_private_spans(answer, values)
    if not spans:
        return GuardResult(answer, (), False)

    labels = list(dict.fromkeys(s.label for s in spans))
    share = sum(len(s) for s in spans) / max(1, len(answer.strip()))
    asked = any(_asked_for(label, question) for label in labels)
    declined = share > config.GUARD_FULL_DECLINE_SHARE or asked
    if declined:
        template = config.DECLINE_TEMPLATES.get(person, config.DECLINE_TEMPLATES["third"])
        text = template.format(label=_join_labels(labels))
    else:
        text = answer
        for span in reversed(spans):
            text = text[:span.start] + config.GUARD_SPAN_REPLACEMENT + text[span.end:]
        text = f"{text}\n\n{config.GUARD_NOTE}"
    log_json_line(config.PRIVACY_LOG_PATH, event="guard", labels=labels, declined=declined)
    return GuardResult(text, tuple(labels), declined)
