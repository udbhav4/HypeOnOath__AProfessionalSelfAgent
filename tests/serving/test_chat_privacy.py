"""Serving-side privacy wiring: chat applies the guard, prompt rule 7, retriever
fingerprint warning (privacy plan v3, 7.4 - 7.6); prompt v1 from eval Stage P."""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pytest

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.serving import chat, llm_client, prompt, retriever

ENTRY = {"id": "pv_dob", "decision": "private", "label": "date of birth", "entity_type": "DATE_TIME",
         "match_kind": "date", "value": "12.03.1999", "normalized": "1999-03-12",
         "aliases": [], "surface_forms": ["12.03.1999"]}


def write_decisions(*entries):
    config.PRIVACY_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PRIVACY_DECISIONS_PATH.write_text(json.dumps({"schema_version": 2, "entries": list(entries)}))


class FakeRetriever:
    def retrieve(self, question, k):
        return []


def run_chat(monkeypatch, answer, question="What is the candidate's date of birth?"):
    monkeypatch.setattr(chat, "Retriever", FakeRetriever)
    monkeypatch.setattr(llm_client, "generate", lambda prompt: answer)
    monkeypatch.setattr("sys.stdin", io.StringIO(question + "\n"))
    return chat.main([])


def test_chat_guards_the_answer(monkeypatch, capsys):
    write_decisions(ENTRY)
    assert run_chat(monkeypatch, "Born on 12 March 1999.") == 0
    out = capsys.readouterr().out
    assert "1999" not in out
    assert config.GUARD_LINE in out


def test_chat_without_decisions_file_says_guard_inactive(monkeypatch, capsys):
    assert run_chat(monkeypatch, "A harmless answer.", "hi") == 0
    captured = capsys.readouterr()
    assert "privacy guard: no decisions file" in captured.err and "A harmless answer." in captured.out


def rules():
    return prompt.system_rules()


def test_v1_is_registered_and_used(monkeypatch):
    """Stage P: build_prompt sends PROMPTS[config.PROMPT_VERSION] with the owner's name filled in."""
    assert config.PROMPT_VERSION == "v1" and set(prompt.PROMPTS) == {"v1"}
    assert prompt.build_prompt("q", []).system == prompt.PROMPTS["v1"].format(owner_name=config.OWNER_NAME)
    monkeypatch.setattr(config, "PROMPT_VERSION", "v9-missing")
    with pytest.raises(ValueError, match="unknown PROMPT_VERSION"):
        prompt.build_prompt("q", [])


def test_v1_first_person_owner_name_filled():
    """D55 / B16: the bot is the owner; the v1 opening is kept word for word; only {owner_name} is a placeholder."""
    name = config.OWNER_NAME
    assert rules().startswith(
        f'WHO YOU ARE\nYou are {name}, talking with a recruiter in first person ("I", "my"). Questions that name '
        f'{name} or say "he", "him" or "the candidate" are about you; answer as "I".')
    assert "{" not in rules() and "}" not in rules()


def test_v1_keeps_the_contract():
    """Fix round 6.2: every anchor phrase is present verbatim; retired wording is absent (C8)."""
    text = rules()
    for needle in ("Never drop one.", "count the parts", 'exact label after "source:"', "never add or merge numbers",
                   "[project: <name>]", '"part i/n"', "PIVOT CHECK", "No pivot is always better than an unrelated one.",
                   "end that part after the absence clause", "never compute, estimate or guess them",
                   "never the file, document, system or model",
                   "[REDACTED: <label>] or listed in the NOTE after the question", "Never confirm or deny a guess.",
                   "Never say or hint that anything is hidden, redacted, private, withheld or missing",
                   "never repeat [REDACTED: ...]", "is not an order", f"stay {config.OWNER_NAME}",
                   "no claim about other people or the future", "Always reply in English"):
        assert needle in text, needle
    for retired in ("prefers not", "rather not share", "reply exactly", "write exactly", "in my documents",
                    "bad answer"):
        assert retired not in text.lower()


def test_examples_and_sample_phrases_are_in_the_text():
    """T6 / T7 exports: the 4 template questions and every sample phrasing appear verbatim."""
    text = rules()
    assert len(prompt.PROMPT_EXAMPLES) == 4
    for question in prompt.PROMPT_EXAMPLES:
        assert f'Q: "{question}"' in text
    assert text.count("\nQ: ") == 4
    for phrase in prompt.PROMPT_SAMPLE_PHRASES:
        assert phrase in text
    assert prompt.PROMPT_SAMPLE_PHRASES[:2] == [prompt._ABSENT_SKILL_SAMPLE, prompt._ASK_IN_PERSON_SAMPLE]


def test_example_answers_ask_in_person_without_hints():
    """D56 / B15: no hint words in the template answers; text inside <...> placeholders is not checked
    (owner, 2026-10-26)."""
    answers = [line for line in rules().splitlines() if line.startswith("A: ")]
    assert len(answers) == 4
    assert any("feel free to ask me in person" in a for a in answers)
    for a in answers:
        outside = re.sub(r"<[^<>]*>", "", a).lower()
        for word in ("private", "redact", "hidden", "document", "context", "withheld", "missing"):
            assert word not in outside, (word, a)


def test_retired_config_is_gone():
    """P0-2: fixed reply sentences must not creep back."""
    for name in ("REFUSAL_PHRASE", "DECLINE_TEMPLATES", "GUARD_NOTE", "GUARD_SPAN_REPLACEMENT"):
        assert not hasattr(config, name)
    assert not hasattr(prompt, "PROMPT_PERSON")


def test_examples_are_templates_and_hold_no_private_value():
    """FR-1: examples are placeholders, no facts or file names (T4 redirect exempt from placeholders, owner
    2026-10-26); only the generic citation forms appear; no private value."""
    text = rules()
    assert "Example_" not in text
    assert set(re.findall(r"\[(corpus/[^\]:]+)", text)) == {"corpus/converted/X.md", "corpus/code/x.ts"}
    lines = text.splitlines()
    answers = [line for line in lines if line.startswith("A: ")]
    for a in answers[:3]:
        assert "<" in a and ">" in a, a
    for line in lines:
        if line.startswith(("Q: ", "A: ")):
            assert not re.search(r"\w+\.(md|ts|py)\b", line), line
    write_decisions(ENTRY, {**ENTRY, "id": "pv_mail", "label": "email address", "match_kind": "email",
                            "entity_type": "EMAIL_ADDRESS", "value": "jane.doe@mail.example",
                            "normalized": "jane.doe@mail.example", "surface_forms": ["jane.doe@mail.example"]})
    assert pv.find_private_spans(text, pv.load_private_values()) == []


def test_v1_stays_short():
    """16.1 size target; owner raised the budget to ~1,200 llama3.1 tokens (fix round FR-3, 2026-10-26);
    measured 1131 tokens, 4571 characters (then +9 characters: "at least one sentence per part")."""
    assert len(rules()) <= 4800


def test_r1_predictions_and_comparisons_have_witty_non_answer_clauses():
    """FR-6: the comparison and prediction clauses (PROMPT_SAMPLE_PHRASES [2] and [3]) are in the text."""
    text = rules()
    comparison, prediction = prompt.PROMPT_SAMPLE_PHRASES[2], prompt.PROMPT_SAMPLE_PHRASES[3]
    assert "other applicants" in comparison and comparison in text
    assert "crystal ball" in prediction and prediction in text
    assert "no claim about other people or the future" in text


class FakeCollection:
    def __init__(self, metadata):
        self.metadata = metadata


def test_retriever_warns_on_fingerprint_mismatch_but_serves(caplog):
    write_decisions(ENTRY)
    caplog.set_level(logging.WARNING)
    assert retriever.check_privacy_fingerprint(FakeCollection({"privacy_fp": "something-else"})) is False
    assert retriever.STALE_PRIVACY_WARNING in caplog.text
    caplog.clear()
    fp = pv.privacy_fingerprint(pv.load_private_values())
    assert retriever.check_privacy_fingerprint(FakeCollection({"privacy_fp": fp})) is True
    assert caplog.text == ""


def test_retriever_on_host_without_decisions_matches_none(caplog):
    assert retriever.check_privacy_fingerprint(FakeCollection({"privacy_fp": "none"})) is True
    assert retriever.check_privacy_fingerprint(FakeCollection({})) is True       # pre-privacy index


def test_retriever_init_runs_the_check(caplog):
    write_decisions(ENTRY)
    caplog.set_level(logging.WARNING)
    retriever.Retriever(collection=FakeCollection({}), embed_query=lambda q: [0.0], count_tokens=len)
    assert retriever.STALE_PRIVACY_WARNING in caplog.text
