"""Serving-side privacy wiring: chat applies the guard, prompt rule 7, retriever
fingerprint warning (privacy plan v3, 7.4 - 7.6)."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.serving import chat, guard, llm_client, retriever
from hypeonoath.serving.prompt import PROMPT_PERSON, SYSTEM_RULES

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
    assert "The candidate prefers not to share their date of birth here." in out


def test_chat_without_decisions_file_says_guard_inactive(monkeypatch, capsys):
    assert run_chat(monkeypatch, "A harmless answer.", "hi") == 0
    captured = capsys.readouterr()
    assert "privacy guard: no decisions file" in captured.err and "A harmless answer." in captured.out


def test_prompt_rule_7_and_person_mapping():
    """Test 14.11 (v13, P-D52): rewritten rule 7; rules 1-6 and the person mapping unchanged."""
    assert "7. Text shown as [REDACTED: <label>] is private." in SYSTEM_RULES
    assert f'write exactly: "{config.DECLINE_TEMPLATES["third"].format(label="<label>")}"' in SYSTEM_RULES
    assert f'Never answer a redacted item with "{config.REFUSAL_PHRASE}"' in SYSTEM_RULES
    assert "an age from a date of birth" in SYSTEM_RULES                                  # derived questions
    assert "Answer every other part of the question normally" in SYSTEM_RULES
    assert "Never guess, reconstruct, hint at or partly reveal a redacted value." in SYSTEM_RULES
    assert "Example. Context:" in SYSTEM_RULES and "[REDACTED: email address]" in SYSTEM_RULES
    assert "6. Be concise and factual.\n7. " in SYSTEM_RULES
    assert PROMPT_PERSON == {"v1": "third"}


def test_rules_2_7a_8_silent_skip_fix():
    """Silent-skip fix plan v2: F1 any wording, F2 per-part refusal (rule 2) + rule 8 every part."""
    assert ('2. If the context covers none of the question, reply exactly: '
            f'"{config.REFUSAL_PHRASE}". If it covers only some parts, answer those and say '
            f'"{config.REFUSAL_PHRASE}" for each part it does not cover.') in SYSTEM_RULES
    assert 'asks for it in any wording (for example "application number" for [REDACTED: applicant number])' \
        in SYSTEM_RULES
    assert SYSTEM_RULES.endswith("8. Address every part of the question. Never leave a part out silently: "
                                 "each part is answered, declined (rule 7) or refused (rule 2).")
    assert "1. Use only facts stated in the CONTEXT." in SYSTEM_RULES and "3. Cite the source" in SYSTEM_RULES


def test_rule_7_decline_sentence_is_the_guard_sentence():
    """Single source (14.11): the prompt's decline sentence is exactly what the guard produces."""
    example_value = pv.values_from_entries([{**ENTRY, "label": "email address", "match_kind": "email",
                                             "value": "x@example.org", "normalized": "x@example.org",
                                             "surface_forms": ["x@example.org"]}])
    guarded = guard.guard_answer("x@example.org", "email?", example_value, person="third")
    assert guarded.declined and f"Example_CV.md]. {guarded.text}\"" in SYSTEM_RULES


def test_rule_7_example_is_invented_and_holds_no_private_value():
    """Example safety (14.11): the example file is not in the corpus; no private value is in the prompt."""
    repo_corpus = Path(__file__).resolve().parents[2] / "corpus"
    assert not list(repo_corpus.rglob("Example_CV*"))
    write_decisions(ENTRY, {**ENTRY, "id": "pv_mail", "label": "email address", "match_kind": "email",
                            "entity_type": "EMAIL_ADDRESS", "value": "jane.doe@mail.example",
                            "normalized": "jane.doe@mail.example", "surface_forms": ["jane.doe@mail.example"]})
    assert pv.find_private_spans(SYSTEM_RULES, pv.load_private_values()) == []


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
