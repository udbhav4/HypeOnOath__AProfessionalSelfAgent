"""Serving-side privacy wiring: chat applies the guard, prompt rule 7, retriever
fingerprint warning (privacy plan v3, 7.4 - 7.6; interim prompt per the P0 guideline)."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.serving import chat, llm_client, prompt, retriever
from hypeonoath.serving.prompt import SYSTEM_RULES

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


def test_prompt_first_person_and_rule_7():
    """P0-1 / D55 / D56: first person as OWNER_NAME; rule 7 answers private parts in the ask-in-person spirit."""
    assert SYSTEM_RULES.startswith(f"You are {config.OWNER_NAME}, answering a recruiter's questions about yourself")
    assert f'Questions that name {config.OWNER_NAME} or say "he", "him" or "the candidate" are about you.' \
        in SYSTEM_RULES
    assert "7. Text shown as [REDACTED: <label>] must never be shared." in SYSTEM_RULES
    assert 'asks for it in any wording (for example "application number" for [REDACTED: applicant number])' \
        in SYSTEM_RULES
    assert "an age from a date of birth), reply to that part as in rule 2b." in SYSTEM_RULES
    assert "Answer every other part of the question normally" in SYSTEM_RULES
    assert ("Never say that anything is hidden, redacted, private, withheld or missing, never mention "
            '"documents" or "context", and never repeat [REDACTED: ...].') in SYSTEM_RULES
    assert "Example. Context:" in SYSTEM_RULES and "[REDACTED: email address]" in SYSTEM_RULES
    assert "6. Be concise and factual.\n7. " in SYSTEM_RULES
    for retired in ("prefers not", "rather not share", "reply exactly", "write exactly", "in my documents"):
        assert retired not in SYSTEM_RULES


def test_rules_2_and_8_every_part():
    """D56: two absence kinds with their sample phrasings; rule 8 is the last line; rules 1, 3, 4, 6 kept."""
    assert "2. Handle every part of the question on its own." in SYSTEM_RULES
    assert f'a. a skill or experience: say so softly and honestly, e.g. "{prompt._ABSENT_SKILL_SAMPLE}".' \
        in SYSTEM_RULES
    assert "ONE sentence about a related skill or experience from the context, with its citation" in SYSTEM_RULES
    assert f'b. a fact or detail: reply in this spirit: "{prompt._ASK_IN_PERSON_SAMPLE}"' in SYSTEM_RULES
    assert SYSTEM_RULES.endswith("8. Address every part of the question. Never leave a part out silently: "
                                 "each part is answered from the context or gets a reply as in rule 2.")
    assert "1. Use only facts stated in the CONTEXT. Never use outside knowledge or guess about yourself." \
        in SYSTEM_RULES
    assert "3. Cite the source of every claim inline" in SYSTEM_RULES
    assert '4. A code chunk marked "part i/n" is only one piece' in SYSTEM_RULES
    assert "5. Entries under \"Projects referenced\" are your own project descriptions" in SYSTEM_RULES


def test_rule_7_example_answer_asks_in_person_without_hints():
    """Replaces the retired guard-sentence equality (D56 / D63: no shared decline sentence any more)."""
    answer = SYSTEM_RULES.split("   Answer: ", 1)[1].split("\n", 1)[0]
    assert answer.startswith('"I work with Python and SQL [corpus/converted/Example_CV.md].')
    assert "feel free to ask me in person" in answer
    assert "[REDACTED" not in answer and "private" not in answer.lower()


def test_retired_config_is_gone():
    """P0-2: fixed reply sentences must not creep back."""
    for name in ("REFUSAL_PHRASE", "DECLINE_TEMPLATES", "GUARD_NOTE", "GUARD_SPAN_REPLACEMENT"):
        assert not hasattr(config, name)
    assert not hasattr(prompt, "PROMPT_PERSON")


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
