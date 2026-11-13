"""serving/query_rewrite.py (eval plan v29, D94; F-R4): third-person retrieval query, pronouns from config."""
from __future__ import annotations

import pytest

from hypeonoath.core import config
from hypeonoath.serving import query_rewrite
from hypeonoath.serving.query_rewrite import expand_contractions, rewrite_query, third_person


def _model_available() -> bool:
    try:
        import lemminflect  # noqa: F401
        import spacy

        return spacy.util.is_package(config.QUERY_REWRITE_SPACY_MODEL)
    except ImportError:
        return False


needs_model = pytest.mark.skipif(not _model_available(),
                                 reason="spaCy en_core_web_sm + lemminflect: pip install -r requirements-eval-models.txt")

# Real question shapes of the gold set + the grammar cases D94 names (risk row: "a test table").
CASES = [
    ("What are you studying right now, and where?", "What is he studying right now, and where?"),
    ("Do you need visa sponsorship to work in Germany?", "Does he need visa sponsorship to work in Germany?"),
    ("Have you designed or analysed A/B tests?", "Has he designed or analysed A/B tests?"),
    ("Where were you admitted for your master's?", "Where was he admitted for his master's?"),
    ("List all the Coursera certificates you've earned.", "List all the Coursera certificates he has earned."),
    ("Why should we hire you as a machine learning engineer?", "Why should we hire him as a machine learning engineer?"),
    ("How can I get in touch with you?", "How can I get in touch with him?"),
    ("Tell me about yourself.", "Tell me about him."),                       # not "about himself"
    ("Did you teach yourself Python?", "Did he teach himself Python?"),
    ("What makes you a good fit for a data analyst role?", "What makes him a good fit for a data analyst role?"),
    ("Why should a company trust that the AI systems you build don't make things up?",
     "Why should a company trust that the AI systems he builds don't make things up?"),
    ("Is this project yours?", "Is this project his?"),
    ("You're based in Darmstadt, right?", "He is based in Darmstadt, right?"),
    ("Can you build dashboards in Tableau?", "Can he build dashboards in Tableau?"),
    ("What do you and your team do?", "What do he and his team do?"),          # plural subject: verb kept
]


@needs_model
@pytest.mark.parametrize("question, expected", CASES)
def test_third_person_table(question, expected):
    assert third_person(question) == expected


@needs_model
def test_never_rewrites_to_the_owner_name():
    for question, _ in CASES:
        assert config.OWNER_NAME.lower() not in third_person(question).lower()


@needs_model
def test_pronouns_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "OWNER_PRONOUNS", ("she", "her", "her", "herself"))
    assert third_person("Are you proud of your work?") == "Is she proud of her work?"
    assert third_person("Why should we hire you?") == "Why should we hire her?"
    assert third_person("Did you teach yourself SQL?") == "Did she teach herself SQL?"


def test_no_second_person_returns_the_question_without_loading_spacy(monkeypatch):
    monkeypatch.setattr(query_rewrite, "_nlp", lambda: pytest.fail("parser loaded"))
    question = "Which tools does the CV list?"
    assert third_person(question) is question


def test_any_failure_falls_back_to_the_original(monkeypatch, caplog):
    def broken():
        raise OSError("model missing")

    monkeypatch.setattr(query_rewrite, "_nlp", broken)
    assert third_person("What are you proud of?") == "What are you proud of?"
    assert "query rewrite failed" in caplog.text


def test_contractions_are_expanded():
    assert expand_contractions("You're here, you've done it, you'll go, you'd say, you’re") == \
        "You are here, you have done it, you will go, you would say, you are"
    assert expand_contractions("Yours, yourself") == "Yours, yourself"


def test_rewrite_query_modes():
    assert rewrite_query("What are you proud of?", "none") == "What are you proud of?"
    assert config.QUERY_REWRITE == "none"                                   # production default off (D98)
    with pytest.raises(ValueError, match="unknown query rewrite"):
        rewrite_query("q", "name")


@needs_model
@pytest.mark.parametrize("question, expected", [
    ("Thank you. What are you proud of?", "Thank you. What is he proud of?"),
    ("Do you guys use Git?", "Do you guys use Git?"),
])
def test_addressing_others_is_left_alone(question, expected):
    assert third_person(question) == expected


def test_rewrite_query_reads_the_config_default_at_call_time(monkeypatch):
    monkeypatch.setattr(query_rewrite, "third_person", lambda q: "REWRITTEN")
    monkeypatch.setattr(config, "QUERY_REWRITE", "third_person")
    assert rewrite_query("q") == "REWRITTEN"


def test_check_available_turns_a_missing_model_into_value_error(monkeypatch):
    def missing():
        raise OSError("[E050] Can't find model")
    monkeypatch.setattr(query_rewrite, "_nlp", missing)
    with pytest.raises(ValueError, match="requirements-eval-models.txt"):
        query_rewrite.check_available()
