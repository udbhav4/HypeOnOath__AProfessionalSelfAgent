"""
privacy.detector + Presidio configuration (privacy plan v3, 6.2, 6.3; tests 14.2).

Tests marked privacy_model load the real spaCy models (en_core_web_lg +
de_core_news_lg, ~45 s once per session). The config / generator / static
leak tests only need presidio-analyzer installed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from hypeonoath.core import config

presidio = pytest.importorskip("presidio_analyzer")

from hypeonoath.privacy import detector  # noqa: E402
from hypeonoath.privacy.conf import _generate_recognizers as generator  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src" / "hypeonoath"
EXTERNAL = {"BasicLangExtractRecognizer", "AzureAILanguageRecognizer", "AzureHealthDeidRecognizer",
            "AzureOpenAILangExtractRecognizer"}


# --- configuration (no models needed) -------------------------------------------------------------

def test_recognizers_yaml_everything_local_on_for_en_and_de():
    conf = yaml.safe_load(detector.RECOGNIZERS_CONF_PATH.read_text(encoding="utf-8"))
    assert conf["supported_languages"] == ["en", "de"] and "supported_countries" not in conf
    names = [r["name"] for r in conf["recognizers"]]
    assert len(names) == len(set(names))                                    # duplicates removed
    assert not EXTERNAL & set(names) and "HuggingFaceNerRecognizer" not in names
    for entry in conf["recognizers"]:
        assert entry.get("enabled", True) is True, entry["name"]
        languages = [lang if isinstance(lang, str) else lang["language"] for lang in entry["supported_languages"]]
        assert languages == ["en", "de"], entry["name"]
    assert {"DeTaxIdRecognizer", "InAadhaarRecognizer", "UsSsnRecognizer", "KrRrnRecognizer",
            "DateOfBirthKeywordRecognizer"} <= set(names)
    assert not set(RETIRED_RECOGNIZERS) & set(names)                         # P-D35, P-D51


RETIRED_RECOGNIZERS = ("DeKfzRecognizer", "UkVehicleRegistrationRecognizer", "InVehicleRegistrationRecognizer",
                       "NgVehicleRegistrationRecognizer", "TrLicensePlateRecognizer", "DePlzRecognizer",
                       "UkPostcodeRecognizer", "UsLicenseRecognizer")


def test_retired_recognizers_match_retired_entities():
    """P-D35 / P-D51: the generator's dropped classes and config.PRIVACY_RETIRED_ENTITIES stay in sync."""
    import presidio_analyzer.predefined_recognizers as predefined

    assert {n for n, why in generator._EXCLUDED_BY_DECISION.items() if "P-D35" in why or "P-D51" in why} == set(RETIRED_RECOGNIZERS)
    entities = set()
    for name in RETIRED_RECOGNIZERS:
        entities.update(getattr(predefined, name)().supported_entities)
    assert entities == set(config.PRIVACY_RETIRED_ENTITIES)


def test_generated_header_names_the_installed_presidio_version():
    from importlib import metadata

    header = detector.RECOGNIZERS_CONF_PATH.read_text(encoding="utf-8").splitlines()[1]
    assert header == f"# Source: presidio-analyzer {metadata.version('presidio-analyzer')} default_recognizers.yaml"


def test_generator_output_is_up_to_date(tmp_path):
    """Re-generating from the installed Presidio gives the committed file (else: re-run the generator)."""
    out = tmp_path / "recognizers.yaml"
    generator.generate(out)
    assert out.read_text(encoding="utf-8") == detector.RECOGNIZERS_CONF_PATH.read_text(encoding="utf-8")


def test_generator_transform_rules():
    class Local:
        __module__ = "presidio_analyzer.predefined_recognizers.generic.x"

    class Remote:
        __module__ = "presidio_analyzer.predefined_recognizers.third_party.y"

    default = {"global_regex_flags": 26, "recognizers": [
        {"name": "A", "type": "predefined", "enabled": False, "supported_languages": ["en"]},
        {"name": "A", "type": "predefined", "supported_languages": ["en"]},                    # duplicate
        {"name": "B", "type": "predefined", "supported_languages": [
            {"language": "en", "context": ["card"]}, {"language": "es", "context": ["tarjeta"]}]},
        {"name": "Remote", "type": "predefined", "enabled": False},
        {"name": "HuggingFaceNerRecognizer", "type": "predefined", "enabled": False},
        {"name": "DeKfzRecognizer", "type": "predefined", "enabled": False, "country_code": "de"},
        "C",
    ]}
    conf, notes = generator.build_config(default, [{"name": "K", "type": "custom"}],
                                         {"A": Local, "B": Local, "C": Local, "Remote": Remote})
    by_name = {r["name"]: r for r in conf["recognizers"]}
    assert list(by_name) == ["A", "B", "C", "K"]
    assert "enabled" not in by_name["A"] and by_name["A"]["supported_languages"] == ["en", "de"]
    assert by_name["B"]["supported_languages"] == [{"language": "en", "context": ["card"]},
                                                    {"language": "de", "context": ["card"]}]
    assert by_name["C"]["supported_languages"] == ["en", "de"]
    assert any("Remote" in n and "external" in n for n in notes) and any("HuggingFace" in n for n in notes)
    assert any("DeKfzRecognizer" in n and "P-D35" in n for n in notes)


def test_nlp_yaml_matches_config_and_keeps_money():
    conf = yaml.safe_load(detector.NLP_CONF_PATH.read_text(encoding="utf-8"))
    assert {m["lang_code"]: m["model_name"] for m in conf["models"]} == config.PRIVACY_SPACY_MODELS
    ner = conf["ner_model_configuration"]
    assert ner["model_to_presidio_entity_mapping"]["MONEY"] == "MONEY"
    assert "MONEY" not in ner["labels_to_ignore"] and "ORGANIZATION" in ner["labels_to_ignore"]


def test_keyword_patterns_match_the_value_only():
    import regex

    custom = yaml.safe_load((detector.CONF_DIR / "custom_recognizers.yaml").read_text(encoding="utf-8"))
    patterns = {r["supported_entity"]: regex.compile(r["patterns"][0]["regex"], flags=26) for r in custom}
    cases = {
        "DATE_OF_BIRTH": [("Geburtsdatum: 12.03.1999", "12.03.1999"), ("Date of birth: March 12, 1999",
                                                                       "March 12, 1999"), ("DOB 1999-03-12", "1999-03-12")],
        "PLACE_OF_BIRTH": [("Geburtsort: Dehradun, Indien", "Dehradun"), ("Place of birth - New Delhi\n", "New Delhi")],
        "NATIONALITY": [("Staatsangehörigkeit: indisch, ledig", "indisch")],
        "STUDENT_ID": [("Matrikelnummer 2876543", "2876543"), ("student ID: AB-1234", "AB-1234")],
        "APPLICANT_ID": [("Bewerbernummer: BEW-2023-4711", "BEW-2023-4711"), ("Applicant no. A-123456", "A-123456")],
    }
    for entity, examples in cases.items():
        for text, expected in examples:
            match = patterns[entity].search(text)
            assert match and match.group(0) == expected, (entity, text)
    assert patterns["STUDENT_ID"].search("Matrikelnummer unknown") is None        # needs a digit


# --- static leak guard (14.1) ---------------------------------------------------------------------------

def test_candidates_are_never_serialised_by_writers():
    """No asdict()/json.dumps of a Candidate (or a value field) anywhere in privacy/ or indexing/."""
    offenders = []
    for folder in ("privacy", "indexing"):
        for path in (SRC / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in (r"asdict\(\s*(cand|candidate|c|group)\b", r"json\.dumps?\([^)]*(cand|candidate)\b",
                            r"log_json_line\([^)]*\bvalue=", r"logger\.\w+\([^)]*\.value\b"):
                offenders += [f"{path.name}: {m.group(0)}" for m in re.finditer(pattern, text)]
    assert offenders == []


def test_candidate_repr_hides_the_value():
    cand = detector.Candidate("a.md", 3, "PHONE_NUMBER", "PhoneRecognizer", 0.75, "+49 151 2345678",
                              "Phone: +49 151 2345678", 7)
    assert "2345678" not in repr(cand) and "2345678" not in str([cand])


def test_dedupe_keeps_best_same_type_and_all_types():
    hits = [detector._Hit(0, 10, "DATE_TIME", 0.6, "en"), detector._Hit(0, 10, "DATE_TIME", 0.85, "de"),
            detector._Hit(2, 8, "DATE_TIME", 0.4, "x"), detector._Hit(0, 10, "DATE_OF_BIRTH", 0.6, "kw")]
    kept = detector._dedupe_hits(hits)
    assert [(h.entity_type, h.score) for h in kept] == [("DATE_OF_BIRTH", 0.6), ("DATE_TIME", 0.85)]


def test_line_mapping_and_context():
    text = "line one\nPhone: X123\nline three"
    hit = detector._Hit(text.index("X123"), text.index("X123") + 4, "ID", 0.9, "r")
    (cand,) = detector._to_candidates(text, [hit], "f.md", 10)
    assert (cand.line, cand.line_text, cand.start, cand.before, cand.after) == (
        11, "Phone: X123", 7, "line one", "line three")
    (no_line,) = detector._to_candidates(text, [hit], "f.md", None)
    assert no_line.line is None


def test_secret_hits_locate_the_value():
    text = 'x = 1\nkey = "AKIAIOSFODNN7EXAMPLE"\n'
    (hit,) = detector._secret_hits(text, "f.py")
    assert text[hit.start:hit.end] == "AKIAIOSFODNN7EXAMPLE" and hit.entity_type == "SECRET:AWS Access Key"


# --- real models (14.2) -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    pytest.importorskip("en_core_web_lg")
    pytest.importorskip("de_core_news_lg")
    return detector.build_analyzer()


EN_TEXT = ("Email jane.doe@example.com. Phone +49 151 23456789. Indian mobile +91 98765 43210.\n"
           "IBAN DE89 3704 0044 0532 0130 00. My social security number is 536-22-1234. Aadhaar 2345 6789 0124. "
           "PAN ABCDE1234F.\nSteuer-ID / tax ID: 12345678995.")


@pytest.mark.privacy_model
def test_english_text_types_and_country_ids_fire_on_en(analyzer):
    en = {(r.entity_type, EN_TEXT[r.start:r.end]) for r in analyzer.analyze(text=EN_TEXT, language="en",
                                                                             score_threshold=config.PRIVACY_MIN_SCORE)}
    assert {("EMAIL_ADDRESS", "jane.doe@example.com"), ("PHONE_NUMBER", "+49 151 23456789"),
            ("PHONE_NUMBER", "+91 98765 43210"), ("IBAN_CODE", "DE89 3704 0044 0532 0130 00"),
            ("US_SSN", "536-22-1234"), ("IN_AADHAAR", "2345 6789 0124"), ("IN_PAN", "ABCDE1234F"),
            ("DE_TAX_ID", "12345678995")} <= en                    # German + Indian IDs on English text (P-D3)


@pytest.mark.privacy_model
@pytest.mark.parametrize("text, entity, value", [
    ("Geburtsdatum: 12.03.1999", "DATE_OF_BIRTH", "12.03.1999"),
    ("Date of birth: March 12, 1999", "DATE_OF_BIRTH", "March 12, 1999"),
    ("Matrikelnummer 2876543", "STUDENT_ID", "2876543"),
    ("Salary expectation: €65,000 per year.", "MONEY", None),
])
def test_keyword_and_money_findings(analyzer, text, entity, value):
    found = [(c.entity_type, c.value) for c in detector.detect_text(text, "x.md", 1, analyzer=analyzer)]
    assert any(e == entity and (value is None or v == value) for e, v in found), found


@pytest.mark.privacy_model
@pytest.mark.parametrize("language", ["en", "de"])
def test_no_plate_or_postcode_findings(analyzer, language):
    text = ("4. Write HYPOTHESIS: followed by a 3 to 5 sentence causal chain.\n"
            "* get_belief_trend for up to 6 changed nodes. Address: Hauptstraße 5, 64283 Darmstadt.")
    found = {r.entity_type for r in analyzer.analyze(text=text, language=language, score_threshold=0)}
    assert not found & set(config.PRIVACY_RETIRED_ENTITIES), found


@pytest.mark.privacy_model
@pytest.mark.parametrize("language", ["en", "de"])
def test_application_number_is_an_applicant_id_not_a_licence(analyzer, language):
    """P-D51: "lic" inside "Application" no longer turns the number into a US driver licence."""
    found = {r.entity_type for r in analyzer.analyze(text="Application Number: 123456", language=language,
                                                     score_threshold=config.PRIVACY_MIN_SCORE)}
    assert "APPLICANT_ID" in found and "US_DRIVER_LICENSE" not in found, found


@pytest.mark.privacy_model
def test_organisations_are_not_findings(analyzer):
    found = detector.detect_text("I worked at Deutsche Bank and Amadeus.", "x.md", 1, analyzer=analyzer)
    assert all(c.entity_type != "ORGANIZATION" for c in found)


@pytest.mark.privacy_model
def test_two_languages_union_without_duplicates(analyzer):
    found = detector.detect_text("Contact jane.doe@example.com today.", "x.md", 1, analyzer=analyzer)
    keys = [(c.entity_type, c.value, c.line) for c in found]
    assert keys.count(("EMAIL_ADDRESS", "jane.doe@example.com", 1)) == 1 and len(keys) == len(set(keys))


@pytest.mark.privacy_model
def test_detect_blocks_covers_documents_code_and_projects(analyzer):
    from hypeonoath.core.records import Block
    from hypeonoath.indexing.block_loader import LoadedDocument
    from hypeonoath.indexing.project_registry import ProjectInfo

    doc = LoadedDocument("corpus/converted/cv.md", [Block("prose", "Mail jane.doe@example.com", "corpus/converted/cv.md",
                                                          meta={"line_start": 7})])
    code = Block("code_chunk", "# [Demo] a.py\nKEY = 'AKIAIOSFODNN7EXAMPLE'", "corpus/code/a.py",
                 meta={"line_start": 20, "chunk_id": "c"})
    projects = {"a.py": ProjectInfo("demo", "Demo", "Contact: owner@example.org")}
    found = {(c.file, c.line, c.entity_type) for c in detector.detect_blocks([doc], [code], projects, analyzer=analyzer)}
    assert ("corpus/converted/cv.md", 7, "EMAIL_ADDRESS") in found
    assert ("corpus/code/a.py", 20, "SECRET:AWS Access Key") in found        # code line 1 = line_start
    assert (detector.project_strings_file(), None, "EMAIL_ADDRESS") in found


# --- (v4) language routing P-D25 + no NER on code P-D26 (plan 6.3.1; 14.2 v4 rows) ---------------------

EN_PARA = "I have worked with the team on several projects and we built them together from the start."
DE_PARA = "Ich habe mit dem Team an mehreren Projekten gearbeitet und wir haben sie gemeinsam aufgebaut."
MIXED = "We did the project und ich habe es mit dem team and with them gebaut."


@pytest.mark.parametrize("text, expected", [
    (EN_PARA, "en"), (DE_PARA, "de"), ("Kontakt und Adresse", None), (MIXED, None),
    ("Zulassung für Studiengang Informatik Bewerbung Prüfung Universität Darmstadt Hörsaal Gebühr", "de"),
])
def test_text_language(text, expected):
    assert detector.text_language(text) == expected


def test_route_languages(monkeypatch):
    assert detector.route_languages(EN_PARA) == ("en",)
    assert detector.route_languages(DE_PARA) == ("de",)
    assert detector.route_languages("Kontakt", "de") == ("de",)          # short block -> document language
    assert detector.route_languages("Kontakt", None) == ("en", "de")     # undecided -> both (v3 recall)
    monkeypatch.setattr(config, "PRIVACY_LANGUAGE_ROUTING", False)
    assert detector.route_languages(EN_PARA) == ("en", "de")


def _doc(*blocks):
    from hypeonoath.indexing.block_loader import LoadedDocument

    return LoadedDocument("corpus/converted/cv.md", list(blocks))


def test_block_texts_route_documents_code_and_fences():
    from hypeonoath.core.records import Block

    doc = _doc(Block("heading", "Kontakt", "cv.md"), Block("prose", DE_PARA, "cv.md"),
               Block("code_fence", "```\nprint('hi')\n```", "cv.md"))
    code = Block("code_chunk", "# [P] a.py\nx = 1", "corpus/code/a.py", meta={"line_start": 5, "chunk_id": "c"})
    items = list(detector._block_texts([doc], [code]))
    assert [(i.is_code, i.languages) for i in items] == [
        (False, ("de",)), (False, ("de",)), (True, ("en",)), (True, ("en",))]
    assert items[-1].first_line == 4


class FakeResult:
    def __init__(self, start, end, entity, recognizer):
        self.start, self.end, self.entity_type, self.score = start, end, entity, 0.85
        self.recognition_metadata = {"recognizer_name": recognizer}


class FakeAnalyzer:
    """Returns one NER and one pattern result per call; records the languages asked for."""

    def __init__(self):
        self.languages = []

    def analyze(self, text, language, score_threshold):
        self.languages.append(language)
        return [FakeResult(0, 3, "PERSON", "SpacyRecognizer"), FakeResult(4, 9, "EMAIL_ADDRESS", "EmailRecognizer")]


def test_code_drops_ner_results_but_keeps_patterns():
    analyzer, stats = FakeAnalyzer(), __import__("collections").Counter()
    found = detector.detect_text("Max a@b.c", "a.py", 1, analyzer=analyzer, languages=("en",), is_code=True,
                                 stats=stats)
    assert [c.entity_type for c in found] == ["EMAIL_ADDRESS"] and analyzer.languages == ["en"]
    assert stats["ner_dropped_on_code"] == 1
    prose = detector.detect_text("Max a@b.c", "a.md", 1, analyzer=FakeAnalyzer())
    assert sorted(c.entity_type for c in prose) == ["EMAIL_ADDRESS", "PERSON"]       # default: both, NER kept


def test_detect_blocks_routing_log_counts_only(isolated_logs):
    import json

    from hypeonoath.core.records import Block
    from hypeonoath.indexing.project_registry import ProjectInfo

    doc = _doc(Block("prose", EN_PARA, "cv.md"), Block("prose", "Max Mustermann", "cv.md"))
    code = Block("code_chunk", "# [P] a.py\nx = 1", "corpus/code/a.py", meta={"line_start": 1, "chunk_id": "c"})
    analyzer = FakeAnalyzer()
    detector.detect_blocks([doc], [code], {"a.py": ProjectInfo("p", "Proj", "Max's tool")}, analyzer=analyzer)
    record = json.loads((isolated_logs / "privacy_log_path.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "detect_routing" and record["blocks"] == {"en": 2, "code": 1}
    assert record["ner_dropped_on_code"] == 1 and "Max" not in json.dumps(record)


@pytest.mark.privacy_model
def test_routing_removes_cross_language_noise(analyzer, monkeypatch):
    text = "Born and raised in a small town, I have always been curious about how the things around me work."
    ner = lambda: [c for c in detector.detect_text(text, "x.md", 1, analyzer=analyzer,
                                                   languages=detector.route_languages(text))
                   if c.recognizer == "SpacyRecognizer"]
    assert ner() == []
    monkeypatch.setattr(config, "PRIVACY_LANGUAGE_ROUTING", False)
    assert ner() != []                                           # the de model's nonsense LOCATION


@pytest.mark.privacy_model
def test_routing_keeps_pattern_findings_on_english_text(analyzer):
    text = ("Please contact me by email at jane.doe@example.com or call +49 151 23456789 any time. "
            "My Steuer-ID / tax ID is 12345678995 and I am happy to share the documents.")
    languages = detector.route_languages(text)
    assert languages == ("en",)
    types = {c.entity_type for c in detector.detect_text(text, "x.md", 1, analyzer=analyzer, languages=languages)}
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "DE_TAX_ID"} <= types


@pytest.mark.privacy_model
def test_german_letter_routed_to_de_with_keyword_and_ner(analyzer):
    text = ("Sehr geehrter Herr Max Mustermann, wir freuen uns, Ihnen mitzuteilen, dass Sie zugelassen sind. "
            "Geburtsdatum: 12.03.1999, Geburtsort: Dehradun, Studienort ist Darmstadt.")
    assert detector.route_languages(text) == ("de",)
    found = {(c.entity_type, c.value) for c in detector.detect_text(text, "x.md", 1, analyzer=analyzer,
                                                                     languages=("de",))}
    assert ("DATE_OF_BIRTH", "12.03.1999") in found and any(t == "PERSON" for t, _ in found)


@pytest.mark.privacy_model
def test_code_has_no_ner_but_patterns_and_secrets(analyzer):
    from hypeonoath.core.records import Block
    from hypeonoath.indexing.project_registry import ProjectInfo

    code = Block("code_chunk", "# [Demo] a.py\n# written by Max Mustermann in Berlin\n"
                 "AUTHOR = 'max@example.org'\nKEY = 'AKIAIOSFODNN7EXAMPLE'", "corpus/code/a.py",
                 meta={"line_start": 1, "chunk_id": "c"})
    notebook_md = _doc(Block("prose", "This notebook was written by Max Mustermann for his thesis project.",
                             "corpus/code/n.ipynb"))
    found = detector.detect_blocks([notebook_md], [code], {"a.py": ProjectInfo("d", "Demo", "Max Mustermann's demo")},
                                   analyzer=analyzer)
    code_types = {c.entity_type for c in found if c.file == "corpus/code/a.py"}
    assert {"EMAIL_ADDRESS", "SECRET:AWS Access Key"} <= code_types
    assert not code_types & {"PERSON", "LOCATION", "NRP", "MONEY"}
    assert any(c.entity_type == "PERSON" for c in found if c.file == "corpus/code/n.ipynb")      # markdown = prose
    assert any(c.entity_type == "PERSON" for c in found if c.file == detector.project_strings_file())


# --- (v8) P-D34: no URL findings on code ------------------------------------------------------------------

class UrlAnalyzer:
    def analyze(self, text, language, score_threshold):
        return [FakeResult(0, 3, "URL", "UrlRecognizer"), FakeResult(4, 9, "EMAIL_ADDRESS", "EmailRecognizer")]


def test_code_drops_url_results_documents_keep_them():
    import collections

    stats = collections.Counter()
    code = detector.detect_text("x.io a@b.c", "a.py", 1, analyzer=UrlAnalyzer(), languages=("en",), is_code=True,
                                stats=stats)
    assert [c.entity_type for c in code] == ["EMAIL_ADDRESS"] and stats["url_dropped_on_code"] == 1
    doc = detector.detect_text("x.io a@b.c", "a.md", 1, analyzer=UrlAnalyzer(), languages=("en",))
    assert sorted(c.entity_type for c in doc) == ["EMAIL_ADDRESS", "URL"]
