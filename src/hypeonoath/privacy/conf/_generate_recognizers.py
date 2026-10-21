"""
Generate privacy/conf/recognizers.yaml from the INSTALLED Presidio's own
default_recognizers.yaml (privacy plan v3, 6.2; P-D3).

    python -m hypeonoath.privacy.conf._generate_recognizers

Re-run after every Presidio upgrade and commit the output: Presidio adds
recognizers between versions, and a hand copy would silently miss them.

What it does to Presidio's list:
  * every recognizer ENABLED, for every country (no supported_countries),
    except vehicle plates / postcodes (P-D35) and US driver licences (P-D51),
    via _EXCLUDED_BY_DECISION;
  * every recognizer registered for PRIVACY_LANGUAGES (en + de), keeping a
    recognizer's own per-language context words where Presidio has them
    (a language without its own list borrows the English one);
  * DROPS recognizers that send text to an external service -- every class
    under presidio_analyzer.predefined_recognizers.third_party (LangExtract,
    Azure AI Language, Azure Health De-identification, ...) -- sending
    private documents out to find private data would defeat the purpose;
  * DROPS recognizers excluded by an owner decision (_EXCLUDED_BY_DECISION);
  * removes duplicate entries (Presidio lists some recognizers twice);
  * appends the hand-written keyword recognizers from custom_recognizers.yaml.
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from hypeonoath.core import config

CONF_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = CONF_DIR / "recognizers.yaml"
CUSTOM_PATH = CONF_DIR / "custom_recognizers.yaml"

_EXTERNAL_MODULE_MARKER = ".predefined_recognizers.third_party."

# Local, but a second NER model needing its own model download + config;
# P-D5 starts with the spaCy lg models only (GLiNER / trf are the upgrade path).
_EXCLUDED_BY_DECISION = {
    "HuggingFaceNerRecognizer": "P-D5: spaCy lg NER only",
    # P-D35: short letter/digit patterns that match ordinary text ("by a 3", any 5 digits);
    # a candidate's documents hold no plates, and addresses are found by spaCy LOCATION.
    **{name: "P-D35: no vehicle plates / postcodes" for name in (
        "DeKfzRecognizer", "UkVehicleRegistrationRecognizer", "InVehicleRegistrationRecognizer",
        "NgVehicleRegistrationRecognizer", "TrLicensePlateRecognizer", "DePlzRecognizer",
        "UkPostcodeRecognizer")},
    # P-D51: its digits pattern is boosted by the context word "lic" inside "Application";
    # application numbers are found by ApplicantIdKeywordRecognizer instead.
    "UsLicenseRecognizer": "P-D51: no US driver licences",
}


class _NoAliasDumper(yaml.SafeDumper):
    """Write shared context lists out in full (no &id001 anchors) -- easier to review."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _presidio_default_path() -> Path:
    import presidio_analyzer

    return Path(presidio_analyzer.__file__).resolve().parent / "conf" / "default_recognizers.yaml"


def _recognizer_classes() -> dict[str, type]:
    from presidio_analyzer.recognizer_registry.recognizers_loader_utils import RecognizerListLoader

    import presidio_analyzer.predefined_recognizers  # noqa: F401 -- registers every subclass

    return {cls.__name__: cls for cls in RecognizerListLoader.get_all_existing_recognizers()}


def _languages(entry: dict[str, Any], languages: tuple[str, ...]) -> list[dict[str, Any]] | list[str]:
    """[en, de]; keeps Presidio's per-language context words when it has them."""
    declared = entry.get("supported_languages")
    if not declared or isinstance(declared[0], str):
        return list(languages)
    contexts = {item["language"]: item.get("context") for item in declared if isinstance(item, dict)}
    fallback = contexts.get("en") or next((c for c in contexts.values() if c), None)
    out: list[dict[str, Any]] = []
    for language in languages:
        context = contexts.get(language) or fallback
        out.append({"language": language, **({"context": context} if context else {})})
    return out


def build_config(default_conf: dict[str, Any], custom: list[dict[str, Any]], classes: dict[str, type],
                 languages: tuple[str, ...] = config.PRIVACY_LANGUAGES) -> tuple[dict[str, Any], list[str]]:
    """Pure transform (tested): Presidio's default config -> ours + notes on what was dropped."""
    recognizers: list[dict[str, Any]] = []
    notes: list[str] = []
    seen: set[str] = set()
    for raw in default_conf.get("recognizers", []):
        entry = {"name": raw} if isinstance(raw, str) else dict(raw)
        name = entry["name"]
        class_name = entry.get("class_name", name)
        if name in seen:
            continue
        seen.add(name)
        if name in _EXCLUDED_BY_DECISION:
            notes.append(f"{name}: off ({_EXCLUDED_BY_DECISION[name]})")
            continue
        cls = classes.get(class_name)
        if cls is not None and _EXTERNAL_MODULE_MARKER in f".{cls.__module__}.":
            notes.append(f"{name}: off (calls an external service)")
            continue
        if entry.get("type", "predefined") == "predefined":
            entry.pop("enabled", None)                    # P-D3: everything local on
            entry.pop("config_path", None)
            entry["supported_languages"] = _languages(entry, languages)
        recognizers.append(entry)
    recognizers += custom
    out = {
        "supported_languages": list(languages),
        "global_regex_flags": default_conf.get("global_regex_flags", 26),
        "recognizers": recognizers,
    }
    return out, notes


def generate(output: Path = OUTPUT_PATH) -> list[str]:
    default_conf = yaml.safe_load(_presidio_default_path().read_text(encoding="utf-8"))
    custom = yaml.safe_load(CUSTOM_PATH.read_text(encoding="utf-8")) or []
    conf, notes = build_config(default_conf, custom, _recognizer_classes())
    version = metadata.version("presidio-analyzer")
    header = [
        "# GENERATED by hypeonoath.privacy.conf._generate_recognizers -- do not edit by hand.",
        f"# Source: presidio-analyzer {version} default_recognizers.yaml",
        "# Every local recognizer enabled for en + de, every country (P-D3), except plates / postcodes (P-D35) and US driver licences (P-D51).",
        "# Edit custom_recognizers.yaml for keyword recognizers, then re-generate.",
        *[f"# Dropped: {note}" for note in notes],
    ]
    body = yaml.dump(conf, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True, width=120)
    output.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
    return notes


def main() -> int:
    notes = generate()
    print(f"Wrote {OUTPUT_PATH}")
    for note in notes:
        print(f"  dropped {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
