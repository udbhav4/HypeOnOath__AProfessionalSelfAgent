"""Code-project registry, corpus-preparation side: file -> project resolution,
re-tagging of unchanged files, and the real registry values (3A.12, Section 16)."""
from __future__ import annotations

import yaml

from hypeonoath.ingestion import config as ingestion_config
from hypeonoath.ingestion import metadata_tagger


def test_umbrella_prefix_and_longest_match(code_map):
    assert metadata_tagger.resolve_code_project("proj/a.py") == "proj"
    assert metadata_tagger.resolve_code_project("proj/sub/b.ts") == "proj"
    assert metadata_tagger.resolve_code_project("proj/special.py") == "solo"
    assert metadata_tagger.resolve_code_project("other.py") is None
    assert metadata_tagger.resolve_code_project("projX/a.py") is None    # prefix needs the "/"


def test_refresh_reapplies_registry_to_unchanged_files(code_map, tmp_path, monkeypatch):
    path = tmp_path / "_metadata.yaml"
    path.write_text(yaml.safe_dump({"proj/a.py": {"content_type": "code", "date": "02.02.2026", "project_name": None}}))
    monkeypatch.setattr(ingestion_config, "CODE_METADATA_PATH", path)
    monkeypatch.setattr(ingestion_config, "TAGGING_LOG_PATH", tmp_path / "tagging.jsonl")
    changed = metadata_tagger.refresh_code_project_fields({"proj/a.py": {"destination_category": "code"}})
    assert changed == ["proj/a.py"]
    entry = yaml.safe_load(path.read_text())["proj/a.py"]
    assert entry == {"content_type": "code", "date": "02.02.2026", "project_key": "proj",
                     "project_name": "Proj", "project_description": "An umbrella project."}
    assert metadata_tagger.refresh_code_project_fields({"proj/a.py": {"destination_category": "code"}}) == []


def test_real_registry_has_verbatim_gliimr_entry():
    """Guideline v4 Section 16.1 -- values must stay verbatim."""
    gliimr = ingestion_config.CODE_PROJECTS["gliimr"]
    assert gliimr["name"] == "Gliimr: A personal health intelligence app"
    assert gliimr["description"] == (
        "On-device personal health intelligence app. No cloud. No hallucinations.\n"
        "Gliimr is a React Native health app that passively senses how you live, extracts meaning from "
        "what you say, and maintains a clinically grounded probabilistic model of your health - entirely "
        "on your phone, entirely in private, entirely explainable."
    )
    assert ingestion_config.CODE_PROJECT_MAP["agent.ts"] == "gliimr"
