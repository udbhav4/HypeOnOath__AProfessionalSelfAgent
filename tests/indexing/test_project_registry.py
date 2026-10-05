"""Code-project registry, indexing side: validation before any chunking and
the project_summary chunks (3A.12, Section 16)."""
from __future__ import annotations

import pytest

from hypeonoath.indexing import project_registry as reg
from hypeonoath.ingestion import metadata_tagger

from tests.conftest import REGISTRY


def metadata_for(*rel_paths):
    return {rel: {"content_type": "code", "date": "01.01.2026", **metadata_tagger.code_project_fields(rel)}
            for rel in rel_paths}


def test_umbrella_files_share_project_and_one_summary(code_map):
    files = ["proj/a.py", "proj/b.ts"]
    resolved = reg.validate_code_projects(files, metadata_for(*files), REGISTRY)
    assert {info.key for info in resolved.values()} == {"proj"}
    summaries = reg.build_project_summary_chunks(resolved)
    assert len(summaries) == 1
    text = summaries[0].text
    assert text == "Project: Proj\nDescription: An umbrella project.\nCode files: proj/a.py, proj/b.ts"
    s = summaries[0]
    assert (s.content_type, s.source_document, s.project_key) == ("project_summary", "corpus/code/_projects/proj", "proj")
    assert reg.build_project_summary_chunks(resolved)[0].chunk_id == s.chunk_id     # stable


def test_missing_projects_listed_all_at_once(code_map):
    files = ["x.py", "y.py", "proj/a.py"]
    with pytest.raises(reg.ProjectValidationError) as err:
        reg.validate_code_projects(files, metadata_for(*files), REGISTRY)
    failures = err.value.failures
    assert len(failures) == 2 and failures[0].startswith("x.py") and failures[1].startswith("y.py")
    assert "CODE_PROJECT_MAP['x.py']" in failures[0]
    assert "existing index was not touched" in str(err.value)


@pytest.mark.parametrize("bad", [{"name": "P", "description": "   "}, {"name": "", "description": "d"}])
def test_empty_name_or_description_names_the_project_once(code_map, monkeypatch, bad):
    registry = {"proj": bad}
    meta = {rel: {"project_key": "proj", "project_name": bad["name"], "project_description": bad["description"]}
            for rel in ("proj/a.py", "proj/b.py")}
    with pytest.raises(reg.ProjectValidationError) as err:
        reg.validate_code_projects(list(meta), meta, registry)
    assert len(err.value.failures) == 1 and "'proj'" in err.value.failures[0]


def test_unknown_key(code_map):
    meta = {"a.py": {"project_key": "ghost"}}
    with pytest.raises(reg.ProjectValidationError, match="not defined"):
        reg.validate_code_projects(["a.py"], meta, REGISTRY)


def test_stale_metadata_asks_for_corpus_prep_rerun(code_map):
    meta = metadata_for("proj/a.py")
    meta["proj/a.py"]["project_description"] = "old text"
    with pytest.raises(reg.ProjectValidationError, match="re-run corpus preparation"):
        reg.validate_code_projects(["proj/a.py"], meta, REGISTRY)
