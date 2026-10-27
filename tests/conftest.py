"""
Shared pytest fixtures.

The real nomic tokenizer/model is used wherever token budgets matter (the
whole point of the budgets is that they match what the model reads); it is
loaded once per session from the local Hugging Face cache.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "code"

# A small code-project registry used by the ingestion (mapping) and indexing
# (validation) tests -- see the code_map fixture.
REGISTRY = {
    "proj": {"name": "Proj", "description": "An umbrella project."},
    "solo": {"name": "Solo", "description": "One file."},
}


@pytest.fixture
def code_map(monkeypatch):
    from hypeonoath.ingestion import config as ingestion_config

    monkeypatch.setattr(ingestion_config, "CODE_PROJECTS", REGISTRY)
    monkeypatch.setattr(ingestion_config, "CODE_PROJECT_MAP", {
        "proj/": "proj",
        "proj/special.py": "solo",       # more specific than the folder prefix
    })


# A small project + document registry for the document-registry tests (plan v2, 6) -- see doc_registry.
DOC_PROJECTS = {
    "chat": {"name": "Chatbot", "description": "A chatbot project."},
    "app": {"name": "App", "description": "An app project."},
}
DOCUMENTS = {
    "design.md": {"description": "Design document of the chatbot.", "project": "chat"},
    "cv.pdf": {"description": "My CV.", "project": None},
}


@pytest.fixture
def doc_registry(monkeypatch):
    """Fake registry in core.registry (what serving and the checks read), so tests never depend on the owner's text."""
    from hypeonoath.core import registry

    monkeypatch.setattr(registry, "CODE_PROJECTS", copy.deepcopy(DOC_PROJECTS))   # tests may edit entries
    monkeypatch.setattr(registry, "DOCUMENTS", copy.deepcopy(DOCUMENTS))
    return registry


@pytest.fixture(scope="session")
def embedder_module():
    from hypeonoath.core import embedder

    return embedder


@pytest.fixture(scope="session")
def count_tokens(embedder_module):
    return embedder_module.count_tokens


@pytest.fixture(scope="session")
def code_chunker_instance(embedder_module):
    from hypeonoath.indexing.code_chunker import CodeChunker, self_check

    assert self_check() == [], "some mapped tree-sitter grammars are unavailable"
    return CodeChunker(embedder_module.count_tokens, embedder_module.get_tokenizer())


@pytest.fixture
def project():
    from hypeonoath.indexing.project_registry import ProjectInfo

    return ProjectInfo("demo", "Demo Project", "A demo project used by the tests.")


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """
    Never write test logs/reports into the real corpus/.logs/, and never read
    or write the owner's real privacy decisions (corpus/.privacy/).
    """
    from hypeonoath.core import config

    for name in ("CODE_CHUNKING_LOG_PATH", "DOCUMENT_LOADING_LOG_PATH", "INDEXING_LOG_PATH", "SCAN_WARNINGS_PATH",
                 "PRIVACY_LOG_PATH"):
        monkeypatch.setattr(config, name, tmp_path / "logs" / f"{name.lower()}.jsonl")
    monkeypatch.setattr(config, "PRIVACY_DIR", tmp_path / "privacy")
    monkeypatch.setattr(config, "PRIVACY_DECISIONS_PATH", tmp_path / "privacy" / "decisions.json")
    monkeypatch.setattr(config, "PRIVACY_SCAN_STATE_PATH", tmp_path / "privacy" / "scan_state.json")
    monkeypatch.setattr(config, "PRIVACY_BUILD_LOCK_PATH", tmp_path / "privacy" / "build.lock")
    return tmp_path / "logs"
