"""
Shared pytest fixtures.

The real nomic tokenizer/model is used wherever token budgets matter (the
whole point of the budgets is that they match what the model reads); it is
loaded once per session from the local Hugging Face cache.
"""
from __future__ import annotations

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
    """Never write test logs/reports into the real corpus/.logs/."""
    from hypeonoath.core import config

    for name in ("CODE_CHUNKING_LOG_PATH", "DOCUMENT_LOADING_LOG_PATH", "INDEXING_LOG_PATH", "SCAN_WARNINGS_PATH"):
        monkeypatch.setattr(config, name, tmp_path / "logs" / f"{name.lower()}.jsonl")
    return tmp_path / "logs"
