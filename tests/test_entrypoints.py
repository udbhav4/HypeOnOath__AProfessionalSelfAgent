"""Entry points and package boundaries.

  * the documented `python -m hypeonoath...` commands exist and parse their CLI
    with no PYTHONPATH set (the package is installed, `pip install -e .`);
  * the pre-restructure module names are gone;
  * the serving side imports ONLY hypeonoath.core -- never the indexing or
    ingestion code or their heavy dependencies. This is the reason for the
    core / indexing / serving split (the deployed backend must stay small).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ENTRY_POINTS = [
    "hypeonoath.ingestion.prepare_corpus",
    "hypeonoath.indexing.build_index",
    "hypeonoath.serving.chat",
    "hypeonoath.privacy.cli",
]


def _clean_env() -> dict[str, str]:
    """Subprocess env without PYTHONPATH, so imports must come from the installed package."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_has_main(module):
    assert callable(importlib.import_module(module).main)


@pytest.mark.parametrize("old", [
    "pipeline", "pipeline.chat", "pipeline.build_index", "pipeline.cli_chat", "pipeline.run_phase1_index",
    "ingestion", "ingestion.prepare_corpus", "ingestion.run_phase0",
])
def test_old_names_are_gone(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)


@pytest.mark.parametrize("command", [
    [sys.executable, "-m", "hypeonoath.serving.chat", "--help"],
    [sys.executable, "eval/run_eval.py", "--help"],
    [sys.executable, "-m", "hypeonoath.indexing.build_index", "--help"],
    [sys.executable, "-m", "hypeonoath.privacy.cli", "--help"],
])
def test_cli_help_runs_without_pythonpath(command):
    result = subprocess.run(command, cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


# Modules the serving side must never load (directly or transitively).
FORBIDDEN_FOR_SERVING = [
    "hypeonoath.indexing", "hypeonoath.ingestion", "hypeonoath.privacy",
    "chonkie", "tree_sitter_language_pack", "tree_sitter", "detect_secrets",
    "docling", "docling_core", "nbformat", "markdown_it", "frontmatter", "fitz", "docx",
    "presidio_analyzer", "spacy", "yaml",
]


def test_serving_imports_only_core():
    probe = (
        "import sys\n"
        "import hypeonoath.serving.chat, hypeonoath.serving.retriever, "
        "hypeonoath.serving.prompt, hypeonoath.serving.llm_client, hypeonoath.serving.guard\n"
        f"forbidden = {FORBIDDEN_FOR_SERVING!r}\n"
        "loaded = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden))\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=_clean_env(),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"serving pulled in: {result.stdout.strip()}"


def test_console_script_hypeonoath_privacy_installed():
    """`hypeonoath-privacy --help` works as an installed script (no PYTHONPATH)."""
    script = Path(sys.executable).with_name("hypeonoath-privacy" + (".exe" if os.name == "nt" else ""))
    assert script.exists(), "re-run: pip install -e ."
    result = subprocess.run([str(script), "--help"], cwd=ROOT, env=_clean_env(), capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0 and "check-git" in result.stdout


def test_private_values_imports_stdlib_and_core_config_only():
    """core.private_values must stay import-light: serving, indexing and eval all use it (plan 4.1)."""
    probe = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import hypeonoath.core.private_values\n"
        "new = sorted(set(sys.modules) - before)\n"
        "stdlib = sys.stdlib_module_names\n"
        "bad = [m for m in new if m.split('.')[0] not in stdlib and m not in "
        "('hypeonoath', 'hypeonoath.core', 'hypeonoath.core.config', 'hypeonoath.core.paths', "
        "'hypeonoath.core.private_values')]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=_clean_env(),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"core.private_values pulled in: {result.stdout.strip()}"


def test_pinned_versions_match_installed():
    """config.PINNED_VERSIONS == what is installed (a version change must be a deliberate re-index)."""
    from importlib import metadata

    from hypeonoath.core import config

    mismatched = {}
    for name, pinned in config.PINNED_VERSIONS.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue          # optional extra not installed in this venv
        if installed != pinned:
            mismatched[name] = (pinned, installed)
    assert mismatched == {}
