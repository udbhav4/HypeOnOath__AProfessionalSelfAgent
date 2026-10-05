"""
Code-project registry check + project-summary chunks (guideline v4, 3A.12).

Every code file must carry a compulsory project_key/project_name/
project_description. The registry itself (CODE_PROJECTS + CODE_PROJECT_MAP)
lives in Phase 0's hypeonoath/ingestion/config.py; Phase 0's metadata_tagger copies
the resolved fields into corpus/code/_metadata.yaml. This module is the FIRST
step of every indexing run:

  * validate_code_projects() collects EVERY failure (not just the first) and
    raises ProjectValidationError, so the user gets one fix-and-re-run cycle.
    Nothing has been read/written by the indexer at that point, so a stopped
    run leaves the previous index untouched.
  * build_project_summary_chunks() makes one searchable chunk per project, so
    project-level questions ("what is Gliimr?") can match the description,
    which is otherwise only un-embedded metadata on code chunks.

Why stop here but only warn in secret_scan.py: a missing project is a
configuration error the user controls and fixes in seconds; a scan hit is a
probabilistic guess about content (guideline 3A.12 vs 3A.13).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from hypeonoath.core import config
from hypeonoath.core.records import Chunk, make_chunk_id
from hypeonoath.ingestion import config as ingestion_config
from hypeonoath.ingestion.metadata_tagger import resolve_code_project


@dataclass(frozen=True)
class ProjectInfo:
    key: str
    name: str
    description: str


class ProjectValidationError(Exception):
    """Raised with every failing code file / project, never just the first."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__(self.format())

    def format(self) -> str:
        lines = [
            f"Indexing stopped: {len(self.failures)} code-project problem(s). "
            "Every code file needs a project name + description (guideline 3A.12).",
            *[f"  - {failure}" for failure in self.failures],
            "Fix src/hypeonoath/ingestion/config.py (CODE_PROJECTS / CODE_PROJECT_MAP), then re-run "
            "corpus preparation (python -m hypeonoath.ingestion.prepare_corpus), then this indexer. "
            "The existing index was not touched.",
        ]
        return "\n".join(lines)


def load_code_metadata(path: Path | None = None) -> dict:
    path = config.CODE_METADATA_PATH if path is None else path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def validate_code_projects(
    code_rel_paths: list[str],
    metadata: dict | None = None,
    registry: dict[str, dict[str, str]] | None = None,
) -> dict[str, ProjectInfo]:
    """
    Check every code file (rel paths under corpus/code/, already filtered by
    the exclusion rules) has a valid, up-to-date project. Returns
    {rel_path: ProjectInfo}; raises ProjectValidationError listing ALL
    failures otherwise.
    """
    metadata = load_code_metadata() if metadata is None else metadata
    registry = ingestion_config.CODE_PROJECTS if registry is None else registry

    failures: list[str] = []
    resolved: dict[str, ProjectInfo] = {}
    bad_projects: set[str] = set()

    for rel_path in sorted(code_rel_paths):
        entry = metadata.get(rel_path) or {}
        key = _clean(entry.get("project_key"))
        if not key:
            failures.append(
                f"{rel_path}: no project. Add CODE_PROJECT_MAP[{rel_path!r}] = '<project_key>' "
                "and CODE_PROJECTS['<project_key>'] = {'name': '...', 'description': '...'}"
            )
            continue

        project = registry.get(key)
        name = _clean(project.get("name")) if project else ""
        description = _clean(project.get("description")) if project else ""
        if project is None:
            failures.append(f"{rel_path}: project_key {key!r} is not defined in CODE_PROJECTS")
            continue
        if not name or not description:
            if key not in bad_projects:  # report a broken project once, not per file
                bad_projects.add(key)
                missing = " and ".join(f for f, v in (("name", name), ("description", description)) if not v)
                failures.append(f"project {key!r}: empty {missing} in CODE_PROJECTS")
            continue

        # Stale check: _metadata.yaml must match the registry exactly, else the
        # registry was edited without a Phase 0 re-run.
        if (
            resolve_code_project(rel_path) != key
            or entry.get("project_name") != project.get("name")
            or entry.get("project_description") != project.get("description")
        ):
            failures.append(f"{rel_path}: metadata out of date with the registry -- re-run corpus preparation (python -m hypeonoath.ingestion.prepare_corpus)")
            continue

        resolved[rel_path] = ProjectInfo(key=key, name=project["name"], description=project["description"])

    if failures:
        raise ProjectValidationError(failures)
    return resolved


def project_summary_text(info: ProjectInfo, code_files: list[str]) -> str:
    return (
        f"Project: {info.name}\n"
        f"Description: {info.description}\n"
        f"Code files: {', '.join(sorted(code_files))}"
    )


def build_project_summary_chunks(resolved: dict[str, ProjectInfo]) -> list[Chunk]:
    """One `project_summary` chunk per project (guideline 3A.12)."""
    files_by_project: dict[str, list[str]] = {}
    info_by_key: dict[str, ProjectInfo] = {}
    for rel_path, info in resolved.items():
        files_by_project.setdefault(info.key, []).append(rel_path)
        info_by_key[info.key] = info

    chunks: list[Chunk] = []
    for key in sorted(files_by_project):
        info = info_by_key[key]
        text = project_summary_text(info, files_by_project[key])
        source = f"{config.PROJECT_SUMMARY_SOURCE_PREFIX}{key}"
        chunks.append(
            Chunk(
                # Content-derived: changes only when name/description/file list change.
                chunk_id=make_chunk_id(source, 0, len(text.encode("utf-8")), text),
                text=text,
                source_document=source,
                content_type="project_summary",
                project_key=key,
                project_name=info.name,
                project_description=info.description,
                line_start=1,
                line_end=text.count("\n") + 1,
            )
        )
    return chunks
