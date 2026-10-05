"""
Step 3.6: metadata tagging -- project_name and date only.

content_type is deliberately NOT written here: Step 3.4 (document-level
prose/structured_doc classification) was removed from Phase 0 entirely, not
relocated -- see plans/phase0-document-prep-subplan-v5.md, Section 3C. See
plans/phase0-steps-3.5-3.8-code-implementation-guideline-v2.md, Section 6/6A.

date: dual mechanism, per explicit user direction (2026-09-20) -- a per-file
custom override (config.CUSTOM_DATE_OVERRIDES) always wins when present;
otherwise the file's mtime in corpus/raw/ is used as the default. Both are
formatted as DD.MM.YYYY (config.DATE_FORMAT), never ISO -- the format itself
was confirmed with the user the same day, resolving what had been an open
question about how the date strings should be interpreted.

project_name (documents): config.PROJECT_NAME_MAP, keyed by path relative
to corpus/raw/. A document with no entry gets project_name: null -- the
documented default, not a gap.

project fields (code): config.CODE_PROJECTS + config.CODE_PROJECT_MAP (the
code-project registry, Phase 1 guideline v4 Section 3A.12). Every code entry
in _metadata.yaml carries project_key/project_name/project_description,
looked up from the registry. They are compulsory for code, but Phase 0 does
not enforce that -- an unmapped file is written with nulls and Phase 1's
project check (hypeonoath/indexing/project_registry.py) stops the indexing run,
listing every such file. Because the registry can change without any code
file changing, refresh_code_project_fields() re-applies it to every code
entry on every run (run_tagging otherwise only touches changed files).

Markdown files are tagged via frontmatter_utils (python-frontmatter) --
unlike cleaner.py, this step always writes real metadata, so there is no
empty-front-matter-block edge case to avoid. Code files cannot carry a YAML
front-matter block without risking their syntax, so they're tracked instead
in corpus/code/_metadata.yaml, per the sub-plan's Section 4.2.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.ingestion import config, frontmatter_utils
from hypeonoath.ingestion.cleaner import resolve_document_target
from hypeonoath.ingestion.manifest import save_manifest


def resolve_date(rel_path: str, raw_path: Path) -> tuple[str, str]:
    """
    Return (date_str, date_source) for a file, where date_source is
    'custom_override' or 'mtime'. The override always wins when present.
    """
    override = config.CUSTOM_DATE_OVERRIDES.get(rel_path)
    if override is not None:
        return override, "custom_override"

    mtime = datetime.fromtimestamp(raw_path.stat().st_mtime)
    return mtime.strftime(config.DATE_FORMAT), "mtime"


def resolve_project_name(rel_path: str) -> str | None:
    """Return the mapped project name for a document, or None if unmapped."""
    return config.PROJECT_NAME_MAP.get(rel_path)


def resolve_code_project(rel_path: str) -> str | None:
    """
    Return the project_key for a code file from config.CODE_PROJECT_MAP, or
    None if unmapped. Keys ending in "/" are folder prefixes (umbrella
    projects); an exact file key is just the longest possible match, so
    "longest matching key wins" covers both cases.
    """
    best_key: str | None = None
    best_len = -1
    for map_key, project_key in config.CODE_PROJECT_MAP.items():
        matches = rel_path.startswith(map_key) if map_key.endswith("/") else rel_path == map_key
        if matches and len(map_key) > best_len:
            best_key, best_len = project_key, len(map_key)
    return best_key


def code_project_fields(rel_path: str) -> dict:
    """
    project_key/project_name/project_description for a code file. All three
    are None when the file is unmapped or its key is missing from
    CODE_PROJECTS -- Phase 1's project check reports both cases.
    """
    project_key = resolve_code_project(rel_path)
    project = config.CODE_PROJECTS.get(project_key) if project_key else None
    return {
        "project_key": project_key,
        "project_name": project.get("name") if project else None,
        "project_description": project.get("description") if project else None,
    }


def tag_markdown_file(path: Path, rel_path: str, raw_path: Path) -> dict:
    """Write project_name/date into a Markdown file's front-matter."""
    date_str, date_source = resolve_date(rel_path, raw_path)
    project_name = resolve_project_name(rel_path)

    post = frontmatter_utils.load_document(path)
    post["project_name"] = project_name
    post["date"] = date_str
    frontmatter_utils.save_document(path, post)

    return {
        "file": path.relative_to(config.CORPUS_DIR).as_posix(),
        "project_name": project_name,
        "date": date_str,
        "date_source": date_source,
    }


def _load_code_metadata() -> dict:
    if not config.CODE_METADATA_PATH.exists():
        return {}
    with config.CODE_METADATA_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_code_metadata(data: dict) -> None:
    config.CODE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CODE_METADATA_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)


def tag_code_entry(rel_path: str, raw_path: Path) -> dict:
    """
    Update this code file's entry in corpus/code/_metadata.yaml. Content is
    never edited in place, per the sub-plan's Section 4.2 -- injecting
    metadata into a source file risks breaking syntax tree-sitter (Phase 1)
    would need to parse.
    """
    date_str, date_source = resolve_date(rel_path, raw_path)
    project = code_project_fields(rel_path)

    data = _load_code_metadata()
    data[rel_path] = {"content_type": "code", "date": date_str, **project}
    _save_code_metadata(data)

    code_path = config.CODE_DIR / rel_path
    return {
        "file": code_path.relative_to(config.CORPUS_DIR).as_posix(),
        "project_key": project["project_key"],
        "project_name": project["project_name"],
        "date": date_str,
        "date_source": date_source,
    }


def refresh_code_project_fields(manifest: dict) -> list[str]:
    """
    Re-apply the code-project registry to EVERY code entry in
    _metadata.yaml, not just the files changed this run. Without this, a
    registry edit (new project, new description) would never reach files
    whose content is unchanged, and Phase 1's stale check would keep asking
    for a Phase 0 re-run that changes nothing (guideline v4, Section 16.2
    step 4). date is left untouched. Returns the rel_paths that changed.
    """
    data = _load_code_metadata()
    changed: list[str] = []
    for rel_path, entry in manifest.items():
        if entry.get("destination_category") != "code" or rel_path not in data:
            continue
        current = data[rel_path] or {}
        project = code_project_fields(rel_path)
        if any(current.get(k) != v for k, v in project.items()):
            data[rel_path] = {**current, **project}
            changed.append(rel_path)
            log_json_line(
                config.TAGGING_LOG_PATH,
                file=(config.CODE_DIR / rel_path).relative_to(config.CORPUS_DIR).as_posix(),
                status="code_project_refreshed",
                project_key=project["project_key"],
            )
    if changed:
        _save_code_metadata(data)
    return changed


def run_tagging(manifest: dict, raw_rel_paths: list[str]) -> dict:
    """
    Tag exactly the files corresponding to raw_rel_paths (the files
    router.py/converter.py/cleaner.py just processed this run). Scoping to
    this list is what makes tagging re-run-safe -- unchanged files are
    never re-tagged.
    """
    for rel_path_str in raw_rel_paths:
        entry = manifest.get(rel_path_str)
        if entry is None:
            continue

        raw_path = config.RAW_DIR / rel_path_str
        category = entry.get("destination_category")

        if category == "document":
            target = resolve_document_target(rel_path_str, entry)
            if target is None:
                continue
            path, _is_converted = target
            result = tag_markdown_file(path, rel_path_str, raw_path)
        elif category == "code":
            result = tag_code_entry(rel_path_str, raw_path)
        else:
            continue  # quarantine -- not tagged

        entry["tagged"] = True
        log_json_line(config.TAGGING_LOG_PATH, **result)

    refresh_code_project_fields(manifest)
    save_manifest(config.MANIFEST_PATH, manifest)
    return manifest
