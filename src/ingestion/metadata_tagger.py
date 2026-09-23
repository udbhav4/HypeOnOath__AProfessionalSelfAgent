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

project_name: config.PROJECT_NAME_MAP, keyed by path relative to
corpus/raw/. A file with no entry gets project_name: null -- the documented
default, not a gap. Exact taxonomy remains deferred to Phase 1 per the
sub-plan.

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

from . import config, frontmatter_utils
from .cleaner import resolve_document_target
from .logging_utils import log_json_line
from .manifest import save_manifest


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
    """Return the mapped project name for a file, or None if unmapped."""
    return config.PROJECT_NAME_MAP.get(rel_path)


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
    project_name = resolve_project_name(rel_path)

    data = _load_code_metadata()
    data[rel_path] = {
        "content_type": "code",
        "project_name": project_name,
        "date": date_str,
    }
    _save_code_metadata(data)

    code_path = config.CODE_DIR / rel_path
    return {
        "file": code_path.relative_to(config.CORPUS_DIR).as_posix(),
        "project_name": project_name,
        "date": date_str,
        "date_source": date_source,
    }


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

    save_manifest(config.MANIFEST_PATH, manifest)
    return manifest
