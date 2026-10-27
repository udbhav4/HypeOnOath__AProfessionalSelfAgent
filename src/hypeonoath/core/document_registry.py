"""
Document registry lookup + checks (document-registry plan v2, 3.2; in core so serving may use it, invariant I5).

registry.DOCUMENTS gives every document a compulsory description and an
optional project (a key of registry.CODE_PROJECTS, the one project registry for
code and documents). serving/prompt.py shows them when the prompt is built;
indexing/build_index.py stops the build when an indexed document has no valid
entry (DR-9). Nothing here is embedded or stored in the index (DR-5).

Indexed paths (corpus/converted/CV_Udbhav.md, corpus/documents/README.md) map
to their registry key (CV_Udbhav.pdf, README.md) by file stem; stems are
unique, and validate_documents() checks that they stay so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from hypeonoath.core.private_values import find_private_spans, load_private_values
from hypeonoath.core import registry

DESCRIPTION_MAX_CHARS = 300   # DR-6
CODE_PREFIX = "corpus/code/"  # code files and project summaries (corpus/code/_projects/<key>) are not documents


@dataclass(frozen=True)
class DocumentInfo:
    description: str
    project_key: str | None
    project_name: str | None
    project_description: str | None


def _stem(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).stem


def document_key(source_document: str) -> str | None:
    """The DOCUMENTS key of an indexed document; None for code, project summaries and unknown files."""
    if source_document.replace("\\", "/").startswith(CODE_PREFIX):
        return None
    stem = _stem(source_document)
    return next((key for key in registry.DOCUMENTS if _stem(key) == stem), None)


def document_info(source_document: str) -> DocumentInfo | None:
    """Description + project fields of an indexed document, or None when it has no entry."""
    key = document_key(source_document)
    if key is None:
        return None
    entry = registry.DOCUMENTS[key]
    project_key = entry.get("project")
    project = registry.CODE_PROJECTS.get(project_key) if project_key else None
    return DocumentInfo(
        description=entry.get("description") or "",
        project_key=project_key if project else None,
        project_name=project.get("name") if project else None,
        project_description=project.get("description") if project else None,
    )


def validate_documents(source_documents: list[str]) -> list[str]:
    """
    Every failure for the given indexed documents (empty list = OK): missing
    entry, empty or too long description, unknown project key, duplicate
    stem, private value in a description / project name / project
    description. Messages name the key and field, never a matched value.
    """
    failures: list[str] = []

    stems: dict[str, str] = {}
    for key in registry.DOCUMENTS:
        if _stem(key) in stems:
            failures.append(f"DOCUMENTS[{key!r}]: same file stem as DOCUMENTS[{stems[_stem(key)]!r}]")
        stems.setdefault(_stem(key), key)

    values = load_private_values()
    used: set[str] = set()         # each entry checked once
    for path in sorted(set(source_documents)):
        key = document_key(path)
        if key is None:
            failures.append(f"{path}: no entry. Add DOCUMENTS[<raw file name>] = "
                            "{'description': '...', 'project': None} in src/hypeonoath/core/registry.py")
            continue
        used.add(key)

    for key in sorted(used):
        entry = registry.DOCUMENTS[key]
        description = (entry.get("description") or "").strip()
        if not description:
            failures.append(f"DOCUMENTS[{key!r}]: empty description")
        elif len(description) > DESCRIPTION_MAX_CHARS:
            failures.append(f"DOCUMENTS[{key!r}]: description over {DESCRIPTION_MAX_CHARS} characters")
        texts = [("description", description)]
        project_key = entry.get("project")
        if project_key:
            project = registry.CODE_PROJECTS.get(project_key)
            if project is None:
                failures.append(f"DOCUMENTS[{key!r}]: project {project_key!r} is not defined in CODE_PROJECTS")
            else:
                texts += [("project name", project.get("name") or ""),
                          ("project description", project.get("description") or "")]
        for field, text in texts:
            if values and find_private_spans(text, values):
                failures.append(f"DOCUMENTS[{key!r}]: {field} contains a private value")
    return failures
