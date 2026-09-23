"""
Shared YAML front-matter load/save helpers for cleaner.py and
metadata_tagger.py.

Wraps `python-frontmatter` so both steps share one implementation instead of
each hand-rolling YAML-block parsing -- per the guideline's Section 6
sequencing note (Step 3.5 needs to preserve an existing front-matter block
while cleaning the body; Step 3.6 needs to write to it).

API verified by direct inspection 2026-09-20 against the installed version
(python-frontmatter 1.3.0), the same discipline already applied to docling's
API in converter.py -- this was not assumed from documentation alone:
  - frontmatter.load(path) -> Post, with .metadata (dict) and .content (str)
  - frontmatter.dumps(post) -> str, re-emitting the '---' YAML block + body,
    with no trailing newline (confirmed by direct test against a real
    corpus file, corpus/documents/README.md) -- callers must add one.
A round trip (load -> mutate .metadata -> dumps -> loads again) was tested
directly against that real file and preserves body content exactly, with no
corruption of the pre-existing Markdown structure.
"""
from __future__ import annotations

from pathlib import Path

import frontmatter


def load_document(path: Path) -> frontmatter.Post:
    """Load a Markdown file's front-matter + body as one Post object."""
    return frontmatter.load(path)


def save_document(path: Path, post: frontmatter.Post) -> None:
    """
    Write a Post back to disk, re-emitting its front-matter block followed
    by the body. Adds the trailing newline frontmatter.dumps() omits.
    """
    path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
