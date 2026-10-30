"""
Gold workflow G3-G4 (eval plan v15, 3.3; V14, D49).

review-gold: one draft at a time with the checklist; y accept, n reject, e open in an editor,
q save and quit. Each answer is written to drafts.yaml at once (only the entry's `status:` line
changes, comments and layout stay), so the review is resumable: entries already approved /
rejected are skipped next time.

freeze: every draft reviewed + the approved set passes V1-V13 -> eval/questions.yaml with
meta.privacy_fp and meta.question_set_sha256. The old flat questions.yaml (pre-harness) is
archived once as questions_v0.yaml (not scored).
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

from hypeonoath.evaluation import paths
from hypeonoath.evaluation.questions import (ValidationContext, current_privacy_fp, load_file,
                                             question_set_sha256, redact_messages, validate)

CHECKLIST = """\
Checklist:
 1. Realistic and in scope (a recruiter would ask it; not ambiguous).
 2. Category, doc_group and edge tags are right.
 3. Each part's `expected` is right: absent from the documents -> no_answer without a tag; in the documents
    but made private by you -> no_answer WITH private_label; everything else -> answer.
 4. Each answer part's reference_answer is accurate (not wrong, outdated, over-claiming); key points atomic + complete.
 5. Every evidence alternative actually states its key point.
 6. The parts together cover the whole question.
Keys: y accept · n reject · e open in editor · q save and quit"""

_ENTRY_RE = re.compile(r"^- id:\s*['\"]?(?P<id>[^\s'\"#]+)", re.MULTILINE)
_STATUS_RE = re.compile(r"^(?P<indent>  )status:[ \t]*\S+[ \t]*$", re.MULTILINE)


def entry_span(text: str, qid: str) -> tuple[int, int]:
    """(start, end) of the YAML block of entry `qid` in the file text."""
    starts = list(_ENTRY_RE.finditer(text))
    for i, m in enumerate(starts):
        if m.group("id") == qid:
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            return m.start(), end
    raise KeyError(f"entry {qid} not found")


def set_status(text: str, qid: str, status: str) -> str:
    start, end = entry_span(text, qid)
    block = text[start:end]
    new_block, n = _STATUS_RE.subn(lambda m: f"{m.group('indent')}status: {status}", block, count=1)
    if not n:
        raise KeyError(f"entry {qid} has no top-level status line")
    return text[:start] + new_block + text[end:]


def _default_editor(path: Path) -> None:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    subprocess.run([editor, str(path)], check=False)


def review(path: Path | None = None, ask: Callable[[str], str] = input,
           say: Callable[[str], None] = print, edit: Callable[[Path], None] = _default_editor) -> dict[str, int]:
    """The interactive G3 loop; returns counts {approved, rejected, left}."""
    path = path or paths.DRAFTS_PATH
    while True:
        _, questions, errors = load_file(path)
        if errors and not questions:
            raise ValueError("; ".join(errors))
        todo = [q for q in questions if q.status == "draft"]
        done = len(questions) - len(todo)
        if not todo:
            break
        q = todo[0]
        text = path.read_text(encoding="utf-8")
        start, end = entry_span(text, q.id)
        say(f"\n=== {q.id}  ({done + 1}/{len(questions)}, {len(todo)} left) ===\n{CHECKLIST}\n")
        say(text[start:end].rstrip())
        key = ask("\n[y/n/e/q] > ").strip().lower()
        if key in ("y", "n"):
            path.write_text(set_status(text, q.id, "approved" if key == "y" else "rejected"), encoding="utf-8")
        elif key == "e":
            edit(path)                                   # the entry is shown again after the edit
        elif key == "q":
            break
        else:
            say("Please answer y, n, e or q.")
    _, questions, _ = load_file(path)
    counts = {s: sum(q.status == s for q in questions) for s in ("approved", "rejected")}
    counts["left"] = sum(q.status == "draft" for q in questions)
    return counts


def freeze(ctx: ValidationContext, drafts: Path | None = None, target: Path | None = None,
           archive: Path | None = None, privacy_fp: str | None = None) -> list[str]:
    """G4. Returns the problems (empty = frozen and written)."""
    drafts, target = drafts or paths.DRAFTS_PATH, target or paths.QUESTIONS_PATH
    archive = archive or paths.QUESTIONS_V0_PATH
    _, questions, errors = load_file(drafts)
    if errors:
        return redact_messages(errors, ctx.private_values)
    left = [q.id for q in questions if q.status == "draft"]
    if left:
        return [f"{len(left)} drafts not reviewed yet (hypeonoath-eval review-gold): {', '.join(left[:10])}"]
    approved = [q for q in questions if q.status == "approved"]
    problems = validate(approved, ctx, approved_only=True)
    if problems:
        return problems
    meta = {"privacy_fp": privacy_fp or current_privacy_fp(), "question_set_sha256": question_set_sha256(approved)}
    if target.exists() and not archive.exists() and isinstance(yaml.safe_load(target.read_text(encoding="utf-8")), list):
        target.replace(archive)                          # the pre-harness flat list, kept but never scored
    header = ("# Frozen gold question set (eval plan v15, 3.1; written by `hypeonoath-eval validate --freeze`).\n"
              "# Do not edit by hand: edit eval/gold_work/drafts.yaml, review, and freeze again.\n")
    body = yaml.safe_dump({"meta": meta, "questions": [q.raw for q in approved]}, sort_keys=False,
                          allow_unicode=True, width=120)
    target.write_text(header + body, encoding="utf-8")
    return []

