"""
Step 6 -- Layer 1 grounded prompt assembly (guideline v4, Section 7).

Order is fixed by the design doc (Section 6): system rules, then retrieved
context, then the question. Context = an optional "Projects referenced"
block (each project's description ONCE, not once per chunk) followed by the
chunks themselves, each labelled with its source. Code chunks keep their
embedded header line (project, file, symbols, lines, "part i/n"), because
Chroma metadata never reaches the model -- only prompt text does.

Layers 2/3 (similarity gate, groundedness check) are out of scope for Phase 1.
Rule 7 + serving.guard handle [REDACTED: <label>] placeholders (privacy plan v13, 7.5, P-D52);
rules 2 / 7a / 8 and the private-items note: prompt-decline-silent-skip-fix-plan-v2 (F1 - F3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem

# Rule 7 decline sentences are rendered from the guard's template (P-D52), so
# the prompt and serving.guard can never word a decline differently.
_DECLINE = config.DECLINE_TEMPLATES["third"].format(label="<label>")
_EXAMPLE_DECLINE = config.DECLINE_TEMPLATES["third"].format(label="email address")

SYSTEM_RULES = f"""You answer recruiters' questions about a candidate, using ONLY the context provided below in a witty and catchy manner.
Aim to sell the candidate to the recruiter.

Rules:
1. Use only facts stated in the CONTEXT. Never use outside knowledge or guess about the candidate.
2. If the context covers none of the question, reply exactly: "{config.REFUSAL_PHRASE}". If it covers only some parts, answer those and say "{config.REFUSAL_PHRASE}" for each part it does not cover.
3. Cite the source of every claim inline in square brackets, exactly as given in the chunk label, e.g. [corpus/converted/CV_Udbhav.md] or, for code, [corpus/code/agent.ts:40-96].
4. A code chunk marked "part i/n" is only one piece of a longer function. Do not describe behaviour from parts that were not provided.
5. Entries under "Projects referenced" are the candidate's own project descriptions; you may use and cite them as [project: <name>].
6. Be concise and factual.
7. Text shown as [REDACTED: <label>] is private. The information exists, but the candidate chose not to share it.
   a. If the question asks for it in any wording (for example "application number" for [REDACTED: applicant number]), or for something worked out from it (for example an age from a date of birth), write exactly: "{_DECLINE}" with <label> taken from the brackets.
   b. Never answer a redacted item with "{config.REFUSAL_PHRASE}": it is private, not missing.
   c. Answer every other part of the question normally, from the context, with citations.
   d. Never guess, reconstruct, hint at or partly reveal a redacted value.
   Example. Context: "[1] source: corpus/converted/Example_CV.md (prose)  Languages: Python, SQL. Email: [REDACTED: email address]"
   Question: "Which languages does she use, and what is her email?"
   Answer: "She works with Python and SQL [corpus/converted/Example_CV.md]. {_EXAMPLE_DECLINE}"
8. Address every part of the question. Never leave a part out silently: each part is answered, declined (rule 7) or refused (rule 2)."""

# Prompt version -> grammatical person of the bot's voice; picks the decline
# template (config.DECLINE_TEMPLATES) the answer guard uses. v1 talks about
# "the candidate" (third person); the persona prompt (eval plan) adds "first".
PROMPT_PERSON = {"v1": "third"}

# [REDACTED: <label>] in the assembled context; built from config so it follows
# REDACTED_PLACEHOLDER (same construction as privacy.review).
_PLACEHOLDER_RE = re.compile(
    re.escape(config.REDACTED_PLACEHOLDER).replace(re.escape("{label}"), r"(?P<label>[^\]]+)"))


@dataclass
class Prompt:
    system: str
    user: str

    def as_text(self) -> str:
        """Single-string form (for logging / eval review)."""
        return f"{self.system}\n\n{self.user}"


def _cite_label(item: RetrievedItem) -> str:
    """The exact citation string the model is told to use for this item."""
    chunks = item.ordered_chunks
    first, last = chunks[0], chunks[-1]
    if first.line_start is not None and first.content_type in ("code", "code_fence"):
        return f"{first.source_document}:{first.line_start}-{last.line_end}"
    return first.source_document


def _projects_block(items: list[RetrievedItem]) -> str:
    """name: description, once per project with >= 1 retrieved CODE chunk."""
    seen: dict[str, tuple[str, str]] = {}
    for item in items:
        chunk = item.chunk
        if chunk.content_type == "code" and chunk.project_key and chunk.project_key not in seen:
            seen[chunk.project_key] = (chunk.project_name or chunk.project_key, chunk.project_description or "")
    if not seen:
        return ""
    lines = ["Projects referenced:"]
    lines += [f"- {name}: {description}" for name, description in seen.values()]
    return "\n".join(lines)


def _private_note(context: str) -> str:
    """
    F3: a line after the question naming the private items in the context, with
    the exact decline sentence per label, so the rule-7 instruction sits next to
    the question. Labels only -- placeholders never hold a value. Empty when the
    context has no placeholder (the prompt is then unchanged).
    """
    labels = list(dict.fromkeys(m.group("label").strip() for m in _PLACEHOLDER_RE.finditer(context)))
    if not labels:
        return ""
    template = config.DECLINE_TEMPLATES[PROMPT_PERSON.get(config.PROMPT_VERSION, "third")]
    declines = "; ".join(f'for {label}: "{template.format(label=label)}"' for label in labels)
    return (f"NOTE: the context contains private items: {', '.join(labels)}. If the question asks for any of "
            f"them in any wording, write exactly the sentence for it ({declines}), and answer the other parts normally.")


def build_prompt(question: str, items: list[RetrievedItem]) -> Prompt:
    """Assemble the Layer 1 prompt from retriever output (already ordered by rank)."""
    sections: list[str] = []
    projects = _projects_block(items)
    if projects:
        sections.append(projects)

    for n, item in enumerate(items, start=1):
        chunks = item.ordered_chunks
        kind = item.chunk.content_type
        if len(chunks) > 1:
            kind += f", parts {chunks[0].part}-{chunks[-1].part} of {item.chunk.total_parts}"
        body = "\n".join(c.text for c in chunks)
        sections.append(f"[{n}] source: {_cite_label(item)} ({kind})\n{body}")

    context = "\n\n".join(sections) if sections else "(no context retrieved)"
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question.strip()}"
    note = _private_note(context)
    if note:
        user += f"\n\n{note}"
    return Prompt(system=SYSTEM_RULES, user=user)
