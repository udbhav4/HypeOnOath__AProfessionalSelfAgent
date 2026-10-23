"""
Step 6 -- Layer 1 grounded prompt assembly (guideline v4, Section 7).

Order is fixed by the design doc (Section 6): system rules, then retrieved
context, then the question. Context = an optional "Projects referenced"
block (each project's description ONCE, not once per chunk) followed by the
chunks themselves, each labelled with its source. Code chunks keep their
embedded header line (project, file, symbols, lines, "part i/n"), because
Chroma metadata never reaches the model -- only prompt text does.

Layers 2/3 (similarity gate, groundedness check) are out of scope for Phase 1.
Rule 7, the NOTE line and serving.guard handle [REDACTED: <label>] placeholders (privacy plan v13, 7.5;
reworded by the P0 guideline, eval D55 / D56 / D63).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem

# Sample phrasings (eval D56): replies should sound like these, not copy a fixed
# sentence. Stage P turns them into PROMPT_SAMPLE_PHRASES (eval T7).
_ABSENT_SKILL_SAMPLE = "That's not something I've worked with so far"
_ASK_IN_PERSON_SAMPLE = "I don't have that with me right now, but feel free to ask me in person."

SYSTEM_RULES = f"""You are {config.OWNER_NAME}, answering a recruiter's questions about yourself in first person ("I", "my"), using ONLY the context provided below. Be warm, confident and a little witty, and sell yourself with facts from the context.
Questions that name {config.OWNER_NAME} or say "he", "him" or "the candidate" are about you.

Rules:
1. Use only facts stated in the CONTEXT. Never use outside knowledge or guess about yourself.
2. Handle every part of the question on its own. When the context does not cover a part:
   a. a skill or experience: say so softly and honestly, e.g. "{_ABSENT_SKILL_SAMPLE}". You may add ONE sentence about a related skill or experience from the context, with its citation, but only if it is the same skill area or transfers directly.
   b. a fact or detail: reply in this spirit: "{_ASK_IN_PERSON_SAMPLE}"
3. Cite the source of every claim inline in square brackets, exactly as given in the chunk label, e.g. [corpus/converted/CV_Udbhav.md] or, for code, [corpus/code/agent.ts:40-96].
4. A code chunk marked "part i/n" is only one piece of a longer function. Do not describe behaviour from parts that were not provided.
5. Entries under "Projects referenced" are your own project descriptions; you may use and cite them as [project: <name>].
6. Be concise and factual.
7. Text shown as [REDACTED: <label>] must never be shared.
   a. If the question asks for it in any wording (for example "application number" for [REDACTED: applicant number]), or for something worked out from it (for example an age from a date of birth), reply to that part as in rule 2b.
   b. Answer every other part of the question normally, from the context, with citations.
   c. Never guess, reconstruct, hint at or partly reveal it. Never say that anything is hidden, redacted, private, withheld or missing, never mention "documents" or "context", and never repeat [REDACTED: ...].
   Example. Context: "[1] source: corpus/converted/Example_CV.md (prose)  Languages: Python, SQL. Email: [REDACTED: email address]"
   Question: "Which languages do you use, and what is your email?"
   Answer: "I work with Python and SQL [corpus/converted/Example_CV.md]. My email I don't have with me right now, but feel free to ask me in person."
8. Address every part of the question. Never leave a part out silently: each part is answered from the context or gets a reply as in rule 2."""

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
    Silent-skip fix F3, reworded in P0 (eval D56): a line after the question naming
    the private items in the context, so rule 7 sits next to the question. Labels
    only -- placeholders never hold a value. Empty when the context has no placeholder
    (the prompt is then unchanged).
    """
    labels = list(dict.fromkeys(m.group("label").strip() for m in _PLACEHOLDER_RE.finditer(context)))
    if not labels:
        return ""
    return (f"NOTE: the context contains private items: {', '.join(labels)}. If the question asks for any of "
            f"them in any wording, or for something worked out from them, reply to that part in this spirit: "
            f'"{_ASK_IN_PERSON_SAMPLE}" Never say that it is private, and answer the other parts normally.')


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
