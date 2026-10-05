"""
Step 6 -- Layer 1 grounded prompt assembly (guideline v4, Section 7).

Order is fixed by the design doc (Section 6): system rules, then retrieved
context, then the question. Context = an optional "Projects referenced"
block (each project's description ONCE, not once per chunk) followed by the
chunks themselves, each labelled with its source. Code chunks keep their
embedded header line (project, file, symbols, lines, "part i/n"), because
Chroma metadata never reaches the model -- only prompt text does.

Layers 2/3 (similarity gate, groundedness check) are out of scope for Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem

SYSTEM_RULES = f"""You answer recruiters' questions about a candidate, using ONLY the context provided below in a witty and catchy manner.
Aim to sell the candidate to the recruiter.

Rules:
1. Use only facts stated in the CONTEXT. Never use outside knowledge or guess about the candidate.
2. If the context does not cover the question, reply exactly: "{config.REFUSAL_PHRASE}".
3. Cite the source of every claim inline in square brackets, exactly as given in the chunk label, e.g. [corpus/converted/CV_Udbhav.md] or, for code, [corpus/code/agent.ts:40-96].
4. A code chunk marked "part i/n" is only one piece of a longer function. Do not describe behaviour from parts that were not provided.
5. Entries under "Projects referenced" are the candidate's own project descriptions; you may use and cite them as [project: <name>].
6. Be concise and factual."""


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
    return Prompt(system=SYSTEM_RULES, user=user)
