"""
The gold question set (eval plan v15, Section 3): schema, loading, validation rules V1-V14,
hashing and the privacy-fingerprint check.

Files: eval/questions.yaml (frozen, approved entries + meta), eval/gold_work/drafts.yaml
(drafts, Stage A), eval/doc_groups.yaml (3.2, D70), eval/edge_tags.yaml (3.4, T8).

Privacy: decisions.json is read for labels / entity types / match kinds only (V6, V11);
the private-value check (V9) uses the shared matcher and reports labels, never a value.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hypeonoath.core import config
from hypeonoath.core.private_values import (_FALLBACK_LABEL, PrivateValue, find_private_spans, load_private_values, placeholder,
                                            privacy_fingerprint, read_decisions_file, redact)
from hypeonoath.evaluation import paths
from hypeonoath.evaluation.evidence import Corpus

CATEGORIES = ("single_fact", "multi_hop", "project", "code", "system_design", "persona_pitch",
              "unanswerable", "adversarial")
SUPPORTING_CATEGORIES = ("persona_pitch", "project")          # V8, D21
EXPECTED = ("answer", "no_answer")
STATUSES = ("draft", "approved", "rejected")
NO_GROUP = "none"                                             # doc_group of pure no_answer questions

MIN_QUESTIONS, MAX_QUESTIONS = 70, 80                         # V1, D10
MIN_NO_ANSWER_SHARE = 0.25                                    # V2
MIN_PER_CATEGORY = 2                                          # V3
MIN_MIXED, MAX_MIXED = 10, 15                                 # V10, D39
MIN_PRIVATE_PARTS = 3                                         # V11, D59
CONTACT_TYPES = ("PHONE_NUMBER", "EMAIL_ADDRESS")             # V11
T6_MAX_SIMILARITY = 0.8      # V13 -- assumption: fuzzy ratio at or above this = overlap; tune if it misfires

# Part kinds and the question-level mix tags set from them (3.4, V10).
ANSWER, PRIVATE, NOT_COVERED = "answer", "private", "not_covered"
MIX_TAGS = {
    frozenset({ANSWER, PRIVATE}): "mix_answer_private",
    frozenset({ANSWER, NOT_COVERED}): "mix_answer_notcovered",
    frozenset({PRIVATE, NOT_COVERED}): "mix_private_notcovered",
    frozenset({ANSWER, PRIVATE, NOT_COVERED}): "mix_all_three",
}

PRIVACY_CHANGED = ("privacy decisions changed since the gold freeze -> re-validate gold, re-baseline "
                   "(hypeonoath-eval validate --freeze, then a new baseline run)")


# --- Schema -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Alternative:
    document: str
    quote: str


@dataclass
class Group:
    point: int | None
    any_of: list[Alternative]


@dataclass
class Part:
    id: str
    ask: str
    expected: str
    private_label: str | None = None
    reference_answer: str | None = None
    key_points: list[str] = field(default_factory=list)
    evidence: list[Group] = field(default_factory=list)
    supporting: list[Group] = field(default_factory=list)
    edge: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.expected == "answer":
            return ANSWER
        return PRIVATE if self.private_label else NOT_COVERED


@dataclass
class Question:
    id: str
    category: str
    doc_group: str
    question: str
    status: str
    parts: list[Part]
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)   # the YAML entry, for hashing / merging

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(p.kind for p in self.parts)

    @property
    def mix_tag(self) -> str | None:
        return MIX_TAGS.get(self.kinds)

    @property
    def answerable(self) -> bool:
        return ANSWER in self.kinds

    def part(self, part_id: str) -> Part | None:
        return next((p for p in self.parts if p.id == part_id), None)


def _groups(raw: Any, where: str, errors: list[str], supporting: bool = False) -> list[Group]:
    groups: list[Group] = []
    for g in raw or []:
        if not isinstance(g, dict) or not isinstance(g.get("any_of"), list):
            errors.append(f"{where}: every group needs an any_of list")
            continue
        alts = []
        for a in g["any_of"]:
            if not isinstance(a, dict) or not a.get("document") or not isinstance(a.get("quote"), str):
                errors.append(f"{where}: every alternative needs a document and a quote")
                continue
            alts.append(Alternative(str(a["document"]), a["quote"]))
        groups.append(Group(None if supporting else g.get("point"), alts))
    return groups


def _str_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def parse_question(raw: Any, errors: list[str]) -> Question | None:
    """One YAML entry -> Question; schema problems go to `errors` (never raises on bad data)."""
    if not isinstance(raw, dict) or not raw.get("id"):
        errors.append("an entry without an id (its content is not shown)")
        return None
    qid = str(raw["id"])
    for key in ("category", "question", "status", "parts"):
        if not raw.get(key):
            errors.append(f"{qid}: missing {key}")
    parts: list[Part] = []
    for i, p in enumerate(raw.get("parts") or []):
        if not isinstance(p, dict) or not p.get("id"):
            errors.append(f"{qid}: part {i + 1} has no id")
            continue
        where = f"{qid}/{p['id']}"
        if p.get("expected") not in EXPECTED:
            errors.append(f"{where}: expected must be one of {EXPECTED}")
        if not p.get("ask"):
            errors.append(f"{where}: missing ask")
        parts.append(Part(
            id=str(p["id"]), ask=str(p.get("ask") or ""), expected=str(p.get("expected")),
            private_label=p.get("private_label") or None,
            reference_answer=(str(p["reference_answer"]).strip() if p.get("reference_answer") else None),
            key_points=_str_list(p.get("key_points")),
            evidence=_groups(p.get("evidence"), where, errors),
            supporting=_groups(p.get("supporting"), where, errors, supporting=True),
            edge=_str_list(p.get("edge")),
        ))
    return Question(id=qid, category=str(raw.get("category") or ""), doc_group=str(raw.get("doc_group") or NO_GROUP),
                    question=str(raw.get("question") or ""), status=str(raw.get("status") or ""),
                    parts=parts, notes=str(raw.get("notes") or ""), raw=raw)


def load_file(path: Path) -> tuple[dict[str, Any], list[Question], list[str]]:
    """(meta, questions, schema errors) of a questions.yaml / drafts.yaml file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, [], [f"{path.name}: cannot read ({type(exc).__name__}: {exc})"]
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        return {}, [], [f"{path.name}: expected a mapping with a 'questions' list (the old flat list "
                        f"format is {paths.QUESTIONS_V0_PATH.name}, not scored)"]
    errors: list[str] = []
    questions = [q for q in (parse_question(r, errors) for r in data["questions"]) if q]
    return data.get("meta") or {}, questions, errors


# --- Configuration files ------------------------------------------------------------------

@dataclass(frozen=True)
class DocGroup:
    name: str
    files: tuple[str, ...]
    minimum: int | None


@dataclass(frozen=True)
class EdgeTag:
    name: str
    applies_to: str            # answer | not_covered | private | adversarial
    minimum: int
    required: bool


def load_doc_groups(path: Path | None = None) -> dict[str, DocGroup]:
    data = yaml.safe_load((path or paths.DOC_GROUPS_PATH).read_text(encoding="utf-8")) or {}
    return {name: DocGroup(name, tuple(g.get("files") or ()), g.get("min"))
            for name, g in (data.get("groups") or {}).items()}


def load_edge_tags(path: Path | None = None) -> dict[str, EdgeTag]:
    data = yaml.safe_load((path or paths.EDGE_TAGS_PATH).read_text(encoding="utf-8")) or {}
    return {name: EdgeTag(name, t.get("applies_to", ""), int(t.get("min", 0)), bool(t.get("required", True)))
            for name, t in (data.get("tags") or {}).items()}


@dataclass(frozen=True)
class PrivateDecision:
    """What validation may know about a private decision: never its value."""
    label: str
    entity_types: frozenset[str]
    match_kinds: frozenset[str]


def private_decisions(path: Path | None = None) -> dict[str, PrivateDecision]:
    """label -> types / match kinds of the private decisions (and their aliases) in decisions.json."""
    out: dict[str, PrivateDecision] = {}
    for entry in read_decisions_file(path)["entries"]:
        if entry.get("decision") != "private":
            continue
        label = entry.get("label") or _FALLBACK_LABEL                  # same fallback as values_from_entries
        types = {entry.get("entity_type") or "MANUAL"} | {a.get("entity_type") for a in entry.get("aliases") or []}
        kinds = {entry.get("match_kind") or "text"} | {a.get("match_kind") for a in entry.get("aliases") or []}
        old = out.get(label, PrivateDecision(label, frozenset(), frozenset()))
        out[label] = PrivateDecision(label, old.entity_types | {t for t in types if t},
                                     old.match_kinds | {k for k in kinds if k})
    return out


@dataclass
class ValidationContext:
    corpus: Corpus
    doc_groups: dict[str, DocGroup]
    edge_tags: dict[str, EdgeTag]
    private: dict[str, PrivateDecision]
    private_values: list[PrivateValue]
    prompt_examples: list[str]


def default_context() -> ValidationContext:
    from hypeonoath.serving.prompt import PROMPT_EXAMPLES

    return ValidationContext(Corpus(), load_doc_groups(), load_edge_tags(), private_decisions(),
                             load_private_values(), list(PROMPT_EXAMPLES))


# --- Validation (V1-V13; V14 = freeze / run gate below) --------------------------------------

def counted(questions: list[Question], approved_only: bool) -> list[Question]:
    """The entries a set is judged by: approved only (frozen set) or every non-rejected entry (drafts)."""
    if approved_only:
        return [q for q in questions if q.status == "approved"]
    return [q for q in questions if q.status != "rejected"]


def _strings(value: Any, where: str = "") -> list[tuple[str, str]]:
    """(key path, text) of every string in a YAML entry, keys included (freeze writes the whole entry)."""
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in [(f"{where}{k}(key)", str(k))] + _strings(v, f"{where}{k}.")]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in _strings(v, f"{where}{i}.")]
    return [(where.rstrip("."), str(value))] if value is not None else []


def _check_entry(q: Question, ctx: ValidationContext, errors: list[str]) -> None:
    if q.status not in STATUSES:
        errors.append(f"{q.id}: V1 status must be one of {STATUSES}")
    if q.category not in CATEGORIES:
        errors.append(f"{q.id}: V3 unknown category {q.category!r}")
    if q.doc_group != NO_GROUP and q.doc_group not in ctx.doc_groups:
        errors.append(f"{q.id}: V3 unknown doc_group {q.doc_group!r}")
    part_ids = [p.id for p in q.parts]
    for dup in sorted(k for k, n in Counter(part_ids).items() if n > 1):
        errors.append(f"{q.id}: V1 duplicate part id {dup!r}")
    if not q.parts:
        errors.append(f"{q.id}: V1 no parts")

    for p in q.parts:
        where = f"{q.id}/{p.id}"

        if p.expected == "answer":                                                  # V4
            if not p.key_points:
                errors.append(f"{where}: V4 answer part without key points")
            if not p.reference_answer:
                errors.append(f"{where}: V4 answer part without reference_answer")
            points = [g.point for g in p.evidence]
            if sorted(x for x in points if isinstance(x, int)) != list(range(len(p.key_points))) \
                    or len(points) != len(p.key_points):
                errors.append(f"{where}: V4 every key point needs exactly one evidence group (points 0..n-1)")
            if p.private_label:
                errors.append(f"{where}: V6 private_label only on no_answer parts")
        elif p.private_label:                                                       # V6
            decision = ctx.private.get(p.private_label)
            if decision is None:
                errors.append(f"{where}: V6 private_label {p.private_label!r} is not a private decision label")
            if p.key_points:
                errors.append(f"{where}: V6 private part must have no key points")
            want = placeholder(p.private_label)
            if len(p.evidence) != 1 or p.evidence[0].point is not None or not p.evidence[0].any_of \
                    or any(a.quote != want for a in p.evidence[0].any_of):
                errors.append(f"{where}: V6 private part needs exactly one group (point: null) quoting {want!r}")
        elif p.key_points or p.evidence:                                            # V5
            errors.append(f"{where}: V5 not-covered no_answer part must have no key points and no evidence")

        if p.supporting and (q.category not in SUPPORTING_CATEGORIES or p.expected != "answer"):
            errors.append(f"{where}: V8 supporting groups only on answer parts of {SUPPORTING_CATEGORIES}")

        grouped = {f for g in ctx.doc_groups.values() for f in g.files}
        for g in p.evidence + p.supporting:                                         # V7 (+ D70 group check)
            if not g.any_of:
                errors.append(f"{where}: V4 evidence group without alternatives")
            for a in g.any_of:
                if not a.document.startswith("corpus/"):
                    errors.append(f"{where}: V7 document outside corpus/: {a.document}")
                elif not ctx.corpus.exists(a.document):
                    errors.append(f"{where}: V7 document not found: {a.document}")
                elif not ctx.corpus.contains(a.document, a.quote):
                    shown = redact(a.quote, ctx.private_values)[0][:60]       # never echo a private value
                    errors.append(f"{where}: V7 quote not found in {a.document}: {shown!r}")
                if a.document not in grouped:
                    errors.append(f"{where}: D70 document belongs to no doc group: {a.document}")

        if not p.edge:
            errors.append(f"{where}: V12 part without an edge tag")
        for tag in p.edge:
            if tag not in ctx.edge_tags:
                errors.append(f"{where}: V12 unknown edge tag {tag!r}")

    for name, text in _strings(q.raw):                                              # V9: every string, any key
        labels = sorted({s.label for s in find_private_spans(text, ctx.private_values)})
        if labels:
            errors.append(f"{q.id}: V9 private value ({', '.join(labels)}) in {name}")

    for example in ctx.prompt_examples:                                             # V13 (T6)
        ratio = difflib.SequenceMatcher(None, q.question.casefold(), example.casefold()).ratio()
        if ratio >= T6_MAX_SIMILARITY:
            errors.append(f"{q.id}: V13 question overlaps prompt example {example!r} (similarity {ratio:.2f})")


def validate(questions: list[Question], ctx: ValidationContext, approved_only: bool = True) -> list[str]:
    """Every V1-V13 violation as a message (empty = valid). Messages never hold a private value."""
    errors: list[str] = []
    for dup in sorted(k for k, n in Counter(q.id for q in questions).items() if n > 1):
        errors.append(f"{dup}: V1 duplicate question id")
    for q in questions:
        _check_entry(q, ctx, errors)

    qs = counted(questions, approved_only)
    which = "approved" if approved_only else "non-rejected"
    n = len(qs)
    if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
        errors.append(f"V1 {n} {which} questions; need {MIN_QUESTIONS}-{MAX_QUESTIONS}")
    with_no_answer = sum(any(p.expected == "no_answer" for p in q.parts) for q in qs)
    if n and with_no_answer / n < MIN_NO_ANSWER_SHARE:
        errors.append(f"V2 {with_no_answer}/{n} questions with a no_answer part; need >= {MIN_NO_ANSWER_SHARE:.0%}")

    per_category = Counter(q.category for q in qs)
    for cat in CATEGORIES:
        if per_category[cat] < MIN_PER_CATEGORY:
            errors.append(f"V3 category {cat}: {per_category[cat]} questions; need >= {MIN_PER_CATEGORY}")
    per_group = Counter(q.doc_group for q in qs)
    for group in ctx.doc_groups.values():
        if group.minimum and per_group[group.name] < group.minimum:
            errors.append(f"V3 doc group {group.name}: {per_group[group.name]} questions; need >= {group.minimum}")

    mixed = [q for q in qs if len(q.kinds) >= 2]
    if not MIN_MIXED <= len(mixed) <= MAX_MIXED:
        errors.append(f"V10 {len(mixed)} mixed questions; need {MIN_MIXED}-{MAX_MIXED}")
    present = Counter(q.mix_tag for q in mixed)
    for tag in MIX_TAGS.values():
        if not present[tag]:
            errors.append(f"V10 no mixed question of combination {tag}")

    private_parts = [p for q in qs for p in q.parts if p.kind == PRIVATE]
    if len(private_parts) < MIN_PRIVATE_PARTS:
        errors.append(f"V11 {len(private_parts)} private parts; need >= {MIN_PRIVATE_PARTS}")
    decisions = [ctx.private.get(p.private_label) for p in private_parts]
    if not any(d and d.entity_types & set(CONTACT_TYPES) for d in decisions):
        errors.append(f"V11 no private part about a contact detail ({' / '.join(CONTACT_TYPES)})")
    if not any(d and "id" in d.match_kinds for d in decisions):
        errors.append("V11 no private part about an ID-like item (match_kind id)")

    tag_counts = Counter(t for q in qs for p in q.parts for t in p.edge)
    for tag in ctx.edge_tags.values():
        if tag.required and tag_counts[tag.name] < tag.minimum:
            errors.append(f"V12 edge tag {tag.name}: {tag_counts[tag.name]}; need >= {tag.minimum}")
    # messages echo labels, tags, categories: redact so a value typed into any field is never printed
    return redact_messages(errors, ctx.private_values)


def redact_messages(messages: list[str], values: list[PrivateValue]) -> list[str]:
    return [redact(m, values)[0] for m in messages]


# --- Hashes and the privacy gate (V14, D49) -------------------------------------------------

def question_set_sha256(questions: list[Question]) -> str:
    """sha256 of the approved entries, independent of key order and file layout."""
    entries = sorted((q.raw for q in questions if q.status == "approved"), key=lambda r: str(r.get("id")))
    return hashlib.sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def current_privacy_fp(path: Path | None = None) -> str:
    return privacy_fingerprint(load_private_values(path or config.PRIVACY_DECISIONS_PATH))


def privacy_gate(meta: dict[str, Any], current_fp: str) -> str | None:
    """None when the frozen set matches the current private decisions, else the D49 message."""
    frozen = meta.get("privacy_fp")
    if not frozen:
        return "the question set is not frozen (no meta.privacy_fp) -> hypeonoath-eval validate --freeze"
    return None if frozen == current_fp else PRIVACY_CHANGED
