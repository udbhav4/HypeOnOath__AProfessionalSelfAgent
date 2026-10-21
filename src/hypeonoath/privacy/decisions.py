"""
Owner decisions: corpus/.privacy/decisions.json (privacy plan v10, 5.1, 6.4, 6.4.1).

  * Decision / Alias       one decision per VALUE (P-D24), private OR keep
                           (P-D32), with every location it was seen at.
  * DecisionFile           the decisions + `pending_restore` (raw files to
                           rebuild because a value went public, P-D33).
  * load / save            JSON (serving has no pyyaml), atomic write, schema
                           1/2 -> 3 migration on load (ids and fingerprint kept).
  * group_candidates       Candidates -> one CandidateGroup per decision id.
  * resolve / resolve_all  rules R1, R5, R2, R3, RS (6.4.1): decisions extend
                           value-wide to new types (P-D30), private wins
                           conflicts, new secrets become private (P-D31); the
                           scan then stores every other new value as keep
                           (P-D27, P-D39, privacy.scan).
  * prune_retired          drop keep decisions of retired types (P-D35).
  * set_private / set_public / set_keep_for_undecided   the review's and the scan's writes.
  * AllowRule              legacy corpus/.scan_allowlist.yaml entries = keep.

The file holds exact values: it is gitignored and NOTHING here logs, prints
or raises with a value (only ids, types, labels, paths and counts).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.core.logging_utils import log_json_line
from hypeonoath.privacy.detector import SECRET_PREFIX, Candidate

SCHEMA_VERSION = 3
DECISIONS = ("private", "keep")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decision_id(entity_type: str, normalized: str) -> str:
    """'pv_' + sha256(entity_type \\x1f normalized)[:12] of the FIRST decision; never changes."""
    return "pv_" + hashlib.sha256(f"{entity_type}\x1f{normalized}".encode("utf-8")).hexdigest()[:12]


# --- Records ---------------------------------------------------------------------------

@dataclass
class Alias:
    entity_type: str
    match_kind: str
    normalized: str
    added_at: str                     # ISO UTC
    rule: str                         # "R2" | "R5" | "migrated_keep" (+ legacy "R4")


@dataclass
class Decision:
    id: str
    decision: str                     # "private" | "keep"
    label: str | None                 # required for private; kept on a switch to keep (for a later switch back)
    entity_type: str                  # type at first decision (unchanged by aliases)
    match_kind: str
    value: str
    normalized: str
    value_keys: list[str]             # sorted; recomputed on load if missing (migration)
    aliases: list[Alias] = field(default_factory=list)
    surface_forms: list[str] = field(default_factory=list)
    source: str = "review"            # review | manual | default_secret | detected
    first_seen: dict[str, Any] = field(default_factory=dict)
    locations: list[dict[str, Any]] = field(default_factory=list)   # [{file, line}] -- paths and lines only
    decided_at: str = field(default_factory=utc_now)

    def __repr__(self) -> str:        # never show the value by accident
        return f"Decision({self.id}, {self.decision}, label={self.label!r}, type={self.entity_type})"

    @property
    def is_private(self) -> bool:
        return self.decision == "private"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Decision":
        value = raw.get("value", "")
        keys = raw.get("value_keys") or sorted(pv.value_keys(value))      # schema 1 migration
        first_seen = dict(raw.get("first_seen") or {})
        locations = raw.get("locations")
        if locations is None:                                              # schema 1/2 migration
            locations = [{"file": first_seen["file"], "line": first_seen.get("line")}] if first_seen.get("file") else []
        return cls(
            id=raw["id"],
            decision=raw["decision"],
            label=raw.get("label"),
            entity_type=raw.get("entity_type", "MANUAL"),
            match_kind=raw.get("match_kind", "text"),
            value=value,
            normalized=raw.get("normalized", ""),
            value_keys=sorted(keys),
            aliases=[Alias(**a) for a in raw.get("aliases") or []],
            surface_forms=list(raw.get("surface_forms") or ([value] if value else [])),
            source=raw.get("source", "detected"),
            first_seen=first_seen,
            locations=[dict(loc) for loc in locations],
            decided_at=raw.get("decided_at", ""),
        )

    def forms(self) -> list[tuple[str, str, str]]:
        """(entity_type, match_kind, normalized) of the decision and every alias."""
        return [(self.entity_type, self.match_kind, self.normalized)] + [
            (a.entity_type, a.match_kind, a.normalized) for a in self.aliases]

    def all_types(self) -> list[str]:
        return list(dict.fromkeys(t for t, _k, _n in self.forms()))

    def keys(self) -> frozenset[str]:
        """Every value key of the decision, its aliases and its surface forms."""
        keys = set(self.value_keys)
        keys.update(f"{kind}:{normalized}" for _t, kind, normalized in self.forms() if normalized)
        for form in self.surface_forms:
            keys.update(pv.value_keys(form))
        return frozenset(keys)

    def files(self) -> list[str]:
        return list(dict.fromkeys(loc["file"] for loc in self.locations))

    def add_alias(self, entity_type: str, match_kind: str, normalized: str, rule: str) -> bool:
        if (entity_type, match_kind, normalized) in self.forms():
            return False
        self.aliases.append(Alias(entity_type, match_kind, normalized, utc_now(), rule))
        return True

    def add_surface_forms(self, forms: list[str]) -> bool:
        new = [f for f in forms if f and f not in self.surface_forms]
        self.surface_forms.extend(new)
        return bool(new)

    def set_file_locations(self, file: str, lines: list[int | None]) -> bool:
        """Replace this file's locations with the lines detected now (refresh)."""
        current = [loc for loc in self.locations if loc["file"] == file]
        fresh = [{"file": file, "line": line} for line in sorted(set(lines), key=lambda n: (n is None, n or 0))]
        if current == fresh:
            return False
        self.locations = [loc for loc in self.locations if loc["file"] != file] + fresh
        return True


@dataclass
class DecisionFile:
    """Everything in decisions.json."""

    decisions: list[Decision] = field(default_factory=list)
    pending_restore: list[str] = field(default_factory=list)   # raw rel paths to rebuild (P-D33)

    def by_id(self, did: str) -> Decision | None:
        return next((d for d in self.decisions if d.id == did), None)


# --- File I/O ------------------------------------------------------------------------------

def load(path: Path | None = None) -> DecisionFile:
    """Decisions + restore queue; missing file -> empty. Accepts schema_version 1, 2 or 3."""
    data = pv.read_decisions_file(path)
    version = data.get("schema_version", 1)
    if version not in (1, 2, SCHEMA_VERSION):
        raise ValueError(f"Unsupported decisions schema_version {version!r}")
    decisions = []
    for raw in data["entries"]:
        if not isinstance(raw, dict) or raw.get("decision") not in DECISIONS or not raw.get("id"):
            raise ValueError("Malformed entry in the decisions file (needs id + decision private|keep)")
        decisions.append(Decision.from_dict(raw))
    pending = data.get("pending_restore") or []
    if not isinstance(pending, list) or not all(isinstance(p, str) for p in pending):
        raise ValueError("Malformed pending_restore in the decisions file (needs a list of paths)")
    return DecisionFile(decisions, list(dict.fromkeys(pending)))


def save(data: DecisionFile, path: Path | None = None) -> None:
    """Atomic: write a temp file next to the target, then os.replace (a crash leaves the old file)."""
    path = config.PRIVACY_DECISIONS_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "entries": [d.to_dict() for d in data.decisions],
               "pending_restore": data.pending_restore}
    fd, tmp = tempfile.mkstemp(prefix=".decisions.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def private_values(decisions: list[Decision]) -> list[pv.PrivateValue]:
    """The matcher's view of the decisions (private ones, all alias forms)."""
    return pv.values_from_entries(d.to_dict() for d in decisions)


def privacy_fingerprint(decisions: list[Decision]) -> str:
    return pv.privacy_fingerprint(private_values(decisions))


# --- Grouping ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Location:
    file: str
    line: int | None

    def __str__(self) -> str:
        return f"{self.file}:{self.line if self.line is not None else '?'}"


@dataclass
class CandidateGroup:
    """Every candidate with the same decision id (= entity type + normalised value)."""

    id: str
    entity_type: str
    match_kind: str
    normalized: str
    candidates: list[Candidate] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"CandidateGroup({self.id}, {self.entity_type}, {len(self.candidates)} hit(s))"

    @property
    def value(self) -> str:
        return self.candidates[0].value

    @property
    def max_score(self) -> float:
        return max(c.score for c in self.candidates)

    @property
    def recognizers(self) -> list[str]:
        return sorted({c.recognizer for c in self.candidates})

    @property
    def locations(self) -> list[Location]:
        return list(dict.fromkeys(Location(c.file, c.line) for c in self.candidates))

    @property
    def files(self) -> list[str]:
        return list(dict.fromkeys(c.file for c in self.candidates))

    @property
    def surface_forms(self) -> list[str]:
        return list(dict.fromkeys(c.value for c in self.candidates))

    def value_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for form in self.surface_forms:
            keys.update(pv.value_keys(form))
        return frozenset(keys)

    def first_seen(self) -> dict[str, Any]:
        first = self.candidates[0]
        return {"file": first.file, "line": first.line}

    def sort_key(self) -> tuple:
        first = min(self.locations, key=lambda loc: (loc.file, loc.line or 0))
        return (first.file, first.line or 0, self.entity_type)


def group_candidates(candidates: list[Candidate]) -> list[CandidateGroup]:
    """By decision id; keeps all locations + surface forms. Unnormalisable values are dropped."""
    groups: dict[str, CandidateGroup] = {}
    for cand in candidates:
        kind = pv.match_kind_for(cand.entity_type, cand.value)
        normalized = pv.normalize(cand.value, kind)
        if not normalized:
            continue
        gid = decision_id(cand.entity_type, normalized)
        group = groups.setdefault(gid, CandidateGroup(gid, cand.entity_type, kind, normalized))
        group.candidates.append(cand)
    return sorted(groups.values(), key=CandidateGroup.sort_key)


def new_decision(group: CandidateGroup, decision: str, label: str | None, source: str = "review") -> Decision:
    return Decision(
        id=group.id, decision=decision, label=label,
        entity_type=group.entity_type, match_kind=group.match_kind, value=group.value,
        normalized=group.normalized, value_keys=sorted(group.value_keys()),
        surface_forms=group.surface_forms, source=source, first_seen=group.first_seen(),
        locations=[{"file": loc.file, "line": loc.line} for loc in group.locations],
    )


# --- Legacy allowlist ---------------------------------------------------------------------------

# Old pii-regex type names -> the Presidio entity types they correspond to.
_LEGACY_TYPES = {
    "email_address": {"EMAIL_ADDRESS"},
    "phone_number": {"PHONE_NUMBER"},
    "street_address": {"LOCATION"},
}


class AllowRule:
    """
    Legacy .scan_allowlist.yaml {file, line?, finding_type}: matching
    candidates count as keep without being stored (the YAML stays the source),
    and the secret default (RS) doesn't apply -- the allowlist exists for
    documented test keys.
    """

    def __init__(self, entries: list[dict[str, Any]]):
        self.entries = entries

    def __bool__(self) -> bool:
        return bool(self.entries)

    def matches(self, cand: Candidate) -> bool:
        etype = cand.entity_type.removeprefix(SECRET_PREFIX).lower()
        for entry in self.entries:
            if entry["file"] != cand.file:
                continue
            if entry.get("line") is not None and entry.get("line") != cand.line:
                continue
            wanted = str(entry["finding_type"]).lower()
            if wanted == etype or cand.entity_type in _LEGACY_TYPES.get(wanted, ()):
                return True
            # id_number covered every digit-run / IBAN-like ID.
            if wanted == "id_number" and pv.match_kind_for(cand.entity_type, cand.value) == "id" \
                    and cand.entity_type not in ("PHONE_NUMBER", "EMAIL_ADDRESS"):
                return True
        return False

    def filter(self, candidates: list[Candidate]) -> tuple[list[Candidate], int]:
        """(candidates not allowlisted, number allowlisted)."""
        kept = [c for c in candidates if not self.matches(c)]
        return kept, len(candidates) - len(kept)


def import_scan_allowlist(path: Path | None = None) -> AllowRule:
    from hypeonoath.indexing.secret_scan import load_allowlist

    return AllowRule(load_allowlist(path))


# --- Resolution: R1, R5, R2, R3, RS (plan 6.4.1) ---------------------------------------------------

class DecisionIndex:
    """In-memory lookup built from the decisions (rebuilt after each change)."""

    def __init__(self, decisions: list[Decision]):
        self.by_id: dict[str, Decision] = {}
        self.by_key: dict[str, list[Decision]] = {}
        for decision in decisions:
            self.by_id[decision.id] = decision
            # Every recorded (type, normalised form) answers to its id too.
            for etype, _kind, normalized in decision.forms():
                self.by_id.setdefault(decision_id(etype, normalized), decision)
            for key in decision.keys():
                self.by_key.setdefault(key, []).append(decision)

    def by_decision_id(self, did: str) -> Decision | None:
        return self.by_id.get(did)

    def by_value(self, keys: frozenset[str]) -> list[Decision]:
        """All decisions sharing >= 1 key, no duplicates, stable order."""
        found: dict[str, Decision] = {}
        for key in sorted(keys):
            for decision in self.by_key.get(key, []):
                found.setdefault(decision.id, decision)
        return list(found.values())


@dataclass
class Resolution:
    rule: str                         # "R1" | "R5" | "R2" | "R3" | "RS" | "NEW"
    decision: Decision | None         # matched / created decision
    conflict_with: list[str] = field(default_factory=list)   # R5: keep ids overridden

    @property
    def state(self) -> str:
        """'private' | 'keep' | 'covered' | 'new' -- what the review screen shows."""
        if self.rule == "R3":
            return "covered"
        if self.decision is None:
            return "new"
        return self.decision.decision


def _covered(text: str, values: list[pv.PrivateValue]) -> bool:
    """Every non-space character of `text` lies inside a private span."""
    spans = pv.find_private_spans(text, values)
    return all(ch.isspace() or any(s.start <= i < s.end for s in spans) for i, ch in enumerate(text))


def resolve(group: CandidateGroup, index: DecisionIndex, values: list[pv.PrivateValue]) -> Resolution:
    """
    First matching rule wins (6.4.1). Read-only: apply_resolution() makes the
    changes. R5 is tested before R2 because R2's condition is a subset of it.
    """
    same = index.by_decision_id(group.id)
    if same is not None:
        return Resolution("R1", same)
    matches = index.by_value(group.value_keys())
    private = [d for d in matches if d.is_private]
    keep = [d for d in matches if not d.is_private]
    if private and keep:
        return Resolution("R5", private[0], [d.id for d in keep])
    if matches:
        return Resolution("R2", matches[0])
    if values and all(_covered(form, values) for form in group.surface_forms):
        return Resolution("R3", None)
    if group.entity_type.startswith(SECRET_PREFIX):
        return Resolution("RS", None)
    return Resolution("NEW", None)


def apply_resolution(res: Resolution, group: CandidateGroup, data: DecisionFile) -> bool:
    """Apply the automatic part of a resolution. Returns True when the decisions changed."""
    if res.rule == "R1" and res.decision is not None:
        return res.decision.add_surface_forms(group.surface_forms)
    if res.rule in ("R2", "R5") and res.decision is not None:
        changed = res.decision.add_alias(group.entity_type, group.match_kind, group.normalized, res.rule)
        changed |= res.decision.add_surface_forms(group.surface_forms)
        if res.rule == "R5":
            log_json_line(config.PRIVACY_LOG_PATH, event="privacy_conflict_resolved",
                          private_id=res.decision.id, keep_ids=res.conflict_with)
        return changed
    if res.rule == "RS":
        res.decision = new_decision(group, "private", pv.default_label(group.entity_type), source="default_secret")
        data.decisions.append(res.decision)
        return True
    return False


@dataclass
class ResolveResult:
    items: list[tuple[CandidateGroup, Resolution]]       # every group, in file / line order
    changed: bool = False                                 # decisions mutated (caller saves)
    rules: Counter = field(default_factory=Counter)       # rule -> count
    extended: Counter = field(default_factory=Counter)    # label/"public" -> new alias forms (R2 / R5)
    default_secrets: int = 0                              # RS decisions created


def _refresh_locations(items: list[tuple[CandidateGroup, Resolution]], data: DecisionFile) -> bool:
    """
    Per decision and file: replace the locations with the lines detected now.
    Files where the value is no longer detected keep their old locations (it
    was redacted there, P-D32) unless the file no longer exists.
    """
    detected: dict[str, dict[str, list[int | None]]] = {}
    for group, res in items:
        if res.decision is None:
            continue
        per_file = detected.setdefault(res.decision.id, {})
        for loc in group.locations:
            per_file.setdefault(loc.file, []).append(loc.line)
    changed = False
    root = config.PROJECT_ROOT
    for decision in data.decisions:
        for file, lines in detected.get(decision.id, {}).items():
            changed |= decision.set_file_locations(file, lines)
        alive = [loc for loc in decision.locations if (root / loc["file"]).exists()]
        if len(alive) != len(decision.locations):
            decision.locations = alive
            changed = True
    return changed


def prune_retired(data: DecisionFile) -> bool:
    """
    Drop keep decisions whose every type is retired (P-D35): no recognizer finds
    them any more, so review would list them forever from their stale locations.
    Private decisions are never dropped -- the value stays redacted (P-D11).
    """
    retired = set(config.PRIVACY_RETIRED_ENTITIES)
    kept = [d for d in data.decisions if d.is_private or not set(d.all_types()) <= retired]
    removed = len(data.decisions) - len(kept)
    if removed:
        data.decisions = kept
        log_json_line(config.PRIVACY_LOG_PATH, event="pruned_retired", count=removed)
    return bool(removed)


def resolve_all(groups: list[CandidateGroup], data: DecisionFile) -> ResolveResult:
    """Run the rules over every group; apply R1/R2/R5/RS in place; refresh locations."""
    result = ResolveResult(items=[], changed=prune_retired(data))
    values = private_values(data.decisions)
    index = DecisionIndex(data.decisions)
    for group in sorted(groups, key=CandidateGroup.sort_key):
        res = resolve(group, index, values)
        result.rules[res.rule] += 1
        changed = apply_resolution(res, group, data)
        result.items.append((group, res))
        if not changed:
            continue
        result.changed = True
        if res.rule == "RS":
            result.default_secrets += 1
        elif res.rule in ("R2", "R5"):
            result.extended[res.decision.label if res.decision.is_private else "public"] += 1
        # A new decision or alias form changes the matcher: later groups must see it.
        values = private_values(data.decisions)
        index = DecisionIndex(data.decisions)
    result.changed |= _refresh_locations(result.items, data)
    return result


# --- Writes made by the review (P-D28, P-D30, P-D33) ---------------------------------------------

def set_private(target: CandidateGroup | Decision, label: str, data: DecisionFile,
                source: str = "review") -> Decision:
    """Make a value private, value-wide: the existing decision for it, or a new one."""
    if isinstance(target, Decision):
        decision = target
    else:
        index = DecisionIndex(data.decisions)
        decision = index.by_decision_id(target.id)
        if decision is None:
            matches = index.by_value(target.value_keys())
            decision = matches[0] if matches else None
        if decision is None:
            decision = new_decision(target, "private", label, source=source)
            data.decisions.append(decision)
            return decision
        decision.add_alias(target.entity_type, target.match_kind, target.normalized, "R2")
        decision.add_surface_forms(target.surface_forms)
    decision.decision, decision.label, decision.decided_at = "private", label, utc_now()
    if decision.source == "default_secret" or source == "manual":
        decision.source = source
    return decision


def set_public(decision: Decision, data: DecisionFile) -> list[str]:
    """
    Make a value public (keep), value-wide. If it was private, the raw sources
    of the derived files holding it are queued for a rebuild from raw (P-D33).
    Returns the newly queued raw paths.
    """
    queued: list[str] = []
    if decision.is_private:
        from hypeonoath.privacy.restore import raw_sources_for

        queued = [p for p in raw_sources_for(decision.files()) if p not in data.pending_restore]
        data.pending_restore.extend(queued)
    decision.decision, decision.decided_at = "keep", utc_now()
    if decision.source == "default_secret":
        decision.source = "review"
    return queued


def remove_decision(decision: Decision, data: DecisionFile) -> list[str]:
    """Delete a decision (value undecided again); a private one queues a rebuild like set_public."""
    queued = set_public(decision, data) if decision.is_private else []
    data.decisions.remove(decision)
    return queued


def set_keep_for_undecided(groups: list[CandidateGroup], data: DecisionFile, source: str = "review") -> int:
    """
    Store every group that still has no decision as keep -- the scan stores
    every found value this way (P-D39, source "detected"). Groups covered by
    private values (R3) are left alone.
    """
    stored = 0
    values = private_values(data.decisions)
    for group in groups:
        index = DecisionIndex(data.decisions)
        if index.by_decision_id(group.id) is not None or index.by_value(group.value_keys()):
            continue
        if values and all(_covered(form, values) for form in group.surface_forms):
            continue
        data.decisions.append(new_decision(group, "keep", None, source=source))
        stored += 1
    return stored
