"""
Per-file review screen in the terminal (privacy plan v10, 6.5; P-D28, P-D30,
P-D40 – P-D45).

Every found value is in decisions.json with its current state (the scan
stores each one at once, P-D39), so a file's screen lists the decisions
located in it, the value highlighted in its line as it is on disk now.
Private values are shown in their original line: the stored value is put
back into the redacted line for display only (never written).

Order (P-D44): files by their number of critical, then high values; after an
upload, files pending review come first ("upload" order), the access-settings
order ignores pending ("access"). Inside a file, values are grouped by
category: critical, high, medium, low.

Keys (typed; arrows / clicks belong to the web UI, plan 17.1):
    Enter     next file (after the last one: the first again)
    2 4 / 2-4 switch values public <-> private
    l<n>      edit value n's label (private values)
    a         public for all: every value of this file public -- secrets
              included, after a confirmation (P-D45)
    x         private for all: every value of this file private
    q         Done: the state shown is the owner's choice (P-D41); the
              caller then builds the index (P-D43)

Every switch applies to the VALUE in every file (P-D30) and is saved at once.
Switching a private value public queues its documents for a rebuild from
corpus/raw/ (P-D33).

Values are shown ON THE TERMINAL ONLY -- never logged or written anywhere but
decisions.json. The terminal's scrollback keeps what was shown; that is
accepted (plan 6.5).
"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

from hypeonoath.core import config
from hypeonoath.core import private_values as pv
from hypeonoath.privacy import decisions as dec
from hypeonoath.privacy.detector import SECRET_PREFIX

RULE = "═" * 78
MENU = ("Enter = next file · numbers = switch private/public (e.g. 2 4, 2-4) · l<n> = label · "
        "a = public for all · x = private for all · q = Done (build the index)")
_ANSI_REVERSE, _ANSI_RESET = "\x1b[7m", "\x1b[0m"
_RANGE_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
# "[REDACTED: {label}]" -> r"\[REDACTED: (?P<label>[^\]]+)\]" (labels holding "]" are not restored)
_PLACEHOLDER_RE = re.compile(
    re.escape(config.REDACTED_PLACEHOLDER).replace(re.escape("{label}"), r"(?P<label>[^\]]+)"))

# --- categories (P-D44) -----------------------------------------------------------------------------

CRITICAL, HIGH, MEDIUM, LOW = range(4)
CATEGORY_NAMES = ("critical", "high", "medium", "low")
_CRITICAL_TYPES = frozenset({"IBAN_CODE", "CREDIT_CARD", "CRYPTO"})      # + secrets + country IDs
_HIGH_TYPES = frozenset({"PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_OF_BIRTH", "APPLICANT_ID", "STUDENT_ID"})
_COUNTRY_ID_RE = re.compile(r"^[A-Z]{2}_[A-Z0-9_]+$")                   # US_SSN, DE_TAX_ID, IN_PAN, ...
_NOT_COUNTRY = frozenset({"IP_ADDRESS"})


def type_category(entity_type: str, value: str) -> int:
    """
    Critical: secrets, government IDs, bank / card. High: phone, email, street
    address (a LOCATION with a number), date of birth, applicant / student IDs.
    Medium: person names, other ID-like numbers. Low: other dates, places,
    URLs and the rest.
    """
    if (entity_type.startswith(SECRET_PREFIX) or entity_type in _CRITICAL_TYPES
            or (_COUNTRY_ID_RE.match(entity_type) and entity_type not in _NOT_COUNTRY)):
        return CRITICAL
    if entity_type in _HIGH_TYPES or (entity_type == "LOCATION" and any(ch.isdigit() for ch in value)):
        return HIGH
    if entity_type == "PERSON" or pv.match_kind_for(entity_type, value) == "id":
        return MEDIUM
    return LOW


def category(decision: dec.Decision) -> int:
    """The most critical category among the decision's types (value-wide)."""
    return min(type_category(t, decision.value) for t in decision.all_types())


def _is_secret(decision: dec.Decision) -> bool:
    return any(t.startswith(SECRET_PREFIX) for t in decision.all_types())


# --- building the screens ----------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """One value in one file's screen."""

    decision: dec.Decision
    line: int | None
    line_text: str                      # the line on disk now ('' when unknown)
    start: int                          # offset of the value in line_text; -1 = not in the line (e.g. redacted)
    shown: str                          # the spelling found in the line (else the stored value)
    count: int = 1                      # lines of this file holding the value
    category: int = LOW


@dataclass
class FileReview:
    file: str
    items: list[ReviewItem] = field(default_factory=list)
    pending: bool = False

    def count(self, cat: int) -> int:
        return sum(item.category == cat for item in self.items)


def _read_lines(file: str) -> list[str]:
    try:
        return (config.PROJECT_ROOT / file).read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return []


def _find(line_text: str, decision: dec.Decision) -> tuple[int, str]:
    """(offset, spelling) of the value in the line, longest known spelling first; (-1, value) if absent."""
    lowered = line_text.lower()
    for form in sorted(set(decision.surface_forms or [decision.value]), key=len, reverse=True):
        at = lowered.find(form.lower()) if form else -1
        if at >= 0:
            return at, line_text[at:at + len(form)]
    return -1, decision.value


def _restore_line(line_text: str, private_here: list[dec.Decision]) -> str:
    """
    Put the stored values back into a line redacted on disk (P-D11 b), so the
    review shows it as it was. private_here = private decisions located at
    this file + line, in decisions.json order. Placeholders are matched by
    label first, then the remaining ones in order; unmatched ones stay.
    Display only -- never written anywhere.
    """
    slots = list(_PLACEHOLDER_RE.finditer(line_text))
    if not slots or not private_here:
        return line_text
    free = list(private_here)
    chosen: dict[int, dec.Decision] = {}
    for i, slot in enumerate(slots):                                   # pass 1: same label
        match = next((d for d in free if d.label == slot.group("label")), None)
        if match is not None:
            chosen[i] = match
            free.remove(match)
    for i in range(len(slots)):                                        # pass 2: the rest, in order
        if i not in chosen and free:
            chosen[i] = free.pop(0)
    for i in sorted(chosen, reverse=True):                             # from the end: offsets stay valid
        slot = slots[i]
        line_text = line_text[:slot.start()] + chosen[i].value + line_text[slot.end():]
    return line_text


def build_reviews(data: dec.DecisionFile, *, pending: list[str] | tuple = (), order: str = "upload",
                  selection: str | None = None, pending_only: bool = False) -> list[FileReview]:
    """
    One screen per file holding stored decisions. order "upload": pending files
    first; "access": by category counts only. selection: one file path.
    """
    per_file: dict[str, dict[str, list[int | None]]] = {}
    private_at: dict[tuple[str, int], list[dec.Decision]] = {}       # (file, line) -> private decisions there
    by_id = {d.id: d for d in data.decisions}
    for decision in data.decisions:
        for loc in decision.locations:
            per_file.setdefault(loc["file"], {}).setdefault(decision.id, []).append(loc.get("line"))
            if decision.is_private and loc.get("line") is not None:
                here = private_at.setdefault((loc["file"], loc["line"]), [])
                if decision not in here:
                    here.append(decision)

    pending_set = set(pending)
    screens = []
    for file, located in per_file.items():
        if selection is not None and file != selection or pending_only and file not in pending_set:
            continue
        lines = _read_lines(file)
        restored: dict[int, str] = {}                                  # line -> rebuilt text (shared by values)
        screen = FileReview(file, pending=file in pending_set)
        for did, numbers in located.items():
            decision = by_id[did]
            known = sorted({n for n in numbers if n is not None})
            line = known[0] if known else None
            text = ""
            if line is not None and 0 < line <= len(lines):
                if line not in restored:
                    restored[line] = _restore_line(lines[line - 1], private_at.get((file, line), []))
                text = restored[line]
            start, shown = _find(text, decision)
            screen.items.append(ReviewItem(decision, line, text, start, shown, max(1, len(set(numbers))),
                                           category(decision)))
        screen.items.sort(key=lambda i: (i.category, i.line is None, i.line or 0, i.decision.entity_type))
        screens.append(screen)

    def key(s: FileReview) -> tuple:
        rank = (-s.count(CRITICAL), -s.count(HIGH), s.file)
        return rank if order == "access" else (not s.pending, *rank)

    return sorted(screens, key=key)


# --- rendering ----------------------------------------------------------------------------------------

def _use_ansi(stdout: TextIO) -> bool:
    isatty = getattr(stdout, "isatty", None)
    return bool(isatty and isatty()) and "NO_COLOR" not in os.environ


def _highlight(value: str, ansi: bool) -> str:
    marked = f"»{value}«"
    return f"{_ANSI_REVERSE}{marked}{_ANSI_RESET}" if ansi else marked


def _all_public(screen: FileReview) -> bool:
    return all(not item.decision.is_private for item in screen.items)


def render(screen: FileReview, position: int, total: int, ansi: bool = False) -> str:
    n_private = sum(item.decision.is_private for item in screen.items)
    head = f"File {position}/{total}   {screen.file}   {len(screen.items)} value(s) · {n_private} private"
    lines = [RULE, head + (" · PENDING REVIEW" if screen.pending else "")]
    toggle = f"public for all: {'on' if _all_public(screen) else 'off'}"
    lines.append(("  ⚠ critical values in this file · " if screen.count(CRITICAL) else "  ") + toggle)
    current = None
    for n, item in enumerate(screen.items, start=1):
        if item.category != current:
            current = item.category
            lines.append(f"  {CATEGORY_NAMES[current]}")
        decision = item.decision
        shown = "PRIVATE " if decision.is_private else "public  "
        note = ""
        if decision.is_private:
            note = (decision.label or pv.default_label(decision.entity_type)) + (
                " (default)" if decision.source == "default_secret" else "")
        where = f"l.{item.line}" if item.line is not None else "l.?"
        if item.start >= 0:
            end = item.start + len(item.shown)
            text = item.line_text[:item.start] + _highlight(item.shown, ansi) + item.line_text[end:]
        else:
            text = f"value {_highlight(item.shown, ansi)}"              # line unknown or value gone from it
        extras = []
        if item.count > 1:
            extras.append(f"×{item.count}")
        if (others := len(set(decision.files()) - {screen.file})):
            extras.append(f"also in {others} file(s)")
        tail = f"   ({', '.join(extras)})" if extras else ""
        lines.append(f"  [{n}] {shown} {decision.entity_type:<16} {note:<20} {where:<6} │ {text}{tail}")
    lines.append(MENU)
    return "\n".join(lines)


# --- input --------------------------------------------------------------------------------------------

def _ask(prompt: str, stdin: TextIO, stdout: TextIO) -> str | None:
    """One answer line; None on EOF."""
    stdout.write(prompt)
    stdout.flush()
    line = stdin.readline()
    return None if line == "" else line.strip()


def _parse_numbers(answer: str, count: int) -> list[int] | None:
    """'2 4', '2,4', '2-4' -> [2, 4] / [2, 3, 4]; None when anything is invalid or out of range."""
    numbers: list[int] = []
    for token in answer.replace(",", " ").split():
        match = _RANGE_RE.match(token)
        if not match:
            return None
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if not 1 <= low <= high <= count:
            return None
        numbers.extend(range(low, high + 1))
    return list(dict.fromkeys(numbers)) or None


def _label_ok(label: str, decision: dec.Decision) -> bool:
    """A label must not contain the value itself."""
    kind = pv.match_kind_for(decision.entity_type, decision.value)
    probe = [pv.PrivateValue("probe", label, kind, pv.normalize(decision.value, kind),
                             tuple(decision.surface_forms or [decision.value]))]
    return bool(label.strip()) and not pv.find_private_spans(label, probe)


def _make_private(decision: dec.Decision, data: dec.DecisionFile) -> None:
    dec.set_private(decision, decision.label or pv.default_label(decision.entity_type), data)


@dataclass(frozen=True)
class ReviewOutcome:
    done: bool                        # q pressed (Done); False = input ended without Done
    changes: int                      # switches / labels saved


def review(
    screens: list[FileReview],
    data: dec.DecisionFile,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    save: Callable[[dec.DecisionFile], None] | None = None,
) -> ReviewOutcome:
    """Walk the screens until Done; every change updates `data` in place and is saved at once."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    save = save or dec.save
    ansi = _use_ansi(stdout)
    changes = 0
    position = 0
    if not screens:
        return ReviewOutcome(True, 0)

    def saved(queued: list[str]) -> None:
        nonlocal changes
        save(data)
        changes += 1
        if queued:
            stdout.write(f"{len(queued)} document(s) will be rebuilt from raw when the index is built.\n")

    while True:
        screen = screens[position]
        items = screen.items
        stdout.write(render(screen, position + 1, len(screens), ansi) + "\n")
        answer = _ask("> ", stdin, stdout)
        if answer is None:
            return ReviewOutcome(False, changes)
        command = answer.lower()
        if command == "q":
            return ReviewOutcome(True, changes)
        if command == "":
            position = (position + 1) % len(screens)
            continue
        if command == "a":
            private = [i.decision for i in items if i.decision.is_private]
            secrets = [d for d in private if _is_secret(d)]
            if secrets:
                confirm = _ask(f"This makes {len(secrets)} secret(s) public everywhere. Type yes to confirm: ",
                               stdin, stdout)
                if confirm is None:
                    return ReviewOutcome(False, changes)
                if confirm.lower() != "yes":
                    stdout.write("Nothing changed.\n")
                    continue
            queued: list[str] = []
            for decision in private:
                queued += dec.set_public(decision, data)
            saved(queued)
            continue
        if command == "x":
            for item in items:
                if not item.decision.is_private:
                    _make_private(item.decision, data)
            saved([])
            continue
        if command.startswith("l"):
            picked = _parse_numbers(answer[1:], len(items))
            if not picked or len(picked) != 1:
                stdout.write("Use l<n>, e.g. l3.\n")
                continue
            decision = items[picked[0] - 1].decision
            if not decision.is_private:
                stdout.write("Only private values have a label. Switch it first.\n")
                continue
            label = _ask(f"Label [{decision.label or pv.default_label(decision.entity_type)}]: ", stdin, stdout)
            if label is None:
                return ReviewOutcome(False, changes)
            if not label:
                continue
            if not _label_ok(label, decision):
                stdout.write("The label must not contain the private value itself.\n")
                continue
            decision.label = label
            saved([])
            continue
        numbers = _parse_numbers(answer, len(items))
        if numbers is None:
            stdout.write("Please press Enter, type value numbers (e.g. 2 4 or 2-4), l<n>, a, x or q.\n")
            continue
        queued = []
        for n in numbers:
            decision = items[n - 1].decision
            if decision.is_private:
                queued += dec.set_public(decision, data)
            else:
                _make_private(decision, data)
            if (others := len(set(decision.files()) - {screen.file})):
                stdout.write(f"[{n}] also changes {others} other file(s).\n")
        saved(queued)
