"""
Secret + personal-info scan -- WARNING-ONLY (guideline v4, Sections 0, 3A.13;
design doc ADR-15, accepted risk in 5.3).

What it does:
  * scans raw decoded code text (code_chunker, Step 3) and every final chunk
    of every content type (build_index, step 3);
  * secrets via detect-secrets' built-in plugins (pip-only, no system binary);
  * personal info via simple named regexes (email, phone, street address,
    ID-number-like digit runs / IBAN) -- cheap and good enough for a warning.

What it never does:
  * stop the run (user decision 2026-09-30) -- the indexer always continues;
  * record the matched value, masked or not -- a Finding has no field for it,
    so neither the report file, the console nor a future API/UI can leak it;
  * change content (no redaction -- it can silently break code).

Reviewed, accepted findings go in corpus/.scan_allowlist.yaml as
{file, line?, finding_type, reason}; they are hidden from the warnings but
counted in the summary. The report (.logs/scan_warnings.jsonl) holds the
LATEST run only -- it is what the Phase 2 API will expose.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

from hypeonoath.core import config
from hypeonoath.core.records import Chunk


@dataclass(frozen=True)
class Finding:
    """One possible secret / personal-info hit. Deliberately has NO value field."""

    file: str
    line: int | None
    finding_type: str
    detector: str


@dataclass
class ScanReport:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    findings: list[Finding] = field(default_factory=list)
    allowlisted: int = 0


# --- personal-info detectors --------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w+.])\+?\(?\d[\d\s()-]{6,}\d(?![\w.])")
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z]+\s+(?:Street|St\.|Road|Rd\.|Avenue|Ave\.?|Lane|Drive|Boulevard)\b"
    r"|\b[A-ZÄÖÜ][a-zäöüß]+(?:straße|strasse|str\.|weg|platz|allee|gasse|ring)\s+\d{1,4}[a-z]?\b"
)
_ID_NUMBER_RE = re.compile(r"(?<![\w.+-])\d{9,}(?![\w.])")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")


def _phone_like(candidate: str) -> bool:
    digits = sum(ch.isdigit() for ch in candidate)
    has_sep = any(ch in candidate for ch in " -()")
    return 9 <= digits <= 15 and (candidate.startswith(("+", "(")) or has_sep)


def pii_findings_in_line(line: str) -> list[str]:
    """Names of personal-info types present in one line (no values returned)."""
    types: list[str] = []
    if _EMAIL_RE.search(line):
        types.append("email_address")
    phones = [m.group(0) for m in _PHONE_CANDIDATE_RE.finditer(line) if _phone_like(m.group(0).strip())]
    if phones:
        types.append("phone_number")
    if _STREET_RE.search(line):
        types.append("street_address")
    phone_text = " ".join(phones)
    if _IBAN_RE.search(line) or any(m.group(0) not in phone_text for m in _ID_NUMBER_RE.finditer(line)):
        types.append("id_number")
    return types


def _secret_hits(lines: list[str], filename: str) -> dict[int, set[str]]:
    """
    detect-secrets in FILE mode over `lines` -> {1-based line index: {types}}.
    Values are dropped right here.

    Uses _process_line_based_plugins (what scan_file() runs per file) rather
    than the public scan_line(): scan_line is detect-secrets' ad-hoc
    "scan this string" mode, which treats every line as a quoted secret and
    flags nearly every line as high-entropy. Private API, but the version is
    pinned (config.PINNED_VERSIONS) and covered by tests/test_secret_scan.py.
    """
    from detect_secrets.core.scan import _process_line_based_plugins
    from detect_secrets.settings import default_settings

    hits: dict[int, set[str]] = {}
    with default_settings():
        for secret in _process_line_based_plugins(lines=list(enumerate(lines, start=1)), filename=filename):
            hits.setdefault(secret.line_number, set()).add(secret.type)
    return hits


# --- scanning -------------------------------------------------------------------

def scan_text(text: str, file: str, first_line: int = 1) -> list[Finding]:
    """Scan verbatim file text; line numbers are first_line-based."""
    lines = text.split("\n")
    secrets = _secret_hits(lines, file)
    findings: list[Finding] = []
    for index, line in enumerate(lines, start=1):
        line_no = first_line + index - 1
        findings += [Finding(file, line_no, t, "detect-secrets") for t in sorted(secrets.get(index, ()))]
        findings += [Finding(file, line_no, t, "pii-regex") for t in pii_findings_in_line(line)]
    return findings


@lru_cache(maxsize=256)
def _source_lines(source_document: str) -> tuple[str, ...]:
    path = config.PROJECT_ROOT / source_document
    if not path.is_file():
        return ()
    return tuple(path.read_text(encoding="utf-8", errors="replace").split("\n"))


def _line_in_source(chunk: Chunk, text_line: str, index: int) -> int | None:
    """Map 0-based line `index` of a chunk's text back to a line of its source file."""
    if chunk.content_type == "code" and chunk.line_start is not None:
        return chunk.line_start + index - 1 if index > 0 else None   # line 0 = header
    if chunk.content_type == "project_summary":
        return index + 1
    needle = text_line.strip()
    lines = _source_lines(chunk.source_document)
    if needle and lines:
        for number, source_line in enumerate(lines, start=1):
            stripped = source_line.strip()
            if needle in source_line or (len(stripped) >= 8 and stripped in needle):
                return number
    return chunk.line_start


def scan_chunks(chunks: list[Chunk]) -> list[Finding]:
    """Scan the exact text that will be stored in Chroma, for every chunk."""
    findings: list[Finding] = []
    for chunk in chunks:
        lines = chunk.text.split("\n")
        secrets = _secret_hits(lines, chunk.source_document)
        for index, line in enumerate(lines):
            types = [(t, "detect-secrets") for t in sorted(secrets.get(index + 1, ()))]
            types += [(t, "pii-regex") for t in pii_findings_in_line(line)]
            if types:
                line_no = _line_in_source(chunk, line, index)
                findings += [Finding(chunk.source_document, line_no, t, d) for t, d in types]
    return findings


# --- allowlist + report -------------------------------------------------------------

def load_allowlist(path: Path | None = None) -> list[dict]:
    path = config.SCAN_ALLOWLIST_PATH if path is None else path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    if not isinstance(entries, list):
        raise ValueError(f"{path} must be a YAML list of {{file, line?, finding_type, reason}}")
    return [e for e in entries if isinstance(e, dict) and e.get("file") and e.get("finding_type")]


def _allowlisted(finding: Finding, allowlist: list[dict]) -> bool:
    for entry in allowlist:
        if (
            entry["file"] == finding.file
            and str(entry["finding_type"]).lower() == finding.finding_type.lower()
            and (entry.get("line") is None or entry.get("line") == finding.line)
        ):
            return True
    return False


def build_report(findings: list[Finding], allowlist: list[dict] | None = None) -> ScanReport:
    """Deduplicate (same file/line/type from raw text and chunk scans) + apply allowlist."""
    allowlist = load_allowlist() if allowlist is None else allowlist
    report = ScanReport()
    unique = sorted(set(findings), key=lambda f: (f.file, f.line or 0, f.finding_type))
    for finding in unique:
        if _allowlisted(finding, allowlist):
            report.allowlisted += 1
        else:
            report.findings.append(finding)
    return report


def write_report(report: ScanReport, path: Path | None = None) -> None:
    """Overwrite with the latest run's findings (one JSON line each, no values)."""
    path = config.SCAN_WARNINGS_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        for finding in report.findings:
            record = {**asdict(finding), "run_id": report.run_id, "timestamp": timestamp}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_summary(report: ScanReport) -> str:
    """The end-of-run block, printed LAST so it can't scroll away."""
    bar = "=" * 72
    if not report.findings:
        tail = f" ({report.allowlisted} allowlisted)" if report.allowlisted else ""
        return f"{bar}\nSecret / personal-info scan: no warnings{tail}.\n{bar}"
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.finding_type] = counts.get(finding.finding_type, 0) + 1
    lines = [
        bar,
        f"WARNING: {len(report.findings)} possible secret / personal-info finding(s) "
        f"({report.allowlisted} allowlisted). The run was NOT stopped:",
        "flagged content IS in the index. Review before any git push of the index folder.",
        "By type: " + ", ".join(f"{t}={n}" for t, n in sorted(counts.items())),
        *[f"  {f.file}:{f.line if f.line is not None else '?'}: {f.finding_type}" for f in report.findings],
        f"Report: {config.SCAN_WARNINGS_PATH}   Allowlist: {config.SCAN_ALLOWLIST_PATH}",
        bar,
    ]
    return "\n".join(lines)
