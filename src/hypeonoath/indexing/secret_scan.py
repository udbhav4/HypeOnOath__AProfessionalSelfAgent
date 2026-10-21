"""
Secret scan for the index-build REPORT (guideline v4, 3A.13; privacy plan v3, 7.2).

Personal information moved to hypeonoath.privacy (Presidio + owner review +
redaction). Secrets are found by detect-secrets in two places:
  * the privacy review (hypeonoath.privacy.detector, via _secret_matches),
    where the owner marks each one private (redacted) or keep;
  * this module, on the raw code text (code_chunker) and on every final
    chunk (build_index) -- so the report lists what is still in the index:
    secrets the owner kept or allowlisted.

What it never does:
  * stop the run -- stopping is the privacy review's job;
  * record the matched value, masked or not -- a Finding has no field for it,
    so neither the report file, the console nor a future API/UI can leak it;
  * change content.

Reviewed, accepted findings may also go in corpus/.scan_allowlist.yaml as
{file, line?, finding_type, reason}; they are hidden from the warnings but
counted in the summary (the privacy review treats them as "keep"). The
report (.logs/scan_warnings.jsonl) holds the LATEST run only.
"""
from __future__ import annotations

import json
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
    """One possible secret hit. Deliberately has NO value field."""

    file: str
    line: int | None
    finding_type: str
    detector: str


@dataclass
class ScanReport:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    findings: list[Finding] = field(default_factory=list)
    allowlisted: int = 0


def _secret_matches(lines: list[str], filename: str) -> list[tuple[int, str, str | None]]:
    """
    detect-secrets in FILE mode over `lines` -> [(1-based line, type, value)].

    The value is for the privacy review ONLY (in memory, never persisted or
    logged); everything in this module drops it right away (_secret_hits).

    Uses _process_line_based_plugins (what scan_file() runs per file) rather
    than the public scan_line(): scan_line is detect-secrets' ad-hoc
    "scan this string" mode, which treats every line as a quoted secret and
    flags nearly every line as high-entropy. Private API, but the version is
    pinned (config.PINNED_VERSIONS) and covered by tests/test_secret_scan.py.
    """
    from detect_secrets.core.scan import _process_line_based_plugins
    from detect_secrets.settings import default_settings

    with default_settings():
        return [
            (secret.line_number, secret.type, getattr(secret, "secret_value", None))
            for secret in _process_line_based_plugins(lines=list(enumerate(lines, start=1)), filename=filename)
        ]


def _secret_hits(lines: list[str], filename: str) -> dict[int, set[str]]:
    """{1-based line: {secret types}} -- values dropped right here."""
    hits: dict[int, set[str]] = {}
    for line_number, secret_type, _value in _secret_matches(lines, filename):
        hits.setdefault(line_number, set()).add(secret_type)
    return hits


# --- scanning -------------------------------------------------------------------

def scan_text(text: str, file: str, first_line: int = 1) -> list[Finding]:
    """Scan verbatim file text; line numbers are first_line-based."""
    lines = text.split("\n")
    secrets = _secret_hits(lines, file)
    findings: list[Finding] = []
    for index in range(1, len(lines) + 1):
        line_no = first_line + index - 1
        findings += [Finding(file, line_no, t, "detect-secrets") for t in sorted(secrets.get(index, ()))]
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
            types = sorted(secrets.get(index + 1, ()))
            if types:
                line_no = _line_in_source(chunk, line, index)
                findings += [Finding(chunk.source_document, line_no, t, "detect-secrets") for t in types]
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
        return f"{bar}\nSecret scan: no secrets left in the index{tail}.\n{bar}"
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.finding_type] = counts.get(finding.finding_type, 0) + 1
    lines = [
        bar,
        f"NOTE: {len(report.findings)} possible secret(s) still in the index "
        f"({report.allowlisted} allowlisted).",
        "Each was kept in the privacy review (hypeonoath-privacy review). If one is real, mark it",
        "private (hypeonoath-privacy set ID --decision private) and rebuild before any git push.",
        "By type: " + ", ".join(f"{t}={n}" for t, n in sorted(counts.items())),
        *[f"  {f.file}:{f.line if f.line is not None else '?'}: {f.finding_type}" for f in report.findings],
        f"Report: {config.SCAN_WARNINGS_PATH}   Allowlist: {config.SCAN_ALLOWLIST_PATH}",
        bar,
    ]
    return "\n".join(lines)
