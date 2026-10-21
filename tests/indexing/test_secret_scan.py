"""secret_scan.py: secrets only (PII moved to hypeonoath.privacy, plan v3 7.2), type + line only,
never the value (3A.13)."""
from __future__ import annotations

import json

import pytest

from hypeonoath.core.records import Chunk
from hypeonoath.indexing import secret_scan as scan

from tests.conftest import FIXTURES

FAKE_KEY = "AKIAIOSFODNN7EXAMPLE"
FAKE_EMAIL = "jane.doe@example.com"


def test_fake_secret_found_with_type_and_line_only():
    findings = scan.scan_text((FIXTURES / "fake_secret.py").read_text(), "fake_secret.py")
    assert [(f.line, f.finding_type) for f in findings] == [(2, "AWS Access Key")]
    assert not hasattr(findings[0], "value") and FAKE_KEY not in repr(findings)


def test_plain_code_has_no_entropy_noise():
    # Regression: detect-secrets' ad-hoc scan_line() flagged nearly every line.
    findings = scan.scan_text((FIXTURES / "small_functions.py").read_text(), "small_functions.py")
    assert findings == []


def test_personal_info_is_no_longer_scanned_here():
    # PII is detected, reviewed and redacted by hypeonoath.privacy now.
    assert scan.scan_text((FIXTURES / "fake_pii.md").read_text(), "fake_pii.md") == []
    assert not hasattr(scan, "pii_findings_in_line")


def test_secret_matches_expose_the_value_in_memory_only():
    lines = (FIXTURES / "fake_secret.py").read_text().split("\n")
    assert scan._secret_matches(lines, "fake_secret.py") == [(2, "AWS Access Key", FAKE_KEY)]
    assert scan._secret_hits(lines, "fake_secret.py") == {2: {"AWS Access Key"}}


def test_scan_chunks_maps_lines_for_code_and_summary():
    code = Chunk("c1", "# [P] f.py > x  [lines 10-12]\nx = 1\nkey = 'AKIAIOSFODNN7EXAMPLE'\n", "corpus/code/f.py",
                 "code", line_start=10, line_end=12)
    summary = Chunk("s1", "Project: P\nDescription: key 'AKIAIOSFODNN7EXAMPLE'\nCode files: f.py",
                    "corpus/code/_projects/p", "project_summary")
    found = {(f.file, f.line, f.finding_type) for f in scan.scan_chunks([code, summary])}
    assert ("corpus/code/f.py", 11, "AWS Access Key") in found
    assert ("corpus/code/_projects/p", 2, "AWS Access Key") in found


def test_report_dedupes_allowlists_and_never_writes_values(tmp_path):
    secret = (FIXTURES / "fake_secret.py").read_text()
    findings = scan.scan_text(secret, "fake_secret.py") * 2                                   # duplicates
    findings += scan.scan_text(secret, "other.py")
    allowlist_path = tmp_path / "allow.yaml"
    allowlist_path.write_text("- {file: other.py, finding_type: AWS Access Key, reason: documented test key}\n")
    report = scan.build_report(findings, scan.load_allowlist(allowlist_path))
    assert report.allowlisted == 1
    assert {(f.file, f.finding_type) for f in report.findings} == {("fake_secret.py", "AWS Access Key")}
    out = tmp_path / "scan_warnings.jsonl"
    scan.write_report(report, out)
    raw = out.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert all(set(r) == {"file", "line", "finding_type", "detector", "run_id", "timestamp"} for r in records)
    summary = scan.format_summary(report)
    for secret in (FAKE_KEY, FAKE_EMAIL, "23456789"):
        assert secret not in raw and secret not in summary
    assert "1 allowlisted" in summary and "privacy review" in summary


def test_allowlist_line_specific(tmp_path):
    finding = scan.Finding("a.md", 3, "email_address", "pii-regex")
    assert scan.build_report([finding], [{"file": "a.md", "line": 4, "finding_type": "email_address"}]).findings
    assert not scan.build_report([finding], [{"file": "a.md", "line": 3, "finding_type": "EMAIL_ADDRESS"}]).findings


def test_empty_summary():
    assert "no secrets left in the index" in scan.format_summary(scan.ScanReport())


def test_bad_allowlist_shape(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("file: x\n")
    with pytest.raises(ValueError):
        scan.load_allowlist(path)
