"""secret_scan.py: warning-only, type + line only, never the value (3A.13)."""
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


def test_fake_pii_types_and_lines():
    findings = scan.scan_text((FIXTURES / "fake_pii.md").read_text(), "fake_pii.md")
    assert {(f.line, f.finding_type) for f in findings} == {(3, "email_address"), (5, "phone_number")}


@pytest.mark.parametrize("line, expected", [
    ("mail me: someone@example.org", ["email_address"]),
    ("+49 1637223004 | x@y.de", ["email_address", "phone_number"]),
    ("Eulerweg 1, 64347 Griesheim.", ["street_address"]),
    ("221 Baker Street, London", ["street_address"]),
    ("IBAN DE89 3704 0044 0532 0130 00", ["id_number"]),
    ("customer id 123456789012", ["id_number"]),
    ("July 2018 - July 2022 Dehradun", []),          # date ranges are not phones
    ("Grade: 8.41/10, 20.09.2026", []),
    ("total += values[12] * 12", []),
])
def test_pii_patterns(line, expected):
    assert scan.pii_findings_in_line(line) == expected


def test_scan_chunks_maps_lines_for_code_and_summary():
    code = Chunk("c1", "# [P] f.py > x  [lines 10-12]\nx = 1\nkey = 'AKIAIOSFODNN7EXAMPLE'\n", "corpus/code/f.py",
                 "code", line_start=10, line_end=12)
    summary = Chunk("s1", "Project: P\nDescription: mail jane.doe@example.com\nCode files: f.py",
                    "corpus/code/_projects/p", "project_summary")
    found = {(f.file, f.line, f.finding_type) for f in scan.scan_chunks([code, summary])}
    assert ("corpus/code/f.py", 11, "AWS Access Key") in found
    assert ("corpus/code/_projects/p", 2, "email_address") in found


def test_report_dedupes_allowlists_and_never_writes_values(tmp_path):
    findings = scan.scan_text((FIXTURES / "fake_pii.md").read_text(), "fake_pii.md") * 2   # duplicates
    findings += scan.scan_text((FIXTURES / "fake_secret.py").read_text(), "fake_secret.py")
    allowlist_path = tmp_path / "allow.yaml"
    allowlist_path.write_text("- {file: fake_pii.md, finding_type: email_address, reason: public test email}\n")
    report = scan.build_report(findings, scan.load_allowlist(allowlist_path))
    assert report.allowlisted == 1
    assert {(f.file, f.finding_type) for f in report.findings} == {("fake_pii.md", "phone_number"),
                                                                   ("fake_secret.py", "AWS Access Key")}
    out = tmp_path / "scan_warnings.jsonl"
    scan.write_report(report, out)
    raw = out.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert all(set(r) == {"file", "line", "finding_type", "detector", "run_id", "timestamp"} for r in records)
    summary = scan.format_summary(report)
    for secret in (FAKE_KEY, FAKE_EMAIL, "23456789"):
        assert secret not in raw and secret not in summary
    assert "1 allowlisted" in summary and "NOT stopped" in summary


def test_allowlist_line_specific(tmp_path):
    finding = scan.Finding("a.md", 3, "email_address", "pii-regex")
    assert scan.build_report([finding], [{"file": "a.md", "line": 4, "finding_type": "email_address"}]).findings
    assert not scan.build_report([finding], [{"file": "a.md", "line": 3, "finding_type": "EMAIL_ADDRESS"}]).findings


def test_empty_summary():
    assert "no warnings" in scan.format_summary(scan.ScanReport())


def test_bad_allowlist_shape(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("file: x\n")
    with pytest.raises(ValueError):
        scan.load_allowlist(path)
