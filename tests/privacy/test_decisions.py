"""privacy.decisions: storage, grouping, allowlist, fingerprint and the v8
rules R1, R5, R2, R3, RS + the review writes (privacy plan v8, 5.1, 6.4, 6.4.1; tests 14.4)."""
from __future__ import annotations

import json
import os

import pytest

from hypeonoath.core import config
from hypeonoath.privacy import decisions as dec

from tests.factories import candidate

PHONE = "+49 151 2345678"
KEY = "AKIAIOSFODNN7EXAMPLE"


@pytest.fixture(autouse=True)
def corpus_root(tmp_path, monkeypatch):
    """Location files must exist (stale locations of deleted files are dropped)."""
    root = tmp_path / "root"
    for rel in ("corpus/converted/cv.md", "corpus/code/a.py", "x.py", "a.md", "b.md", "c.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    return root


def groups(*cands):
    return dec.group_candidates(list(cands))


def decide(value, entity_type, decision, label=None, **kw):
    (group,) = groups(candidate(value, entity_type, **kw))
    return dec.new_decision(group, decision, label)


def data_with(*decisions):
    return dec.DecisionFile(list(decisions))


def resolve(data, *cands):
    return dec.resolve_all(groups(*cands), data)


# --- storage -----------------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    path = tmp_path / "d" / "decisions.json"
    original = data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number"), decide("Jane Doe", "PERSON", "keep"))
    original.decisions[0].add_alias("DE_TAX_ID", "id", "491512345678", "R2")
    original.pending_restore = ["CV.pdf"]
    dec.save(original, path)
    loaded = dec.load(path)
    assert [d.to_dict() for d in loaded.decisions] == [d.to_dict() for d in original.decisions]
    assert loaded.pending_restore == ["CV.pdf"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 3 and saved["pending_restore"] == ["CV.pdf"]


def test_missing_file_loads_empty():
    data = dec.load()
    assert data.decisions == [] and data.pending_restore == []


def test_atomic_save_leaves_old_file_on_crash(tmp_path, monkeypatch):
    path = tmp_path / "decisions.json"
    dec.save(data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number")), path)
    before = path.read_text(encoding="utf-8")

    def crash(src, dst):
        raise OSError("power cut between tmp write and replace")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        dec.save(dec.DecisionFile(), path)
    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob(".decisions.*.tmp"))          # temp file cleaned up


@pytest.mark.parametrize("version", [1, 2])
def test_migration_to_schema_3_keeps_ids_keep_entries_and_fingerprint(tmp_path, version):
    path = tmp_path / "decisions.json"
    entries = [
        {"id": "pv_3f2a9c1e07b4", "decision": "private", "label": "date of birth", "entity_type": "DATE_TIME",
         "match_kind": "date", "value": "12.03.1999", "normalized": "1999-03-12",
         "surface_forms": ["12.03.1999"], "source": "detected",
         "first_seen": {"file": "a.md", "line": 1}, "decided_at": "2026-10-15T18:02:11Z"},
        {"id": "pv_a81c55d0e9f2", "decision": "keep", "label": None, "entity_type": "PERSON",
         "match_kind": "text", "value": "Jane Doe", "normalized": "jane doe",
         "first_seen": {"file": "b.md", "line": 2}},
    ]
    if version == 2:
        for e in entries:
            e["value_keys"], e["aliases"] = [], []
    path.write_text(json.dumps({"schema_version": version, "entries": entries}), encoding="utf-8")
    data = dec.load(path)
    assert [(d.id, d.decision) for d in data.decisions] == [("pv_3f2a9c1e07b4", "private"), ("pv_a81c55d0e9f2", "keep")]
    assert data.decisions[0].locations == [{"file": "a.md", "line": 1}] and data.pending_restore == []
    assert "date:1999-03-12" in data.decisions[0].value_keys
    fp = dec.privacy_fingerprint(data.decisions)
    dec.save(data, path)
    again = dec.load(path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3
    assert dec.privacy_fingerprint(again.decisions) == fp


@pytest.mark.parametrize("payload", [{"schema_version": 9, "entries": []},
                                     {"schema_version": 3, "entries": [{"id": "x", "decision": "maybe"}]},
                                     {"schema_version": 3, "entries": [], "pending_restore": "nope"}])
def test_bad_files_rejected(tmp_path, payload):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        dec.load(path)


def test_repr_never_shows_the_value():
    decision = decide(PHONE, "PHONE_NUMBER", "private", "phone number")
    (group,) = groups(candidate(PHONE))
    for text in (repr(decision), repr(group), repr(group.candidates[0]), repr(data_with(decision))):
        assert "2345678" not in text


# --- grouping ---------------------------------------------------------------------------------------

def test_same_value_in_three_files_is_one_group_with_locations():
    (group,) = groups(*(candidate(PHONE, file=f"corpus/converted/{n}.md") for n in "abc"))
    assert len(group.locations) == 3 and group.surface_forms == [PHONE]
    decision = dec.new_decision(group, "keep", None)
    assert decision.files() == ["corpus/converted/a.md", "corpus/converted/b.md", "corpus/converted/c.md"]


def test_spellings_group_together():
    found = groups(candidate(PHONE), candidate("+49-151-2345678"), candidate("Jane Doe", "PERSON"))
    phone = next(g for g in found if g.entity_type == "PHONE_NUMBER")
    assert phone.surface_forms == [PHONE, "+49-151-2345678"]


def test_unnormalisable_values_are_dropped():
    assert groups(candidate("---", "SOMETHING")) == []


# --- fingerprint ---------------------------------------------------------------------------------------

def test_fingerprint_changes_only_with_private_entries():
    private = decide(PHONE, "PHONE_NUMBER", "private", "phone number")
    fp = dec.privacy_fingerprint([private])
    assert dec.privacy_fingerprint([]) == "none"
    assert dec.privacy_fingerprint([private, decide("Jane Doe", "PERSON", "keep")]) == fp
    private.label = "mobile"
    assert dec.privacy_fingerprint([private]) != fp


# --- legacy allowlist --------------------------------------------------------------------------------

def test_legacy_allowlist_entries_count_as_keep(tmp_path):
    allow = tmp_path / "allow.yaml"
    allow.write_text("- {file: corpus/converted/cv.md, finding_type: phone_number, reason: public}\n"
                     "- {file: x.py, line: 2, finding_type: AWS Access Key}\n"
                     "- {file: corpus/converted/cv.md, finding_type: id_number}\n")
    rule = dec.import_scan_allowlist(allow)
    kept, n = rule.filter([
        candidate(PHONE),
        candidate(KEY, "SECRET:AWS Access Key", file="x.py", line=2),
        candidate(KEY, "SECRET:AWS Access Key", file="x.py", line=3),
        candidate("DE89 3704 0044 0532 0130 00", "IBAN_CODE"),
        candidate("Jane Doe", "PERSON"),
    ])
    assert n == 3
    assert [(c.entity_type, c.line) for c in kept] == [("SECRET:AWS Access Key", 3), ("PERSON", 4)]


def test_missing_allowlist_is_empty(tmp_path):
    assert not dec.import_scan_allowlist(tmp_path / "none.yaml")


# --- rules R1, R5, R2, R3, RS, NEW (6.4.1) ------------------------------------------------------------------

def test_r1_same_id_keeps_state_and_records_spelling():
    data = data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number"))
    result = resolve(data, candidate("+49-151-2345678"))
    assert result.rules == {"R1": 1} and result.items[0][1].state == "private" and result.changed
    assert data.decisions[0].surface_forms == [PHONE, "+49-151-2345678"]


@pytest.mark.parametrize("state, label", [("private", "phone number"), ("keep", None)])
def test_r2_value_under_a_new_type_follows_value_wide(state, label):
    data = data_with(decide(PHONE, "PHONE_NUMBER", state, label))
    result = resolve(data, candidate("491512345678", "DE_TAX_ID"))
    (_, res), = result.items
    assert result.rules == {"R2": 1} and res.state == state
    assert [(a.entity_type, a.rule) for a in data.decisions[0].aliases] == [("DE_TAX_ID", "R2")]
    assert result.extended == {(label or "public"): 1}


def test_r2_alias_form_is_matched_by_redaction():
    data = data_with(decide("12.03.1999", "DATE_TIME", "private", "date of birth"))
    resolve(data, candidate("12031999", "DE_ID_DATE_CUSTOM"))
    values = dec.private_values(data.decisions)
    assert {(v.match_kind, v.normalized) for v in values} >= {("date", "1999-03-12"), ("id", "12031999")}


def test_r5_conflict_private_wins_and_is_logged_without_values(isolated_logs):
    private = decide(PHONE, "PHONE_NUMBER", "private", "phone number")
    keep = decide("491512345678", "MONEY", "keep")
    data = data_with(private, keep)
    result = resolve(data, candidate(PHONE, "DE_TAX_ID"))
    assert result.rules == {"R5": 1} and result.items[0][1].state == "private"
    log = (isolated_logs / "privacy_log_path.jsonl").read_text(encoding="utf-8")
    record = json.loads(log.splitlines()[-1])
    assert record["event"] == "privacy_conflict_resolved" and record["keep_ids"] == [keep.id]
    assert "2345678" not in log


def test_r3_fully_covered_finding_stores_nothing():
    data = data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number"),
                     decide("12.03.1999", "DATE_TIME", "private", "date of birth"))
    before = [d.to_dict() for d in data.decisions]
    result = resolve(data, candidate(f"{PHONE} 12.03.1999", "SOME_COMBINED"))
    assert result.rules == {"R3": 1} and result.items[0][1].state == "covered"
    assert [d.to_dict() for d in data.decisions] == before


def test_rs_new_secret_becomes_private_by_default():
    data = dec.DecisionFile()
    result = resolve(data, candidate(KEY, "SECRET:AWS Access Key", file="corpus/code/a.py"))
    (decision,) = data.decisions
    assert result.rules == {"RS": 1} and result.default_secrets == 1
    assert (decision.decision, decision.source, decision.label) == ("private", "default_secret", "secret")


def test_existing_keep_beats_the_secret_default():
    data = data_with(decide(KEY, "URL", "keep"))
    result = resolve(data, candidate(KEY, "SECRET:AWS Access Key"))
    assert result.rules == {"R2": 1} and result.items[0][1].state == "keep"


def test_new_value_is_left_for_the_scan_to_store():
    data = dec.DecisionFile()
    groups = dec.group_candidates([candidate("Jane Doe", "PERSON", file="corpus/converted/cv.md"),
                                   candidate("Darmstadt", "LOCATION", file="corpus/converted/cv.md")])
    result = dec.resolve_all(groups, data)
    assert data.decisions == [] and result.changed is False
    assert [res.state for _g, res in result.items] == ["new", "new"]
    assert dec.set_keep_for_undecided(groups, data, source="detected") == 2           # what privacy.scan does
    assert {(d.decision, d.source) for d in data.decisions} == {("keep", "detected")}


def test_simulated_presidio_type_rename_adds_one_alias_once():
    data = data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number"))
    renamed = [candidate(PHONE, "PHONE_NUMBER_V2"), candidate(PHONE, "PHONE_NUMBER_V2", file="b.md")]
    first = resolve(data, *renamed)
    assert first.rules == {"R2": 1} and [a.entity_type for a in data.decisions[0].aliases] == ["PHONE_NUMBER_V2"]
    second = resolve(data, *renamed)
    assert second.rules == {"R1": 1} and second.changed is False


# --- locations (P-D32) -----------------------------------------------------------------------------------

def test_locations_refreshed_kept_when_redacted_dropped_when_file_gone():
    decision = decide(PHONE, "PHONE_NUMBER", "private", "phone number", file="a.md", line=4)
    decision.locations += [{"file": "b.md", "line": 9}, {"file": "gone.md", "line": 1}]
    data = data_with(decision)
    resolve(data, candidate(PHONE, file="a.md", line=7), candidate(PHONE, file="c.md", line=2))
    assert sorted((loc["file"], loc["line"]) for loc in decision.locations) == [
        ("a.md", 7), ("b.md", 9), ("c.md", 2)]       # a refreshed, b kept (redacted there), gone dropped


# --- review writes (P-D28, P-D30, P-D33) ---------------------------------------------------------------------

def test_set_private_on_a_new_value_and_value_wide_on_an_existing_one():
    data = dec.DecisionFile()
    (group,) = groups(candidate("Jane Doe", "PERSON"))
    created = dec.set_private(group, "name", data)
    assert (created.decision, created.label, created.source) == ("private", "name", "review")
    (other_type,) = groups(candidate("Jane Doe", "US_DRIVER_LICENSE"))
    same = dec.set_private(other_type, "name", data)
    assert same is created and len(data.decisions) == 1
    assert ("US_DRIVER_LICENSE", "id", "JANEDOE") in created.forms()


def test_set_private_turns_keep_and_default_secret_into_review_decisions():
    keep = decide(PHONE, "PHONE_NUMBER", "keep")
    secret = decide(KEY, "SECRET:AWS Access Key", "private", "secret")
    secret.source = "default_secret"
    data = data_with(keep, secret)
    dec.set_private(keep, "phone number", data)
    dec.set_private(secret, "api key", data)
    assert keep.is_private and keep.label == "phone number"
    assert (secret.label, secret.source) == ("api key", "review")


def test_set_public_queues_raw_sources_only_for_private_values(monkeypatch):
    from hypeonoath.privacy import restore

    monkeypatch.setattr(restore, "raw_sources_for", lambda files: [f"raw:{f}" for f in files])
    private = decide(PHONE, "PHONE_NUMBER", "private", "phone number", file="corpus/converted/cv.md")
    keep = decide("Jane Doe", "PERSON", "keep")
    data = data_with(private, keep)
    assert dec.set_public(private, data) == ["raw:corpus/converted/cv.md"]
    assert private.decision == "keep" and private in data.decisions           # not deleted
    assert dec.set_public(private, data) == [] and dec.set_public(keep, data) == []
    assert data.pending_restore == ["raw:corpus/converted/cv.md"]
    assert dec.privacy_fingerprint(data.decisions) == "none"


def test_remove_decision(monkeypatch):
    from hypeonoath.privacy import restore

    monkeypatch.setattr(restore, "raw_sources_for", lambda files: ["CV.pdf"])
    private = decide(PHONE, "PHONE_NUMBER", "private", "phone number")
    data = data_with(private)
    assert dec.remove_decision(private, data) == ["CV.pdf"] and data.decisions == []


def test_set_keep_for_undecided_skips_decided_and_covered():
    data = data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number"),
                     decide("12.03.1999", "DATE_TIME", "private", "date of birth"))
    stored = dec.set_keep_for_undecided(groups(
        candidate("Jane Doe", "PERSON"),                        # new -> keep
        candidate("491512345678", "DE_TAX_ID"),                 # same value as a private one -> untouched
        candidate(f"{PHONE} 12.03.1999", "COMBINED"),           # covered by private values -> untouched
    ), data)
    assert stored == 1
    assert [(d.entity_type, d.decision) for d in data.decisions] == [
        ("PHONE_NUMBER", "private"), ("DATE_TIME", "private"), ("PERSON", "keep")]


def test_alias_and_surface_helpers_are_idempotent():
    decision = decide(PHONE, "PHONE_NUMBER", "private", "phone number")
    assert decision.add_alias("PHONE_NUMBER", "id", "491512345678", "R2") is False
    assert decision.add_alias("X", "id", "1", "R2") and not decision.add_alias("X", "id", "1", "R2")
    assert decision.add_surface_forms([PHONE]) is False
    assert decision.all_types() == ["PHONE_NUMBER", "X"]


def test_prune_retired_drops_only_keep_decisions_of_retired_types(isolated_logs):
    plate = decide("by a 3", "DE_KFZ", "keep")
    hidden = decide("DA X 12", "DE_KFZ", "private", "number")
    postcode = decide("64283", "DE_PLZ", "keep")
    postcode.add_alias("DATE_TIME", "text", "64283", "R2")                # still detected under another type
    phone = decide(PHONE, "PHONE_NUMBER", "keep")
    data = data_with(plate, hidden, postcode, phone)
    assert dec.prune_retired(data) is True
    assert [d.id for d in data.decisions] == [hidden.id, postcode.id, phone.id]
    assert dec.prune_retired(data) is False
    log = (isolated_logs / "privacy_log_path.jsonl").read_text(encoding="utf-8")
    assert '"pruned_retired"' in log and '"count": 1' in log and "by a 3" not in log


def test_prune_retired_us_driver_licence_keeps_private_aliases():
    licence = decide("658336", "US_DRIVER_LICENSE", "keep")                      # P-D51
    applicant = decide("123456", "APPLICANT_ID", "private", "applicant number")
    applicant.add_alias("US_DRIVER_LICENSE", "id", "123456", "R2")
    data = data_with(licence, applicant)
    assert dec.prune_retired(data) is True
    assert data.decisions == [applicant] and len(applicant.aliases) == 1


def test_resolve_all_prunes_and_reports_a_change():
    data = data_with(decide("by a 3", "DE_KFZ", "keep"))
    result = resolve(data)
    assert result.changed and data.decisions == []


def test_default_path_used():
    dec.save(data_with(decide(PHONE, "PHONE_NUMBER", "private", "phone number")))
    assert config.PRIVACY_DECISIONS_PATH.exists() and len(dec.load().decisions) == 1
