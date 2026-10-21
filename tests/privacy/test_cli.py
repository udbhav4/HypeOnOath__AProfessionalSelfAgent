"""hypeonoath-privacy commands: mark, list, aliases, unmark, set, apply, check-git, review
(privacy plan v10, 6.4.1, 6.10, 11; tests 14.4 unmark, 14.7 check-git, 14.10 review = Done -> build)."""
from __future__ import annotations

import io
import subprocess

import pytest

from hypeonoath.core import config
from hypeonoath.privacy import cli
from hypeonoath.privacy import decisions as dec

from tests.factories import candidate

DOB = "12.03.1999"
PHONE = "+49 151 2345678"


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for name, path in {"PROJECT_ROOT": root, "CORPUS_DIR": root / "corpus",
                       "CONVERTED_DIR": root / "corpus" / "converted",
                       "DOCUMENTS_DIR": root / "corpus" / "documents",
                       "CHROMA_DIR": root / "index" / "chroma",
                       "PRIVACY_DIR": root / "corpus" / ".privacy",
                       "PRIVACY_DECISIONS_PATH": root / "corpus" / ".privacy" / "decisions.json"}.items():
        monkeypatch.setattr(config, name, path)
    for folder in ("converted", "documents", "raw"):
        (root / "corpus" / folder).mkdir(parents=True)
    letter = root / "corpus" / "converted" / "Letter.md"
    letter.write_text(f"# Zulassung\n\nGeburtsdatum: {DOB}\nTelefon: {PHONE}\n", encoding="utf-8")
    return root


def save(*decisions):
    dec.save(dec.DecisionFile(list(decisions)))


def loaded():
    return dec.load().decisions


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def private_decision(value=PHONE, etype="PHONE_NUMBER", label="phone number", file="corpus/converted/Letter.md"):
    (group,) = dec.group_candidates([candidate(value, etype, file=file)])
    return dec.new_decision(group, "private", label)


# --- mark ------------------------------------------------------------------------------------------

def test_mark_stores_manual_private_value(root, capsys):
    code, out, _ = run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", DOB,
                       "--label", "date of birth", "--kind", "date")
    assert code == 0 and "Marked private" in out and DOB not in out
    (decision,) = loaded()
    assert (decision.decision, decision.source, decision.label, decision.match_kind, decision.normalized) == (
        "private", "manual", "date of birth", "date", "1999-03-12")
    assert decision.first_seen == {"file": "corpus/converted/Letter.md", "line": 3}


def test_mark_finds_other_spelling_and_refuses_typos(root, capsys):
    assert run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", "1999-03-12", "--kind", "date")[0] == 0
    assert DOB in loaded()[0].surface_forms                       # the corpus spelling is recorded too
    code, _, err = run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", "+49 999 0000000",
                       "--kind", "id")
    assert code == 1 and "not found" in err and "0000000" not in err


def test_mark_rejects_missing_file_bad_kind_value_and_label_with_value(root, capsys):
    assert run(capsys, "mark", "--file", "nope.md", "--text", "x")[0] == 1
    assert run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", "not a date", "--kind", "date")[0] == 1
    code, _, err = run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", PHONE, "--kind", "id",
                       "--label", f"phone {PHONE}")
    assert code == 1 and "must not contain" in err and loaded() == []


def test_mark_converts_an_earlier_keep_of_the_same_value(root, capsys):
    (group,) = dec.group_candidates([candidate(PHONE, file="corpus/converted/Letter.md")])
    save(dec.new_decision(group, "keep", None))
    assert run(capsys, "mark", "--file", "corpus/converted/Letter.md", "--text", PHONE, "--kind", "id")[0] == 0
    (decision,) = loaded()                                         # same value -> same decision (P-D30)
    assert (decision.id, decision.decision, decision.source) == (group.id, "private", "manual")
    assert ("MANUAL", "id", "491512345678") in decision.forms()


# --- list / aliases / unmark / set -------------------------------------------------------------------

def test_mask():
    assert cli.mask("12.03.1999") == "12.••.••••" and cli.mask("+49 151") == "+49 •••"


def test_list_masks_values_unless_reveal(root, capsys):
    decision = private_decision()
    decision.add_alias("DE_TAX_ID", "id", "491512345678", "R2")
    save(decision)
    code, out, _ = run(capsys, "list")
    assert code == 0 and decision.id in out and "PHONE_NUMBER, DE_TAX_ID (R2)" in out
    assert PHONE not in out and "+49 •••" in out and "fingerprint" in out
    assert PHONE in run(capsys, "list", "--reveal")[1]


def test_list_empty(root, capsys):
    assert "No decisions yet" in run(capsys, "list")[1]


def test_aliases(root, capsys):
    decision = private_decision()
    save(decision)
    assert "has no aliases" in run(capsys, "aliases", decision.id)[1]
    decision.add_alias("DE_TAX_ID", "id", "491512345678", "R2")
    save(decision)
    out = run(capsys, "aliases", decision.id)[1]
    assert "DE_TAX_ID" in out and "R2" in out and "491512345678" not in out
    assert "491512345678" in run(capsys, "aliases", decision.id, "--reveal")[1]
    assert run(capsys, "aliases", "pv_missing")[0] == 1


def test_unmark_removes_decision_and_aliases_and_changes_fingerprint(root, capsys):
    decision = private_decision()
    decision.add_alias("DE_TAX_ID", "id", "491512345678", "R2")
    save(decision)
    before = dec.privacy_fingerprint(loaded())
    code, out, _ = run(capsys, "unmark", decision.id)
    assert code == 0 and "PHONE_NUMBER, DE_TAX_ID" in out and "hypeonoath-index" in out
    assert loaded() == [] and dec.privacy_fingerprint([]) != before
    assert run(capsys, "unmark", decision.id)[0] == 1


def test_set_changes_decision_and_label(root, capsys):
    decision = private_decision()
    save(decision)
    assert run(capsys, "set", decision.id, "--decision", "keep")[0] == 0
    assert loaded()[0].decision == "keep"
    code, out, _ = run(capsys, "set", decision.id, "--decision", "private", "--label", "mobile")
    assert code == 0 and loaded()[0].label == "mobile" and "Rebuild the index" in out
    assert run(capsys, "set", decision.id, "--decision", "private", "--label", f"m {PHONE}")[0] == 1
    assert run(capsys, "set", "pv_missing", "--decision", "keep")[0] == 1


def test_apply_redacts_disk_files(root, capsys):
    assert "nothing to redact" in run(capsys, "apply")[1]
    save(private_decision())
    code, out, _ = run(capsys, "apply")
    assert code == 0 and "corpus/converted/Letter.md" in out
    assert "[REDACTED: phone number]" in (root / "corpus" / "converted" / "Letter.md").read_text(encoding="utf-8")


def test_unreadable_decisions_file_is_a_clean_error(root, capsys):
    config.PRIVACY_DECISIONS_PATH.parent.mkdir(parents=True)
    config.PRIVACY_DECISIONS_PATH.write_text("{broken " + PHONE)
    code, _, err = run(capsys, "list")
    assert code == 1 and "unreadable" in err and PHONE not in err


# --- check-git (Section 11) -------------------------------------------------------------------------------

def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(root):
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "corpus/converted/Letter.md")
    git(root, "commit", "-q", "-m", "add letter")
    (root / "corpus" / "raw" / "Letter.pdf").write_bytes(b"%PDF-1.4 binary")
    return root


def test_check_git_reports_history_and_staged_raw_file_without_values(repo, capsys):
    save(private_decision())
    git(repo, "add", "corpus/raw/Letter.pdf")
    code, out, err = run(capsys, "check-git")
    assert code == 5
    assert "corpus/converted/Letter.md · phone number" in out
    assert "corpus/raw/Letter.pdf: STAGED" in out
    assert PHONE not in out + err and "2345678" not in out + err
    assert "never rewrites history" in out


def test_check_git_ignored_raw_file_is_not_reported(repo, capsys):
    save(private_decision())
    (repo / ".gitignore").write_text("corpus/raw/Letter.pdf\ncorpus/.privacy/\n")
    out = run(capsys, "check-git")[1]
    assert "Letter.pdf" not in out


def test_check_git_untracked_unignored_raw_file_is_a_risk(repo, capsys):
    save(private_decision())
    code, out, _ = run(capsys, "check-git")
    assert code == 5 and "corpus/raw/Letter.pdf: UNTRACKED, NOT IGNORED" in out


def test_check_git_clean_repo(repo, capsys):
    save()
    (repo / ".gitignore").write_text("corpus/.privacy/\n")
    code, out, _ = run(capsys, "check-git")
    assert code == 0 and out.rstrip().endswith("check-git: clean.") and "pending review" not in out


def test_check_git_reports_pending_files_as_info(repo, capsys):
    from hypeonoath.privacy import scan

    save()
    (repo / ".gitignore").write_text("corpus/.privacy/\n")
    scan.save_state(scan.ScanState("s", {"corpus/converted/Letter.md": {"hash": "h", "pending": True}}))
    code, out, _ = run(capsys, "check-git")
    assert code == 0 and "Info: 1 file(s) pending review -- hypeonoath-privacy review." in out


def test_check_git_tracked_privacy_folder_and_index_without_fp(repo, capsys):
    from hypeonoath.core.store import get_client, release_client

    save()
    git(repo, "add", "-f", "corpus/.privacy/decisions.json")
    client = get_client(config.CHROMA_DIR)
    client.create_collection(config.COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
    release_client(config.CHROMA_DIR)
    code, out, _ = run(capsys, "check-git")
    assert code == 5 and "are tracked by git" in out and "no privacy_fp" in out


def test_check_git_searches_committed_eval_questions_only(repo, capsys):
    """Test 14.11 (v13, P-D53): eval/questions.yaml is searched; eval run output is not."""
    assert "eval/questions.yaml" in cli.GIT_SEARCH_PATHS
    (repo / "eval" / "runs" / "r1").mkdir(parents=True)
    (repo / "eval" / "questions.yaml").write_text(f"- quote: 'Ruf: {PHONE}'\n", encoding="utf-8")
    (repo / "eval" / "runs" / "r1" / "answers.jsonl").write_text(f'{{"reply": "{PHONE}"}}\n', encoding="utf-8")
    git(repo, "add", "eval/questions.yaml", "eval/runs/r1/answers.jsonl")    # even a committed run file is not searched
    git(repo, "commit", "-q", "-m", "eval")
    save(private_decision())
    code, out, err = run(capsys, "check-git")
    assert code == 5 and "· eval/questions.yaml · phone number" in out
    assert "answers.jsonl" not in out and PHONE not in out + err


def test_check_git_clean_message_names_all_search_paths(repo, capsys):
    save()
    (repo / ".gitignore").write_text("corpus/.privacy/\n")
    out = run(capsys, "check-git")[1]
    assert "corpus/, index/ or eval/questions.yaml" in out


def test_check_git_outside_a_repo(root, capsys, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(root.parent))     # never find a repo further up
    code, _, err = run(capsys, "check-git")
    assert code == 1 and "not a git repository" in err


# --- review command: scan -> screens -> q = Done -> build (P-D43) ----------------------------------------------

@pytest.fixture
def review_env(root, monkeypatch):
    """Fake corpus + scan (one pending file holding a phone number) + a recorded build call."""
    from hypeonoath.indexing import build_index
    from hypeonoath.privacy import scan

    file = "corpus/converted/Letter.md"
    calls = {"build": [], "scan": []}

    def fake_scan(corpus, full=False):
        calls["scan"].append(full)
        data = dec.load()
        if not data.decisions:
            (group,) = dec.group_candidates([candidate(PHONE, file=file, line=4)])
            data.decisions.append(dec.new_decision(group, "keep", None, source="detected"))
            dec.save(data)
            scan.save_state(scan.ScanState("s", {file: {"hash": "h", "pending": True}}))
        return scan.ScanResult([file], scan.load_state().pending(), 1, [])

    monkeypatch.setattr(build_index, "load_corpus", lambda: object())
    monkeypatch.setattr(scan, "run", fake_scan)
    monkeypatch.setattr(build_index, "main", lambda argv: calls["build"].append(argv) or 0)
    return calls


def test_review_quit_is_done_and_builds(review_env, capsys, monkeypatch):
    from hypeonoath.privacy import scan

    monkeypatch.setattr("sys.stdin", io.StringIO("1\nq\n"))
    code, out, _ = run(capsys, "review")
    assert code == 0 and "1 file(s) pending review" in out and "Done -- building the index." in out
    assert loaded()[0].decision == "private" and review_env["build"] == [[]]
    assert scan.load_state().pending() == []


def test_review_without_done_keeps_files_pending_and_does_not_build(review_env, capsys, monkeypatch):
    from hypeonoath.privacy import scan

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code, out, _ = run(capsys, "review", "--access", "--full-scan")
    assert code == 0 and "ended without Done" in out and review_env["build"] == []
    assert review_env["scan"] == [True] and scan.load_state().pending() == ["corpus/converted/Letter.md"]


def test_review_nothing_to_show_still_builds(review_env, capsys, monkeypatch):
    code, out, _ = run(capsys, "review", "--file", "corpus/converted/Other.md")
    assert code == 0 and "Nothing to show" in out and review_env["build"] == [[]]


def test_restore_command(root, capsys, monkeypatch):
    from hypeonoath.privacy import restore

    assert "Nothing queued" in run(capsys, "restore")[1]
    dec.save(dec.DecisionFile([], ["Letter.pdf"]))
    monkeypatch.setattr(restore, "restore_pending", lambda data: [data.pending_restore.pop()])
    code, out, _ = run(capsys, "restore")
    assert code == 0 and "Rebuilt 1 document(s)" in out and dec.load().pending_restore == []


def test_set_keep_and_unmark_queue_a_rebuild(root, capsys, monkeypatch):
    from hypeonoath.privacy import restore

    monkeypatch.setattr(restore, "raw_sources_for", lambda files: ["Letter.pdf"])
    decision = private_decision()
    save(decision)
    code, out, _ = run(capsys, "set", decision.id, "--decision", "keep")
    assert code == 0 and "1 document(s) will be rebuilt from raw" in out
    assert dec.load().pending_restore == ["Letter.pdf"]
    run(capsys, "set", decision.id, "--decision", "private")
    code, out, _ = run(capsys, "unmark", decision.id)
    assert code == 0 and "undecided again" in out and loaded() == []
