"""
hypeonoath-privacy -- the owner's privacy tool (privacy plan v10, 6.5, 6.7.1, 6.10, 11).

    review [--access] [--full-scan] [--new | --file F]
                               scan new / changed files, then the per-file
                               screens; q = Done -> the index is built
    mark --file F --text T     manually mark text the detectors missed as private
    list [--reveal]            all decisions (values masked unless --reveal)
    aliases ID [--reveal]      the other types one decision also covers
    set ID --decision keep|private [--label L]
    unmark ID                  delete a decision (the value is undecided again)
    apply                      redact the derived corpus files on disk only
    restore                    rebuild queued documents from corpus/raw/ now
    check-git                  private values in git history / tracked raw files /
                               index fingerprint (prints commit, path, label --
                               never a value)

Exit codes: 0 ok / clean, 1 error, 2 project check failed (review),
5 check-git found something; review ends with the index build's exit code.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from hypeonoath.core import config, paths
from hypeonoath.core import private_values as pv
from hypeonoath.privacy import decisions as dec

EXIT_OK, EXIT_ERROR, EXIT_PROJECT, EXIT_GIT_FOUND = 0, 1, 2, 5
REBUILD_HINT = "Rebuild the index: hypeonoath-index"
GIT_SEARCH_PATHS = ("corpus", "index", "eval/questions.yaml")   # v13, P-D53: committed eval questions too
_RAW_SUFFIXES = (".pdf", ".docx", ".doc")


# --- helpers ---------------------------------------------------------------------------------

def mask(value: str) -> str:
    """Keep separators and the first 2 alphanumerics: '12.03.1999' -> '12.••.••••'."""
    shown = 0
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch if shown < 2 else "•")
            shown += 1
        else:
            out.append(ch)
    return "".join(out)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)


def _types(decision: dec.Decision) -> str:
    parts = [decision.entity_type] + [f"{a.entity_type} ({a.rule})" for a in decision.aliases
                                      if a.entity_type != decision.entity_type]
    return ", ".join(dict.fromkeys(parts))


def _label_ok(label: str, decision: dec.Decision) -> bool:
    """A label must not contain the value itself."""
    probe = dec.private_values([dec.Decision.from_dict({**decision.to_dict(), "decision": "private"})])
    return bool(label.strip()) and not pv.find_private_spans(label, probe)


def _queued_note(queued: list[str]) -> str:
    return (f" {len(queued)} document(s) will be rebuilt from raw on the next index build."
            if queued else "")


# --- commands ---------------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    """Scan (B) -> review (C) -> Done (D) -> build (E) (plan v10, P-D37, P-D43)."""
    from hypeonoath.indexing import build_index, project_registry
    from hypeonoath.privacy import review, scan

    try:
        corpus = build_index.load_corpus()
    except project_registry.ProjectValidationError as exc:
        print(exc.format(), file=sys.stderr)
        return EXIT_PROJECT
    print(scan.run(corpus, full=args.full_scan).summary())
    data = dec.load()
    screens = review.build_reviews(data, pending=scan.load_state().pending(),
                                   order="access" if args.access else "upload",
                                   selection=_rel(Path(args.file)) if args.file else None, pending_only=args.new)
    if screens:
        print(f"{len(screens)} file(s). Values are shown here only -- never logged. q = Done (builds the index).")
        outcome = review.review(screens, data)
        if not outcome.done:
            print("Review ended without Done: files stay pending review; the index was not built.")
            return EXIT_OK
    else:
        print("Nothing to show for this selection.")
    scan.mark_reviewed([s.file for s in screens])
    print("Done -- building the index.")
    return build_index.main([])


def cmd_mark(args: argparse.Namespace) -> int:
    from hypeonoath.privacy.detector import Candidate

    path = Path(args.file)
    path = path if path.is_absolute() else config.PROJECT_ROOT / path
    if not path.is_file():
        print(f"No such file: {args.file}", file=sys.stderr)
        return EXIT_ERROR
    text = args.text.strip()
    normalized = pv.normalize(text, args.kind)
    if not text or not normalized:
        print(f"The text can't be used as a '{args.kind}' value (empty after normalising).", file=sys.stderr)
        return EXIT_ERROR
    content = path.read_text(encoding="utf-8", errors="replace")
    probe = [pv.PrivateValue("probe", "probe", args.kind, normalized, (text,))]
    spans = pv.find_private_spans(content, probe)
    if not spans:
        # Refuse: probably a typo. The text itself is not echoed.
        print(f"The text was not found in {_rel(path)} (in any spelling). Nothing stored.", file=sys.stderr)
        return EXIT_ERROR

    lines = content.split("\n")
    candidates = []
    for span in spans:                                  # every occurrence -> locations
        line_no = content.count("\n", 0, span.start) + 1
        offset = span.start - (content.rfind("\n", 0, span.start) + 1)
        candidates.append(Candidate(_rel(path), line_no, "MANUAL", "manual", 1.0, content[span.start:span.end],
                                    lines[line_no - 1], offset))
    candidates.append(Candidate(_rel(path), candidates[0].line, "MANUAL", "manual", 1.0, text,
                                candidates[0].line_text, 0))      # the typed spelling too
    group = dec.CandidateGroup(dec.decision_id("MANUAL", normalized), "MANUAL", args.kind, normalized, candidates)

    label = args.label or pv.default_label("MANUAL")
    if pv.find_private_spans(label, probe):
        print("The label must not contain the private value itself.", file=sys.stderr)
        return EXIT_ERROR
    data = dec.load()
    decision = dec.set_private(group, label, data, source="manual")
    if args.label:
        decision.label = args.label
    dec.save(data)
    print(f"Marked private as {decision.id} (label '{decision.label}', kind {decision.match_kind}). {REBUILD_HINT}")
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    data = dec.load()
    if not data.decisions:
        print(f"No decisions yet ({config.PRIVACY_DECISIONS_PATH}).")
        return EXIT_OK
    rows = [["id", "decision", "label", "types", "kind", "source", "#files", "#forms", "value"]]
    for d in sorted(data.decisions, key=lambda d: (d.decision, d.label or "", d.id)):
        rows.append([d.id, d.decision, d.label or "-", _types(d), d.match_kind, d.source, str(len(d.files())),
                     str(len(d.surface_forms)), d.value if args.reveal else mask(d.value)])
    print(_table(rows))
    print(f"\n{len(data.decisions)} decision(s); fingerprint {dec.privacy_fingerprint(data.decisions)}"
          + (f"; {len(data.pending_restore)} document(s) queued for a rebuild from raw" if data.pending_restore else ""))
    return EXIT_OK


def cmd_aliases(args: argparse.Namespace) -> int:
    decision = dec.load().by_id(args.id)
    if decision is None:
        print(f"No decision {args.id}", file=sys.stderr)
        return EXIT_ERROR
    if not decision.aliases:
        print(f"{decision.id} has no aliases (type {decision.entity_type}).")
        return EXIT_OK
    rows = [["type", "kind", "rule", "added_at", "normalized"]]
    rows += [[a.entity_type, a.match_kind, a.rule, a.added_at, a.normalized if args.reveal else mask(a.normalized)]
             for a in decision.aliases]
    print(_table(rows))
    return EXIT_OK


def cmd_unmark(args: argparse.Namespace) -> int:
    data = dec.load()
    decision = data.by_id(args.id)
    if decision is None:
        print(f"No decision {args.id}", file=sys.stderr)
        return EXIT_ERROR
    kind = decision.decision
    queued = dec.remove_decision(decision, data)
    dec.save(data)
    print(f"Removed {decision.id} ({kind}) and its types: {', '.join(decision.all_types())}. "
          f"The value is undecided again (public, or private if it is a secret).{_queued_note(queued)}")
    print(REBUILD_HINT)
    return EXIT_OK


def cmd_set(args: argparse.Namespace) -> int:
    data = dec.load()
    decision = data.by_id(args.id)
    if decision is None:
        print(f"No decision {args.id}", file=sys.stderr)
        return EXIT_ERROR
    queued: list[str] = []
    if args.decision == "private":
        label = args.label or decision.label or pv.default_label(decision.entity_type)
        if not _label_ok(label, decision):
            print("The label must not contain the private value itself.", file=sys.stderr)
            return EXIT_ERROR
        dec.set_private(decision, label, data)
    else:
        queued = dec.set_public(decision, data)
    dec.save(data)
    print(f"{decision.id} is now {decision.decision}"
          f"{f' (label {decision.label!r})' if decision.is_private else ''} for types: "
          f"{', '.join(decision.all_types())}.{_queued_note(queued)}")
    print(REBUILD_HINT)
    return EXIT_OK


def cmd_apply(args: argparse.Namespace) -> int:
    from hypeonoath.privacy import redactor

    values = dec.private_values(dec.load().decisions)
    if not values:
        print("No private decisions -- nothing to redact.")
        return EXIT_OK
    changed = redactor.redact_files_on_disk(values)
    print(f"Redacted {len(changed)} file(s)" + (":" if changed else "."))
    for path in changed:
        print(f"  {path}")
    return EXIT_OK


def cmd_restore(args: argparse.Namespace) -> int:
    from hypeonoath.privacy import restore

    data = dec.load()
    if not data.pending_restore:
        print("Nothing queued for a rebuild from raw.")
        return EXIT_OK
    try:
        rebuilt = restore.restore_pending(data)
    finally:
        dec.save(data)
    print(f"Rebuilt {len(rebuilt)} document(s) from raw. {REBUILD_HINT}")
    return EXIT_OK


# --- check-git (plan Section 11) ----------------------------------------------------------------------

def _git(*args: str) -> subprocess.CompletedProcess:
    """Run git with an argument LIST (no shell): a value is never on a printed command line."""
    return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False)


def history_hits(decisions: list[dec.Decision]) -> list[tuple[str, str, str]]:
    """G2: (commit, path, label) for every private surface form found by git's pickaxe search."""
    hits: set[tuple[str, str, str]] = set()
    for decision in decisions:
        if not decision.is_private:
            continue
        for form in dict.fromkeys(decision.surface_forms or [decision.value]):
            if len(form.strip()) < 3:
                continue
            result = _git("log", "--all", "-S", form, "--format=commit %h", "--name-only", "--",
                          *GIT_SEARCH_PATHS)
            commit = None
            for line in result.stdout.splitlines():
                if line.startswith("commit "):
                    commit = line.split(" ", 1)[1].strip()
                elif line.strip() and commit:
                    hits.add((commit, line.strip(), decision.label or "private detail"))
    return sorted(hits)


def _documents_with_private_values(decisions: list[dec.Decision]) -> set[str]:
    """
    Stems of documents holding private values: every converted / document file
    in a private decision's locations, plus any converted file that still
    contains one (PDFs are binary; their converted text is the same document).
    """
    stems: set[str] = set()
    converted = config.CONVERTED_DIR
    for decision in decisions:
        if decision.is_private:
            for file in decision.files():
                seen = Path(file)
                if seen.parent.name in (converted.name, config.DOCUMENTS_DIR.name):
                    stems.add(seen.name.split(".")[0])
    values = dec.private_values(decisions)
    if values and converted.is_dir():
        for md in converted.glob("*.md"):
            if pv.find_private_spans(md.read_text(encoding="utf-8", errors="replace"), values):
                stems.add(md.name.split(".")[0])
    return stems


def raw_file_status(stems: set[str]) -> list[tuple[str, str]]:
    """P-D21: (path, status) for each raw original / routed copy of those documents."""
    staged = set(_git("diff", "--cached", "--name-only").stdout.split("\n"))
    rows = []
    for folder in (config.CORPUS_DIR / paths.RAW_DIR.name, config.DOCUMENTS_DIR):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _RAW_SUFFIXES or path.stem not in stems:
                continue
            rel = _rel(path)
            if rel in staged:
                status = "STAGED"
            elif _git("ls-files", "--error-unmatch", rel).returncode == 0:
                status = "TRACKED"
            elif _git("check-ignore", "-q", rel).returncode == 0:
                status = "ignored"
            else:
                status = "UNTRACKED, NOT IGNORED"
            rows.append((rel, status))
    return rows


def index_status(decisions: list[dec.Decision]) -> str | None:
    """P-D20: a problem with the working-tree index, or None when it is fine to commit."""
    from hypeonoath.core.store import get_client, release_client

    if not config.CHROMA_DIR.exists():
        return None
    try:
        metadata = get_client(config.CHROMA_DIR).get_collection(config.COLLECTION_NAME).metadata or {}
    except Exception:  # noqa: BLE001 -- no collection: nothing to commit
        return None
    finally:
        release_client(config.CHROMA_DIR)
    if "privacy_fp" not in metadata:
        return "the index was built before the privacy plan (no privacy_fp) -- rebuild before committing it"
    if metadata["privacy_fp"] != dec.privacy_fingerprint(decisions):
        return "the index was built with different privacy decisions -- rebuild before committing it"
    return None


def cmd_check_git(args: argparse.Namespace) -> int:
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        print(f"{config.PROJECT_ROOT} is not a git repository.", file=sys.stderr)
        return EXIT_ERROR
    data = dec.load()
    decisions = data.decisions
    found = 0

    tracked_privacy = [p for p in _git("ls-files", "--", _rel(config.PRIVACY_DIR)).stdout.splitlines() if p]
    if tracked_privacy:
        found += 1
        print(f"WARNING: {len(tracked_privacy)} file(s) under {_rel(config.PRIVACY_DIR)} are tracked by git -- "
              "untrack them (git rm --cached) before any push.")

    hits = history_hits(decisions)
    if hits:
        found += 1
        print(f"Private values found in git history ({len(hits)} commit/path pair(s)):")
        for commit, path, label in hits:
            print(f"  {commit} · {path} · {label}")
        print("If any of these commits were PUSHED, only a history rewrite (git filter-repo) + force-push removes\n"
              "them -- irreversible for every clone, and GitHub may keep cached views. That is the owner's\n"
              "decision (plan Section 11, G4); this tool never rewrites history.")
    else:
        print("Git history: no private value found in corpus/, index/ or eval/questions.yaml (pickaxe search over every commit).")

    raw_rows = raw_file_status(_documents_with_private_values(decisions))
    risky = [(p, s) for p, s in raw_rows if s != "ignored"]
    if risky:
        found += 1
        print("Raw files with private findings that could be committed (P-D21) -- add each to .gitignore by name:")
        for path, status in risky:
            print(f"  {path}: {status}")

    problem = index_status(decisions)
    if problem:
        found += 1
        print(f"Index: {problem} (P-D20).")
    elif config.CHROMA_DIR.exists():
        print("Index: built with the current privacy decisions.")

    # Information only: never changes the exit code.
    from hypeonoath.privacy import scan

    pending = scan.load_state().pending()
    if pending:
        print(f"Info: {len(pending)} file(s) pending review -- hypeonoath-privacy review.")
    if data.pending_restore:
        print(f"Info: {len(data.pending_restore)} document(s) queued for a rebuild from raw "
              "(next hypeonoath-index or hypeonoath-privacy restore).")

    if found:
        print(f"check-git: {found} problem area(s) found.")
        return EXIT_GIT_FOUND
    print("check-git: clean.")
    return EXIT_OK


# --- entry point ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hypeonoath-privacy",
                                     description="Review, mark and redact personal information in the corpus.")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="scan new / changed files, review, q = Done -> build the index")
    review.add_argument("--access", action="store_true",
                        help="access-settings order: all files by critical / high counts (not pending first)")
    review.add_argument("--full-scan", action="store_true", help="re-scan every file, not only new / changed ones")
    which = review.add_mutually_exclusive_group()
    which.add_argument("--new", action="store_true", help="only files pending review")
    which.add_argument("--file", help="one file (path relative to the project root)")
    review.set_defaults(func=cmd_review)

    mark = sub.add_parser("mark", help="mark text the detectors missed as private")
    mark.add_argument("--file", required=True, help="file (relative to the project root) that contains the text")
    mark.add_argument("--text", required=True, help="the exact text to mark")
    mark.add_argument("--label", help="placeholder label, e.g. 'date of birth' (default: private detail)")
    mark.add_argument("--kind", choices=pv.MATCH_KINDS, default="text", help="how the value is matched")
    mark.set_defaults(func=cmd_mark)

    listing = sub.add_parser("list", help="list all decisions (values masked)")
    listing.add_argument("--reveal", action="store_true", help="show the values (terminal only)")
    listing.set_defaults(func=cmd_list)

    aliases = sub.add_parser("aliases", help="list the other types one decision covers")
    aliases.add_argument("id")
    aliases.add_argument("--reveal", action="store_true")
    aliases.set_defaults(func=cmd_aliases)

    unmark = sub.add_parser("unmark", help="delete a decision (the value is undecided again)")
    unmark.add_argument("id")
    unmark.set_defaults(func=cmd_unmark)

    setter = sub.add_parser("set", help="change a decision (and all its aliases)")
    setter.add_argument("id")
    setter.add_argument("--decision", required=True, choices=dec.DECISIONS)
    setter.add_argument("--label")
    setter.set_defaults(func=cmd_set)

    sub.add_parser("apply", help="redact the derived corpus files on disk").set_defaults(func=cmd_apply)
    sub.add_parser("restore", help="rebuild queued documents from corpus/raw/ now").set_defaults(func=cmd_restore)
    sub.add_parser("check-git", help="private values in git history / raw files / index").set_defaults(
        func=cmd_check_git)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # masks / box chars on legacy consoles
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except ValueError as exc:        # e.g. an unreadable decisions file (messages never hold values)
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
