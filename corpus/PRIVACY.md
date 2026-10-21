# Privacy in the corpus

Owner notes for `plans/privacy-pii-redaction-plan-v11.md`. This file is not part of the indexed corpus (the index only reads files Phase 0 routed from `corpus/raw/`).

## Never commit

| Path | Why |
|---|---|
| `corpus/.privacy/` | `decisions.json` holds the exact values of every finding (private and public); `scan_state.json` and `build.lock` hold paths and hashes only (gitignored). Back up `decisions.json` locally — losing it resets every value to its default. |
| `corpus/raw/AcceptanceLetter1_Zulassung.pdf` and its `corpus/documents/` copy | Raw originals are never redacted; files with private findings stay out of git (gitignored by name). |
| Any other raw file with private findings | Add it to `.gitignore` by name. `hypeonoath-privacy check-git` warns about tracked or staged ones. |

## Day to day

| Step | Command |
|---|---|
| New or changed raw files | `hypeonoath-prepare` |
| Review (required): scans only new / changed files, then one screen per file; `q` = Done → the index is built | `hypeonoath-privacy review` |
| Change decisions later (all files by critical / high counts) | `hypeonoath-privacy review --access` |
| Re-scan every file (e.g. after changing recognizers) | `hypeonoath-privacy review --full-scan` |
| Something the detectors missed | `hypeonoath-privacy mark --file F --text "…" --label L` |
| Before a push | `hypeonoath-privacy check-git` |
| Rebuild the index only (refuses while files are pending review or not scanned) | `hypeonoath-index` |

Review keys: Enter = next file · numbers = switch private / public · `l<n>` = label · `a` = public for all (asks before making secrets public) · `x` = private for all · `q` = Done.

Defaults: secrets (API keys, tokens) are private, everything else is public until switched. A choice applies to the value everywhere. Switching a value back to public rebuilds its documents from `corpus/raw/` during the next build.
