"""
Phase 0 ingestion pipeline (Steps 3.1-3.3): intake -> route -> convert.

Decisions recorded per plans/phase0-steps-3.1-3.3-code-implementation-guideline.md,
Section 1A (confirmed with user 2026-09-15), documented here per that guideline's
own requirement that developer decisions be written down, not left silent:

- Orchestration: a single orchestrator entry point (run_phase0.py) runs
  routing then conversion in sequence. Router and converter logic stay in
  separate, independently importable modules -- the orchestrator is a thin
  composition layer, not a monolith.
- Automated tests: skipped for now, by explicit user decision. No tests/
  package has been built out alongside this one.
- corpus/raw/ immutability: enforced by code discipline only, not an OS-level
  read-only flag. No function in this package ever opens a path under
  corpus/raw/ in a write or delete mode.
"""
