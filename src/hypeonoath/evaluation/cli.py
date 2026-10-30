"""
hypeonoath-eval (eval plan v15, Section 9).

  validate [PATH] [--freeze]   V1-V13 on the drafts (default when present) or questions.yaml; --freeze = G4 / V14
  metrics [NAME]               definitions; regenerates eval/METRICS.md
  review-gold                  G3, resumable
  run --retrieval-only [...]   retrieval stage (the full stage needs the judge: Stage D, owner decision B-2)
  run --resume RUN             continue an interrupted run
  score RUN                    recompute scores, report and the runs.csv row (no API calls)
  compare RUN_A RUN_B          B vs A over the metrics present in both
  promote RUN --note TEXT      set the baseline (full stage, 100 % coverages)

Every run refuses before anything is generated when the private decisions changed since the gold
freeze (D49). Output never holds a private value (labels and counts only).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypeonoath.core import config
from hypeonoath.core.private_values import load_private_values
from hypeonoath.evaluation import glossary, gold, paths, report, runner, tracking
from hypeonoath.evaluation.answerer import ANSWERERS, make_answerer
from hypeonoath.evaluation.metrics import METRICS_BY_NAME, REPORT_TERMS, score_run
from hypeonoath.evaluation.questions import (current_privacy_fp, default_context, load_file, privacy_gate,
                                             question_set_sha256, redact_messages, validate)

FULL_STAGE_NOT_BUILT = ("the full stage needs the judge, which is built in Stage D; "
                        "use --retrieval-only for now")
SCORECARD_LINES = ("hit@8", "in_context_hit", "mrr", "answer_rate", "correct_no_answer")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- validate / metrics / review-gold -------------------------------------------------------

def cmd_validate(args) -> int:
    path = Path(args.path) if args.path else (paths.DRAFTS_PATH if paths.DRAFTS_PATH.exists() else paths.QUESTIONS_PATH)
    ctx = default_context()
    if args.freeze:
        problems = gold.freeze(ctx, drafts=path, privacy_fp=current_privacy_fp())
        for p in problems:
            _err(p)
        if problems:
            _err(f"FREEZE FAILED: {len(problems)} problem(s)")
            return 1
        print(f"frozen -> {paths.QUESTIONS_PATH}")
        return 0
    meta, questions, errors = load_file(path)
    is_frozen_set = path.resolve() == paths.QUESTIONS_PATH.resolve()
    errors = redact_messages(errors, ctx.private_values) + validate(questions, ctx, approved_only=is_frozen_set)
    if is_frozen_set:
        problem = privacy_gate(meta, current_privacy_fp())
        if problem:
            errors.append(f"V14 {problem}")
        elif meta.get("question_set_sha256") != question_set_sha256(questions):
            errors.append("V14 question_set_sha256 does not match the approved entries (edited after the freeze?)")
    for e in errors:
        print(e)
    print(f"{path.name}: {len(questions)} entries, {'OK' if not errors else f'{len(errors)} problem(s)'}")
    return 1 if errors else 0


def cmd_metrics(args) -> int:
    try:
        print(glossary.text(args.name))
    except ValueError as exc:
        _err(str(exc))
        return 1
    paths.METRICS_MD_PATH.write_text(glossary.markdown(), encoding="utf-8")
    return 0


def cmd_review_gold(args) -> int:
    counts = gold.review()
    print(f"\napproved {counts['approved']}, rejected {counts['rejected']}, left {counts['left']}")
    if not counts["left"]:
        print("All reviewed. Next: hypeonoath-eval validate --freeze")
    return 0


# --- run ------------------------------------------------------------------------------------

def load_frozen() -> tuple[dict[str, Any], list]:
    """The approved, frozen set, after V1-V13 and the D49 gate; raises SystemExit(1) with the reasons."""
    meta, questions, errors = load_file(paths.QUESTIONS_PATH)
    ctx = default_context()
    errors = redact_messages(errors, ctx.private_values) + validate(questions, ctx, approved_only=True)
    problem = privacy_gate(meta, current_privacy_fp())
    if problem:
        errors.insert(0, problem)
    if errors:
        for e in errors:
            _err(e)
        raise SystemExit(1)
    return meta, [q for q in questions if q.status == "approved"]


def _index_ids(answerer) -> list[str]:
    return answerer.index_ids()


def _count_tokens():
    from hypeonoath.core import embedder
    return embedder.count_tokens


def cmd_run(args) -> int:
    if args.resume:
        run_dir = _run_dir(args.resume)
        manifest = runner.read_manifest(run_dir)
        stage = manifest["stage"]
    else:
        stage = "retrieval" if args.retrieval_only else "full"
    if stage == "full":
        _err(FULL_STAGE_NOT_BUILT)                     # owner decision B-2 (2026-10-28)
        return 2
    if args.pin_contexts:
        _err("--pin-contexts replays contexts into generation: full stage only")
        return 2

    meta, questions = load_frozen()                    # step 1: validate + privacy gate (D49), before anything
    values = load_private_values()
    if args.resume:
        mmr = manifest.get("mmr") or {}
        answerer = make_answerer(manifest["answerer"], top_k=manifest["top_k"],
                                 mmr_lambda=mmr.get("lambda") if mmr.get("enabled") else None,
                                 fetch_k=mmr.get("fetch_k", config.MMR_FETCH_K),
                                 prompt_version=manifest["prompt_version"])
    else:
        answerer = make_answerer(args.answerer, top_k=args.k, mmr_lambda=args.mmr_lambda, fetch_k=args.fetch_k,
                                 prompt_version=args.prompt)
    current = {"index_fp": runner.index_fingerprint(_index_ids(answerer)),
               "question_set_sha256": meta["question_set_sha256"], "privacy_fp": meta["privacy_fp"]}
    if args.resume:
        problem = runner.resume_problem(manifest, current)
        if problem:
            _err(problem)
            return 1
        manifest["resumed_from"] = manifest.get("generation_status")     # 6.1: the state the resume started from
        manifest["generation_status"] = "running"
        runner.write_manifest(run_dir, manifest)
    else:
        run_id = runner.new_run_id()
        run_dir = paths.RUNS_DIR / run_id
        manifest = {"run_id": run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "stage": stage, "answerer": answerer.name, **answerer.settings(), **current,
                    **runner.git_state(), "pinned_from": None, "resumed_from": None, "note": args.note,
                    "generation_status": "running", "judge": {"status": "not_run"}}
        runner.write_manifest(run_dir, manifest)

    started = datetime.now(timezone.utc)
    try:
        status = runner.run_questions(questions, answerer, run_dir, stage, values, count_tokens=_count_tokens())
    except runner.RunAborted as exc:
        manifest["generation_status"] = "aborted"
        runner.write_manifest(run_dir, manifest)
        _err(str(exc))
        return 3
    manifest["generation_status"] = status
    manifest.setdefault("timings", {})["run_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    runner.write_manifest(run_dir, manifest)
    scores = finish(run_dir, questions)
    m = scores["metrics"]
    print(f"report: {run_dir / 'report.md'}")
    for name in SCORECARD_LINES:
        if m.get(name) is not None:
            print(f"  {name:<18} {m[name]:.3f}")
    return 0


# --- score / compare / promote --------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finish(run_dir: Path, questions: list | None = None) -> dict[str, Any]:
    """Scores + report + runs.csv row of one run directory (no API calls)."""
    from hypeonoath.serving.prompt import PROMPT_EXAMPLES, PROMPT_SAMPLE_PHRASES, system_rules

    manifest = runner.read_manifest(run_dir)
    if questions is None:
        _, loaded, _ = load_file(paths.QUESTIONS_PATH)
        questions = [q for q in loaded if q.status == "approved"]
        if question_set_sha256(questions) != manifest.get("question_set_sha256"):
            _err("warning: the question set changed since this run; scoring with the current set")
    answers = runner.read_answers(run_dir)
    judgments = _read_jsonl(run_dir / "judgments.jsonl")
    system = system_rules(manifest["prompt_version"]) if manifest.get("stage") == "full" else None
    scores = score_run(questions, answers, judgments, manifest, system,
                       list(PROMPT_EXAMPLES) + list(PROMPT_SAMPLE_PHRASES))
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False, default=str) + "\n",
                                         encoding="utf-8")
    baseline = (tracking.read_baseline() or {}).get("current")
    baseline_scores = None
    if baseline:
        path = paths.RUNS_DIR / str(baseline["run_id"]) / "scores.json"
        baseline_scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    warnings = tracking.comparability_warnings(manifest, scores["metrics"], baseline)
    changed = tracking.changed_settings(manifest, baseline.get("settings") or {}) if baseline else None
    (run_dir / "report.md").write_text(
        report.render(manifest, scores, questions, answers, judgments, baseline, baseline_scores, warnings, changed),
        encoding="utf-8")
    tracking.write_row(tracking.csv_row(manifest, scores["metrics"]))
    return scores


def _run_dir(run_id: str) -> Path:
    path = paths.RUNS_DIR / run_id
    if not (path / "manifest.json").exists():
        raise SystemExit(f"no run {run_id!r} in {paths.RUNS_DIR}")
    return path


def cmd_score(args) -> int:
    finish(_run_dir(args.run))
    print(f"report: {paths.RUNS_DIR / args.run / 'report.md'}")
    return 0


def _scores(run_id: str) -> dict[str, Any]:
    path = _run_dir(run_id) / "scores.json"
    if not path.exists():
        raise SystemExit(f"run {run_id} has no scores.json; run: hypeonoath-eval score {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_compare(args) -> int:
    a, b = _scores(args.run_a)["metrics"], _scores(args.run_b)["metrics"]
    print(f"{'metric':<26} {'A':>8} {'B':>8} {'B-A':>8}")
    for row in tracking.compare(a, b):
        mark = "" if row["better"] is None else ("better" if row["better"] else "worse")
        print(f"{row['metric']:<26} {row['a']:>8.3f} {row['b']:>8.3f} {row['delta']:>+8.3f} {mark}")
    return 0


def cmd_promote(args) -> int:
    run_dir = _run_dir(args.run)
    try:
        tracking.promote(runner.read_manifest(run_dir), _scores(args.run), args.note)
    except ValueError as exc:
        _err(f"refused: {exc}")
        return 1
    print(f"baseline -> {args.run}")
    return 0


# --- parser ---------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from hypeonoath.serving.prompt import PROMPTS

    parser = argparse.ArgumentParser(prog="hypeonoath-eval", description="Offline evaluation harness (eval plan v15)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check the question set (V1-V13); --freeze writes questions.yaml (V14)")
    p.add_argument("path", nargs="?", help="drafts.yaml or questions.yaml (default: the drafts when present)")
    p.add_argument("--freeze", action="store_true", help="merge approved drafts into eval/questions.yaml")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("metrics", help="print metric definitions; regenerate eval/METRICS.md")
    p.add_argument("name", nargs="?", choices=sorted(METRICS_BY_NAME) + sorted(REPORT_TERMS), metavar="NAME")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("review-gold", help="review the drafts one at a time (resumable)")
    p.set_defaults(func=cmd_review_gold)

    p = sub.add_parser("run", help="run every approved question")
    p.add_argument("--answerer", choices=sorted(ANSWERERS), default="baseline")
    p.add_argument("--retrieval-only", action="store_true", help="stop after retrieval (no LLM)")
    p.add_argument("--k", type=int, default=config.DEFAULT_TOP_K, help="top_k context items")
    p.add_argument("--mmr-lambda", type=float, default=config.MMR_LAMBDA)
    p.add_argument("--fetch-k", type=int, default=config.MMR_FETCH_K)
    p.add_argument("--prompt", choices=sorted(PROMPTS), default=config.PROMPT_VERSION)
    p.add_argument("--pin-contexts", metavar="RUN", help="replay the contexts of an earlier run (full stage)")
    p.add_argument("--note", default="")
    p.add_argument("--resume", metavar="RUN", help="continue an interrupted run")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="recompute metrics, report and the runs.csv row")
    p.add_argument("run")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="compare two runs")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("promote", help="make a run the baseline")
    p.add_argument("run")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_promote)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
