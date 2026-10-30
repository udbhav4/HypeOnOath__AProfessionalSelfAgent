"""
Experiment tracking (eval plan v15, 6.2, 6.3; D12, D15, D28): eval/runs.csv (fixed columns, empty
cells for metrics a stage did not compute, row rewritten on a re-score), eval/baseline.json
(promote + history), comparability checks and run-vs-run deltas.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypeonoath.evaluation import paths
from hypeonoath.evaluation.metrics import METRICS, METRICS_BY_NAME, RUNS_CSV_COLUMNS

# Settings whose change makes a run a different experiment ("Changed vs baseline", D13).
SETTING_KEYS = ("stage", "answerer", "prompt_version", "prompt_sha256", "ollama_model", "top_k", "mmr", "index_fp",
                "question_set_sha256", "privacy_fp")


def manifest_columns(manifest: dict[str, Any]) -> dict[str, Any]:
    mmr = manifest.get("mmr") or {}
    sha = manifest.get("git_sha") or ""
    return {
        "run_id": manifest.get("run_id"), "date": manifest.get("date"), "note": manifest.get("note"),
        "stage": manifest.get("stage"), "git_sha": sha + ("-dirty" if manifest.get("git_dirty") else ""),
        "answerer": manifest.get("answerer"), "prompt_version": manifest.get("prompt_version"),
        "top_k": manifest.get("top_k"), "mmr_lambda": mmr.get("lambda") if mmr.get("enabled") else None,
        "index_fp": manifest.get("index_fp"), "privacy_fp": manifest.get("privacy_fp"),
        "question_set_sha": manifest.get("question_set_sha256"),
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""                                         # D28: empty, never 0
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def csv_row(manifest: dict[str, Any], metrics: dict[str, Any]) -> dict[str, str]:
    """One runs.csv row: settings from the manifest, metrics from scores (stage-limited)."""
    stage = manifest.get("stage")
    row = {}
    values = {**metrics, **manifest_columns(manifest)}
    for column in RUNS_CSV_COLUMNS:
        definition = METRICS_BY_NAME[column]
        value = values.get(column)
        row[column] = _cell(value if stage in definition.stages else None)
    return row


def write_row(row: dict[str, str], path: Path | None = None) -> None:
    """Append the row, or rewrite the existing row of the same run_id (re-score / resumed judging)."""
    path = path or paths.RUNS_CSV_PATH
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("run_id") != row["run_id"]] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- baseline -------------------------------------------------------------------------------

def read_baseline(path: Path | None = None) -> dict[str, Any] | None:
    path = path or paths.BASELINE_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def promote_problem(manifest: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    """6.3: only a full run with 100 % judge and part-outcome coverage can become the baseline."""
    if manifest.get("stage") != "full":
        return "promote needs a full run (stage = full)"
    for key in ("judge_coverage", "part_outcome_coverage"):
        if metrics.get(key) != 1.0:
            return f"promote needs {key} = 100 % (is {metrics.get(key)})"
    return None


def promote(manifest: dict[str, Any], scores: dict[str, Any], note: str,
            path: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    path = path or paths.BASELINE_PATH
    problem = promote_problem(manifest, scores.get("metrics") or {})
    if problem:
        raise ValueError(problem)
    old = read_baseline(path) or {"current": None, "history": []}
    entry = {"run_id": manifest.get("run_id"), "promoted": (now or datetime.now(timezone.utc)).isoformat(),
             "note": note, "settings": {k: manifest.get(k) for k in SETTING_KEYS}, "metrics": scores.get("metrics")}
    history = old.get("history") or []
    if old.get("current"):
        history.append(old["current"])
    data = {"current": entry, "history": history}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


# --- comparability ---------------------------------------------------------------------------

def changed_settings(manifest: dict[str, Any], baseline_settings: dict[str, Any]) -> list[str]:
    return [k for k in SETTING_KEYS if manifest.get(k) != baseline_settings.get(k)]


def comparability_warnings(manifest: dict[str, Any], metrics: dict[str, Any],
                           baseline: dict[str, Any] | None) -> list[str]:
    warnings = []
    if manifest.get("generation_status") == "aborted":
        warnings.append("generation was aborted: not every question has a reply")
    for key in ("judge_coverage", "part_outcome_coverage"):
        value = metrics.get(key)
        if manifest.get("stage") == "full" and value is not None and value < 1.0:
            warnings.append(f"{key} {value:.0%} < 100 %: judged deltas withheld")
    if metrics.get("errors"):
        warnings.append(f"{metrics['errors']} generation errors")
    if baseline:
        settings = baseline.get("settings") or {}
        changed = changed_settings(manifest, settings)
        if "privacy_fp" in changed:
            warnings.append("privacy decisions differ from the baseline: not comparable (D49)")
        if "question_set_sha256" in changed:
            warnings.append("different question set than the baseline: not comparable")
        if "stage" in changed:
            warnings.append(f"stage {manifest.get('stage')} differs from the baseline's {settings.get('stage')}")
        if "index_fp" in changed:
            warnings.append("different index than the baseline")
        if len([c for c in changed if c not in ("privacy_fp", "question_set_sha256", "stage")]) > 1:
            warnings.append(f"more than one setting changed vs baseline ({', '.join(changed)}): "
                            "the cause of a delta is ambiguous (D13)")
    return warnings


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Run B vs run A over the metrics present (non-empty) in both; delta = B - A."""
    rows = []
    for m in METRICS:
        if m.section == "setting":
            continue
        va, vb = a.get(m.name), b.get(m.name)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            delta = vb - va
            better = None if m.direction == "none" or delta == 0 else (delta > 0) == (m.direction == "up")
            rows.append({"metric": m.name, "a": va, "b": vb, "delta": delta, "better": better})
    return rows
