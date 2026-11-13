"""
Experiment tracking (eval plan v15, 6.2, 6.3; D12, D15, D28): eval/runs.csv (fixed columns, empty
cells for metrics a stage did not compute, row rewritten on a re-score), eval/baseline.json
(promote + history; v20 D82: a separate `retrieval` slot for retrieval-stage runs), comparability
checks and run-vs-run deltas.
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
SETTING_KEYS = ("stage", "answerer", "prompt_version", "prompt_sha256", "ollama_model", "top_k", "mmr",
                "doc_description_in_chunks", "small_sections_merged", "query_rewrite", "index_fp",
                "question_set_sha256", "privacy_fp")
# Settings baked into the index: changing one always changes index_fp too, so index_fp is not counted as a
# second changed setting next to them (D13 count; owner decision 2026-11-11).
INDEX_BUILD_KEYS = ("doc_description_in_chunks", "small_sections_merged")
# Value of a setting in manifests / baseline entries written before it existed: what every run did then.
SETTING_DEFAULTS: dict[str, Any] = {"doc_description_in_chunks": False,     # D88: no index had descriptions
                                    "small_sections_merged": False,         # D95: no index merged small chunks
                                    "query_rewrite": "none"}                # D94: no run rewrote the query


def setting(manifest: dict[str, Any], key: str) -> Any:
    """A setting of a run; a manifest written before a setting existed reads its SETTING_DEFAULTS value."""
    value = manifest.get(key)
    return SETTING_DEFAULTS.get(key) if value is None else value


def manifest_columns(manifest: dict[str, Any]) -> dict[str, Any]:
    mmr = manifest.get("mmr") or {}
    sha = manifest.get("git_sha") or ""
    on = bool(mmr.get("enabled"))
    return {
        "run_id": manifest.get("run_id"), "date": manifest.get("date"), "note": manifest.get("note"),
        "stage": manifest.get("stage"), "git_sha": sha + ("-dirty" if manifest.get("git_dirty") else ""),
        "answerer": manifest.get("answerer"), "prompt_version": manifest.get("prompt_version"),
        "top_k": manifest.get("top_k"), "mmr_lambda": mmr.get("lambda") if on else None,
        "mmr_fetch_k": mmr.get("fetch_k") if on else None,                                     # D88
        "doc_description_in_chunks": setting(manifest, "doc_description_in_chunks"),           # D88
        "small_sections_merged": setting(manifest, "small_sections_merged"),                   # D95
        "query_rewrite": setting(manifest, "query_rewrite"),                                   # D94
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


def read_rows(path: Path | None = None) -> list[dict[str, str]]:
    """Every runs.csv row as strings (empty cell = ""); [] when the file does not exist."""
    path = path or paths.RUNS_CSV_PATH
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_row(row: dict[str, str], path: Path | None = None) -> None:
    """Append the row, or rewrite the existing row of the same run_id (re-score / resumed judging)."""
    path = path or paths.RUNS_CSV_PATH
    rows = [r for r in read_rows(path) if r.get("run_id") != row["run_id"]] + [row]
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


def baseline_slot(stage: str | None) -> tuple[str, str]:
    """(slot, history key) in baseline.json for a run of this stage (v20, D82)."""
    return ("retrieval", "retrieval_history") if stage == "retrieval" else ("current", "history")


def baseline_for(stage: str | None, path: Path | None = None) -> dict[str, Any] | None:
    """The baseline a run of this stage is compared against: retrieval slot or current (D82)."""
    return (read_baseline(path) or {}).get(baseline_slot(stage)[0])


def promote_problem(manifest: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    """6.3: a full run needs 100 % judge and part-outcome coverage; a retrieval run goes to its own slot (D82)."""
    stage = manifest.get("stage")
    if stage == "retrieval":
        return None
    if stage != "full":
        return f"promote needs a full or retrieval run (stage = {stage})"
    for key in ("judge_coverage", "part_outcome_coverage"):
        if metrics.get(key) != 1.0:
            return f"promote needs {key} = 100 % (is {metrics.get(key)})"
    return None


def promote(manifest: dict[str, Any], scores: dict[str, Any], note: str,
            path: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Write the run into its stage's slot; the previous entry of that slot moves to its history."""
    path = path or paths.BASELINE_PATH
    problem = promote_problem(manifest, scores.get("metrics") or {})
    if problem:
        raise ValueError(problem)
    slot, history_key = baseline_slot(manifest.get("stage"))
    data = read_baseline(path) or {"current": None, "history": []}     # other slots are kept as they are
    entry = {"run_id": manifest.get("run_id"), "promoted": (now or datetime.now(timezone.utc)).isoformat(),
             "note": note, "settings": {k: setting(manifest, k) for k in SETTING_KEYS}, "metrics": scores.get("metrics")}
    history = data.get(history_key) or []
    if data.get(slot):
        history.append(data[slot])
    data[slot], data[history_key] = entry, history
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


# --- comparability ---------------------------------------------------------------------------

def _comparable(settings: dict[str, Any], key: str) -> Any:
    """A setting as compared between runs: MMR off is one value whatever its unused lambda / fetch_k, so the
    "Changed vs baseline" line agrees with the report's Settings section (D85)."""
    if key == "mmr":
        mmr = settings.get("mmr") or {}
        return (mmr.get("lambda"), mmr.get("fetch_k")) if mmr.get("enabled") else None
    return setting(settings, key)


def changed_settings(manifest: dict[str, Any], baseline_settings: dict[str, Any]) -> list[str]:
    return [k for k in SETTING_KEYS if _comparable(manifest, k) != _comparable(baseline_settings, k)]


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
        counted = [c for c in changed if c not in ("privacy_fp", "question_set_sha256", "stage")]
        if any(k in changed for k in INDEX_BUILD_KEYS):
            counted = [c for c in counted if c != "index_fp"]      # a consequence of the index flag, not a 2nd lever
        if len(counted) > 1:
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
