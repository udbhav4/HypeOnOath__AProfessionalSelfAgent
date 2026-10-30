"""Every eval test writes into tmp_path: the real eval/ outputs (runs, runs.csv, baseline, questions,
drafts, METRICS.md) are never touched. The committed config files (doc_groups, edge_tags) stay readable."""
from __future__ import annotations

import pytest

from hypeonoath.evaluation import paths

REDIRECTED = {
    "QUESTIONS_PATH": "questions.yaml", "QUESTIONS_V0_PATH": "questions_v0.yaml",
    "DRAFTS_PATH": "gold_work/drafts.yaml", "RUNS_DIR": "runs", "RUNS_CSV_PATH": "runs.csv",
    "BASELINE_PATH": "baseline.json", "METRICS_MD_PATH": "METRICS.md",
}


@pytest.fixture(autouse=True)
def isolated_eval_dir(tmp_path, monkeypatch):
    root = tmp_path / "eval"
    for name, rel in REDIRECTED.items():
        monkeypatch.setattr(paths, name, root / rel)
    root.mkdir(parents=True, exist_ok=True)
    return root
