"""
Where the eval files live (eval plan v15, Sections 6 and 8) and the fixed eval settings.

Committed (D50): runs.csv, baseline.json, METRICS.md, questions.yaml, doc_groups.yaml,
edge_tags.yaml (+ judge.yaml, judge_rubric.md from Stage D). Everything else under eval/
stays local (.gitignore).
"""
from __future__ import annotations

from hypeonoath.core import paths

EVAL_DIR = paths.PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.yaml"
QUESTIONS_V0_PATH = EVAL_DIR / "questions_v0.yaml"     # the pre-harness 25 questions, archived, not scored
DRAFTS_PATH = EVAL_DIR / "gold_work" / "drafts.yaml"
DOC_GROUPS_PATH = EVAL_DIR / "doc_groups.yaml"
EDGE_TAGS_PATH = EVAL_DIR / "edge_tags.yaml"
RUNS_DIR = EVAL_DIR / "runs"
RUNS_CSV_PATH = EVAL_DIR / "runs.csv"
BASELINE_PATH = EVAL_DIR / "baseline.json"
METRICS_MD_PATH = EVAL_DIR / "METRICS.md"

EVAL_DEPTH = 20                                   # D14: raw ranking depth of every eval query
K_VALUES = (1, 3, 4, 5, 8, 10, 12, 15, 20)        # D14 / D74 (v16): Hit@k / Recall@k cut-offs
REFERENCE_K = 8          # the fixed k of recall@8, supporting_recall@8 and the hit@8 breakdowns (6.2 column names)
MAX_CONSECUTIVE_LLM_ERRORS = 3                    # D26: abort the run after this many in a row


def check_k_values(k_values: tuple[int, ...], eval_depth: int = EVAL_DEPTH, reference_k: int = REFERENCE_K) -> None:
    """D74: ascending, unique, within 1..eval_depth, and including the reference k (recall@8)."""
    if (reference_k not in k_values or list(k_values) != sorted(set(k_values)) or k_values[0] < 1
            or k_values[-1] > eval_depth):
        raise ValueError(f"K_VALUES must be ascending, unique, within 1-{eval_depth} (EVAL_DEPTH) and include "
                         f"{reference_k} (recall@{reference_k}); got {k_values}")


check_k_values(K_VALUES)                                 # fail at import, not mid-run
