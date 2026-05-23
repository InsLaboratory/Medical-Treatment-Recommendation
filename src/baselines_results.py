"""
baselines_results.py
====================
Utilities for saving and loading baseline evaluation results in a format
fully compatible with the CPQ-IQL experiment artefacts.

All results are stored under:
    experiments/baselines_<method>.json        per-method JSON
    experiments/baselines_summary.csv          aggregated comparison table

Schema (per-method JSON)
------------------------
{
  "method"       : "random" | "dqn" | "iql" | "cql",
  "cvr_rollout"  : { total_cvr, C1_hypotension, C2_metabolic,
                     C3_cumulative, C4_withdrawal },
  "safe_actions" : { safe_total_cvr, safe_C1_hypotension, safe_C2_metabolic,
                     safe_C3_cumulative, safe_C4_withdrawal,
                     n_interventions, intervention_rate },
  "bc_accuracy"  : { top1, top3, top5, top1_norm, top3_norm,
                     fluid_top1, vaso_top1 },
  "survival_rate": { sr_clinician, sr_policy, delta_sr, ci_lo, ci_hi,
                     n_patients },
  "fqe"          : { v_fqe, v_wis_clinician, delta_v,
                     val_bellman_loss, n_epochs_trained },
  "training"     : { n_epochs_trained, best_val_loss },
}

Public API
----------
save_baseline_results(method, results, exp_dir)
load_baseline_results(method, exp_dir) -> dict
build_summary_csv(exp_dir, methods, out_path) -> pd.DataFrame
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_baseline_results(
    method: str,
    results: Dict,
    exp_dir: str = "../experiments",
) -> str:
    """Persist a baseline's evaluation results as JSON.

    Parameters
    ----------
    method : str
        One of ``"random"``, ``"dqn"``, ``"iql"``, ``"cql"``.
    results : dict
        Dictionary containing the evaluation sub-dicts defined in the
        module docstring.
    exp_dir : str
        Directory where artefacts are stored.

    Returns
    -------
    str  — absolute path to the written JSON file.
    """
    os.makedirs(exp_dir, exist_ok=True)
    payload = {"method": method, **_to_python(results)}
    path = os.path.join(exp_dir, f"baselines_{method}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_baseline_results(
    method: str,
    exp_dir: str = "../experiments",
) -> Dict:
    """Load a previously saved baseline result.

    Returns
    -------
    dict  — same schema as written by :func:`save_baseline_results`.

    Raises
    ------
    FileNotFoundError  — if the JSON file does not exist.
    """
    path = os.path.join(exp_dir, f"baselines_{method}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No results found for method '{method}' at {path}. "
            "Run the corresponding training cell first."
        )
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

_SUMMARY_COLUMNS = [
    "method",
    # CVR (no filter)
    "cvr_total", "cvr_C1", "cvr_C2", "cvr_C3", "cvr_C4",
    # CVR (with Safe Actions)
    "safe_cvr_total", "safe_cvr_C1", "safe_cvr_C2",
    "safe_cvr_C3", "safe_cvr_C4", "intervention_rate",
    # Efficacy
    "sr_policy", "delta_sr", "v_fqe", "delta_v",
    # Behavioral consistency
    "bc_top1", "bc_top3",
]


def build_summary_csv(
    exp_dir: str,
    methods: List[str],
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate results from all methods into a comparison DataFrame.

    Parameters
    ----------
    exp_dir : str
        Directory containing ``baselines_<method>.json`` files.
    methods : list[str]
        Methods to include, e.g. ``["random", "dqn", "iql", "cql"]``.
    out_path : str or None
        If set, the DataFrame is also written to this CSV path.

    Returns
    -------
    pd.DataFrame  with columns defined in ``_SUMMARY_COLUMNS``.
    """
    rows = []
    for m in methods:
        try:
            r = load_baseline_results(m, exp_dir)
        except FileNotFoundError:
            continue

        cvr  = r.get("cvr_rollout", {})
        safe = r.get("safe_actions", {})
        bc   = r.get("bc_accuracy", {})
        sr   = r.get("survival_rate", {})
        fqe  = r.get("fqe", {})

        rows.append({
            "method"           : m,
            "cvr_total"        : cvr.get("total_cvr",        float("nan")) * 100,
            "cvr_C1"           : cvr.get("C1_hypotension",   float("nan")) * 100,
            "cvr_C2"           : cvr.get("C2_metabolic",     float("nan")) * 100,
            "cvr_C3"           : cvr.get("C3_cumulative",    float("nan")) * 100,
            "cvr_C4"           : cvr.get("C4_withdrawal",    float("nan")) * 100,
            "safe_cvr_total"   : safe.get("safe_total_cvr",        float("nan")) * 100,
            "safe_cvr_C1"      : safe.get("safe_C1_hypotension",   float("nan")) * 100,
            "safe_cvr_C2"      : safe.get("safe_C2_metabolic",     float("nan")) * 100,
            "safe_cvr_C3"      : safe.get("safe_C3_cumulative",    float("nan")) * 100,
            "safe_cvr_C4"      : safe.get("safe_C4_withdrawal",    float("nan")) * 100,
            "intervention_rate": safe.get("intervention_rate",     float("nan")) * 100,
            "sr_policy"        : sr.get("sr_policy",  float("nan")) * 100,
            "delta_sr"         : sr.get("delta_sr",   float("nan")) * 100,
            "v_fqe"            : fqe.get("v_fqe",     float("nan")),
            "delta_v"          : fqe.get("delta_v",   float("nan")),
            "bc_top1"          : bc.get("top1",        float("nan")) * 100,
            "bc_top3"          : bc.get("top3",        float("nan")) * 100,
        })

    df = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.4f")

    return df