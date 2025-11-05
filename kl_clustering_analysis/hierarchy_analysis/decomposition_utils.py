from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def binary_kl(q: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    """Calculates binary Kullback-Leibler divergence."""
    q = np.clip(q, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)
    v = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
    return float(v.sum())


def generate_decomposition_report(
    decomposition_results: Dict[str, object],
) -> pd.DataFrame:
    """Generates a DataFrame report from decomposition results."""
    cluster_assignments = decomposition_results.get("cluster_assignments", {})
    if not cluster_assignments:
        return pd.DataFrame(columns=["cluster_id", "cluster_root", "cluster_size"])
    rows = {}
    for cid, info in cluster_assignments.items():
        for lbl in info["leaves"]:
            rows[lbl] = {
                "cluster_id": cid,
                "cluster_root": info["root_node"],
                "cluster_size": info["size"],
            }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "sample_id"
    return df.sort_values("cluster_id")
