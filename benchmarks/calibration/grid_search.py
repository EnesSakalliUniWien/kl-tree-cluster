"""Grid search: inflation calibration × BH alpha interaction.

Runs the KL decomposition across a grid of (sibling_test_method, sibling_alpha,
edge_alpha) combinations on a curated set of benchmark cases, emitting a
per-node diagnostic table and a case-level summary.

Usage
-----
    python benchmarks/calibration/grid_search.py
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ---------------------------------------------------------------------------
# Grid axes
# ---------------------------------------------------------------------------

SIBLING_METHODS = ["wald", "cousin_adjusted_wald"]
SIBLING_ALPHAS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]
EDGE_ALPHAS = [0.01, 0.05]

# ---------------------------------------------------------------------------
# Benchmark cases (self-contained, no generator dependency)
# ---------------------------------------------------------------------------


def _make_block_diagonal(n_per: int, n_feat: int, k: int, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    block_size = n_feat // k
    X = rng.binomial(1, noise, size=(n_per * k, n_feat)).astype(float)
    labels = np.repeat(np.arange(k), n_per)
    for c in range(k):
        rows = slice(c * n_per, (c + 1) * n_per)
        cols = slice(c * block_size, (c + 1) * block_size)
        X[rows, cols] = rng.binomial(1, 1.0 - noise, size=(n_per, block_size)).astype(float)
    idx = [f"S{i}" for i in range(len(labels))]
    feat = [f"F{j}" for j in range(n_feat)]
    return {"data": pd.DataFrame(X, index=idx, columns=feat), "labels": labels}


CASES = [
    {"name": "easy_2c", "n_per": 30, "n_feat": 40, "k": 2, "noise": 0.15, "seed": 0},
    {"name": "easy_3c", "n_per": 25, "n_feat": 60, "k": 3, "noise": 0.15, "seed": 1},
    {"name": "easy_4c", "n_per": 20, "n_feat": 80, "k": 4, "noise": 0.15, "seed": 2},
    {"name": "noisy_2c", "n_per": 30, "n_feat": 40, "k": 2, "noise": 0.30, "seed": 3},
    {"name": "noisy_3c", "n_per": 25, "n_feat": 60, "k": 3, "noise": 0.30, "seed": 4},
    {"name": "noisy_4c", "n_per": 20, "n_feat": 80, "k": 4, "noise": 0.30, "seed": 5},
    {"name": "hard_3c", "n_per": 25, "n_feat": 60, "k": 3, "noise": 0.40, "seed": 6},
    {"name": "small_2c", "n_per": 12, "n_feat": 30, "k": 2, "noise": 0.20, "seed": 7},
    {"name": "large_5c", "n_per": 40, "n_feat": 100, "k": 5, "noise": 0.15, "seed": 8},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_one(
    data: pd.DataFrame,
    y_true: np.ndarray,
    sibling_method: str,
    sibling_alpha: float,
    edge_alpha: float,
) -> dict:
    """Run a single decomposition with overridden config and return metrics."""
    # Save originals
    orig_method = config.SIBLING_TEST_METHOD
    orig_sib_alpha = config.SIBLING_ALPHA
    orig_edge_alpha = config.EDGE_ALPHA

    try:
        config.SIBLING_TEST_METHOD = sibling_method
        config.SIBLING_ALPHA = sibling_alpha
        config.EDGE_ALPHA = edge_alpha

        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(
                leaf_data=data,
                alpha_local=edge_alpha,
                sibling_alpha=sibling_alpha,
            )

        k_found = decomp.get("num_clusters", 0)
        assignments = decomp.get("cluster_assignments", {})
        labels_pred = np.full(len(data), -1, dtype=int)
        for cid, info in assignments.items():
            for leaf in info["leaves"]:
                idx = data.index.get_loc(leaf)
                labels_pred[idx] = cid

        ari = adjusted_rand_score(y_true, labels_pred)

        # Extract calibration audit
        annotations_df = tree.annotations_df
        audit = (
            annotations_df.attrs.get("sibling_divergence_audit", {})
            if annotations_df is not None
            else {}
        )
        cal_method = audit.get("calibration_method", "n/a")
        cal_n = audit.get("calibration_n", 0)
        c_hat = audit.get("global_inflation_factor", 1.0)
        n_focal = audit.get("focal_pairs", audit.get("total_tests", 0))
        n_null = audit.get("null_like_pairs", 0)

        # Count BH rejections from annotations_df
        n_bh_reject = 0
        n_tested = 0
        if annotations_df is not None and "Sibling_BH_Different" in annotations_df.columns:
            tested = annotations_df["Sibling_Divergence_Skipped"] == False  # noqa: E712
            n_tested = int(tested.sum())
            n_bh_reject = int(annotations_df.loc[tested, "Sibling_BH_Different"].sum())

        return {
            "k_found": k_found,
            "ari": ari,
            "calibration_method": cal_method,
            "calibration_n": cal_n,
            "c_hat": c_hat,
            "n_focal": n_focal,
            "n_null": n_null,
            "n_tested": n_tested,
            "n_bh_reject": n_bh_reject,
        }
    finally:
        config.SIBLING_TEST_METHOD = orig_method
        config.SIBLING_ALPHA = orig_sib_alpha
        config.EDGE_ALPHA = orig_edge_alpha


def main() -> None:
    grid = list(itertools.product(CASES, SIBLING_METHODS, SIBLING_ALPHAS, EDGE_ALPHAS))
    print(
        f"Grid: {len(CASES)} cases × {len(SIBLING_METHODS)} methods "
        f"× {len(SIBLING_ALPHAS)} sib_α × {len(EDGE_ALPHAS)} edge_α "
        f"= {len(grid)} runs\n"
    )

    rows: list[dict] = []
    for i, (case_cfg, method, sib_a, edge_a) in enumerate(grid, 1):
        gen = _make_block_diagonal(
            case_cfg["n_per"],
            case_cfg["n_feat"],
            case_cfg["k"],
            case_cfg["noise"],
            case_cfg["seed"],
        )
        if i % 20 == 1:
            print(
                f"[{i}/{len(grid)}] {case_cfg['name']}  method={method}  "
                f"sib_α={sib_a}  edge_α={edge_a}"
            )

        result = _run_one(gen["data"], gen["labels"], method, sib_a, edge_a)
        rows.append(
            {
                "case": case_cfg["name"],
                "true_k": case_cfg["k"],
                "noise": case_cfg["noise"],
                "sibling_method": method,
                "sibling_alpha": sib_a,
                "edge_alpha": edge_a,
                **result,
            }
        )

    df = pd.DataFrame(rows)

    # --- Summary tables ---
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    # 1. Mean ARI by method × sibling_alpha (averaged over cases and edge_alpha)
    pivot_ari = df.pivot_table(
        values="ari", index="sibling_alpha", columns="sibling_method", aggfunc="mean"
    )
    print("\n--- Mean ARI by sibling_alpha × method ---")
    print(pivot_ari.round(3).to_string())

    # 2. Exact-K rate
    df["exact_k"] = (df["k_found"] == df["true_k"]).astype(int)
    pivot_k = df.pivot_table(
        values="exact_k", index="sibling_alpha", columns="sibling_method", aggfunc="mean"
    )
    print("\n--- Exact-K rate by sibling_alpha × method ---")
    print(pivot_k.round(3).to_string())

    # 3. Mean K found
    pivot_kf = df.pivot_table(
        values="k_found", index="sibling_alpha", columns="sibling_method", aggfunc="mean"
    )
    print("\n--- Mean K found by sibling_alpha × method ---")
    print(pivot_kf.round(2).to_string())

    # 4. Calibration summary for adjusted wald
    adj_df = df[df["sibling_method"] == "cousin_adjusted_wald"]
    if not adj_df.empty:
        print("\n--- Inflation calibration summary (cousin_adjusted_wald) ---")
        cal_summary = adj_df.groupby("sibling_alpha").agg(
            mean_c_hat=("c_hat", "mean"),
            mean_n_null=("n_null", "mean"),
            mean_n_focal=("n_focal", "mean"),
            mean_n_bh_reject=("n_bh_reject", "mean"),
        )
        print(cal_summary.round(3).to_string())

    # 5. Per-case breakdown at best alpha
    best_alpha = pivot_ari.mean(axis=1).idxmax()
    print(f"\n--- Per-case detail at best sibling_alpha={best_alpha} (edge_alpha=0.01) ---")
    detail = df[(df["sibling_alpha"] == best_alpha) & (df["edge_alpha"] == 0.01)]
    detail_cols = [
        "case",
        "true_k",
        "noise",
        "sibling_method",
        "k_found",
        "ari",
        "c_hat",
        "n_null",
        "n_focal",
        "n_bh_reject",
    ]
    print(detail[detail_cols].to_string(index=False))

    # 6. Edge alpha effect
    print("\n--- Mean ARI by edge_alpha × method ---")
    pivot_edge = df.pivot_table(
        values="ari", index="edge_alpha", columns="sibling_method", aggfunc="mean"
    )
    print(pivot_edge.round(3).to_string())

    # Save to CSV
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_search_calibration_bh.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")
    print(f"Total runs: {len(df)}")


if __name__ == "__main__":
    main()
