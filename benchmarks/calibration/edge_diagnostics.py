"""Edge-only calibration diagnostics.

Produces edge-level artifacts to isolate inflation sources without changing
production inference code:
1) Conditional p-value histograms (overall + stratified).
2) KS tests against Uniform(0,1).
3) Fixed-tree permutation checks (selection removed).
4) Stratification by n_child / n_parent and branch-length usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import kstest

from benchmarks.shared.util.time import format_timestamp_utc
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _default_null_scenarios() -> list[dict[str, Any]]:
    return [
        {"scenario": "null_small", "n_samples": 64, "n_features": 32, "p_one": 0.5},
        {"scenario": "null_medium", "n_samples": 128, "n_features": 64, "p_one": 0.5},
        {"scenario": "null_large", "n_samples": 192, "n_features": 96, "p_one": 0.5},
    ]


def _make_null_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    p_one: float,
) -> pd.DataFrame:
    matrix = (rng.random((n_samples, n_features)) < p_one).astype(int)
    rows = [f"S{i}" for i in range(n_samples)]
    cols = [f"F{j}" for j in range(n_features)]
    return pd.DataFrame(matrix, index=rows, columns=cols)


def _permute_features(
    data_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    arr = data_df.to_numpy(copy=True)
    for j in range(arr.shape[1]):
        rng.shuffle(arr[:, j])
    return pd.DataFrame(arr, index=data_df.index, columns=data_df.columns)


def _ratio_bin(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "missing"
    if ratio <= 0.125:
        return "<=0.125"
    if ratio <= 0.25:
        return "(0.125,0.25]"
    if ratio <= 0.5:
        return "(0.25,0.5]"
    return ">0.5"


def _run_tree(
    linkage_matrix: np.ndarray,
    leaf_names: list[str],
    data_df: pd.DataFrame,
    alpha: float,
) -> tuple[PosetTree, pd.DataFrame]:
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=leaf_names)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No eligible parent nodes for sibling tests",
            category=UserWarning,
        )
        tree.decompose(
            leaf_data=data_df,
            alpha_local=float(alpha),
            sibling_alpha=float(alpha),
        )
    stats_df = tree.stats_df if tree.stats_df is not None else pd.DataFrame(index=tree.nodes())
    return tree, stats_df


def _extract_edge_rows(
    tree: PosetTree,
    stats_df: pd.DataFrame,
    *,
    scenario: str,
    mode: str,
    replicate: int,
    permutation_id: int,
    alpha: float,
) -> list[dict[str, Any]]:
    mean_bl = compute_mean_branch_length(tree)
    out: list[dict[str, Any]] = []

    p_col = "Child_Parent_Divergence_P_Value"
    sig_col = "Child_Parent_Divergence_Significant"
    inv_col = "Child_Parent_Divergence_Invalid"

    for parent, child in tree.edges():
        p_value = (
            float(stats_df.at[child, p_col])
            if p_col in stats_df.columns and child in stats_df.index
            else np.nan
        )
        reject = (
            bool(stats_df.at[child, sig_col])
            if sig_col in stats_df.columns and child in stats_df.index
            else False
        )
        invalid = (
            bool(stats_df.at[child, inv_col])
            if inv_col in stats_df.columns and child in stats_df.index
            else False
        )

        n_child = float(tree.nodes[child].get("leaf_count", np.nan))
        n_parent = float(tree.nodes[parent].get("leaf_count", np.nan))
        sample_ratio = n_child / n_parent if n_parent > 0 else np.nan

        bl_raw = tree.edges[parent, child].get("branch_length")
        branch_length = sanitize_positive_branch_length(bl_raw)
        bl_usable = bool(
            branch_length is not None
            and mean_bl is not None
            and np.isfinite(mean_bl)
            and mean_bl > 0
        )
        bl_normalized = (
            float(1.0 + branch_length / mean_bl)
            if bl_usable and branch_length is not None and mean_bl is not None
            else np.nan
        )

        out.append(
            {
                "scenario": scenario,
                "mode": mode,
                "replicate": int(replicate),
                "permutation_id": int(permutation_id),
                "parent": parent,
                "child": child,
                "alpha": float(alpha),
                "p_value": p_value,
                "reject": int(reject),
                "invalid": int(invalid),
                "n_child": n_child,
                "n_parent": n_parent,
                "sample_ratio": sample_ratio,
                "sample_ratio_bin": _ratio_bin(sample_ratio),
                "branch_length": float(branch_length) if branch_length is not None else np.nan,
                "mean_branch_length": float(mean_bl) if mean_bl is not None else np.nan,
                "branch_length_usable": int(bl_usable),
                "bl_normalized": bl_normalized,
            }
        )

    return out


def _ks_summary(df_edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mode, scenario), group in df_edges.groupby(["mode", "scenario"], dropna=False):
        pvals = group["p_value"].to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size > 0:
            ks_stat, ks_p = kstest(pvals, "uniform")
        else:
            ks_stat, ks_p = np.nan, np.nan
        rows.append(
            {
                "mode": mode,
                "scenario": scenario,
                "n_edges": int(len(group)),
                "n_finite_p": int(pvals.size),
                "type1_rate": float(group["reject"].mean()) if len(group) > 0 else np.nan,
                "invalid_rate": float(group["invalid"].mean()) if len(group) > 0 else np.nan,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }
        )
    return pd.DataFrame(rows)


def _ks_summary_stratified(df_edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["mode", "scenario", "sample_ratio_bin", "branch_length_usable"]
    for key_vals, group in df_edges.groupby(keys, dropna=False):
        pvals = group["p_value"].to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size >= 20:
            ks_stat, ks_p = kstest(pvals, "uniform")
        else:
            ks_stat, ks_p = np.nan, np.nan
        rows.append(
            {
                "mode": key_vals[0],
                "scenario": key_vals[1],
                "sample_ratio_bin": key_vals[2],
                "branch_length_usable": int(key_vals[3]),
                "n_edges": int(len(group)),
                "n_finite_p": int(pvals.size),
                "type1_rate": float(group["reject"].mean()) if len(group) > 0 else np.nan,
                "invalid_rate": float(group["invalid"].mean()) if len(group) > 0 else np.nan,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }
        )
    return pd.DataFrame(rows)


def _histogram_summary(df_edges: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    keys = ["mode", "scenario", "sample_ratio_bin", "branch_length_usable"]

    for key_vals, group in df_edges.groupby(keys, dropna=False):
        pvals = group["p_value"].to_numpy(dtype=float)
        pvals = pvals[np.isfinite(pvals)]
        counts, edges = np.histogram(pvals, bins=bins)
        total = int(np.sum(counts))
        for i, count in enumerate(counts):
            rows.append(
                {
                    "mode": key_vals[0],
                    "scenario": key_vals[1],
                    "sample_ratio_bin": key_vals[2],
                    "branch_length_usable": int(key_vals[3]),
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(count),
                    "proportion": float(count / total) if total > 0 else np.nan,
                    "expected_uniform": float(1.0 / n_bins),
                    "n_finite_p_in_group": total,
                }
            )

    return pd.DataFrame(rows)


def _replicate_summary(df_edges: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_edges.groupby(["mode", "scenario", "replicate", "permutation_id"], as_index=False)
        .agg(
            n_edges=("reject", "size"),
            rejects=("reject", "sum"),
            invalid=("invalid", "sum"),
        )
        .sort_values(["mode", "scenario", "replicate", "permutation_id"])
    )
    grouped["type1_rate"] = grouped["rejects"] / grouped["n_edges"].replace(0, np.nan)
    grouped["invalid_rate"] = grouped["invalid"] / grouped["n_edges"].replace(0, np.nan)
    return grouped


def run_edge_diagnostics(
    output_dir: Path,
    *,
    alpha: float = 0.05,
    n_reps: int = 20,
    n_fixed_perms: int = 10,
    seed: int = 20260213,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    all_rows: list[dict[str, Any]] = []
    scenarios = _default_null_scenarios()

    for scenario in scenarios:
        scenario_name = str(scenario["scenario"])
        n_samples = int(scenario["n_samples"])
        n_features = int(scenario["n_features"])
        p_one = float(scenario["p_one"])

        for rep in range(int(n_reps)):
            base_data = _make_null_data(rng, n_samples, n_features, p_one)
            dist = pdist(base_data.values, metric=config.TREE_DISTANCE_METRIC)
            linkage_matrix = linkage(dist, method=config.TREE_LINKAGE_METHOD)
            leaf_names = base_data.index.tolist()

            # In-sample: tree topology and tests estimated on same data.
            tree_in, stats_in = _run_tree(linkage_matrix, leaf_names, base_data, alpha)
            all_rows.extend(
                _extract_edge_rows(
                    tree_in,
                    stats_in,
                    scenario=scenario_name,
                    mode="in_sample",
                    replicate=rep,
                    permutation_id=-1,
                    alpha=alpha,
                )
            )

            # Fixed-tree permutations: same topology, new permuted leaf data.
            for perm_id in range(int(n_fixed_perms)):
                perm_data = _permute_features(base_data, rng)
                tree_fix, stats_fix = _run_tree(linkage_matrix, leaf_names, perm_data, alpha)
                all_rows.extend(
                    _extract_edge_rows(
                        tree_fix,
                        stats_fix,
                        scenario=scenario_name,
                        mode="fixed_tree_permutation",
                        replicate=rep,
                        permutation_id=perm_id,
                        alpha=alpha,
                    )
                )

    edges_df = pd.DataFrame(all_rows)
    ks_df = _ks_summary(edges_df)
    ks_strat_df = _ks_summary_stratified(edges_df)
    hist_df = _histogram_summary(edges_df, n_bins=20)
    rep_df = _replicate_summary(edges_df)

    edge_csv = output_dir / "edge_diagnostic_edge_level.csv"
    rep_csv = output_dir / "edge_diagnostic_replicate_summary.csv"
    ks_csv = output_dir / "edge_diagnostic_ks_summary.csv"
    ks_strat_csv = output_dir / "edge_diagnostic_ks_stratified.csv"
    hist_csv = output_dir / "edge_diagnostic_histogram_stratified.csv"

    edges_df.to_csv(edge_csv, index=False)
    rep_df.to_csv(rep_csv, index=False)
    ks_df.to_csv(ks_csv, index=False)
    ks_strat_df.to_csv(ks_strat_csv, index=False)
    hist_df.to_csv(hist_csv, index=False)

    return {
        "edge_level_csv": str(edge_csv),
        "replicate_summary_csv": str(rep_csv),
        "ks_summary_csv": str(ks_csv),
        "ks_stratified_csv": str(ks_strat_csv),
        "histogram_stratified_csv": str(hist_csv),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run edge-only calibration diagnostics.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--n-fixed-perms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260213)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/run_<timestamp>/calibration/edge_diagnostics"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_dir = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "results"
            / f"run_{format_timestamp_utc()}"
            / "calibration"
            / "edge_diagnostics"
        )
        out_dir = run_dir

    outputs = run_edge_diagnostics(
        out_dir,
        alpha=float(args.alpha),
        n_reps=int(args.n_reps),
        n_fixed_perms=int(args.n_fixed_perms),
        seed=int(args.seed),
    )
    print("Edge diagnostics complete:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
