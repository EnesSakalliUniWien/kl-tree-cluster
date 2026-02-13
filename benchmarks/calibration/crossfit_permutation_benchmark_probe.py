"""Probe cross-fit permutation diagnostic on benchmark cases.

This is a diagnostic-only script that compares:
1) Current in-sample edge/sibling rejection rates from the existing pipeline.
2) Cross-fit + fixed-topology permutation rejection rates on the same case data.

It uses the decomposition-respecting candidate selection implemented in
`crossfit_permutation_diagnostic.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.calibration.crossfit_permutation_diagnostic import (
    _combine_two_fold_pvals,
    _evaluate_fold,
    _feature_split_indices,
)
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.time_utils import format_timestamp_utc
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _default_probe_cases() -> list[str]:
    """Representative benchmark cases across data families."""
    return [
        "gauss_clear_medium",
        "gauss_noisy_many",
        "binary_perfect_8c",
        "binary_hard_8c",
        "sbm_moderate",
        "cat_mod_4cat_6c",
        "phylo_dna_8taxa_med_mut",
        "phylo_divergent_8taxa",
        "overlap_heavy_8c_large_feat",
        "gauss_overlap_extreme_6c",
    ]


def _run_in_sample_summary(
    data_df: pd.DataFrame,
    *,
    alpha: float,
) -> dict[str, Any]:
    dist = pdist(data_df.to_numpy(dtype=np.float64), metric=config.TREE_DISTANCE_METRIC)
    linkage_matrix = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.decompose(leaf_data=data_df, alpha_local=float(alpha), sibling_alpha=float(alpha))
    stats_df = tree.stats_df if tree.stats_df is not None else pd.DataFrame(index=tree.nodes())

    edge_mask = (
        stats_df["Child_Parent_Divergence_P_Value_BH"].notna()
        if "Child_Parent_Divergence_P_Value_BH" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )
    sibling_mask = (
        stats_df["Sibling_Divergence_P_Value_Corrected"].notna()
        if "Sibling_Divergence_P_Value_Corrected" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )
    edge_tests = int(edge_mask.sum())
    sibling_tests = int(sibling_mask.sum())
    edge_rejects = int(
        stats_df.loc[edge_mask, "Child_Parent_Divergence_Significant"].astype(bool).sum()
    ) if edge_tests > 0 else 0
    sibling_rejects = int(
        stats_df.loc[sibling_mask, "Sibling_BH_Different"].astype(bool).sum()
    ) if sibling_tests > 0 else 0

    return {
        "edge_tests_in_sample": edge_tests,
        "edge_rejects_in_sample": edge_rejects,
        "edge_reject_rate_in_sample": float(edge_rejects / edge_tests) if edge_tests > 0 else np.nan,
        "sibling_tests_in_sample": sibling_tests,
        "sibling_rejects_in_sample": sibling_rejects,
        "sibling_reject_rate_in_sample": float(sibling_rejects / sibling_tests) if sibling_tests > 0 else np.nan,
    }


def _run_crossfit_perm_summary(
    data_df: pd.DataFrame,
    *,
    alpha: float,
    n_perms: int,
    seed: int,
) -> dict[str, Any]:
    n_features = int(data_df.shape[1])
    idx_a, idx_b = _feature_split_indices(n_features)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return {
            "edge_hypotheses_crossfit_perm": 0,
            "edge_rejects_crossfit_perm": 0,
            "edge_reject_rate_crossfit_perm": np.nan,
            "sibling_hypotheses_crossfit_perm": 0,
            "sibling_rejects_crossfit_perm": 0,
            "sibling_reject_rate_crossfit_perm": np.nan,
        }

    cols_a = [data_df.columns[i] for i in idx_a]
    cols_b = [data_df.columns[i] for i in idx_b]
    data_a = data_df.loc[:, cols_a]
    data_b = data_df.loc[:, cols_b]
    leaf_names = data_df.index.tolist()

    # A -> B
    dist_a = pdist(data_a.to_numpy(dtype=np.float64), metric=config.TREE_DISTANCE_METRIC)
    linkage_a = linkage(dist_a, method=config.TREE_LINKAGE_METHOD)
    p_edge_ab, p_sib_ab = _evaluate_fold(
        linkage_matrix=linkage_a,
        leaf_names=leaf_names,
        select_df=data_a,
        infer_df=data_b,
        alpha=float(alpha),
        n_perms=int(n_perms),
        base_seed=int(seed + 101),
    )

    # B -> A
    dist_b = pdist(data_b.to_numpy(dtype=np.float64), metric=config.TREE_DISTANCE_METRIC)
    linkage_b = linkage(dist_b, method=config.TREE_LINKAGE_METHOD)
    p_edge_ba, p_sib_ba = _evaluate_fold(
        linkage_matrix=linkage_b,
        leaf_names=leaf_names,
        select_df=data_b,
        infer_df=data_a,
        alpha=float(alpha),
        n_perms=int(n_perms),
        base_seed=int(seed + 202),
    )

    p_edge = _combine_two_fold_pvals(p_edge_ab, p_edge_ba)
    p_sib = _combine_two_fold_pvals(p_sib_ab, p_sib_ba)

    edge_vals = np.array(list(p_edge.values()), dtype=np.float64)
    sib_vals = np.array(list(p_sib.values()), dtype=np.float64)
    edge_rej = int(np.sum(edge_vals <= alpha)) if edge_vals.size > 0 else 0
    sib_rej = int(np.sum(sib_vals <= alpha)) if sib_vals.size > 0 else 0

    return {
        "edge_hypotheses_crossfit_perm": int(edge_vals.size),
        "edge_rejects_crossfit_perm": edge_rej,
        "edge_reject_rate_crossfit_perm": float(edge_rej / edge_vals.size) if edge_vals.size > 0 else np.nan,
        "sibling_hypotheses_crossfit_perm": int(sib_vals.size),
        "sibling_rejects_crossfit_perm": sib_rej,
        "sibling_reject_rate_crossfit_perm": float(sib_rej / sib_vals.size) if sib_vals.size > 0 else np.nan,
    }


def run_probe(
    *,
    case_names: list[str],
    alpha: float,
    n_perms: int,
    seed: int,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = {str(c.get("name")): c for c in get_default_test_cases()}
    rows: list[dict[str, Any]] = []

    for i, case_name in enumerate(case_names):
        if case_name not in all_cases:
            rows.append(
                {
                    "case_name": case_name,
                    "status": "missing_case",
                }
            )
            continue

        case = all_cases[case_name].copy()
        data_df, _, _, meta = generate_case_data(case)
        data_df = data_df.astype(np.float64)

        row: dict[str, Any] = {
            "case_name": case_name,
            "status": "ok",
            "generator": case.get("generator", "blobs"),
            "category": case.get("category", ""),
            "n_samples": int(data_df.shape[0]),
            "n_features": int(data_df.shape[1]),
            "alpha": float(alpha),
        }

        row.update(_run_in_sample_summary(data_df, alpha=alpha))
        row.update(
            _run_crossfit_perm_summary(
                data_df,
                alpha=alpha,
                n_perms=n_perms,
                seed=int(seed + 1000 * (i + 1)),
            )
        )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "crossfit_permutation_benchmark_probe.csv"
    out_df.to_csv(csv_path, index=False)

    summary_cols = [
        "case_name",
        "generator",
        "n_samples",
        "n_features",
        "edge_reject_rate_in_sample",
        "edge_reject_rate_crossfit_perm",
        "sibling_reject_rate_in_sample",
        "sibling_reject_rate_crossfit_perm",
        "edge_tests_in_sample",
        "edge_hypotheses_crossfit_perm",
        "sibling_tests_in_sample",
        "sibling_hypotheses_crossfit_perm",
    ]
    md_lines = ["# Cross-fit Permutation Benchmark Probe", ""]
    md_lines.append(out_df[summary_cols].to_markdown(index=False))
    md_path = output_dir / "crossfit_permutation_benchmark_probe.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"csv": str(csv_path), "md": str(md_path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-fit permutation diagnostic on benchmark cases."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260213)
    parser.add_argument(
        "--case-names",
        type=str,
        default="",
        help="Comma-separated benchmark case names. Uses a representative default if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/run_<timestamp>/calibration/crossfit_benchmark_probe"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cases = (
        [c.strip() for c in args.case_names.split(",") if c.strip()]
        if args.case_names
        else _default_probe_cases()
    )
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "results"
            / f"run_{format_timestamp_utc()}"
            / "calibration"
            / "crossfit_benchmark_probe"
        )
    outputs = run_probe(
        case_names=cases,
        alpha=float(args.alpha),
        n_perms=int(args.n_perms),
        seed=int(args.seed),
        output_dir=out_dir,
    )
    print("Cross-fit benchmark probe complete:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
