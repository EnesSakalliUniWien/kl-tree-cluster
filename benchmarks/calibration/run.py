"""Null/Type-I and TreeBH calibration benchmark suite.

This module provides empirical calibration reports for:
1) Null Type-I behavior of edge/sibling tests in the full KL-TE pipeline.
2) TreeBH FDR behavior on tree-structured synthetic p-values with known truth.

Outputs are written to:
    <run_dir>/calibration/
        null_replicate_level.csv
        null_type1_summary.csv
        treebh_replicate_level.csv
        treebh_summary.csv
        calibration_plots.pdf
        calibration_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.calibration.crossfit_permutation_benchmark_probe import (
    _default_probe_cases as _default_crossfit_probe_cases,
)
from benchmarks.calibration.crossfit_permutation_benchmark_probe import (
    run_probe as run_crossfit_benchmark_probe_fn,
)
from benchmarks.calibration.crossfit_permutation_diagnostic import (
    run_crossfit_permutation_diagnostic,
)
from benchmarks.shared.util.time import format_timestamp_utc
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction import (
    tree_bh_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    compute_projection_dimension,
    derive_projection_seed,
    generate_projection_matrix,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _binomial_ci_95(successes: int, trials: int) -> tuple[float, float]:
    """Normal-approximation 95% interval for a Bernoulli proportion."""
    if trials <= 0:
        return np.nan, np.nan
    p_hat = successes / trials
    se = np.sqrt(max(p_hat * (1.0 - p_hat) / trials, 0.0))
    z = 1.96
    low = max(0.0, p_hat - z * se)
    high = min(1.0, p_hat + z * se)
    return float(low), float(high)


def _default_null_scenarios() -> list[dict[str, Any]]:
    return [
        {"scenario": "null_small", "n_samples": 64, "n_features": 32, "p_one": 0.5},
        {"scenario": "null_medium", "n_samples": 128, "n_features": 64, "p_one": 0.5},
        {"scenario": "null_large", "n_samples": 192, "n_features": 96, "p_one": 0.5},
    ]


def _default_treebh_scenarios() -> list[dict[str, Any]]:
    return [
        {"scenario": "treebh_sparse_strong", "alt_fraction": 0.20, "alt_beta_a": 0.20},
        {"scenario": "treebh_balanced_strong", "alt_fraction": 0.50, "alt_beta_a": 0.20},
        {"scenario": "treebh_sparse_moderate", "alt_fraction": 0.20, "alt_beta_a": 0.50},
        {"scenario": "treebh_balanced_moderate", "alt_fraction": 0.50, "alt_beta_a": 0.50},
        {"scenario": "treebh_all_null", "alt_fraction": 0.00, "alt_beta_a": 0.50},
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


def _run_one_null_replicate(
    rng: np.random.Generator,
    scenario: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    data_df = _make_null_data(
        rng,
        n_samples=int(scenario["n_samples"]),
        n_features=int(scenario["n_features"]),
        p_one=float(scenario["p_one"]),
    )

    # Build hierarchy and run full annotation/decomposition path.
    distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    linkage_matrix = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.decompose(
        leaf_data=data_df,
        alpha_local=float(alpha),
        sibling_alpha=float(alpha),
    )
    stats_df = tree.stats_df if tree.stats_df is not None else pd.DataFrame(index=tree.nodes())

    edge_tested_mask = (
        stats_df["Child_Parent_Divergence_P_Value_BH"].notna()
        if "Child_Parent_Divergence_P_Value_BH" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )
    sibling_tested_mask = (
        stats_df["Sibling_Divergence_P_Value_Corrected"].notna()
        if "Sibling_Divergence_P_Value_Corrected" in stats_df.columns
        else pd.Series(False, index=stats_df.index)
    )

    edge_tests = int(edge_tested_mask.sum())
    sibling_tests = int(sibling_tested_mask.sum())
    edge_rejects = (
        int(
            stats_df.loc[edge_tested_mask, "Child_Parent_Divergence_Significant"].astype(bool).sum()
        )
        if edge_tests > 0
        else 0
    )
    sibling_rejects = (
        int(stats_df.loc[sibling_tested_mask, "Sibling_BH_Different"].astype(bool).sum())
        if sibling_tests > 0
        else 0
    )

    edge_invalid = (
        int(stats_df.loc[edge_tested_mask, "Child_Parent_Divergence_Invalid"].astype(bool).sum())
        if "Child_Parent_Divergence_Invalid" in stats_df.columns and edge_tests > 0
        else 0
    )
    sibling_invalid = (
        int(stats_df.loc[sibling_tested_mask, "Sibling_Divergence_Invalid"].astype(bool).sum())
        if "Sibling_Divergence_Invalid" in stats_df.columns and sibling_tests > 0
        else 0
    )

    cp_audit = stats_df.attrs.get("child_parent_divergence_audit", {})
    sib_audit = stats_df.attrs.get("sibling_divergence_audit", {})

    return {
        "scenario": scenario["scenario"],
        "n_samples": int(scenario["n_samples"]),
        "n_features": int(scenario["n_features"]),
        "alpha": float(alpha),
        "edge_tests": edge_tests,
        "edge_rejects": edge_rejects,
        "edge_invalid": edge_invalid,
        "sibling_tests": sibling_tests,
        "sibling_rejects": sibling_rejects,
        "sibling_invalid": sibling_invalid,
        "edge_invalid_audit": int(cp_audit.get("invalid_tests", 0)),
        "sibling_invalid_audit": int(sib_audit.get("invalid_tests", 0)),
    }


def _binary_shrinkage_wald_from_samples(
    left: np.ndarray,
    right: np.ndarray,
    *,
    test_id: str,
    ridge: float = 1e-8,
) -> tuple[float, float, float, float]:
    """Diagnostic-only binary test with shrinkage covariance + JL projection.

    Projection is always activated on the whitened vector.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n_left, d = left.shape
    n_right = right.shape[0]
    if right.shape[1] != d:
        raise ValueError(f"Feature mismatch: left has d={d}, right has d={right.shape[1]}.")
    if n_left <= 1 or n_right <= 1:
        return np.nan, np.nan, np.nan, np.nan

    theta_left = np.mean(left, axis=0)
    theta_right = np.mean(right, axis=0)
    delta = theta_left - theta_right

    cov_left = np.cov(left, rowvar=False, ddof=1)
    cov_right = np.cov(right, rowvar=False, ddof=1)
    if np.ndim(cov_left) == 0:
        cov_left = np.array([[float(cov_left)]], dtype=np.float64)
    if np.ndim(cov_right) == 0:
        cov_right = np.array([[float(cov_right)]], dtype=np.float64)

    dof = float(n_left + n_right - 2)
    cov_pooled = (
        (float(n_left - 1) * np.asarray(cov_left, dtype=np.float64))
        + (float(n_right - 1) * np.asarray(cov_right, dtype=np.float64))
    ) / dof
    cov_diff = cov_pooled * (1.0 / n_left + 1.0 / n_right)
    diag_cov = np.diag(np.diag(cov_diff))
    centered = np.vstack([left - theta_left, right - theta_right])
    if d <= 1 or centered.shape[0] <= 2:
        lam = 1.0
    else:
        off_diag = cov_pooled - np.diag(np.diag(cov_pooled))
        denom = float(np.sum(off_diag**2))
        if not np.isfinite(denom) or denom <= 0:
            lam = 1.0
        else:
            n = float(centered.shape[0])
            scale = n / ((n - 1.0) ** 3)
            beta = 0.0
            for i in range(d):
                xi = centered[:, i]
                for j in range(i + 1, d):
                    xj = centered[:, j]
                    sij = float(cov_pooled[i, j])
                    beta += scale * float(np.sum((xi * xj - sij) ** 2))
            lam = float(np.clip(beta / denom, 0.0, 1.0))
    cov_shrunk = (1.0 - lam) * cov_diff + lam * diag_cov

    ridge_scale = ridge * max(float(np.mean(np.diag(cov_shrunk))), 1e-8)
    cov_shrunk = cov_shrunk + ridge_scale * np.eye(d, dtype=np.float64)

    try:
        chol = np.linalg.cholesky(cov_shrunk)
        z = np.linalg.solve(chol, delta)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(cov_shrunk)
        evals = np.clip(evals, 1e-12, None)
        inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
        z = inv_sqrt @ delta

    z = np.asarray(z, dtype=np.float64)
    n_eff = max(int(round((2.0 * n_left * n_right) / (n_left + n_right))), 1)
    k = compute_projection_dimension(n_eff, d)
    seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)
    R = generate_projection_matrix(d, k, seed, use_cache=False)
    projected = R.dot(z) if hasattr(R, "dot") else (R @ z)

    stat = float(np.sum(projected**2))
    df = float(k)
    pval = float(chi2.sf(stat, df=k))
    return stat, df, pval, float(lam)


def _run_one_binary_covariance_replicate(
    rng: np.random.Generator,
    scenario: dict[str, Any],
    alpha: float,
    *,
    rep: int,
) -> dict[str, Any]:
    """Run one null replicate comparing projection vs shrinkage Wald on binary data."""
    data_df = _make_null_data(
        rng,
        n_samples=int(scenario["n_samples"]),
        n_features=int(scenario["n_features"]),
        p_one=float(scenario["p_one"]),
    )
    X = data_df.to_numpy(dtype=np.float64)
    n_total = X.shape[0]
    perm = rng.permutation(n_total)
    n_left = n_total // 2
    n_right = n_total - n_left
    left = X[perm[:n_left], :]
    right = X[perm[n_left:], :]

    theta_left = np.mean(left, axis=0)
    theta_right = np.mean(right, axis=0)
    proj_stat, proj_df, proj_p = sibling_divergence_test(
        theta_left,
        theta_right,
        float(n_left),
        float(n_right),
        test_id=f"binary_cov_diag:{scenario['scenario']}:rep={rep}",
    )
    proj_invalid = not (np.isfinite(proj_stat) and np.isfinite(proj_df) and np.isfinite(proj_p))

    shrink_stat, shrink_df, shrink_p, shrink_lam = _binary_shrinkage_wald_from_samples(
        left,
        right,
        test_id=f"binary_shrinkage_proj_diag:{scenario['scenario']}:rep={rep}",
    )
    shrink_invalid = not (
        np.isfinite(shrink_stat) and np.isfinite(shrink_df) and np.isfinite(shrink_p)
    )

    return {
        "scenario": scenario["scenario"],
        "n_samples": int(scenario["n_samples"]),
        "n_features": int(scenario["n_features"]),
        "alpha": float(alpha),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "projection_stat": float(proj_stat) if np.isfinite(proj_stat) else np.nan,
        "projection_df": float(proj_df) if np.isfinite(proj_df) else np.nan,
        "projection_p": float(proj_p) if np.isfinite(proj_p) else np.nan,
        "projection_invalid": bool(proj_invalid),
        "projection_reject": bool((not proj_invalid) and (proj_p <= alpha)),
        "shrinkage_lambda_inferred": float(shrink_lam),
        "shrinkage_stat": float(shrink_stat) if np.isfinite(shrink_stat) else np.nan,
        "shrinkage_df": float(shrink_df) if np.isfinite(shrink_df) else np.nan,
        "shrinkage_p": float(shrink_p) if np.isfinite(shrink_p) else np.nan,
        "shrinkage_invalid": bool(shrink_invalid),
        "shrinkage_reject": bool((not shrink_invalid) and (shrink_p <= alpha)),
    }


def _summarize_binary_covariance_type1(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Type-I for projection vs shrinkage covariance binary diagnostics."""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("scenario", as_index=False).agg(
        n_samples=("n_samples", "first"),
        n_features=("n_features", "first"),
        alpha=("alpha", "first"),
        projection_tests=("projection_invalid", lambda s: int((~s.astype(bool)).sum())),
        projection_rejects=("projection_reject", "sum"),
        projection_invalid=("projection_invalid", "sum"),
        shrinkage_tests=("shrinkage_invalid", lambda s: int((~s.astype(bool)).sum())),
        shrinkage_rejects=("shrinkage_reject", "sum"),
        shrinkage_invalid=("shrinkage_invalid", "sum"),
        shrinkage_lambda_inferred_mean=("shrinkage_lambda_inferred", "mean"),
    )

    grouped["projection_type1"] = grouped["projection_rejects"] / grouped[
        "projection_tests"
    ].replace(0, np.nan)
    grouped["shrinkage_type1"] = grouped["shrinkage_rejects"] / grouped["shrinkage_tests"].replace(
        0, np.nan
    )
    grouped["projection_invalid_rate"] = grouped["projection_invalid"] / grouped[
        "projection_tests"
    ].replace(0, np.nan)
    grouped["shrinkage_invalid_rate"] = grouped["shrinkage_invalid"] / grouped[
        "shrinkage_tests"
    ].replace(0, np.nan)

    proj_ci = [
        _binomial_ci_95(int(r), int(t))
        for r, t in zip(grouped["projection_rejects"], grouped["projection_tests"])
    ]
    shrink_ci = [
        _binomial_ci_95(int(r), int(t))
        for r, t in zip(grouped["shrinkage_rejects"], grouped["shrinkage_tests"])
    ]
    grouped["projection_type1_ci_low"] = [ci[0] for ci in proj_ci]
    grouped["projection_type1_ci_high"] = [ci[1] for ci in proj_ci]
    grouped["shrinkage_type1_ci_low"] = [ci[0] for ci in shrink_ci]
    grouped["shrinkage_type1_ci_high"] = [ci[1] for ci in shrink_ci]
    return grouped


def _summarize_null_type1(null_df: pd.DataFrame) -> pd.DataFrame:
    if null_df.empty:
        return pd.DataFrame()

    grouped = null_df.groupby("scenario", as_index=False).agg(
        n_samples=("n_samples", "first"),
        n_features=("n_features", "first"),
        alpha=("alpha", "first"),
        edge_tests=("edge_tests", "sum"),
        edge_rejects=("edge_rejects", "sum"),
        edge_invalid=("edge_invalid", "sum"),
        sibling_tests=("sibling_tests", "sum"),
        sibling_rejects=("sibling_rejects", "sum"),
        sibling_invalid=("sibling_invalid", "sum"),
    )

    grouped["edge_type1"] = grouped["edge_rejects"] / grouped["edge_tests"].replace(0, np.nan)
    grouped["sibling_type1"] = grouped["sibling_rejects"] / grouped["sibling_tests"].replace(
        0, np.nan
    )
    grouped["edge_invalid_rate"] = grouped["edge_invalid"] / grouped["edge_tests"].replace(
        0, np.nan
    )
    grouped["sibling_invalid_rate"] = grouped["sibling_invalid"] / grouped["sibling_tests"].replace(
        0, np.nan
    )

    edge_ci = [
        _binomial_ci_95(int(r), int(t))
        for r, t in zip(grouped["edge_rejects"], grouped["edge_tests"])
    ]
    sib_ci = [
        _binomial_ci_95(int(r), int(t))
        for r, t in zip(grouped["sibling_rejects"], grouped["sibling_tests"])
    ]
    grouped["edge_type1_ci_low"] = [ci[0] for ci in edge_ci]
    grouped["edge_type1_ci_high"] = [ci[1] for ci in edge_ci]
    grouped["sibling_type1_ci_low"] = [ci[0] for ci in sib_ci]
    grouped["sibling_type1_ci_high"] = [ci[1] for ci in sib_ci]
    return grouped


def _make_reference_tree() -> tuple[nx.DiGraph, list[str]]:
    tree = nx.DiGraph()
    edges = [
        ("root", "A"),
        ("root", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "E"),
        ("B", "F"),
    ]
    tree.add_edges_from(edges)
    child_ids = [child for _, child in edges]
    return tree, child_ids


def _sample_treebh_truth_and_pvalues(
    rng: np.random.Generator,
    n_edges: int,
    alt_fraction: float,
    alt_beta_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    true_alt = rng.random(n_edges) < alt_fraction
    if alt_fraction > 0 and not np.any(true_alt):
        true_alt[int(rng.integers(0, n_edges))] = True

    p_values = np.empty(n_edges, dtype=float)
    null_mask = ~true_alt
    p_values[null_mask] = rng.random(int(np.sum(null_mask)))
    if np.any(true_alt):
        p_values[true_alt] = rng.beta(float(alt_beta_a), 1.0, int(np.sum(true_alt)))
    return p_values, true_alt


def _run_treebh_replicates(
    rng: np.random.Generator,
    scenarios: list[dict[str, Any]],
    alpha: float,
    n_reps: int,
) -> pd.DataFrame:
    tree, child_ids = _make_reference_tree()
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        scenario_name = str(scenario["scenario"])
        alt_fraction = float(scenario["alt_fraction"])
        alt_beta_a = float(scenario["alt_beta_a"])

        for rep in range(n_reps):
            p_values, true_alt = _sample_treebh_truth_and_pvalues(
                rng=rng,
                n_edges=len(child_ids),
                alt_fraction=alt_fraction,
                alt_beta_a=alt_beta_a,
            )

            result = tree_bh_correction(
                tree=tree,
                p_values=p_values,
                child_ids=child_ids,
                alpha=float(alpha),
            )
            reject = np.asarray(result.reject, dtype=bool)

            n_reject = int(np.sum(reject))
            n_null = int(np.sum(~true_alt))
            n_alt = int(np.sum(true_alt))
            false_reject = int(np.sum((~true_alt) & reject))
            true_reject = int(np.sum(true_alt & reject))
            fdr_value = false_reject / max(n_reject, 1)
            type1_null = false_reject / max(n_null, 1)
            power = true_reject / max(n_alt, 1) if n_alt > 0 else np.nan

            rows.append(
                {
                    "scenario": scenario_name,
                    "replicate": rep,
                    "alpha": float(alpha),
                    "n_edges": len(child_ids),
                    "n_null": n_null,
                    "n_alt": n_alt,
                    "n_reject": n_reject,
                    "false_reject": false_reject,
                    "true_reject": true_reject,
                    "fdr": fdr_value,
                    "null_type1": type1_null,
                    "power": power,
                    "alt_fraction": alt_fraction,
                    "alt_beta_a": alt_beta_a,
                }
            )

    return pd.DataFrame(rows)


def _summarize_treebh(df_treebh: pd.DataFrame) -> pd.DataFrame:
    if df_treebh.empty:
        return pd.DataFrame()

    summary = (
        df_treebh.groupby("scenario", as_index=False)[
            ["alpha", "alt_fraction", "alt_beta_a", "fdr", "null_type1", "power", "n_reject"]
        ]
        .mean(numeric_only=True)
        .rename(columns={"n_reject": "mean_rejections"})
    )
    return summary


def _build_calibration_plots_pdf(
    null_summary: pd.DataFrame,
    treebh_summary: pd.DataFrame,
    binary_cov_summary: pd.DataFrame,
    output_pdf: Path,
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        # Page 1: Null Type-I
        fig, ax = plt.subplots(figsize=(11, 8.5))
        if not null_summary.empty:
            x = np.arange(len(null_summary))
            width = 0.36
            edge_vals = null_summary["edge_type1"].to_numpy(dtype=float)
            sibling_vals = null_summary["sibling_type1"].to_numpy(dtype=float)

            edge_low = null_summary["edge_type1_ci_low"].to_numpy(dtype=float)
            edge_high = null_summary["edge_type1_ci_high"].to_numpy(dtype=float)
            sib_low = null_summary["sibling_type1_ci_low"].to_numpy(dtype=float)
            sib_high = null_summary["sibling_type1_ci_high"].to_numpy(dtype=float)
            edge_yerr = np.vstack([edge_vals - edge_low, edge_high - edge_vals])
            sib_yerr = np.vstack([sibling_vals - sib_low, sib_high - sibling_vals])

            ax.bar(
                x - width / 2,
                edge_vals,
                width=width,
                yerr=edge_yerr,
                capsize=4,
                label="Edge Type-I",
            )
            ax.bar(
                x + width / 2,
                sibling_vals,
                width=width,
                yerr=sib_yerr,
                capsize=4,
                label="Sibling Type-I",
            )
            alpha_ref = float(null_summary["alpha"].iloc[0])
            ax.axhline(
                alpha_ref,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"alpha={alpha_ref:.2f}",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"{row['scenario']}\n(n={int(row['n_samples'])}, d={int(row['n_features'])})"
                    for _, row in null_summary.iterrows()
                ],
                rotation=0,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Empirical Type-I Error")
            ax.set_title("Null Calibration: Empirical Type-I by Scenario")
            ax.legend(frameon=False, loc="upper left")
            ax.grid(True, axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No null calibration data", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: TreeBH calibration
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        if not treebh_summary.empty:
            x = np.arange(len(treebh_summary))
            labels = treebh_summary["scenario"].tolist()
            alpha_ref = float(treebh_summary["alpha"].iloc[0])

            axes[0].bar(x, treebh_summary["fdr"].to_numpy(dtype=float), color="#4C72B0")
            axes[0].axhline(
                alpha_ref,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"alpha={alpha_ref:.2f}",
            )
            axes[0].set_title("TreeBH: Empirical FDR")
            axes[0].set_ylabel("FDR")
            axes[0].set_ylim(0.0, 1.0)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=40, ha="right")
            axes[0].legend(frameon=False, loc="upper left")
            axes[0].grid(True, axis="y", alpha=0.25)

            axes[1].bar(
                x, treebh_summary["power"].fillna(0.0).to_numpy(dtype=float), color="#55A868"
            )
            axes[1].set_title("TreeBH: Empirical Power")
            axes[1].set_ylabel("Power")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=40, ha="right")
            axes[1].grid(True, axis="y", alpha=0.25)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No TreeBH calibration data", ha="center", va="center")
                ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Binary projection vs shrinkage covariance Type-I
        fig, ax = plt.subplots(figsize=(11, 8.5))
        if not binary_cov_summary.empty:
            x = np.arange(len(binary_cov_summary))
            width = 0.36
            proj_vals = binary_cov_summary["projection_type1"].to_numpy(dtype=float)
            shrink_vals = binary_cov_summary["shrinkage_type1"].to_numpy(dtype=float)

            proj_low = binary_cov_summary["projection_type1_ci_low"].to_numpy(dtype=float)
            proj_high = binary_cov_summary["projection_type1_ci_high"].to_numpy(dtype=float)
            shrink_low = binary_cov_summary["shrinkage_type1_ci_low"].to_numpy(dtype=float)
            shrink_high = binary_cov_summary["shrinkage_type1_ci_high"].to_numpy(dtype=float)
            proj_yerr = np.vstack([proj_vals - proj_low, proj_high - proj_vals])
            shrink_yerr = np.vstack([shrink_vals - shrink_low, shrink_high - shrink_vals])

            ax.bar(
                x - width / 2,
                proj_vals,
                width=width,
                yerr=proj_yerr,
                capsize=4,
                label="Current Projection",
            )
            ax.bar(
                x + width / 2,
                shrink_vals,
                width=width,
                yerr=shrink_yerr,
                capsize=4,
                label="Shrinkage + Projection",
            )
            alpha_ref = float(binary_cov_summary["alpha"].iloc[0])
            ax.axhline(
                alpha_ref,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"alpha={alpha_ref:.2f}",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"{row['scenario']}\n(n={int(row['n_samples'])}, d={int(row['n_features'])})"
                    for _, row in binary_cov_summary.iterrows()
                ],
                rotation=0,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Empirical Type-I Error")
            lam = float(binary_cov_summary["shrinkage_lambda_inferred_mean"].iloc[0])
            ax.set_title(
                f"Binary Null Calibration: Projection vs Shrinkage+Projection (inferred lambda~{lam:.2f})"
            )
            ax.legend(frameon=False, loc="upper left")
            ax.grid(True, axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No binary covariance diagnostic data", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return output_pdf


def _write_markdown_report(
    output_md: Path,
    alpha: float,
    null_summary: pd.DataFrame,
    treebh_summary: pd.DataFrame,
    binary_cov_summary: pd.DataFrame,
    n_reps_null: int,
    n_reps_treebh: int,
    crossfit_perm_summary: pd.DataFrame | None = None,
    crossfit_probe_df: pd.DataFrame | None = None,
    n_reps_crossfit: int | None = None,
    n_perms_crossfit: int | None = None,
    n_perms_probe: int | None = None,
) -> Path:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("# Calibration Report")
    lines.append("")
    lines.append(f"- Generated: `{now_utc}`")
    lines.append(f"- alpha: `{alpha:.4f}`")
    lines.append(f"- null replicates per scenario: `{n_reps_null}`")
    lines.append(f"- TreeBH replicates per scenario: `{n_reps_treebh}`")
    if n_reps_crossfit is not None and n_perms_crossfit is not None:
        lines.append(
            f"- cross-fit permutation reps/perms: `{n_reps_crossfit}` / `{n_perms_crossfit}`"
        )
    if n_perms_probe is not None:
        lines.append(f"- cross-fit benchmark probe perms: `{n_perms_probe}`")
    lines.append("")

    lines.append("## Null / Type-I Summary")
    lines.append("")
    if null_summary.empty:
        lines.append("No null summary data.")
    else:
        lines.append("```")
        lines.append(
            null_summary[
                [
                    "scenario",
                    "n_samples",
                    "n_features",
                    "edge_tests",
                    "edge_type1",
                    "edge_invalid_rate",
                    "sibling_tests",
                    "sibling_type1",
                    "sibling_invalid_rate",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    lines.append("## TreeBH Summary")
    lines.append("")
    if treebh_summary.empty:
        lines.append("No TreeBH summary data.")
    else:
        lines.append("```")
        lines.append(
            treebh_summary[
                [
                    "scenario",
                    "alt_fraction",
                    "alt_beta_a",
                    "fdr",
                    "null_type1",
                    "power",
                    "mean_rejections",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    lines.append("## Binary Covariance Diagnostic (Null Type-I, Projection Always Active)")
    lines.append("")
    if binary_cov_summary.empty:
        lines.append("No binary covariance diagnostic data.")
    else:
        lines.append("```")
        lines.append(
            binary_cov_summary[
                [
                    "scenario",
                    "n_samples",
                    "n_features",
                    "projection_tests",
                    "projection_type1",
                    "projection_invalid_rate",
                    "shrinkage_tests",
                    "shrinkage_type1",
                    "shrinkage_invalid_rate",
                    "shrinkage_lambda_inferred_mean",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    lines.append("## Cross-Fit + Permutation Diagnostic (Null)")
    lines.append("")
    if crossfit_perm_summary is None or crossfit_perm_summary.empty:
        lines.append("Not run.")
    else:
        lines.append("```")
        lines.append(
            crossfit_perm_summary[
                [
                    "scenario",
                    "edge_hypotheses",
                    "edge_rejects",
                    "edge_type1",
                    "sibling_hypotheses",
                    "sibling_rejects",
                    "sibling_type1",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    lines.append("## Cross-Fit + Permutation Benchmark Probe")
    lines.append("")
    if crossfit_probe_df is None or crossfit_probe_df.empty:
        lines.append("Not run.")
    else:
        lines.append("```")
        _probe_cols = [
            "case_name",
            "generator",
            "edge_reject_rate_in_sample",
            "edge_reject_rate_crossfit_perm",
            "sibling_reject_rate_in_sample",
            "sibling_reject_rate_crossfit_perm",
            "edge_tests_in_sample",
            "edge_hypotheses_crossfit_perm",
            "sibling_tests_in_sample",
            "sibling_hypotheses_crossfit_perm",
        ]
        keep_cols = [c for c in _probe_cols if c in crossfit_probe_df.columns]
        lines.append(crossfit_probe_df[keep_cols].to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Pass/Fail Heuristic")
    lines.append("")
    lines.append("- Target: empirical Type-I and FDR should stay close to alpha.")
    lines.append("- Rule used: pass if metric <= alpha + 0.02.")
    lines.append("")

    if not null_summary.empty:
        null_pass = bool(
            (null_summary["edge_type1"].fillna(0.0) <= alpha + 0.02).all()
            and (null_summary["sibling_type1"].fillna(0.0) <= alpha + 0.02).all()
        )
        lines.append(f"- Null Type-I status: {'PASS' if null_pass else 'FAIL'}")
    if not treebh_summary.empty:
        fdr_pass = bool((treebh_summary["fdr"].fillna(0.0) <= alpha + 0.02).all())
        lines.append(f"- TreeBH FDR status: {'PASS' if fdr_pass else 'FAIL'}")
    if not binary_cov_summary.empty:
        proj_pass = bool((binary_cov_summary["projection_type1"].fillna(0.0) <= alpha + 0.02).all())
        shrink_pass = bool(
            (binary_cov_summary["shrinkage_type1"].fillna(0.0) <= alpha + 0.02).all()
        )
        lines.append(f"- Binary projection status: {'PASS' if proj_pass else 'FAIL'}")
        lines.append(f"- Binary shrinkage+projection status: {'PASS' if shrink_pass else 'FAIL'}")
    if crossfit_perm_summary is not None and not crossfit_perm_summary.empty:
        crossfit_pass = bool(
            (crossfit_perm_summary["edge_type1"].fillna(0.0) <= alpha + 0.02).all()
            and (crossfit_perm_summary["sibling_type1"].fillna(0.0) <= alpha + 0.02).all()
        )
        lines.append(f"- Cross-fit permutation null status: {'PASS' if crossfit_pass else 'FAIL'}")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def run_calibration_suite(
    run_dir: Path,
    *,
    alpha: float | None = None,
    n_reps_null: int = 30,
    n_reps_treebh: int = 200,
    seed: int = 42,
    run_crossfit_perm_diag: bool = False,
    crossfit_diag_reps: int = 8,
    crossfit_diag_perms: int = 50,
    run_crossfit_benchmark_probe: bool = False,
    crossfit_probe_perms: int = 40,
    crossfit_probe_case_names: list[str] | None = None,
) -> dict[str, str]:
    """Run null/Type-I and TreeBH calibration, saving CSV/MD/PDF artifacts."""
    run_dir = Path(run_dir)
    calibration_dir = run_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    alpha_eff = float(config.SIBLING_ALPHA if alpha is None else alpha)
    rng = np.random.default_rng(seed)

    # Null calibration
    null_rows: list[dict[str, Any]] = []
    for scenario in _default_null_scenarios():
        for rep in range(int(n_reps_null)):
            row = _run_one_null_replicate(rng, scenario, alpha_eff)
            row["replicate"] = rep
            null_rows.append(row)
    null_df = pd.DataFrame(null_rows)
    null_summary = _summarize_null_type1(null_df)

    # Binary diagnostic: current projection vs shrinkage covariance Wald
    binary_diag_rows: list[dict[str, Any]] = []
    for scenario in _default_null_scenarios():
        for rep in range(int(n_reps_null)):
            row = _run_one_binary_covariance_replicate(
                rng,
                scenario,
                alpha_eff,
                rep=rep,
            )
            row["replicate"] = rep
            binary_diag_rows.append(row)
    binary_diag_df = pd.DataFrame(binary_diag_rows)
    binary_diag_summary = _summarize_binary_covariance_type1(binary_diag_df)

    # TreeBH calibration
    treebh_df = _run_treebh_replicates(
        rng=rng,
        scenarios=_default_treebh_scenarios(),
        alpha=alpha_eff,
        n_reps=int(n_reps_treebh),
    )
    treebh_summary = _summarize_treebh(treebh_df)

    # Optional: cross-fit + permutation diagnostics
    crossfit_perm_outputs: dict[str, str] = {}
    crossfit_probe_outputs: dict[str, str] = {}
    crossfit_perm_summary: pd.DataFrame | None = None
    crossfit_probe_df: pd.DataFrame | None = None

    if run_crossfit_perm_diag:
        crossfit_perm_dir = calibration_dir / "crossfit_permutation_diagnostic"
        crossfit_perm_outputs = run_crossfit_permutation_diagnostic(
            output_dir=crossfit_perm_dir,
            alpha=alpha_eff,
            n_reps=int(crossfit_diag_reps),
            n_perms=int(crossfit_diag_perms),
            seed=int(seed + 12345),
        )
        summary_csv = Path(crossfit_perm_outputs["summary_csv"])
        if summary_csv.exists():
            crossfit_perm_summary = pd.read_csv(summary_csv)

    if run_crossfit_benchmark_probe:
        crossfit_probe_dir = calibration_dir / "crossfit_benchmark_probe"
        probe_cases = (
            list(crossfit_probe_case_names)
            if crossfit_probe_case_names
            else _default_crossfit_probe_cases()
        )
        crossfit_probe_outputs = run_crossfit_benchmark_probe_fn(
            case_names=probe_cases,
            alpha=alpha_eff,
            n_perms=int(crossfit_probe_perms),
            seed=int(seed + 54321),
            output_dir=crossfit_probe_dir,
        )
        probe_csv = Path(crossfit_probe_outputs["csv"])
        if probe_csv.exists():
            crossfit_probe_df = pd.read_csv(probe_csv)

    # Persist artifacts
    null_csv = calibration_dir / "null_replicate_level.csv"
    null_summary_csv = calibration_dir / "null_type1_summary.csv"
    binary_diag_csv = calibration_dir / "binary_covariance_replicate_level.csv"
    binary_diag_summary_csv = calibration_dir / "binary_covariance_type1_summary.csv"
    treebh_csv = calibration_dir / "treebh_replicate_level.csv"
    treebh_summary_csv = calibration_dir / "treebh_summary.csv"
    plots_pdf = calibration_dir / "calibration_plots.pdf"
    report_md = calibration_dir / "calibration_report.md"

    null_df.to_csv(null_csv, index=False)
    null_summary.to_csv(null_summary_csv, index=False)
    binary_diag_df.to_csv(binary_diag_csv, index=False)
    binary_diag_summary.to_csv(binary_diag_summary_csv, index=False)
    treebh_df.to_csv(treebh_csv, index=False)
    treebh_summary.to_csv(treebh_summary_csv, index=False)
    _build_calibration_plots_pdf(
        null_summary,
        treebh_summary,
        binary_diag_summary,
        plots_pdf,
    )
    _write_markdown_report(
        report_md,
        alpha=alpha_eff,
        null_summary=null_summary,
        treebh_summary=treebh_summary,
        binary_cov_summary=binary_diag_summary,
        n_reps_null=int(n_reps_null),
        n_reps_treebh=int(n_reps_treebh),
        crossfit_perm_summary=crossfit_perm_summary,
        crossfit_probe_df=crossfit_probe_df,
        n_reps_crossfit=int(crossfit_diag_reps) if run_crossfit_perm_diag else None,
        n_perms_crossfit=int(crossfit_diag_perms) if run_crossfit_perm_diag else None,
        n_perms_probe=int(crossfit_probe_perms) if run_crossfit_benchmark_probe else None,
    )

    outputs = {
        "calibration_dir": str(calibration_dir),
        "null_csv": str(null_csv),
        "null_summary_csv": str(null_summary_csv),
        "binary_diag_csv": str(binary_diag_csv),
        "binary_diag_summary_csv": str(binary_diag_summary_csv),
        "treebh_csv": str(treebh_csv),
        "treebh_summary_csv": str(treebh_summary_csv),
        "plots_pdf": str(plots_pdf),
        "report_md": str(report_md),
    }
    outputs.update(crossfit_perm_outputs)
    outputs.update(crossfit_probe_outputs)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibration suite with optional cross-fit diagnostics."
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--n-reps-null", type=int, default=30)
    parser.add_argument("--n-reps-treebh", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run-crossfit-perm-diag", action="store_true")
    parser.add_argument("--crossfit-diag-reps", type=int, default=8)
    parser.add_argument("--crossfit-diag-perms", type=int, default=50)

    parser.add_argument("--run-crossfit-benchmark-probe", action="store_true")
    parser.add_argument("--crossfit-probe-perms", type=int, default=40)
    parser.add_argument(
        "--crossfit-probe-cases",
        type=str,
        default="",
        help="Comma-separated benchmark case names for probe. Uses default set if empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    timestamp = format_timestamp_utc()
    run_dir = project_root / "benchmarks" / "results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_cases = (
        [c.strip() for c in args.crossfit_probe_cases.split(",") if c.strip()]
        if args.crossfit_probe_cases
        else None
    )
    outputs = run_calibration_suite(
        run_dir,
        alpha=args.alpha,
        n_reps_null=int(args.n_reps_null),
        n_reps_treebh=int(args.n_reps_treebh),
        seed=int(args.seed),
        run_crossfit_perm_diag=bool(args.run_crossfit_perm_diag),
        crossfit_diag_reps=int(args.crossfit_diag_reps),
        crossfit_diag_perms=int(args.crossfit_diag_perms),
        run_crossfit_benchmark_probe=bool(args.run_crossfit_benchmark_probe),
        crossfit_probe_perms=int(args.crossfit_probe_perms),
        crossfit_probe_case_names=probe_cases,
    )
    print("Calibration complete:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
