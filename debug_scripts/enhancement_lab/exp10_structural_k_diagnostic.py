#!/usr/bin/env python
"""Experiment 10 — Structure-Aware Sibling Projection Diagnostic.

For every sibling pair, collect:
  - Spectral features: k_left, k_right, k_parent
  - Structural features: n_leaves_left, n_leaves_right, depth, n_edges
  - Ground truth: is this a true split or a false split?

Then evaluate candidate k formulas that combine spectral and structural
information, computing what the Wald p-value would be under each formula
and whether it gives the correct decision.

Candidate formulas:
  1. min(k_L, k_R)                    — conservative, from exp9
  2. max(k_L, k_R)                    — aggressive, from exp9
  3. k_parent                         — parent spectral k (no whitening, random proj)
  4. weighted: (n_L·k_L + n_R·k_R) / (n_L + n_R)  — leaf-weighted average
  5. smaller child's k (by leaf count) — always use dimensionality of smaller subtree
  6. larger child's k (by leaf count)  — always use dimensionality of larger subtree
  7. min(k_L, k_R) + log2(n_leaves_ratio)  — min adjusted by asymmetry
  8. effective_rank of parent data     — data-adaptive, no spectral gating
  9. sqrt(n_min)                       — sample-based scaling
  10. baseline (JL capped at 5)

For each pair × formula, compute:
  - The projected Wald T statistic under that k
  - The p-value from χ²(k)
  - Whether the decision (split/merge at α=0.01) matches ground truth
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from lab_helpers import FAILURE_CASES, REGRESSION_GUARD_CASES, build_tree_and_data

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.pooled_variance import (
    standardize_proportion_difference,
)


def _count_leaves(tree: nx.DiGraph, node: str) -> int:
    """Count leaf descendants of a node."""
    if tree.out_degree(node) == 0:
        return 1
    return sum(1 for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


def _count_edges(tree: nx.DiGraph, node: str) -> int:
    """Count edges in the subtree rooted at node."""
    if tree.out_degree(node) == 0:
        return 0
    desc = nx.descendants(tree, node)
    return sum(1 for u, v in tree.edges() if u == node or u in desc)


def _node_depth(tree: nx.DiGraph, node: str, root: str) -> int:
    """Depth of node from root."""
    try:
        return nx.shortest_path_length(tree, root, node)
    except nx.NetworkXNoPath:
        return -1


def _get_true_split_labels(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
    y_true: np.ndarray | None,
    leaf_data: pd.DataFrame,
) -> str | None:
    """Determine if this split is a 'true split' or 'false split' based on ground truth.

    Returns:
        "true_split"  — children contain leaves from different true clusters
        "false_split" — children contain leaves from the same true cluster (should be merged)
        None          — no ground truth available
    """
    if y_true is None:
        return None

    label_to_idx = {label: i for i, label in enumerate(leaf_data.index)}

    def _get_leaf_labels(node):
        if tree.out_degree(node) == 0:
            lbl = tree.nodes[node].get("label", node)
            idx = label_to_idx.get(lbl)
            return {y_true[idx]} if idx is not None else set()
        labels = set()
        for d in nx.descendants(tree, node):
            if tree.out_degree(d) == 0:
                lbl = tree.nodes[d].get("label", d)
                idx = label_to_idx.get(lbl)
                if idx is not None:
                    labels.add(y_true[idx])
        return labels

    left_labels = _get_leaf_labels(left)
    right_labels = _get_leaf_labels(right)

    if not left_labels or not right_labels:
        return None

    # If they share exactly the same label set → false split
    # If they have any difference → true split
    if left_labels == right_labels:
        return "false_split"
    elif left_labels & right_labels:
        return "mixed"  # some overlap but not identical
    else:
        return "true_split"


# ── z-vector computation ────────────────────────────────────────────────────


def _compute_z_and_norm(
    tree: nx.DiGraph,
    left: str,
    right: str,
) -> Tuple[np.ndarray | None, float]:
    """Compute standardized proportion difference z and ||z||²."""
    try:
        left_dist = extract_node_distribution(tree, left)
        right_dist = extract_node_distribution(tree, right)
        n_left = extract_node_sample_size(tree, left)
        n_right = extract_node_sample_size(tree, right)
    except Exception:
        return None, 0.0

    if n_left < 2 or n_right < 2:
        return None, 0.0

    try:
        z, _var = standardize_proportion_difference(
            left_dist,
            right_dist,
            float(n_left),
            float(n_right),
        )
    except Exception:
        return None, 0.0

    if not np.isfinite(z).all():
        return None, 0.0

    return z, float(np.sum(z**2))


# ── Candidate k formulas ────────────────────────────────────────────────────


def _compute_candidate_ks(
    k_left: int,
    k_right: int,
    k_parent: int,
    n_leaves_left: int,
    n_leaves_right: int,
    n_left: int,
    n_right: int,
    n_features: int,
    jl_k_capped: int,
) -> Dict[str, int]:
    """Compute all candidate projection dimensions."""
    n_min_leaves = min(n_leaves_left, n_leaves_right)
    n_max_leaves = max(n_leaves_left, n_leaves_right)
    leaf_ratio = n_max_leaves / max(n_min_leaves, 1)

    # k of the child with fewer/more leaves
    if n_leaves_left <= n_leaves_right:
        k_smaller_child = k_left
        k_larger_child = k_right
    else:
        k_smaller_child = k_right
        k_larger_child = k_left

    # Weighted average by leaf count
    total_leaves = n_leaves_left + n_leaves_right
    k_weighted = round((n_leaves_left * k_left + n_leaves_right * k_right) / max(total_leaves, 1))

    # Min + asymmetry correction
    k_min_adj = max(1, min(k_left, k_right) + round(math.log2(max(leaf_ratio, 1))))

    # sqrt of minimum sample size
    n_min_samples = min(n_left, n_right)
    k_sqrt_n = max(1, round(math.sqrt(n_min_samples)))

    candidates = {
        "baseline_jl5": jl_k_capped,
        "min_child": max(1, min(k_left, k_right)),
        "max_child": max(1, max(k_left, k_right)),
        "k_parent": max(1, k_parent),
        "k_weighted": max(1, k_weighted),
        "k_smaller_sub": max(1, k_smaller_child),
        "k_larger_sub": max(1, k_larger_child),
        "k_min_adj_asym": max(1, min(k_min_adj, n_features)),
        "k_sqrt_nmin": max(1, min(k_sqrt_n, n_features)),
    }

    # Clamp all to n_features
    return {k: min(v, n_features) for k, v in candidates.items()}


# ── p-value under each k ────────────────────────────────────────────────────


def _pvalue_at_k(z_norm_sq: float, k: int) -> float:
    """Compute p-value of z_norm_sq under χ²(k).

    This is an approximation: the actual test uses a random projection of z
    into k dimensions.  For large d and random R, E[||Rz||²] = (k/d)||z||²
    and ||Rz||² ~ (k/d)||z||² · χ²(k)/k approximately.

    We use T_approx = (k / d) * ||z||² is wrong — the proper statistic is
    T = ||R·z||² where R is k×d random orthonormal.  Under H₀ (z ~ N(0,I_d)),
    T ~ χ²(k) exactly.  Under H₁, T has non-central χ² with
    ncp = (k/d)||z||².   So p = P(χ²(k) ≥ (k/d)||z||²) is the projection p-value.
    """
    d_implicit = z_norm_sq  # ||z||² is the sum over all d features
    # Expected projection statistic: (k / d_features) * ||z||² but we don't know d_features here
    # Instead: use chi2.sf directly on the projected statistic
    # Under random k-dim projection: E[T] = k * (||z||² / d) where d = len(z)
    # But we only have ||z||², not d.  We'll pass d separately.
    # For now, just compute p = P(chi2(k) >= z_norm_sq * k / d)
    # This needs d — we'll handle it in the caller.
    return chi2.sf(z_norm_sq, k)


def _projected_pvalue(z: np.ndarray, k: int, seed: int = 42) -> Tuple[float, float]:
    """Compute actual projected Wald statistic and p-value.

    Draws a random k×d orthonormal matrix R, computes T = ||Rz||², p = P(χ²(k) ≥ T).
    """
    d = len(z)
    k = min(k, d)
    if k <= 0:
        return 0.0, 1.0

    rng = np.random.RandomState(seed)
    # Random orthonormal projection
    G = rng.randn(k, d)
    Q, _ = np.linalg.qr(G.T)  # d × k
    R = Q[:, :k].T  # k × d

    projected = R @ z
    T = float(np.sum(projected**2))
    p = float(chi2.sf(T, k))
    return T, p


# ── Main diagnostic ─────────────────────────────────────────────────────────


def diagnose_case(case_name: str) -> pd.DataFrame:
    """Collect per-pair structural + spectral features with candidate k evaluations."""

    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")
    n_features = data_df.shape[1]

    root = next(n for n, d in tree.in_degree() if d == 0)

    # Spectral decomposition
    spectral_dims, pca_projections, pca_eigenvalues = compute_spectral_decomposition(
        tree,
        data_df,
        method="marchenko_pastur",
        minimum_projection_dimension=config.SPECTRAL_MINIMUM_DIMENSION,
        compute_projections=True,
    )

    rows: List[dict] = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        # z-vector
        z, z_norm_sq = _compute_z_and_norm(tree, left, right)
        if z is None:
            continue

        n_left = extract_node_sample_size(tree, left)
        n_right = extract_node_sample_size(tree, right)

        # Structural features
        n_leaves_left = _count_leaves(tree, left)
        n_leaves_right = _count_leaves(tree, right)
        n_edges_left = _count_edges(tree, left)
        n_edges_right = _count_edges(tree, right)
        depth = _node_depth(tree, parent, root)
        leaf_ratio = max(n_leaves_left, n_leaves_right) / max(min(n_leaves_left, n_leaves_right), 1)

        # Spectral features
        k_left = spectral_dims.get(left, 2)
        k_right = spectral_dims.get(right, 2)
        k_parent = spectral_dims.get(parent, 2)

        # JL baseline
        jl_k_capped = compute_projection_dimension_backend(n_left + n_right, n_features)

        # Ground truth
        gt_label = _get_true_split_labels(tree, parent, left, right, y_t, data_df)

        # Candidate k values
        candidates = _compute_candidate_ks(
            k_left,
            k_right,
            k_parent,
            n_leaves_left,
            n_leaves_right,
            n_left,
            n_right,
            n_features,
            jl_k_capped,
        )

        # Compute projected p-value for each candidate
        # Use a deterministic seed per parent for reproducibility
        seed = hash(parent) % (2**31)

        row = {
            "case": case_name,
            "true_k": true_k,
            "parent": parent,
            "depth": depth,
            "n_left": n_left,
            "n_right": n_right,
            "n_leaves_left": n_leaves_left,
            "n_leaves_right": n_leaves_right,
            "n_edges_left": n_edges_left,
            "n_edges_right": n_edges_right,
            "leaf_ratio": round(leaf_ratio, 2),
            "k_left": k_left,
            "k_right": k_right,
            "k_parent": k_parent,
            "z_norm_sq": round(z_norm_sq, 2),
            "gt_label": gt_label,
        }

        # For each candidate formula, compute T and p
        for cname, ck in candidates.items():
            T, p = _projected_pvalue(z, ck, seed=seed)
            row[f"k_{cname}"] = ck
            row[f"T_{cname}"] = round(T, 2)
            row[f"p_{cname}"] = p
            # Decision at sibling alpha
            row[f"split_{cname}"] = p < config.SIBLING_ALPHA

        rows.append(row)

    return pd.DataFrame(rows)


# ── Reporting ────────────────────────────────────────────────────────────────


def summarize(all_results: pd.DataFrame) -> None:
    """Analyze decision accuracy of each candidate k formula."""

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    formulas = [
        "baseline_jl5",
        "min_child",
        "max_child",
        "k_parent",
        "k_weighted",
        "k_smaller_sub",
        "k_larger_sub",
        "k_min_adj_asym",
        "k_sqrt_nmin",
    ]

    # ── 1. Structural overview ──
    print("\n" + "=" * 100)
    print("  Structural overview per case")
    print("=" * 100)
    struct_agg = (
        all_results.groupby("case")
        .agg(
            true_k=("true_k", "first"),
            n_pairs=("parent", "count"),
            mean_leaves_L=("n_leaves_left", "mean"),
            mean_leaves_R=("n_leaves_right", "mean"),
            mean_leaf_ratio=("leaf_ratio", "mean"),
            mean_depth=("depth", "mean"),
            mean_k_left=("k_left", "mean"),
            mean_k_right=("k_right", "mean"),
            mean_k_parent=("k_parent", "mean"),
        )
        .sort_values("case")
    )
    print(struct_agg.round(1).to_string())

    # ── 2. k value distributions per formula ──
    print("\n" + "=" * 100)
    print("  Candidate k distributions (global)")
    print("=" * 100)
    for f in formulas:
        col = f"k_{f}"
        if col in all_results.columns:
            vals = all_results[col].dropna()
            print(
                f"  {f:20s}  mean={vals.mean():5.1f}  median={vals.median():4.0f}"
                f"  min={vals.min():3.0f}  max={vals.max():5.0f}"
            )

    # ── 3. Decision accuracy vs ground truth ──
    labeled = all_results[all_results["gt_label"].isin(["true_split", "false_split"])].copy()

    if len(labeled) == 0:
        print("\n  No ground-truth-labeled pairs found.")
        return

    print(
        f"\n  Ground truth pairs: {len(labeled)} "
        f"(true_split={sum(labeled['gt_label']=='true_split')}, "
        f"false_split={sum(labeled['gt_label']=='false_split')})"
    )

    print("\n" + "=" * 100)
    print("  Decision accuracy by formula (true_split → should split, false_split → should merge)")
    print("=" * 100)

    gt_should_split = labeled["gt_label"] == "true_split"

    accuracy_rows = []
    for f in formulas:
        split_col = f"split_{f}"
        if split_col not in labeled.columns:
            continue

        predicted_split = labeled[split_col].astype(bool)

        tp = int((predicted_split & gt_should_split).sum())
        tn = int((~predicted_split & ~gt_should_split).sum())
        fp = int((predicted_split & ~gt_should_split).sum())
        fn = int((~predicted_split & gt_should_split).sum())

        n_total = tp + tn + fp + fn
        accuracy = (tp + tn) / n_total if n_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy_rows.append(
            {
                "formula": f,
                "k_median": labeled[f"k_{f}"].median(),
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )

    acc_df = pd.DataFrame(accuracy_rows).sort_values("f1", ascending=False)
    print(acc_df.to_string(index=False))

    # ── 4. Per-case breakdown for top formulas ──
    print("\n" + "=" * 100)
    print("  Per-case FP/FN breakdown (top formulas)")
    print("=" * 100)

    top_formulas = acc_df.head(4)["formula"].tolist()
    for f in top_formulas:
        split_col = f"split_{f}"
        predicted_split = labeled[split_col].astype(bool)

        case_stats = []
        for case_name in labeled["case"].unique():
            mask = labeled["case"] == case_name
            case_gt = gt_should_split[mask]
            case_pred = predicted_split[mask]
            tp = int((case_pred & case_gt).sum())
            fp = int((case_pred & ~case_gt).sum())
            fn = int((~case_pred & case_gt).sum())
            tn = int((~case_pred & ~case_gt).sum())
            case_stats.append(
                {
                    "case": case_name,
                    "n": int(mask.sum()),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                }
            )
        cs_df = pd.DataFrame(case_stats).sort_values("case")
        print(f"\n  {f}:")
        print(cs_df.to_string(index=False))

    # ── 5. Structural feature analysis: when does min vs max matter? ──
    print("\n" + "=" * 100)
    print("  Asymmetry analysis: min_child vs max_child accuracy by leaf_ratio")
    print("=" * 100)

    labeled["asym_bin"] = pd.cut(
        labeled["leaf_ratio"],
        bins=[0, 1.5, 3, 10, 1000],
        labels=["symmetric", "mild_asym", "moderate_asym", "extreme_asym"],
    )

    for f in ["min_child", "max_child", "k_smaller_sub", "k_larger_sub", "k_weighted"]:
        split_col = f"split_{f}"
        if split_col not in labeled.columns:
            continue

        pred = labeled[split_col].astype(bool)
        correct = pred == gt_should_split

        asym_acc = labeled.groupby("asym_bin", observed=False).apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "accuracy": correct[g.index].mean(),
                    "FP_rate": ((pred[g.index]) & (~gt_should_split[g.index])).sum()
                    / max((~gt_should_split[g.index]).sum(), 1),
                    "FN_rate": ((~pred[g.index]) & (gt_should_split[g.index])).sum()
                    / max((gt_should_split[g.index]).sum(), 1),
                }
            )
        )
        print(f"\n  {f}:")
        print(asym_acc.round(3).to_string())


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    cases = FAILURE_CASES + REGRESSION_GUARD_CASES
    all_frames: List[pd.DataFrame] = []

    for i, name in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {name}", end=" ", flush=True)
        try:
            df = diagnose_case(name)
            all_frames.append(df)
            print(f"→ {len(df)} pairs")
        except Exception as e:
            print(f"→ ERROR: {e}")

    if not all_frames:
        print("No results collected.")
        return

    all_results = pd.concat(all_frames, ignore_index=True)
    summarize(all_results)


if __name__ == "__main__":
    print("=" * 100)
    print("  EXPERIMENT 10: Structure-Aware Sibling Projection Diagnostic")
    print("=" * 100)
    main()
