#!/usr/bin/env python
"""Exp 35 — Comprehensive Eigenvalue Approximation Comparison.

Read-only offline analysis.  No production code is modified.

For every binary-parent sibling pair across the sentinel case set, this
script computes and compares three eigenvalue populations:

  k_parent  — MP rank of the parent's full descendant correlation matrix
               (what Gate 2 already computes via compute_spectral_decomposition)

  k_edge    — round(sqrt(k_left * k_right)): current Gate 3 production approximation
               (geometric mean of children's spectral dims; implemented in
               derive_sibling_projection_dimensions_from_child_edge_comparisons)

  k_sibling — MP rank of the *pooled left+right leaf* correlation matrix:
               the "ground truth" sibling correlation matrix not in production.
               Uses the identical eigendecompose_correlation + estimate_k_marchenko_pastur
               pipeline as Gate 2.

Metrics computed per pair
--------------------------
  Degree-of-freedom ratios       : r_parent = k_parent/k_sibling,  r_edge = k_edge/k_sibling
  Signed dimension error         : delta_parent, delta_edge  (positive = sibling has more dims)
  Spectral distance (Wasserstein-1) : W1 between full eigenvalue spectra (parent vs sibling,
                                      edge-implied vs sibling)
  MSE on paired eigenvalues      : after zero-padding to equal length
  Subspace overlap               : cos² of principal angles between leading eigenvectors
                                   (parent↔sibling, left-child↔sibling, right-child↔sibling)
  Between-group loading fraction : fraction of ||δ||² explained by leading k components,
                                   where δ = dist_L - dist_R  (between-split direction)
  Tree-depth eigenvalue accumulation : k values binned by depth from root

Report sections
---------------
  1. Overall bias summary
  2. Per-case median k table
  3. Pearson correlation: n_parent vs DoF ratio
  4. Anti-conservative / conservative outlier pairs
  5. Spectral distance (W1, MSE) summary
  6. Subspace alignment summary
  7. Between-group loading curves (k_parent vs k_sibling eigenvectors)
  8. Tree-level eigenvalue accumulation profile
  9. Conclusions
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg

# ── path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from lab_helpers import (  # noqa: E402
    FAILURE_CASES,
    INTERMEDIATE_CASES,
    REGRESSION_GUARD_CASES,
    build_tree_and_data,
)

from kl_clustering_analysis import config  # noqa: E402

# ── production imports (read-only consumers) ─────────────────────────────────
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.decomposition import (  # noqa: E402
    eigendecompose_correlation,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (  # noqa: E402
    effective_rank,
    estimate_k_marchenko_pastur,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (  # noqa: E402
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_helpers import (  # noqa: E402
    is_leaf,
)

# ── sentinel cases ────────────────────────────────────────────────────────────
# Use the same 28-case set used by the repaired lab harness; exclude phylogenetic
# and categorical cases (type-mismatch: integer category indices ≠ Bernoulli probs).
_PHYLO_CAT_PREFIX = ("phylo_", "cat_")

_SENTINEL_RAW = REGRESSION_GUARD_CASES + INTERMEDIATE_CASES[:15] + FAILURE_CASES[:8]
_SEEN: set[str] = set()
CASES: list[str] = []
for _c in _SENTINEL_RAW:
    if _c not in _SEEN and not any(_c.startswith(p) for p in _PHYLO_CAT_PREFIX):
        _SEEN.add(_c)
        CASES.append(_c)

_SEP = "=" * 115


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers — leaf index / data extraction
# ─────────────────────────────────────────────────────────────────────────────


def _build_label_to_idx(leaf_data: pd.DataFrame) -> dict[str, int]:
    return {lbl: i for i, lbl in enumerate(leaf_data.index)}


def _leaf_row_indices(tree, node: str, label_to_idx: dict[str, int]) -> list[int]:
    """Collect leaf data-matrix row indices for all leaves under *node*."""
    if is_leaf(tree, node):
        lbl = tree.nodes[node].get("label", node)
        idx = label_to_idx.get(lbl)
        return [idx] if idx is not None else []
    indices: list[int] = []
    for desc in nx.descendants(tree, node):
        if is_leaf(tree, desc):
            lbl = tree.nodes[desc].get("label", desc)
            idx = label_to_idx.get(lbl)
            if idx is not None:
                indices.append(idx)
    return indices


def _node_data_matrix(tree, node: str, X: np.ndarray, label_to_idx: dict[str, int]) -> np.ndarray:
    """Return (n_leaves, d) float64 array of leaf data under *node*."""
    idxs = _leaf_row_indices(tree, node, label_to_idx)
    return X[idxs, :].astype(np.float64) if idxs else np.empty((0, X.shape[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue utilities
# ─────────────────────────────────────────────────────────────────────────────


def _compute_k(data: np.ndarray, minimum_k: int) -> tuple[int, np.ndarray | None, int]:
    """Return (k, eigenvalues, n_active_features) for a data matrix."""
    if data.shape[0] < 2:
        return minimum_k, None, 0
    eig = eigendecompose_correlation(data, compute_eigenvectors=False)
    if eig is None:
        return minimum_k, None, 0
    k = max(
        estimate_k_marchenko_pastur(
            eig.eigenvalues,
            n_samples=data.shape[0],
            n_features=int(eig.active_feature_count),
        ),
        minimum_k,
    )
    return k, eig.eigenvalues, int(eig.active_feature_count)


def _recover_eigenvectors(data: np.ndarray, k: int) -> np.ndarray | None:
    """Return the leading k eigenvectors as shape (d, k) in feature space.

    Handles both primal (n >= d) and dual (n < d) forms.
    """
    if data.shape[0] < 2:
        return None
    eig = eigendecompose_correlation(data, compute_eigenvectors=True)
    if eig is None:
        return None

    k = min(k, int(eig.active_feature_count), data.shape[0])
    if k < 1:
        return None

    active_idx = np.where(eig.is_active_feature)[0]
    d_total = len(eig.is_active_feature)

    if eig.use_dual:
        # Recover d_active-space eigenvectors from dual sample eigenvectors
        if eig.dual_sample_eigenvectors is None or eig.standardized_data_active is None:
            return None
        top_eigs = np.maximum(eig.eigenvalues[:k], 1e-12)
        vecs_active = (
            eig.standardized_data_active.T
            @ eig.dual_sample_eigenvectors[:, :k]
            / (np.sqrt(top_eigs) * np.sqrt(eig.active_feature_count))
        )
    else:
        if eig.eigenvectors_active is None:
            return None
        vecs_active = eig.eigenvectors_active[:, :k]

    # Embed into full d-space
    V_full = np.zeros((d_total, k), dtype=np.float64)
    V_full[active_idx, :] = vecs_active
    norms = np.linalg.norm(V_full, axis=0)
    norms[norms == 0] = 1.0
    return V_full / norms


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculations
# ─────────────────────────────────────────────────────────────────────────────


def _zero_padded_sorted(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two eigenvalue spectra zero-padded to equal length, sorted descending."""
    p = max(len(a), len(b))
    ea = np.zeros(p)
    ea[: len(a)] = np.sort(a)[::-1]
    eb = np.zeros(p)
    eb[: len(b)] = np.sort(b)[::-1]
    return ea, eb


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between two eigenvalue spectra (sorted, zero-padded)."""
    ea, eb = _zero_padded_sorted(a, b)
    return float(np.mean(np.abs(ea - eb)))


def _paired_mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE between paired eigenvalues (zero-padded to equal length)."""
    ea, eb = _zero_padded_sorted(a, b)
    return float(np.mean((ea - eb) ** 2))


def _subspace_overlap(V_a: np.ndarray | None, V_b: np.ndarray | None) -> float:
    """Mean cos² of principal angles between two column-orthonormal matrices.

    Returns nan when eigenvectors are unavailable.
    """
    if V_a is None or V_b is None:
        return float("nan")
    # Thin SVD of V_a.T @ V_b gives cos(principal angles) as singular values
    try:
        # Both V columns are unit-normed; compute the cross-product matrix
        C = V_a.T @ V_b  # shape (k_a, k_b)
        sv = linalg.svd(C, compute_uv=False, check_finite=False)
        sv = np.clip(sv, 0.0, 1.0)
        return float(np.mean(sv**2))
    except Exception:
        return float("nan")


def _between_group_loading(delta: np.ndarray, V: np.ndarray | None, k: int) -> float:
    """Fraction of ||delta||² captured by the leading k columns of V.

    delta = dist_L - dist_R  (between-group direction, shape d)
    V     = (d, k) eigenvector matrix
    """
    if V is None or np.linalg.norm(delta) < 1e-12:
        return float("nan")
    delta_norm = delta / np.linalg.norm(delta)
    loadings = (V.T @ delta_norm) ** 2  # shape (k,)
    return float(np.sum(loadings))


def _tree_depth(tree, root: str) -> dict[str, int]:
    """BFS depth from root for all nodes."""
    depths: dict[str, int] = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in tree.successors(node):
            depths[child] = depths[node] + 1
            queue.append(child)
    return depths


# ─────────────────────────────────────────────────────────────────────────────
# Per-case analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyse_case(case_name: str) -> list[dict]:
    """Return one dict per binary-parent sibling pair in the tree."""
    try:
        tree, leaf_data, _y, tc = build_tree_and_data(case_name)
    except Exception as exc:
        print(f"  [SKIP] {case_name}: build failed — {exc}")
        return []

    true_k = tc.get("n_clusters", "?")
    minimum_k = max(getattr(config, "SPECTRAL_MINIMUM_DIMENSION", 2), 1)
    X = leaf_data.values.astype(np.float64)
    label_to_idx = _build_label_to_idx(leaf_data)

    # Gate 2 spectral context (read-only call)
    try:
        spectral_dims, _proj, _pca_eig = compute_spectral_decomposition(
            tree,
            leaf_data,
            minimum_projection_dimension=minimum_k,
            compute_projections=False,
        )
    except Exception as exc:
        print(f"  [SKIP] {case_name}: spectral decomp failed — {exc}")
        return []

    if spectral_dims is None:
        spectral_dims = {}

    # Root for depth calculation
    root = next((n for n in tree.nodes if tree.in_degree(n) == 0), None)
    depths = _tree_depth(tree, root) if root else {}

    rows: list[dict] = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left, right = children[0], children[1]

        # ── data matrices ────────────────────────────────────────────────
        X_parent = _node_data_matrix(tree, parent, X, label_to_idx)
        X_left = _node_data_matrix(tree, left, X, label_to_idx)
        X_right = _node_data_matrix(tree, right, X, label_to_idx)

        if X_left.shape[0] < 2 or X_right.shape[0] < 2:
            continue

        X_sibling = np.vstack([X_left, X_right])  # pooled sibling matrix

        n_left = X_left.shape[0]
        n_right = X_right.shape[0]
        n_parent = X_parent.shape[0]

        # ── k values ─────────────────────────────────────────────────────
        k_parent = int(spectral_dims.get(parent, minimum_k))
        k_left = int(spectral_dims.get(left, minimum_k))
        k_right = int(spectral_dims.get(right, minimum_k))
        # Production formula: geometric mean from the child-edge-derived projection rule
        if k_left > 0 and k_right > 0:
            k_edge = max(round(math.sqrt(k_left * k_right)), minimum_k)
        elif k_left > 0:
            k_edge = k_left
        elif k_right > 0:
            k_edge = k_right
        else:
            k_edge = minimum_k

        k_sibling, eig_sib, n_active_sib = _compute_k(X_sibling, minimum_k)
        _, eig_par, n_active_par = _compute_k(X_parent, minimum_k)

        # ── DoF ratios / deltas ───────────────────────────────────────────
        ratio_parent = k_parent / k_sibling if k_sibling > 0 else float("nan")
        ratio_edge = k_edge / k_sibling if k_sibling > 0 else float("nan")
        delta_parent = k_sibling - k_parent
        delta_edge = k_sibling - k_edge

        # ── spectral distances ────────────────────────────────────────────
        w1_parent_sib = (
            _wasserstein1(eig_par, eig_sib)
            if eig_par is not None and eig_sib is not None
            else float("nan")
        )
        mse_parent_sib = (
            _paired_mse(eig_par, eig_sib)
            if eig_par is not None and eig_sib is not None
            else float("nan")
        )

        # ── effective ranks ───────────────────────────────────────────────
        er_parent = float(effective_rank(eig_par)) if eig_par is not None else float("nan")
        er_sibling = float(effective_rank(eig_sib)) if eig_sib is not None else float("nan")

        # ── eigenvectors for subspace / loading analysis ──────────────────
        V_parent = _recover_eigenvectors(X_parent, min(k_parent, n_parent - 1, 30))
        V_left = _recover_eigenvectors(X_left, min(k_left, n_left - 1, 30))
        V_right = _recover_eigenvectors(X_right, min(k_right, n_right - 1, 30))
        V_sibling = _recover_eigenvectors(X_sibling, min(k_sibling, n_left + n_right - 1, 30))

        # Trim to k_sibling for fair comparison
        if V_parent is not None:
            V_parent = V_parent[:, :k_sibling]
        if V_sibling is not None:
            V_sibling = V_sibling[:, :k_sibling]
        if V_left is not None:
            V_left = V_left[:, :k_sibling]
        if V_right is not None:
            V_right = V_right[:, :k_sibling]

        overlap_parent_sib = _subspace_overlap(V_parent, V_sibling)
        overlap_left_sib = _subspace_overlap(V_left, V_sibling)
        overlap_right_sib = _subspace_overlap(V_right, V_sibling)

        # ── between-group loading ─────────────────────────────────────────
        dist_L = tree.nodes[left].get("distribution")
        dist_R = tree.nodes[right].get("distribution")
        if dist_L is not None and dist_R is not None:
            delta = np.asarray(dist_L, dtype=np.float64) - np.asarray(dist_R, dtype=np.float64)
            loading_parent = _between_group_loading(delta, V_parent, k_parent)
            loading_sibling = _between_group_loading(delta, V_sibling, k_sibling)
        else:
            loading_parent = float("nan")
            loading_sibling = float("nan")

        rows.append(
            {
                "case": case_name,
                "true_k": true_k,
                "parent": parent,
                "left": left,
                "right": right,
                "depth": depths.get(parent, -1),
                "n_left": n_left,
                "n_right": n_right,
                "n_parent": n_parent,
                "n_active_par": n_active_par,
                "n_active_sib": n_active_sib,
                # k values
                "k_parent": k_parent,
                "k_left": k_left,
                "k_right": k_right,
                "k_edge": k_edge,
                "k_sibling": k_sibling,
                # DoF metrics
                "ratio_parent": round(ratio_parent, 4),
                "ratio_edge": round(ratio_edge, 4),
                "delta_parent": delta_parent,
                "delta_edge": delta_edge,
                # Spectral distances
                "w1_parent_sib": round(w1_parent_sib, 4),
                "mse_parent_sib": round(mse_parent_sib, 4),
                # Effective rank
                "er_parent": round(er_parent, 3),
                "er_sibling": round(er_sibling, 3),
                # Subspace alignment
                "overlap_par_sib": round(overlap_parent_sib, 4),
                "overlap_L_sib": round(overlap_left_sib, 4),
                "overlap_R_sib": round(overlap_right_sib, 4),
                # Between-group loading
                "loading_parent": round(loading_parent, 4),
                "loading_sibling": round(loading_sibling, 4),
            }
        )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_bias_summary(df: pd.DataFrame) -> None:
    _section("1. Overall Bias Summary  (k_source vs k_sibling proxy)")
    for source_label, delta_col, ratio_col in [
        ("k_parent  [Gate 2 source]       ", "delta_parent", "ratio_parent"),
        ("k_edge    [Gate 3 production]   ", "delta_edge", "ratio_edge"),
    ]:
        d = df[delta_col].dropna()
        r = df[ratio_col].dropna()
        over = (d < 0).sum()  # source > sibling → conservative
        under = (d > 0).sum()  # source < sibling → anti-conservative
        exact = (d == 0).sum()
        print(f"\n  {source_label}")
        print(f"    Pairs analysed       : {len(d)}")
        print(
            f"    Over-estimate (cons) : {over:4d}   Under-estimate (anti-cons): {under:4d}   Exact: {exact:4d}"
        )
        print(
            f"    Mean delta           : {d.mean():+.3f}  (pos = sibling has MORE dims than source)"
        )
        print(f"    Median delta         : {d.median():+.3f}")
        print(f"    Mean DoF ratio       : {r.mean():.4f}  (>1 conservative, <1 anti-conservative)")
        print(f"    Median DoF ratio     : {r.median():.4f}")
        print(f"    Std  DoF ratio       : {r.std():.4f}")
        print(f"    Fraction ratio < 0.8 : {(r < 0.8).mean():.2%}  ← anti-conservative pairs")
        print(
            f"    Fraction ratio 0.8–1.2: {((r >= 0.8) & (r <= 1.2)).mean():.2%}  ← well-calibrated"
        )
        print(f"    Fraction ratio > 1.2 : {(r > 1.2).mean():.2%}  ← conservative pairs")


def _print_per_case_table(df: pd.DataFrame) -> None:
    _section("2. Per-Case Median k Values and DoF Ratios")
    agg = (
        df.groupby("case")
        .agg(
            true_k=("true_k", "first"),
            pairs=("parent", "count"),
            k_par_med=("k_parent", "median"),
            k_edg_med=("k_edge", "median"),
            k_sib_med=("k_sibling", "median"),
            r_par_med=("ratio_parent", "median"),
            r_edg_med=("ratio_edge", "median"),
            d_par_mean=("delta_parent", "mean"),
            d_edg_mean=("delta_edge", "mean"),
        )
        .round(3)
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 80)
    print(agg.to_string())


def _print_n_correlation(df: pd.DataFrame) -> None:
    _section("3. Pearson Correlations: n_parent vs DoF Ratio / Delta")
    pairs = [
        ("n_parent", "ratio_parent", "n_parent vs r_parent (Gate 2 source)"),
        ("n_parent", "ratio_edge", "n_parent vs r_edge   (Gate 3 production)"),
        ("n_parent", "delta_parent", "n_parent vs delta_parent"),
        ("n_parent", "delta_edge", "n_parent vs delta_edge"),
        ("depth", "ratio_edge", "tree depth vs r_edge"),
        ("depth", "delta_edge", "tree depth vs delta_edge"),
    ]
    sub = df.dropna(subset=["ratio_parent", "ratio_edge"])
    for col_a, col_b, label in pairs:
        if col_a in sub.columns and col_b in sub.columns:
            corr = sub[col_a].corr(sub[col_b])
            print(f"    {label:<45s}: {corr:+.4f}")
    print()
    print(
        "  Interpretation: positive n corr → larger subtrees produce MORE conservative test (ratio > 1)"
    )
    print(
        "                  negative n corr → larger subtrees produce MORE anti-conservative test (ratio < 1)"
    )


def _print_outliers(df: pd.DataFrame, n: int = 12) -> None:
    _section("4. Extreme Pair Analysis")
    anti = df[df["ratio_edge"] < 1.0].nsmallest(n, "ratio_edge")
    print(f"\n  Top {n} most anti-conservative pairs  (ratio_edge < 1, ascending):")
    cols = [
        "case",
        "parent",
        "depth",
        "n_parent",
        "k_edge",
        "k_sibling",
        "ratio_edge",
        "delta_edge",
    ]
    print(anti[cols].to_string(index=False) if len(anti) else "  (none)")

    cons = df[df["ratio_edge"] > 1.0].nlargest(n, "ratio_edge")
    print(f"\n  Top {n} most conservative pairs  (ratio_edge > 1, descending):")
    print(cons[cols].to_string(index=False) if len(cons) else "  (none)")


def _print_spectral_distances(df: pd.DataFrame) -> None:
    _section("5. Spectral Distance Summary  (W1 and MSE vs sibling proxy)")
    sub = df.dropna(subset=["w1_parent_sib", "mse_parent_sib"])
    if sub.empty:
        print("  No spectral distance data available.")
        return
    agg = (
        sub.groupby("case")[["w1_parent_sib", "mse_parent_sib", "er_parent", "er_sibling"]]
        .agg(["mean", "median"])
        .round(4)
    )
    print(agg.to_string())
    print(
        f"\n  Global W1 (parent vs sibling)  — mean: {sub['w1_parent_sib'].mean():.4f}  "
        f"median: {sub['w1_parent_sib'].median():.4f}"
    )
    print(
        f"  Global MSE (parent vs sibling) — mean: {sub['mse_parent_sib'].mean():.4f}  "
        f"median: {sub['mse_parent_sib'].median():.4f}"
    )
    er_diff = (sub["er_parent"] - sub["er_sibling"]).dropna()
    if len(er_diff):
        print("\n  Effective-rank gap (er_parent - er_sibling):")
        print(
            f"    mean={er_diff.mean():+.3f}  median={er_diff.median():+.3f}  "
            f"std={er_diff.std():.3f}"
        )
        print("  (positive = parent has higher effective rank → includes ancestor noise)")


def _print_subspace_alignment(df: pd.DataFrame) -> None:
    _section("6. Subspace Alignment Summary  (mean cos² of principal angles)")
    cols = ["overlap_par_sib", "overlap_L_sib", "overlap_R_sib"]
    labels = [
        "parent eigenvecs  ↔ sibling eigenvecs",
        "left-child eigenvecs ↔ sibling eigenvecs",
        "right-child eigenvecs ↔ sibling eigenvecs",
    ]
    sub = df.dropna(subset=cols)
    if sub.empty:
        print("  No subspace data available.")
        return
    print(f"\n  {'Source pair':<45s}  {'mean':>8}  {'median':>8}  {'std':>8}  {'<0.5 frac':>10}")
    for col, lbl in zip(cols, labels):
        v = sub[col].dropna()
        if len(v) == 0:
            continue
        print(
            f"  {lbl:<45s}  {v.mean():>8.4f}  {v.median():>8.4f}  "
            f"{v.std():>8.4f}  {(v < 0.5).mean():>10.2%}"
        )

    # Per-case breakdown for parent↔sibling overlap
    print("\n  Per-case median parent↔sibling overlap:")
    pc = sub.groupby("case")["overlap_par_sib"].median().round(4)
    for case, val in pc.items():
        bar = "█" * int(val * 20)
        print(f"    {case:<40s}  {val:.4f}  {bar}")


def _print_loading_analysis(df: pd.DataFrame) -> None:
    _section("7. Between-Group Loading  (fraction of ||δ||² explained by leading k eigenvecs)")
    sub = df.dropna(subset=["loading_parent", "loading_sibling"])
    if sub.empty:
        print("  Distributions not present on tree nodes — loading analysis skipped.")
        return
    lp = sub["loading_parent"]
    ls = sub["loading_sibling"]
    diff = ls - lp
    print("\n  loading_parent  (fraction with parent eigenvecs, k=k_parent) :")
    print(f"    mean={lp.mean():.4f}  median={lp.median():.4f}  std={lp.std():.4f}")
    print("\n  loading_sibling (fraction with sibling eigenvecs, k=k_sibling):")
    print(f"    mean={ls.mean():.4f}  median={ls.median():.4f}  std={ls.std():.4f}")
    print("\n  Δloading = loading_sibling − loading_parent:")
    print(f"    mean={diff.mean():+.4f}  median={diff.median():+.4f}  std={diff.std():.4f}")
    print(f"    Fraction where sibling better captures δ : {(diff > 0).mean():.2%}")
    print(f"    Fraction where parent  better captures δ : {(diff < 0).mean():.2%}")

    # Per-case summary
    print("\n  Per-case mean Δloading:")
    pc = (
        sub.groupby("case")
        .apply(lambda g: (g["loading_sibling"] - g["loading_parent"]).mean())
        .round(4)
    )
    for case, val in pc.items():
        sign = "+" if val >= 0 else ""
        print(
            f"    {case:<40s}  {sign}{val:.4f}  {'↑ sibling better' if val > 0.01 else ('↓ parent better' if val < -0.01 else '≈ equal')}"
        )


def _print_depth_accumulation(df: pd.DataFrame) -> None:
    _section("8. Tree-Level Eigenvalue Accumulation  (mean k at each depth)")
    sub = df[df["depth"] >= 0].copy()
    if sub.empty:
        print("  Depth data not available.")
        return
    grp = sub.groupby("depth")
    agg = pd.DataFrame(
        {
            "n_pairs": grp["k_sibling"].count(),
            "k_parent_mean": grp["k_parent"].mean().round(3),
            "k_parent_std": grp["k_parent"].std().round(3),
            "k_edge_mean": grp["k_edge"].mean().round(3),
            "k_edge_std": grp["k_edge"].std().round(3),
            "k_sibling_mean": grp["k_sibling"].mean().round(3),
            "k_sibling_std": grp["k_sibling"].std().round(3),
        }
    )
    print(agg.to_string())

    # Inline ASCII chart of mean k vs depth
    print("\n  Mean k by depth (ASCII bar chart):")
    depth_vals = sub.groupby("depth")[["k_parent", "k_edge", "k_sibling"]].mean()
    for depth, row in depth_vals.iterrows():
        scale = 4
        bar_p = "P" * int(row["k_parent"] * scale)
        bar_e = "e" * int(row["k_edge"] * scale)
        bar_s = "S" * int(row["k_sibling"] * scale)
        print(f"    depth={depth:2d}  k_par={row['k_parent']:5.2f} {bar_p}")
        print(f"           k_edg={row['k_edge']:5.2f} {bar_e}")
        print(f"           k_sib={row['k_sibling']:5.2f} {bar_s}")


def _print_conclusions(df: pd.DataFrame) -> None:
    _section("9. Conclusions")
    sub = df.dropna(subset=["ratio_edge", "ratio_parent"])
    if sub.empty:
        print("  Insufficient data for conclusions.")
        return

    r_e = sub["ratio_edge"]
    r_p = sub["ratio_parent"]

    mean_re = r_e.mean()
    mean_rp = r_p.mean()
    anti_frac = (r_e < 0.8).mean()
    cons_frac = (r_e > 1.2).mean()
    well_frac = ((r_e >= 0.8) & (r_e <= 1.2)).mean()

    print(f"\n  k_edge mean DoF ratio : {mean_re:.4f}")
    print(f"  k_parent mean DoF ratio: {mean_rp:.4f}")
    print()

    if mean_re < 0.9:
        bias = "systematically ANTI-CONSERVATIVE"
        gate3_implication = "Gate 3 uses fewer df than the sibling distribution warrants → inflated T/k → false SPLITs."
    elif mean_re > 1.1:
        bias = "systematically CONSERVATIVE"
        gate3_implication = (
            "Gate 3 uses more df than needed → deflated T/k → missed SPLITs (Type II error)."
        )
    else:
        bias = "approximately WELL-CALIBRATED"
        gate3_implication = (
            "No strong systematic bias detected — k_edge is a reasonable proxy for k_sibling."
        )

    print(f"  k_edge (Gate 3 production) is {bias}.")
    print(f"  Implication: {gate3_implication}")
    print()
    print(
        f"  Distribution: {anti_frac:.2%} anti-cons | {well_frac:.2%} well-cal | {cons_frac:.2%} conservative."
    )
    print()

    # Subspace alignment conclusion
    ov = df["overlap_par_sib"].dropna()
    if len(ov):
        mean_ov = ov.mean()
        if mean_ov > 0.7:
            print(
                f"  Subspace alignment parent↔sibling: HIGH ({mean_ov:.4f}) — parent eigenvectors are a good"
            )
            print(
                "  basis for the sibling test.  Switching to sibling PCA basis likely provides minimal gain."
            )
        elif mean_ov > 0.4:
            print(
                f"  Subspace alignment parent↔sibling: MODERATE ({mean_ov:.4f}) — partial overlap."
            )
            print("  Sibling-specific eigenvectors may improve test power for some nodes.")
        else:
            print(
                f"  Subspace alignment parent↔sibling: LOW ({mean_ov:.4f}) — parent and sibling subspaces"
            )
            print(
                "  diverge substantially.  Sibling-specific eigenvectors are likely better for Gate 3."
            )

    # Loading conclusion
    ld = df.dropna(subset=["loading_parent", "loading_sibling"])
    if len(ld):
        gain = (ld["loading_sibling"] - ld["loading_parent"]).mean()
        print(f"\n  Mean between-group loading gain (sibling eigen − parent eigen): {gain:+.4f}")
        if gain > 0.05:
            print(
                "  → Sibling eigenvectors capture the split direction BETTER → recommend k_sibling for Gate 3."
            )
        elif gain < -0.05:
            print(
                "  → Parent eigenvectors capture the split direction better → current basis is adequate."
            )
        else:
            print("  → No meaningful difference in split-direction capture between bases.")

    # Effective rank conclusion
    er_sub = df.dropna(subset=["er_parent", "er_sibling"])
    if len(er_sub):
        er_gap = (er_sub["er_parent"] - er_sub["er_sibling"]).mean()
        print(f"\n  Mean effective-rank gap (parent − sibling): {er_gap:+.3f}")
        if er_gap > 0.5:
            print("  → Parent accumulates extra dimensionality from ancestor noise.")
            print("    This inflates k_parent relative to the true sibling signal dimension.")
        else:
            print("  → No meaningful effective-rank inflation from ancestor pooling.")

    print("\n  NOTE: Both k_sibling computation AND this entire analysis are READ-ONLY.")
    print("  No production code under kl_clustering_analysis/ has been altered.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(_SEP)
    print("  Exp 35 — Comprehensive Eigenvalue Approximation Comparison")
    print("  k_parent  |  k_edge (geom-mean, Gate 3 production)  |  k_sibling (pooled proxy)")
    print(_SEP)
    minimum_k = max(getattr(config, "SPECTRAL_MINIMUM_DIMENSION", 2), 1)
    print(f"  Cases (binary only, phylo/cat excluded): {len(CASES)}")
    print(f"  Minimum k floor (SPECTRAL_MINIMUM_DIMENSION): {minimum_k}")
    print()

    all_rows: list[dict] = []
    for i, case_name in enumerate(CASES):
        print(f"  [{i+1:02d}/{len(CASES)}] {case_name} ...", end=" ", flush=True)
        rows = analyse_case(case_name)
        print(f"{len(rows)} pairs")
        all_rows.extend(rows)

    if not all_rows:
        print("\n  No data collected — all cases failed or produced no binary-parent nodes.")
        return

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["k_sibling"])

    print(f"\n  Total sibling pairs analysed : {len(df)}")
    print(f"  Cases with data              : {df['case'].nunique()}")
    print(f"  Total binary-parent nodes    : {len(df)}")

    _print_bias_summary(df)
    _print_per_case_table(df)
    _print_n_correlation(df)
    _print_outliers(df)
    _print_spectral_distances(df)
    _print_subspace_alignment(df)
    _print_loading_analysis(df)
    _print_depth_accumulation(df)
    _print_conclusions(df)

    print(f"\n{_SEP}")
    print("  Analysis complete.")
    print(_SEP)


if __name__ == "__main__":
    main()
