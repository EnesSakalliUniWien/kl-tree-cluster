#!/usr/bin/env python
"""Experiment 34 — Comparative Eigenvalue Analysis: parent-child vs edge-derived vs sibling proxy.

For every binary-parent sibling pair in the tree this experiment collects:

  k_parent   — Marchenko-Pastur rank of the *parent's* descendant correlation matrix.
               This is what Gate 2 computes (the existing spectral decomposition).

  k_edge     — min(k_left, k_right): the current Gate 3 production approximation,
               derived from the two children's individual spectral ranks.

  k_sibling  — Marchenko-Pastur rank of the *pooled left+right leaf* correlation
               matrix. This is the proxy "ground truth" for a hypothetical sibling
               correlation matrix that the framework does NOT currently compute.

No production code is modified.  All computations are offline comparisons only.

Comparison metrics (per pair)
-----------------------------
  ratio_parent  = k_parent / k_sibling   — >1 means parent over-estimates sibling rank
  ratio_edge    = k_edge   / k_sibling   — >1 means edge approx over-estimates
  delta_parent  = k_sibling - k_parent   — signed error (positive = sibling has more dims)
  delta_edge    = k_sibling - k_edge     — signed error (positive = sibling has more dims)

DoF ratio interpretation (Gate 3 χ² test calibration):
  ratio < 1  → anti-conservative test (too few df, inflated T/k)
  ratio ≈ 1  → well-calibrated
  ratio > 1  → conservative test (too many df, deflated T/k)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

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
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen.decomposition import (  # noqa: E402
    eigendecompose_correlation,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (  # noqa: E402
    estimate_k_marchenko_pastur,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (  # noqa: E402
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_helpers import (  # noqa: E402
    is_leaf,
)

# ---------------------------------------------------------------------------
# Cases to analyse
# ---------------------------------------------------------------------------
_SENTINEL_CASES = (
    REGRESSION_GUARD_CASES[:10]  # well-behaved — should show ratio ≈ 1
    + INTERMEDIATE_CASES[:10]  # partial-success — interesting mid-range
    + FAILURE_CASES[:8]  # failures — look for systematic bias
)
# Deduplicate while preserving insertion order
_SEEN: set[str] = set()
CASES: list[str] = []
for _c in _SENTINEL_CASES:
    if _c not in _SEEN:
        _SEEN.add(_c)
        CASES.append(_c)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_label_to_idx(leaf_data: pd.DataFrame) -> dict[str, int]:
    return {label: i for i, label in enumerate(leaf_data.index)}


def _get_leaf_row_indices(tree, node: str, label_to_idx: dict[str, int]) -> list[int]:
    """Collect leaf data-matrix row indices for all leaves under *node*."""
    if is_leaf(tree, node):
        label = tree.nodes[node].get("label", node)
        idx = label_to_idx.get(label)
        return [idx] if idx is not None else []
    indices: list[int] = []
    for desc in tree.nodes:
        pass
    # Use precomputed descendants (efficient O(N) traversal already available)
    # We call nx.descendants because we need it for arbitrary nodes at analysis time.
    import networkx as nx

    for desc_node in nx.descendants(tree, node):
        if is_leaf(tree, desc_node):
            label = tree.nodes[desc_node].get("label", desc_node)
            idx = label_to_idx.get(label)
            if idx is not None:
                indices.append(idx)
    return indices


def _compute_k_sibling_proxy(
    tree,
    leaf_data: pd.DataFrame,
    parent: str,
    left: str,
    right: str,
    label_to_idx: dict[str, int],
    minimum_k: int,
) -> int:
    """Compute proxy sibling k from the pooled left+right leaf correlation matrix.

    Re-uses the same eigendecompose_correlation + estimate_k_marchenko_pastur
    pipeline that Gate 2 uses, but on the combined sibling leaf pool rather
    than the full parent descendant set.
    """
    left_indices = _get_leaf_row_indices(tree, left, label_to_idx)
    right_indices = _get_leaf_row_indices(tree, right, label_to_idx)

    if len(left_indices) < 1 or len(right_indices) < 1:
        return minimum_k

    combined_indices = left_indices + right_indices
    X = leaf_data.values[combined_indices, :]

    eig = eigendecompose_correlation(X, compute_eigenvectors=False)
    if eig is None:
        return minimum_k

    n_samples, n_active_features = X.shape[0], int(eig.active_feature_count)
    return max(
        estimate_k_marchenko_pastur(
            eig.eigenvalues,
            n_samples=n_samples,
            n_features=n_active_features,
        ),
        minimum_k,
    )


# ---------------------------------------------------------------------------
# Per-case analysis
# ---------------------------------------------------------------------------


def analyse_case(case_name: str) -> list[dict]:
    """Collect per-sibling-pair eigenvalue source comparison rows."""
    try:
        tree, leaf_data, _y, tc = build_tree_and_data(case_name)
    except Exception as exc:
        print(f"  [SKIP] {case_name}: build failed — {exc}")
        return []

    true_k = tc.get("n_clusters", "?")
    minimum_k = max(getattr(config, "SPECTRAL_MINIMUM_DIMENSION", 2), 1)
    label_to_idx = _build_label_to_idx(leaf_data)

    # Gate 2 spectral context — produces k_parent (indexed by node)
    try:
        spectral_dims, _pca_proj, _pca_eig = compute_spectral_decomposition(
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

    rows: list[dict] = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue  # Gate 1 requires binary parent

        left, right = children[0], children[1]

        k_parent = int(spectral_dims.get(parent, minimum_k))
        k_left = int(spectral_dims.get(left, minimum_k))
        k_right = int(spectral_dims.get(right, minimum_k))
        k_edge = max(min(k_left, k_right), minimum_k)  # current production path

        k_sibling = _compute_k_sibling_proxy(
            tree, leaf_data, parent, left, right, label_to_idx, minimum_k
        )

        # Sample sizes
        n_left = len(_get_leaf_row_indices(tree, left, label_to_idx))
        n_right = len(_get_leaf_row_indices(tree, right, label_to_idx))
        n_parent = n_left + n_right

        # Branch lengths
        bl_left = tree.edges[parent, left].get("branch_length")
        bl_right = tree.edges[parent, right].get("branch_length")

        ratio_parent = k_parent / k_sibling if k_sibling > 0 else float("nan")
        ratio_edge = k_edge / k_sibling if k_sibling > 0 else float("nan")
        delta_parent = k_sibling - k_parent
        delta_edge = k_sibling - k_edge

        rows.append(
            {
                "case": case_name,
                "true_k": true_k,
                "parent": parent,
                "left": left,
                "right": right,
                "n_left": n_left,
                "n_right": n_right,
                "n_parent": n_parent,
                "bl_left": round(bl_left, 6) if bl_left is not None else None,
                "bl_right": round(bl_right, 6) if bl_right is not None else None,
                "k_parent": k_parent,
                "k_left": k_left,
                "k_right": k_right,
                "k_edge": k_edge,
                "k_sibling": k_sibling,
                "delta_parent": delta_parent,
                "delta_edge": delta_edge,
                "ratio_parent": round(ratio_parent, 3),
                "ratio_edge": round(ratio_edge, 3),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEP = "=" * 110


def _print_metric_table(df: pd.DataFrame, metric: str, title: str) -> None:
    """Pivot and print a per-case summary table for one metric."""
    agg = (
        df.groupby("case")[metric]
        .agg(
            mean="mean",
            median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
            n="count",
        )
        .round(3)
    )
    if "true_k" in df.columns:
        agg.insert(0, "true_k", df.drop_duplicates("case").set_index("case")["true_k"])
    print(f"\n  {title}:")
    print(agg.to_string())


def _bias_summary(df: pd.DataFrame) -> None:
    """Print the directional bias of each source vs the sibling proxy."""
    print(f"\n{_SEP}")
    print("  Bias summary (k_used vs k_sibling proxy)")
    print(_SEP)

    for source, delta_col, ratio_col in [
        ("k_parent (Gate 2 source)", "delta_parent", "ratio_parent"),
        ("k_edge   (Gate 3 production)", "delta_edge", "ratio_edge"),
    ]:
        d = df[delta_col]
        r = df[ratio_col]
        over = (d < 0).sum()  # source > sibling → over-estimate → conservative Gate 3
        under = (d > 0).sum()  # source < sibling → under-estimate → anti-conservative Gate 3
        exact = (d == 0).sum()
        print(f"\n  {source}")
        print(
            f"    Pairs: {len(d)}  |  over-estimate: {over}  exact: {exact}  under-estimate: {under}"
        )
        print(
            f"    Mean delta           : {d.mean():+.3f}   (positive = sibling has more dims than source)"
        )
        print(f"    Median delta         : {d.median():+.3f}")
        print(
            f"    Mean DoF ratio       : {r.mean():.3f}    (>1 conservative, <1 anti-conservative)"
        )
        print(f"    Median DoF ratio     : {r.median():.3f}")
        print(f"    Fraction ratio < 0.8 : {(r < 0.8).mean():.2%}  ← anti-conservative pairs")
        print(f"    Fraction ratio > 1.2 : {(r > 1.2).mean():.2%}  ← conservative pairs")


def _per_case_summary(df: pd.DataFrame) -> None:
    print(f"\n{_SEP}")
    print("  Per-case median k values and DoF ratios")
    print(_SEP)
    agg = (
        df.groupby("case")
        .agg(
            true_k=("true_k", "first"),
            n_pairs=("parent", "count"),
            k_parent=("k_parent", "median"),
            k_edge=("k_edge", "median"),
            k_sibling=("k_sibling", "median"),
            r_parent=("ratio_parent", "median"),
            r_edge=("ratio_edge", "median"),
            d_parent=("delta_parent", "mean"),
            d_edge=("delta_edge", "mean"),
        )
        .round(2)
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(agg.to_string())


def _correlation_with_n(df: pd.DataFrame) -> None:
    """Check whether DoF ratio bias correlates with subtree size."""
    print(f"\n{_SEP}")
    print("  Pearson correlations: n_parent vs DoF ratio")
    print(_SEP)
    corr_p = df["n_parent"].corr(df["ratio_parent"])
    corr_e = df["n_parent"].corr(df["ratio_edge"])
    corr_dp = df["n_parent"].corr(df["delta_parent"])
    corr_de = df["n_parent"].corr(df["delta_edge"])
    print(f"  n_parent vs ratio_parent : {corr_p:+.3f}")
    print(f"  n_parent vs ratio_edge   : {corr_e:+.3f}")
    print(f"  n_parent vs delta_parent : {corr_dp:+.3f}")
    print(f"  n_parent vs delta_edge   : {corr_de:+.3f}")
    print()
    print("  (Positive correlation means larger subtrees are more conservative.)")
    print("  (Negative correlation means larger subtrees are anti-conservative.)")


def _outlier_pairs(df: pd.DataFrame, n: int = 10) -> None:
    """Print the most extreme k_edge vs k_sibling discrepancies."""
    print(f"\n{_SEP}")
    print(f"  Top {n} most anti-conservative pairs  (ratio_edge < 1, ranked by ascending ratio)")
    print(_SEP)
    anti = df[df["ratio_edge"] < 1.0].nsmallest(n, "ratio_edge")[
        ["case", "parent", "n_parent", "k_edge", "k_sibling", "ratio_edge", "delta_edge"]
    ]
    print(anti.to_string(index=False) if len(anti) else "  (none)")

    print(f"\n{_SEP}")
    print(f"  Top {n} most conservative pairs  (ratio_edge > 1, ranked by descending ratio)")
    print(_SEP)
    cons = df[df["ratio_edge"] > 1.0].nlargest(n, "ratio_edge")[
        ["case", "parent", "n_parent", "k_edge", "k_sibling", "ratio_edge", "delta_edge"]
    ]
    print(cons.to_string(index=False) if len(cons) else "  (none)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(_SEP)
    print("  Exp 34 — Eigenvalue Source Comparison")
    print(
        "  k_parent (Gate 2 corr matrix)  vs  k_edge (min-child approx)  vs  k_sibling (pooled proxy)"
    )
    print(_SEP)
    print(
        f"  Cases: {len(CASES)}  |  min_k floor: {max(getattr(config, 'SPECTRAL_MINIMUM_DIMENSION', 2), 1)}"
    )
    print()

    all_rows: list[dict] = []
    for i, case_name in enumerate(CASES):
        print(f"  [{i+1:02d}/{len(CASES)}] {case_name} ...", end=" ", flush=True)
        rows = analyse_case(case_name)
        print(f"{len(rows)} pairs")
        all_rows.extend(rows)

    if not all_rows:
        print("  No data collected — all cases failed.")
        return

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["ratio_parent", "ratio_edge"])

    print(f"\n  Total sibling pairs analysed: {len(df)}")
    print(f"  Cases with data           : {df['case'].nunique()}")

    # ── Summary tables ─────────────────────────────────────────────────────
    _per_case_summary(df)

    _print_metric_table(df, "ratio_edge", "DoF ratio — k_edge / k_sibling (Gate 3 production)")
    _print_metric_table(df, "ratio_parent", "DoF ratio — k_parent / k_sibling (Gate 2 source)")

    _bias_summary(df)
    _correlation_with_n(df)
    _outlier_pairs(df)

    # ── Raw pair data ───────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  Full pair table (sorted by abs delta_edge, descending)")
    print(_SEP)
    display_cols = [
        "case",
        "parent",
        "n_left",
        "n_right",
        "k_parent",
        "k_left",
        "k_right",
        "k_edge",
        "k_sibling",
        "delta_edge",
        "delta_parent",
        "ratio_edge",
        "ratio_parent",
    ]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 240)
    print(
        df.assign(abs_delta=df["delta_edge"].abs())
        .sort_values("abs_delta", ascending=False)
        .drop(columns="abs_delta")[display_cols]
        .head(80)
        .to_string(index=False)
    )

    print(f"\n{_SEP}")
    print("  Done.")
    print(_SEP)


if __name__ == "__main__":
    main()
