"""Diagnose spectral-dimension flips between two PosetTree builds.

Two PosetTree instances built from the same linkage matrix can produce
different Gate 2 spectral dimensions for nodes near the Marchenko-Pastur
threshold.  This script builds two trees from the same data (exactly
replicating what compare_interpolation.py does) and diffs every per-node
spectral dimension, sibling spectral dimension, and downstream calibrator summary.

Usage:
    python debug_scripts/sibling_calibration/diagnose_spectral_dim_flip.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_spectral_dims,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


@dataclass(frozen=True)
class GateTwoSnapshot:
    """Immutable snapshot of Gate 2 spectral state for one tree build."""

    edge_spectral_dims: dict[str, int]
    sibling_spectral_dims: dict[str, int]
    annotations_df: pd.DataFrame
    pca_projections: dict[str, np.ndarray]
    pca_eigenvalues: dict[str, np.ndarray]


CASES: list[dict] = [
    {
        "generator": "blobs",
        "n_samples": 30,
        "n_features": 30,
        "n_clusters": 3,
        "cluster_std": 0.6,
        "seed": 42,
        "name": "gaussian_clear_1",
    },
    {
        "generator": "blobs",
        "n_samples": 40,
        "n_features": 40,
        "n_clusters": 4,
        "cluster_std": 0.5,
        "seed": 42,
        "name": "gaussian_clear_2",
    },
    {
        "generator": "blobs",
        "n_samples": 30,
        "n_features": 20,
        "n_clusters": 3,
        "cluster_std": 0.6,
        "seed": 42,
        "name": "gauss_clear_small",
    },
    {
        "generator": "blobs",
        "n_samples": 60,
        "n_features": 40,
        "n_clusters": 4,
        "cluster_std": 0.6,
        "seed": 42,
        "name": "gauss_clear_medium",
    },
]


def _build_and_annotate(
    Z: np.ndarray, leaf_names: list[str], data_bin: pd.DataFrame
) -> GateTwoSnapshot:
    """Build a fresh tree, run Gate 2, extract spectral state."""
    tree = PosetTree.from_linkage(Z, leaf_names=leaf_names)
    tree.populate_node_divergences(leaf_data=data_bin)

    ann = annotate_child_parent_divergence(
        tree, tree.annotations_df, significance_level_alpha=0.05, leaf_data=data_bin
    )

    edge_dims: dict[str, int] = dict(ann.attrs.get("_spectral_dims", {}))
    pca_proj: dict[str, np.ndarray] = dict(ann.attrs.get("_pca_projections", {}))
    pca_eig: dict[str, np.ndarray] = dict(ann.attrs.get("_pca_eigenvalues", {}))
    sib_dims = derive_sibling_spectral_dims(tree, ann) or {}

    return GateTwoSnapshot(
        edge_spectral_dims=edge_dims,
        sibling_spectral_dims=dict(sib_dims),
        annotations_df=ann,
        pca_projections=pca_proj,
        pca_eigenvalues=pca_eig,
    )


def _diff_dicts(a: dict[str, int], b: dict[str, int], label: str) -> list[str]:
    """Print and return nodes whose spectral dim differs between builds."""
    all_keys = sorted(set(a) | set(b))
    flips: list[str] = []
    for key in all_keys:
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            flips.append(key)
            print(f"    {key:>8}:  build_A={va}  build_B={vb}  delta={_delta(va, vb)}")
    return flips


def _delta(a: int | None, b: int | None) -> str:
    if a is None or b is None:
        return "N/A"
    return f"{b - a:+d}"


def diagnose_case(case: dict) -> None:
    """Run two independent tree builds and diff their spectral dims."""
    data_bin, _labels, _, _ = generate_case_data(case)
    Z = linkage(
        pdist(data_bin.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )

    snap_a = _build_and_annotate(Z, data_bin.index.tolist(), data_bin)
    snap_b = _build_and_annotate(Z, data_bin.index.tolist(), data_bin)

    n_edge = len(snap_a.edge_spectral_dims)
    n_sib = len(snap_a.sibling_spectral_dims)
    label = case["name"]

    print("=" * 90)
    print(f"CASE: {label}  (n={case['n_samples']}, p={case['n_features']}, K={case['n_clusters']})")
    print("=" * 90)

    # ── Edge spectral dims (per-node, from Gate 2) ──────────────────────
    print(
        f"\n  Edge spectral dims: {n_edge} nodes (build A), {len(snap_b.edge_spectral_dims)} nodes (build B)"
    )
    edge_flips = _diff_dicts(snap_a.edge_spectral_dims, snap_b.edge_spectral_dims, "edge")
    if not edge_flips:
        print("    (all match)")
    else:
        print(f"    >>> {len(edge_flips)} node(s) flipped")

    # ── Sibling spectral dims (per-parent, geometric mean) ─────────────
    print(
        f"\n  Sibling spectral dims: {n_sib} parents (build A), {len(snap_b.sibling_spectral_dims)} parents (build B)"
    )
    sib_flips = _diff_dicts(snap_a.sibling_spectral_dims, snap_b.sibling_spectral_dims, "sibling")
    if not sib_flips:
        print("    (all match)")
    else:
        print(f"    >>> {len(sib_flips)} parent(s) flipped")

    # ── Downstream impact: which flipped nodes are focal vs null-like? ──
    if edge_flips:
        print("\n  Downstream impact of flipped edge dims:")
        ann_a = snap_a.annotations_df
        sig_col = "Child_Parent_Divergence_Significant"
        pval_col = "Child_Parent_Divergence_P_Value_BH"
        for node in edge_flips:
            sig = (
                ann_a.loc[node, sig_col]
                if node in ann_a.index and sig_col in ann_a.columns
                else None
            )
            pval = (
                ann_a.loc[node, pval_col]
                if node in ann_a.index and pval_col in ann_a.columns
                else None
            )
            k_a = snap_a.edge_spectral_dims.get(node)
            k_b = snap_b.edge_spectral_dims.get(node)
            pval_str = f"{pval:.6f}" if pval is not None and np.isfinite(pval) else "N/A"
            print(f"    {node:>8}  k_A={k_a}  k_B={k_b}  edge_sig={sig}  p_bh={pval_str}")

    # ── Full dimension distribution per build ───────────────────────────
    print("\n  Edge dim distribution (build A):")
    _print_dim_histogram(snap_a.edge_spectral_dims)
    print("  Edge dim distribution (build B):")
    _print_dim_histogram(snap_b.edge_spectral_dims)

    # ── Sibling dim distribution ────────────────────────────────────────
    print("  Sibling dim distribution (build A):")
    _print_dim_histogram(snap_a.sibling_spectral_dims)
    print("  Sibling dim distribution (build B):")
    _print_dim_histogram(snap_b.sibling_spectral_dims)

    # ── PCA projection matrix comparison ────────────────────────────────
    print("\n  PCA projection matrix comparison:")
    all_pca_nodes = sorted(set(snap_a.pca_projections) | set(snap_b.pca_projections))
    pca_flips = []
    for node in all_pca_nodes:
        pa = snap_a.pca_projections.get(node)
        pb = snap_b.pca_projections.get(node)
        if pa is None or pb is None:
            pca_flips.append(node)
            print(f"    {node:>8}  present_A={pa is not None}  present_B={pb is not None}")
            continue
        if pa.shape != pb.shape:
            pca_flips.append(node)
            print(f"    {node:>8}  shape_A={pa.shape}  shape_B={pb.shape}")
            continue
        max_diff = float(np.max(np.abs(pa - pb)))
        # PCA eigenvectors can have sign flips — check via absolute cosine
        if max_diff > 1e-10:
            # Check column-wise sign consistency
            sign_consistent = True
            for col in range(pa.shape[1]):
                cos = np.dot(pa[:, col], pb[:, col]) / (
                    np.linalg.norm(pa[:, col]) * np.linalg.norm(pb[:, col]) + 1e-30
                )
                if abs(abs(cos) - 1.0) > 1e-6:
                    sign_consistent = False
                    break
            if not sign_consistent:
                pca_flips.append(node)
                print(f"    {node:>8}  max_element_diff={max_diff:.2e}  (NOT sign-flip)")
    if not pca_flips:
        print("    (all match or differ only by sign flips)")
    else:
        print(f"    >>> {len(pca_flips)} node(s) with non-trivial PCA differences")

    # ── PCA eigenvalue comparison ───────────────────────────────────────
    print("\n  PCA eigenvalue comparison:")
    all_eig_nodes = sorted(set(snap_a.pca_eigenvalues) | set(snap_b.pca_eigenvalues))
    eig_flips = []
    for node in all_eig_nodes:
        ea = snap_a.pca_eigenvalues.get(node)
        eb = snap_b.pca_eigenvalues.get(node)
        if ea is None or eb is None:
            eig_flips.append(node)
            continue
        if ea.shape != eb.shape:
            eig_flips.append(node)
            print(f"    {node:>8}  shape_A={ea.shape}  shape_B={eb.shape}")
            continue
        max_diff = float(np.max(np.abs(ea - eb)))
        if max_diff > 1e-10:
            eig_flips.append(node)
            print(f"    {node:>8}  max_eigenvalue_diff={max_diff:.2e}  eig_A={ea}  eig_B={eb}")
    if not eig_flips:
        print("    (all match)")
    else:
        print(f"    >>> {len(eig_flips)} node(s) with eigenvalue differences")

    print()


def _print_dim_histogram(dims: dict[str, int]) -> None:
    """Print a compact histogram of dimension values."""
    if not dims:
        print("    (empty)")
        return
    vals = list(dims.values())
    unique, counts = np.unique(vals, return_counts=True)
    parts = [f"k={u}: {c}" for u, c in zip(unique, counts)]
    print(f"    {', '.join(parts)}  (total={len(vals)})")


def main() -> None:
    print("Diagnose spectral-dimension flips between two PosetTree builds")
    print(f"  SPECTRAL_MINIMUM_DIMENSION={config.SPECTRAL_MINIMUM_DIMENSION}")
    print(f"  SIBLING_WHITENING={config.SIBLING_WHITENING}")
    print()
    for case in CASES:
        diagnose_case(case)


if __name__ == "__main__":
    main()
