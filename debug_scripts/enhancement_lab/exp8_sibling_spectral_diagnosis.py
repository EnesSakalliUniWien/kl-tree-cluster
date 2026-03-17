#!/usr/bin/env python
"""Experiment 8 — Sibling Spectral Dimension Diagnostic.

For every sibling pair in the tree that Gate 3 must evaluate, compute:
  1. JL k            — current JL-based projection dimension (baseline)
  2. k_left          — left child's spectral k   (MP on left descendants)
  3. k_right         — right child's spectral k   (MP on right descendants)
  4. k_parent        — parent's spectral k        (MP on parent descendants)
  5. k_union         — rank of span(V_left ∪ V_right)
  6. k_z_signal      — z-vector signal dims: MP on |z| projected into combined basis

The goal is to understand what projection dimension the sibling test
*should* use, compared to the JL dimension it currently uses.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

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
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.pooled_variance import (
    standardize_proportion_difference,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_binary_children(tree, parent: str):
    children = list(tree.successors(parent))


def _get_leaf_indices(tree, node: str, leaf_label_to_index: dict) -> list[int]:
    """Collect leaf row indices for a subtree."""
    from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_helpers import (
        is_leaf,
    )

    if is_leaf(tree, node):
        label = tree.nodes[node].get("label", node)
        idx = leaf_label_to_index.get(label)
        return [idx] if idx is not None else []

    indices = []
    for desc in tree.nodes:
        # walk descendants
        pass
    # Use networkx descendants
    import networkx as nx

    for desc in nx.descendants(tree, node):
        if is_leaf(tree, desc):
            label = tree.nodes[desc].get("label", desc)
            idx = leaf_label_to_index.get(label)
            if idx is not None:
                indices.append(idx)
    return indices


def _compute_z_signal_dim(
    z: np.ndarray,
    pca_left: np.ndarray | None,
    pca_right: np.ndarray | None,
    eig_left: np.ndarray | None,
    eig_right: np.ndarray | None,
    n_left: int,
    n_right: int,
) -> Tuple[int, int, float]:
    """Compute z-vector signal dimensionality in the combined eigenbasis.

    1. Stack eigenvectors from both children (V_left, V_right).
    2. SVD to get the orthonormal union basis U.
    3. Project z into U: z_proj = U @ z.
    4. Treat z_proj² as "eigenvalues" and use MP to count signal dims.

    Returns (k_union_rank, k_z_signal, z_energy_fraction).
    """
    n_total = n_left + n_right
    d = len(z)

    # Build combined basis
    bases = []
    if pca_left is not None and pca_left.shape[0] > 0:
        bases.append(pca_left)
    if pca_right is not None and pca_right.shape[0] > 0:
        bases.append(pca_right)

    if not bases:
        # No spectral info — fall back to z itself
        return 0, 0, 0.0

    combined = np.vstack(bases)  # shape: (k_L + k_R) × d

    # SVD to get orthonormal union basis
    U, S, Vt = linalg.svd(combined, full_matrices=False)
    # Keep only non-negligible singular values
    tol = max(S) * 1e-8 if len(S) > 0 else 1e-12
    rank = int(np.sum(S > tol))
    k_union = rank

    if rank == 0:
        return 0, 0, 0.0

    # Orthonormal basis: top 'rank' rows of Vt
    basis = Vt[:rank]  # (rank × d)

    # Project z into this basis
    z_proj = basis @ z  # (rank,)
    z_proj_sq = z_proj**2

    # Total z energy
    z_total_energy = float(np.sum(z**2))
    z_captured_energy = float(np.sum(z_proj_sq))
    energy_fraction = z_captured_energy / z_total_energy if z_total_energy > 0 else 0.0

    # Use MP on the projected squared components to determine signal dims.
    # Think of z_proj²_i as "pseudo-eigenvalues" in the combined basis.
    # Under H₀ (no sibling difference), z ~ N(0, I) so z_proj²_i ~ χ²(1)
    # all ≈ 1.0.  Signal dims will have z_proj²_i >> 1.
    # We use MP with n_samples = n_total (total observations) and
    # n_features = rank (basis dimension).
    sorted_z_proj_sq = np.sort(z_proj_sq)[::-1]
    k_z_signal = marchenko_pastur_signal_count(
        sorted_z_proj_sq,
        n_samples=n_total,
        n_features=rank,
    )

    return k_union, k_z_signal, energy_fraction


def _compute_symmetric_difference_dim(
    pca_left: np.ndarray | None,
    pca_right: np.ndarray | None,
    *,
    cos_threshold: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    """Compute symmetric difference of eigenvector subspaces.

    Uses principal angles between span(V_L) and span(V_R):
    - cos(θ) ≈ 1 → shared direction (intersection)
    - cos(θ) ≈ 0 → unique direction (symmetric difference)

    Parameters
    ----------
    cos_threshold : float
        Cosine similarity threshold.  Angles with cos(θ) > threshold
        are counted as "shared".

    Returns
    -------
    (k_shared, k_sym_diff, k_union, principal_cosines)
        k_shared     : # of principal angles with cos(θ) > threshold
        k_sym_diff   : k_union - k_shared  (unique dimensions)
        k_union      : rank of combined subspace
        principal_cosines : list of cos(θ) values
    """
    if pca_left is None or pca_right is None:
        k_l = 0 if pca_left is None else pca_left.shape[0]
        k_r = 0 if pca_right is None else pca_right.shape[0]
        return 0, k_l + k_r, k_l + k_r, []

    if pca_left.shape[0] == 0 or pca_right.shape[0] == 0:
        k_l, k_r = pca_left.shape[0], pca_right.shape[0]
        return 0, k_l + k_r, k_l + k_r, []

    # Principal angles via SVD of V_L^T @ V_R
    # V_L: (k_L × d), V_R: (k_R × d) → product: (k_L × k_R)
    cross = pca_left @ pca_right.T  # (k_L × k_R)
    singular_values = linalg.svd(cross, compute_uv=False)
    # Clamp to [0, 1] for numerical safety
    cos_angles = np.clip(singular_values, 0.0, 1.0)

    # Union rank via SVD of stacked bases
    combined = np.vstack([pca_left, pca_right])
    S_union = linalg.svd(combined, compute_uv=False)
    tol = max(S_union) * 1e-8 if len(S_union) > 0 else 1e-12
    k_union = int(np.sum(S_union > tol))

    # Count shared vs unique
    k_shared = int(np.sum(cos_angles > cos_threshold))
    k_sym_diff = max(k_union - k_shared, 0)

    return k_shared, k_sym_diff, k_union, cos_angles.tolist()


def _compute_z_direct_signal_dim(
    z: np.ndarray,
    n_total: int,
) -> int:
    """Estimate z-vector signal dimensionality directly from z² components.

    Under H₀, each z_j² ~ χ²(1) with E[z_j²] = 1.
    Signal features have z_j² >> 1.
    Use MP on sorted z² to count signal dimensions.
    """
    z_sq = np.sort(z**2)[::-1]
    d = len(z_sq)
    if d == 0:
        return 0
    return marchenko_pastur_signal_count(z_sq, n_samples=n_total, n_features=d)


# ── main diagnostic ──────────────────────────────────────────────────────────


def diagnose_case(case_name: str) -> pd.DataFrame:
    """Run the spectral diagnostic for one case, return per-pair DataFrame."""

    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")

    leaf_data = data_df
    n_features = data_df.shape[1]
    leaf_label_to_index = {label: i for i, label in enumerate(data_df.index)}

    # Compute spectral decomposition for all nodes
    spectral_dims, pca_projections, pca_eigenvalues = compute_spectral_decomposition(
        tree,
        leaf_data,
        method="marchenko_pastur",
        minimum_projection_dimension=config.SPECTRAL_MINIMUM_DIMENSION,
        compute_projections=True,
    )

    # Collect results for each binary sibling pair
    rows: List[dict] = []

    for parent in tree.nodes:
        pair = _get_binary_children(tree, parent)
        if pair is None:
            continue
        left, right = pair

        # Distributions and sample sizes
        try:
            left_dist = extract_node_distribution(tree, left)
            right_dist = extract_node_distribution(tree, right)
            n_left = extract_node_sample_size(tree, left)
            n_right = extract_node_sample_size(tree, right)
        except Exception:
            continue

        if n_left < 2 or n_right < 2:
            continue

        # Z-scores (standardized proportion difference)
        try:
            z, _var = standardize_proportion_difference(
                left_dist,
                right_dist,
                float(n_left),
                float(n_right),
            )
        except Exception:
            continue
        if not np.isfinite(z).all():
            continue

        # 1. JL k (current baseline)
        jl_k = compute_projection_dimension_backend(
            n_left + n_right,
            n_features,
        )
        # JL k (same — no cap)
        jl_k_capped = jl_k

        # 2-4. Spectral k for left, right, parent
        k_left = spectral_dims.get(left, 0)
        k_right = spectral_dims.get(right, 0)
        k_parent = spectral_dims.get(parent, 0)

        # PCA projections (k × d matrices)
        pca_left = pca_projections.get(left)
        pca_right = pca_projections.get(right)
        eig_left = pca_eigenvalues.get(left)
        eig_right = pca_eigenvalues.get(right)

        # 5-6. Union basis and z-signal dimensionality
        k_union, k_z_signal, energy_frac = _compute_z_signal_dim(
            z,
            pca_left,
            pca_right,
            eig_left,
            eig_right,
            n_left,
            n_right,
        )

        # 7. Symmetric difference of subspaces (multiple thresholds)
        k_shared_50, k_sym_diff_50, _, cos_angles = _compute_symmetric_difference_dim(
            pca_left,
            pca_right,
            cos_threshold=0.5,
        )
        k_shared_70, k_sym_diff_70, _, _ = _compute_symmetric_difference_dim(
            pca_left,
            pca_right,
            cos_threshold=0.7,
        )
        k_shared_30, k_sym_diff_30, _, _ = _compute_symmetric_difference_dim(
            pca_left,
            pca_right,
            cos_threshold=0.3,
        )

        # 8. z-vector signal directly from z² spectrum
        k_z_direct = _compute_z_direct_signal_dim(z, n_left + n_right)

        # z-vector magnitude (total χ² under random projection)
        z_norm_sq = float(np.sum(z**2))

        rows.append(
            {
                "case": case_name,
                "true_k": true_k,
                "parent": parent,
                "left": left,
                "right": right,
                "n_left": n_left,
                "n_right": n_right,
                "n_total": n_left + n_right,
                "d": n_features,
                "jl_k_raw": jl_k,
                "jl_k_capped": jl_k_capped,
                "k_left": k_left,
                "k_right": k_right,
                "k_parent": k_parent,
                "k_union": k_union,
                "k_z_signal": k_z_signal,
                "k_shared_50": k_shared_50,
                "k_sym_diff_50": k_sym_diff_50,
                "k_shared_70": k_shared_70,
                "k_sym_diff_70": k_sym_diff_70,
                "k_shared_30": k_shared_30,
                "k_sym_diff_30": k_sym_diff_30,
                "k_z_direct": k_z_direct,
                "n_principal_angles": len(cos_angles),
                "mean_cos_angle": round(float(np.mean(cos_angles)), 4) if cos_angles else 0.0,
                "min_cos_angle": round(float(np.min(cos_angles)), 4) if cos_angles else 0.0,
                "energy_fraction": round(energy_frac, 4),
                "z_norm_sq": round(z_norm_sq, 2),
            }
        )

    return pd.DataFrame(rows)


# ── reporting ────────────────────────────────────────────────────────────────


def summarize(all_results: pd.DataFrame) -> None:
    """Print human-readable summary of the diagnostic."""

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 80)
    print("  Per-case aggregates")
    print("=" * 80)

    agg = (
        all_results.groupby("case")
        .agg(
            true_k=("true_k", "first"),
            n_pairs=("parent", "count"),
            jl_k_raw_mean=("jl_k_raw", "mean"),
            jl_k_capped_mean=("jl_k_capped", "mean"),
            k_left_mean=("k_left", "mean"),
            k_right_mean=("k_right", "mean"),
            k_parent_mean=("k_parent", "mean"),
            k_union_mean=("k_union", "mean"),
            k_z_signal_mean=("k_z_signal", "mean"),
            k_sym_diff_50_mean=("k_sym_diff_50", "mean"),
            k_sym_diff_70_mean=("k_sym_diff_70", "mean"),
            k_sym_diff_30_mean=("k_sym_diff_30", "mean"),
            k_shared_50_mean=("k_shared_50", "mean"),
            k_z_direct_mean=("k_z_direct", "mean"),
            mean_cos_angle=("mean_cos_angle", "mean"),
            energy_frac_mean=("energy_fraction", "mean"),
            z_norm_sq_mean=("z_norm_sq", "mean"),
        )
        .sort_values("case")
    )
    print(agg.to_string())

    print("\n" + "=" * 80)
    print("  Global dimension comparison")
    print("=" * 80)

    for col in [
        "jl_k_raw",
        "jl_k_capped",
        "k_left",
        "k_right",
        "k_parent",
        "k_union",
        "k_z_signal",
        "k_sym_diff_50",
        "k_sym_diff_70",
        "k_sym_diff_30",
        "k_shared_50",
        "k_z_direct",
    ]:
        vals = all_results[col].dropna()
        print(
            f"  {col:20s}  mean={vals.mean():6.1f}  median={vals.median():5.0f}  min={vals.min():3.0f}  max={vals.max():5.0f}"
        )

    print("\n" + "=" * 80)
    print("  Principal angles between sibling subspaces")
    print("=" * 80)
    for col in ["mean_cos_angle", "min_cos_angle"]:
        vals = all_results[col].dropna()
        print(
            f"  {col:20s}  mean={vals.mean():.4f}  median={vals.median():.4f}  min={vals.min():.4f}  max={vals.max():.4f}"
        )

    print("\n" + "=" * 80)
    print("  Energy captured by combined PCA basis (how much of z lives in V_L ∪ V_R)")
    print("=" * 80)

    ef = all_results.groupby("case")["energy_fraction"].agg(["mean", "median", "min", "max"])
    print(ef.to_string())

    print("\n" + "=" * 80)
    print("  Ratio: k_z_signal / jl_k_capped")
    print("=" * 80)

    all_results["ratio_z_to_jl"] = np.where(
        all_results["jl_k_capped"] > 0,
        all_results["k_z_signal"] / all_results["jl_k_capped"],
        np.nan,
    )
    ratio_agg = all_results.groupby("case")["ratio_z_to_jl"].agg(["mean", "median"])
    print(ratio_agg.to_string())

    # Top-level signal pairs: where z_norm_sq is large
    print("\n" + "=" * 80)
    print("  Top 20 pairs by z_norm_sq (strongest sibling differences)")
    print("=" * 80)
    top = all_results.nlargest(20, "z_norm_sq")[
        [
            "case",
            "parent",
            "n_total",
            "jl_k_capped",
            "k_left",
            "k_right",
            "k_union",
            "k_z_signal",
            "k_sym_diff_50",
            "k_shared_50",
            "mean_cos_angle",
            "k_z_direct",
            "z_norm_sq",
        ]
    ]
    print(top.to_string(index=False))

    # Bottom pairs (weakest — these are the noise pairs)
    print("\n" + "=" * 80)
    print("  Bottom 20 pairs by z_norm_sq (weakest / noise-like)")
    print("=" * 80)
    bot = all_results.nsmallest(20, "z_norm_sq")[
        [
            "case",
            "parent",
            "n_total",
            "jl_k_capped",
            "k_left",
            "k_right",
            "k_union",
            "k_z_signal",
            "k_sym_diff_50",
            "k_shared_50",
            "mean_cos_angle",
            "k_z_direct",
            "z_norm_sq",
        ]
    ]
    print(bot.to_string(index=False))

    # ── New: symmetric difference deep dive ──
    print("\n" + "=" * 80)
    print("  Symmetric difference by threshold (global)")
    print("=" * 80)
    for thresh in ["30", "50", "70"]:
        sh = all_results[f"k_shared_{thresh}"]
        sd = all_results[f"k_sym_diff_{thresh}"]
        print(
            f"  cos_threshold=0.{thresh}:  shared mean={sh.mean():.1f} median={sh.median():.0f}"
            f"  |  sym_diff mean={sd.mean():.1f} median={sd.median():.0f} max={sd.max():.0f}"
        )

    print("\n" + "=" * 80)
    print("  Dimension comparison: k_sym_diff(0.5) vs JL_capped vs k_z_signal vs k_z_direct")
    print("=" * 80)
    comparison = (
        all_results.groupby("case")
        .agg(
            true_k=("true_k", "first"),
            n_pairs=("parent", "count"),
            jl_capped=("jl_k_capped", "median"),
            k_z_signal=("k_z_signal", "median"),
            k_z_direct=("k_z_direct", "median"),
            k_sym_50=("k_sym_diff_50", "median"),
            k_sym_70=("k_sym_diff_70", "median"),
            k_sym_30=("k_sym_diff_30", "median"),
            mean_cos=("mean_cos_angle", "mean"),
        )
        .sort_values("case")
    )
    print(comparison.to_string())


# ── main ─────────────────────────────────────────────────────────────────────


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
    print("=" * 80)
    print("  EXPERIMENT 8: Sibling Spectral Dimension Diagnostic")
    print("=" * 80)
    main()
