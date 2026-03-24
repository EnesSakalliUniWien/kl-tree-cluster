"""Lab: map eigenvector/eigenvalue relationships between siblings and parent.

For each binary parent node, computes the eigendecomposition of:
  - Left child's descendant data
  - Right child's descendant data
  - Parent's (pooled) descendant data

Measures:
  1. MP signal count (k) for each of left, right, parent
  2. Effective rank (Shannon entropy) for each
  3. Principal angles between child signal subspaces and parent signal subspace
     (how much of each child's signal is "contained" in the parent's)
  4. Subspace overlap between left and right signal subspaces
  5. Whether parent k ≈ k_L + k_R (additive) or parent k ≈ max(k_L, k_R)
  6. Relationship to Gate 3 decision (SPLIT vs MERGE)
"""
from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import networkx as nx
import numpy as np
from lab_helpers import build_tree_and_data
from scipy import linalg

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    effective_rank as compute_effective_rank,
    marchenko_pastur_signal_count,
)

CASES = [
    "binary_balanced_low_noise__2",
    "gauss_clear_small",
    "binary_low_noise_12c",
    "binary_perfect_8c",
    "binary_hard_4c",
    "gauss_noisy_3c",
    "gauss_overlap_4c_med",
    "gauss_moderate_3c",
    "binary_low_noise_4c",
    "binary_perfect_4c",
    "gauss_clear_large",
    "binary_multiscale_4c",
    "binary_many_features",
    "gauss_noisy_many",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def descendant_leaves(tree, node):
    """Return list of leaf node IDs under *node*."""
    if tree.out_degree(node) == 0:
        return [node]
    leaves = []
    for desc in nx.descendants(tree, node):
        if tree.out_degree(desc) == 0:
            leaves.append(desc)
    return sorted(leaves)


def node_data_matrix(tree, node, leaf_data):
    """Return (n_leaves, d) array of leaf data under *node*."""
    leaf_ids = descendant_leaves(tree, node)
    labels = [tree.nodes[lid].get("label", lid) for lid in leaf_ids]
    return leaf_data.loc[labels].values


def get_signal_eigenvectors(data_matrix):
    """Compute eigendecomposition and return (eigenvalues, signal_vecs, k_mp, eff_rank, d_active).

    signal_vecs: (k_mp × d_active) matrix of top-k eigenvectors.
    """
    eig = eigendecompose_correlation_backend(data_matrix, need_eigh=True)
    if eig is None:
        return None, None, 0, 1.0, 0

    n_samples, _ = data_matrix.shape
    k_mp = marchenko_pastur_signal_count(eig.eigenvalues, n_samples, eig.d_active)
    eff_rank = compute_effective_rank(eig.eigenvalues)

    # Get top-k eigenvectors in d_active space
    if eig.use_dual and eig.gram_vecs is not None and eig.X_std is not None:
        # Recover d-space eigenvectors from dual form
        top_gram = eig.gram_vecs[:, :k_mp]
        top_evals = eig.eigenvalues[:k_mp]
        vecs = []
        for i in range(k_mp):
            if top_evals[i] > 1e-12:
                v = eig.X_std.T @ top_gram[:, i]
                v = v / (np.sqrt(top_evals[i]) * np.sqrt(eig.d_active))
                norm = np.linalg.norm(v)
                if norm > 1e-12:
                    v = v / norm
                vecs.append(v)
        if not vecs:
            return eig.eigenvalues, None, k_mp, eff_rank, eig.d_active
        signal_vecs = np.array(vecs)  # (k_mp, d_active)
    elif eig.eigenvectors_active is not None:
        signal_vecs = eig.eigenvectors_active[:, :k_mp].T  # (k_mp, d_active)
    else:
        return eig.eigenvalues, None, k_mp, eff_rank, eig.d_active

    return eig.eigenvalues, signal_vecs, k_mp, eff_rank, eig.d_active


def principal_angles(A, B):
    """Compute principal angles (in degrees) between subspaces spanned by rows of A and B.

    A: (k_a, d) — rows are basis vectors of subspace A
    B: (k_b, d) — rows are basis vectors of subspace B

    Returns array of min(k_a, k_b) angles in degrees.
    """
    if A is None or B is None:
        return np.array([])
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    # Ensure same dimensionality — align on active features
    d = min(A.shape[1], B.shape[1])
    A = A[:, :d]
    B = B[:, :d]

    # QR for numerical stability
    Qa, _ = np.linalg.qr(A.T, mode='reduced')
    Qb, _ = np.linalg.qr(B.T, mode='reduced')

    # SVD of cross-product → singular values are cos(principal angles)
    M = Qa.T @ Qb
    s = np.clip(linalg.svdvals(M), 0.0, 1.0)
    return np.degrees(np.arccos(s))


def subspace_overlap(A, B):
    """Grassmann distance-based overlap score between two subspaces.

    Returns fraction in [0, 1]: 1 = identical subspaces, 0 = orthogonal.
    Computed as mean cos²(principal_angle).
    """
    angles = principal_angles(A, B)
    if len(angles) == 0:
        return 0.0
    return float(np.mean(np.cos(np.radians(angles)) ** 2))


# ── Per-case analysis ──────────────────────────────────────────────────────

def analyze_case(case_name):
    """Analyze eigenvector relationships for all binary parents in a case."""
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)

    # Run decomposition to get gate decisions
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.SIBLING_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    annotations_df = tree.annotations_df

    rows = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        # Skip if children are both leaves (nothing interesting)
        n_left_leaves = len(descendant_leaves(tree, left))
        n_right_leaves = len(descendant_leaves(tree, right))
        if n_left_leaves < 2 and n_right_leaves < 2:
            continue

        try:
            data_L = node_data_matrix(tree, left, data_df)
            data_R = node_data_matrix(tree, right, data_df)
            data_P = np.vstack([data_L, data_R])
        except (KeyError, IndexError):
            continue

        # Eigendecompose all three
        evals_L, vecs_L, k_L, er_L, d_L = get_signal_eigenvectors(data_L)
        evals_R, vecs_R, k_R, er_R, d_R = get_signal_eigenvectors(data_R)
        evals_P, vecs_P, k_P, er_P, d_P = get_signal_eigenvectors(data_P)

        n_features = data_df.shape[1]
        jl_k = compute_jl_dim(len(data_P), n_features)

        # Subspace overlaps (align to common d_active via parent)
        # We need vecs in same feature space — use d_active of parent
        # For simplicity, recompute all with the same active mask (parent's)
        overlap_LP = subspace_overlap(vecs_L, vecs_P) if vecs_L is not None and vecs_P is not None else float("nan")
        overlap_RP = subspace_overlap(vecs_R, vecs_P) if vecs_R is not None and vecs_P is not None else float("nan")
        overlap_LR = subspace_overlap(vecs_L, vecs_R) if vecs_L is not None and vecs_R is not None else float("nan")

        # Principal angles (first angle = most aligned direction)
        pa_LP = principal_angles(vecs_L, vecs_P)
        pa_RP = principal_angles(vecs_R, vecs_P)
        pa_LR = principal_angles(vecs_L, vecs_R)

        # Gate 3 decision
        sibling_diff = False
        if parent in annotations_df.index:
            sibling_diff = bool(annotations_df.loc[parent].get("Sibling_BH_Different", False))

        rows.append({
            "parent": parent,
            "n_L": len(data_L),
            "n_R": len(data_R),
            "n_P": len(data_P),
            # MP signal counts
            "k_L": k_L,
            "k_R": k_R,
            "k_P": k_P,
            "k_min": min(k_L, k_R) if k_L > 0 and k_R > 0 else max(k_L, k_R),
            "k_max": max(k_L, k_R),
            "k_sum": k_L + k_R,
            # Effective rank
            "er_L": round(er_L, 1),
            "er_R": round(er_R, 1),
            "er_P": round(er_P, 1),
            # JL reference
            "jl_k": jl_k,
            "jl_qrt": jl_k // 4,
            # Subspace overlaps (cos² based, 1=identical)
            "ovl_LP": round(overlap_LP, 3),
            "ovl_RP": round(overlap_RP, 3),
            "ovl_LR": round(overlap_LR, 3),
            # First principal angle (degrees) — 0°=aligned, 90°=orthogonal
            "ang1_LP": round(float(pa_LP[0]), 1) if len(pa_LP) > 0 else float("nan"),
            "ang1_RP": round(float(pa_RP[0]), 1) if len(pa_RP) > 0 else float("nan"),
            "ang1_LR": round(float(pa_LR[0]), 1) if len(pa_LR) > 0 else float("nan"),
            # Gate 3 outcome
            "SPLIT": sibling_diff,
        })

    return rows, tc


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print()

    for case_name in CASES:
        rows, tc = analyze_case(case_name)
        true_k = tc.get("n_clusters", "?")
        print(f"═══ {case_name} (true K={true_k}) ═══")

        if not rows:
            print("  No binary parents with ≥2 leaves on each side.\n")
            continue

        # Header
        hdr = (
            f"  {'Parent':<8} {'nL':>3} {'nR':>3} │"
            f" {'kL':>3} {'kR':>3} {'kP':>3} │"
            f" {'min':>3} {'max':>3} {'sum':>3} {'JL/4':>4} │"
            f" {'erL':>5} {'erR':>5} {'erP':>5} │"
            f" {'ovlLP':>5} {'ovlRP':>5} {'ovlLR':>5} │"
            f" {'∠LP':>5} {'∠RP':>5} {'∠LR':>5} │"
            f" {'Gate3':>5}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for r in rows:
            gate = "SPLIT" if r["SPLIT"] else "MERGE"
            print(
                f"  {r['parent']:<8} {r['n_L']:>3} {r['n_R']:>3} │"
                f" {r['k_L']:>3} {r['k_R']:>3} {r['k_P']:>3} │"
                f" {r['k_min']:>3} {r['k_max']:>3} {r['k_sum']:>3} {r['jl_qrt']:>4} │"
                f" {r['er_L']:>5.1f} {r['er_R']:>5.1f} {r['er_P']:>5.1f} │"
                f" {r['ovl_LP']:>5.3f} {r['ovl_RP']:>5.3f} {r['ovl_LR']:>5.3f} │"
                f" {r['ang1_LP']:>5.1f} {r['ang1_RP']:>5.1f} {r['ang1_LR']:>5.1f} │"
                f" {gate:>5}"
            )

        # Summary statistics for this case
        splits = [r for r in rows if r["SPLIT"]]
        merges = [r for r in rows if not r["SPLIT"]]

        print()
        print("  Summary:")
        print(f"    Nodes: {len(rows)} binary parents ({len(splits)} SPLIT, {len(merges)} MERGE)")

        if rows:
            k_Ps = [r["k_P"] for r in rows]
            k_sums = [r["k_sum"] for r in rows]
            k_maxs = [r["k_max"] for r in rows]
            ratios_sum = [r["k_P"] / r["k_sum"] if r["k_sum"] > 0 else 0 for r in rows]
            ratios_max = [r["k_P"] / r["k_max"] if r["k_max"] > 0 else 0 for r in rows]
            print(f"    k_P vs k_L+k_R: k_P/sum median={np.median(ratios_sum):.2f}, mean={np.mean(ratios_sum):.2f}")
            print(f"    k_P vs max(k_L,k_R): k_P/max median={np.median(ratios_max):.2f}, mean={np.mean(ratios_max):.2f}")

        if splits:
            ovl_LR_splits = [r["ovl_LR"] for r in splits if not np.isnan(r["ovl_LR"])]
            ang_LR_splits = [r["ang1_LR"] for r in splits if not np.isnan(r["ang1_LR"])]
            if ovl_LR_splits:
                print(f"    SPLIT nodes: sibling overlap median={np.median(ovl_LR_splits):.3f}, ∠LR median={np.median(ang_LR_splits):.1f}°")
        if merges:
            ovl_LR_merges = [r["ovl_LR"] for r in merges if not np.isnan(r["ovl_LR"])]
            ang_LR_merges = [r["ang1_LR"] for r in merges if not np.isnan(r["ang1_LR"])]
            if ovl_LR_merges:
                print(f"    MERGE nodes: sibling overlap median={np.median(ovl_LR_merges):.3f}, ∠LR median={np.median(ang_LR_merges):.1f}°")

        print()

    # Global cross-case analysis
    print("═══ GLOBAL CROSS-CASE ANALYSIS ═══")
    all_rows = []
    for case_name in CASES:
        rows, tc = analyze_case(case_name)
        for r in rows:
            r["case"] = case_name
        all_rows.extend(rows)

    splits_all = [r for r in all_rows if r["SPLIT"]]
    merges_all = [r for r in all_rows if not r["SPLIT"]]

    print(f"Total: {len(all_rows)} binary parents across {len(CASES)} cases")
    print(f"  SPLIT: {len(splits_all)}, MERGE: {len(merges_all)}")
    print()

    # k_P relationship to children
    print("k_P vs children (ALL nodes):")
    for label, subset in [("ALL", all_rows), ("SPLIT", splits_all), ("MERGE", merges_all)]:
        if not subset:
            continue
        rs = [r["k_P"] / r["k_sum"] if r["k_sum"] > 0 else 0 for r in subset]
        rm = [r["k_P"] / r["k_max"] if r["k_max"] > 0 else 0 for r in subset]
        print(f"  {label:>6}: k_P/sum = {np.median(rs):.2f} (med), {np.mean(rs):.2f} (mean)"
              f"  │  k_P/max = {np.median(rm):.2f} (med), {np.mean(rm):.2f} (mean)")

    # Sibling subspace overlap: SPLIT vs MERGE
    print()
    print("Sibling subspace overlap (ovl_LR) — SPLIT vs MERGE:")
    for label, subset in [("SPLIT", splits_all), ("MERGE", merges_all)]:
        vals = [r["ovl_LR"] for r in subset if not np.isnan(r["ovl_LR"])]
        if vals:
            print(f"  {label:>6}: median={np.median(vals):.3f}, mean={np.mean(vals):.3f}, "
                  f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    # First principal angle between siblings: SPLIT vs MERGE
    print()
    print("First principal angle ∠LR (degrees) — SPLIT vs MERGE:")
    for label, subset in [("SPLIT", splits_all), ("MERGE", merges_all)]:
        vals = [r["ang1_LR"] for r in subset if not np.isnan(r["ang1_LR"])]
        if vals:
            print(f"  {label:>6}: median={np.median(vals):.1f}°, mean={np.mean(vals):.1f}°, "
                  f"min={np.min(vals):.1f}°, max={np.max(vals):.1f}°")

    # Child-parent overlap: SPLIT vs MERGE
    print()
    print("Child→Parent subspace containment (ovl_LP, ovl_RP) — SPLIT vs MERGE:")
    for label, subset in [("SPLIT", splits_all), ("MERGE", merges_all)]:
        lp = [r["ovl_LP"] for r in subset if not np.isnan(r["ovl_LP"])]
        rp = [r["ovl_RP"] for r in subset if not np.isnan(r["ovl_RP"])]
        both = lp + rp
        if both:
            print(f"  {label:>6}: median={np.median(both):.3f}, mean={np.mean(both):.3f}, "
                  f"min={np.min(both):.3f}, max={np.max(both):.3f}")

    # Effective rank relationship
    print()
    print("Effective rank ratio (er_P / (er_L + er_R)) — SPLIT vs MERGE:")
    for label, subset in [("SPLIT", splits_all), ("MERGE", merges_all)]:
        vals = [r["er_P"] / (r["er_L"] + r["er_R"]) if (r["er_L"] + r["er_R"]) > 0 else 0
                for r in subset]
        if vals:
            print(f"  {label:>6}: median={np.median(vals):.2f}, mean={np.mean(vals):.2f}")

    # Can we predict Gate 3 from spectral features?
    print()
    print("── Spectral features as Gate 3 predictors ──")
    print("  (Which spectral feature best separates SPLIT from MERGE?)")
    features = {
        "ovl_LR": lambda r: r["ovl_LR"],
        "ang1_LR": lambda r: r["ang1_LR"],
        "k_P/k_sum": lambda r: r["k_P"] / r["k_sum"] if r["k_sum"] > 0 else 0,
        "k_P/k_max": lambda r: r["k_P"] / r["k_max"] if r["k_max"] > 0 else 0,
        "er_P/er_sum": lambda r: r["er_P"] / (r["er_L"] + r["er_R"]) if (r["er_L"] + r["er_R"]) > 0 else 0,
        "k_P": lambda r: r["k_P"],
        "k_sum": lambda r: r["k_sum"],
    }
    for fname, fn in features.items():
        s_vals = [fn(r) for r in splits_all if not np.isnan(fn(r))]
        m_vals = [fn(r) for r in merges_all if not np.isnan(fn(r))]
        if s_vals and m_vals:
            # Simple separation metric: |mean_S - mean_M| / (std_S + std_M + 1e-6)
            sep = abs(np.mean(s_vals) - np.mean(m_vals)) / (np.std(s_vals) + np.std(m_vals) + 1e-6)
            print(f"  {fname:<12}: SPLIT mean={np.mean(s_vals):.3f}  MERGE mean={np.mean(m_vals):.3f}  sep={sep:.3f}")
