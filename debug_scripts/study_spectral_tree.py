#!/usr/bin/env python3
"""Study how eigenvalues and eigenvectors evolve along the hierarchical tree.

For each internal node, computes:
  - Correlation and covariance eigendecompositions side-by-side
  - Effective rank from each
  - Between-group direction δ = dist_L − dist_R at each split
  - Eigenvector alignment matrices (parent ↔ child cosine similarities)
  - δ loading: how much of the between-group direction each parent eigenvector captures
  - Between-within decomposition residual for covariance (exact) vs correlation (approximate)

Usage:
    python debug_scripts/study_spectral_tree.py [--clusters 4] [--n 100] [--d 50]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import norm as sp_norm

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    count_active_features,
    effective_rank,
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ── KAK-inspired k estimators ────────────────────────────────────────────────
#
# Quantum KAK decomposition: U = K₁ · A · K₂
#   K₁, K₂ = local (block-diagonal) rotations — don't mix qubit groups
#   A       = entangling part — the CNOTs that create correlations
#   CNOT count = how many axes are genuinely coupled
#
# For a correlation matrix C of d features:
#   Under H₀ (independence): C = I, all eigenvalues = 1 — zero entanglement
#   Under H₁: some eigenvalues ≠ 1 — features are entangled
#
# Two complementary ways to count "entangled modes":


def mp_two_sided_signal_count(
    eigenvalues: np.ndarray,
    n_desc: int,
    d_active: int,
    *,
    is_correlation: bool = True,
) -> int:
    """Count eigenvalues outside BOTH Marchenko-Pastur bounds.

    Standard MP counts only λ > upper_bound (modes with excess variance).
    The KAK insight: eigenvalues BELOW the lower bound also represent genuine
    structure — they are "anti-correlated" modes where the correlation matrix
    suppresses variance below what random noise would produce.

    In quantum terms: both |↑↑⟩+|↓↓⟩ (positive correlation, λ > 1) and
    |↑↓⟩+|↓↑⟩ (anti-correlation, λ < 1) require a CNOT to prepare.
    Only |↑⟩⊗|↑⟩ (λ = 1, product state) needs zero CNOTs.

    For correlation matrices, the null σ² = 1 (known), so we don't need
    to estimate it from the data — cleaner than standard MP.

    Parameters
    ----------
    eigenvalues : array of eigenvalues (descending)
    n_desc : number of samples
    d_active : number of active features
    is_correlation : if True, use σ² = 1 (exact for correlation matrices)
                     if False, estimate σ² from median (for covariance)
    """
    if n_desc <= 0 or d_active <= 0:
        return 1

    eigs = np.asarray(eigenvalues, dtype=np.float64)

    if is_correlation:
        sigma2 = 1.0  # Exact under H₀ for correlation matrices
    else:
        sigma2 = float(np.median(eigs[eigs > 0])) if np.any(eigs > 0) else 0.0
        if sigma2 <= 0:
            return 1

    q = float(d_active) / float(n_desc)
    sq = np.sqrt(q)
    mp_upper = sigma2 * (1.0 + sq) ** 2
    mp_lower = sigma2 * max(1.0 - sq, 0.0) ** 2  # Can be 0 when d > n

    # Count eigenvalues outside the bulk
    above = np.sum(eigs > mp_upper)
    below = np.sum(eigs < mp_lower) if mp_lower > 0 else 0
    k = int(above + below)
    return max(k, 1)


def correlation_entanglement_rank(
    data_sub: np.ndarray,
    alpha: float = 0.05,
) -> int:
    """Count entangled feature groups via significant pairwise correlations.

    Analogy to quantum circuits:
    - Each feature = one qubit
    - Significant correlation between features i,j = a CNOT linking them
    - Connected component of correlated features = one entangled register
    - k = number of registers with ≥ 2 qubits (trivial 1-qubit registers
      are "local" — they don't contribute entangled degrees of freedom)

    Algorithm:
    1. Fisher z-test each pairwise correlation ρ_ij for significance
    2. BH-correct across all d(d-1)/2 tests
    3. Build correlation graph (edge if significant after BH)
    4. k = number of connected components with ≥ 2 nodes

    This is the most direct operationalization of "how many axes are
    correlated with each other" — it counts correlation communities,
    not eigenvalue magnitudes.
    """
    data = np.asarray(data_sub, dtype=np.float64)
    n, d = data.shape

    # Only active features
    col_var = np.var(data, axis=0)
    active_mask = col_var > 0
    active_idx = np.where(active_mask)[0]
    d_active = len(active_idx)
    if d_active < 2 or n < 4:
        return 1

    X = data[:, active_mask]
    C = np.corrcoef(X.T)

    # Fisher z-test for each pair with BH correction
    se = 1.0 / np.sqrt(max(n - 3, 1))
    n_pairs = d_active * (d_active - 1) // 2
    p_values = np.empty(n_pairs)
    pair_i = np.empty(n_pairs, dtype=int)
    pair_j = np.empty(n_pairs, dtype=int)

    idx = 0
    for i in range(d_active):
        for j in range(i + 1, d_active):
            r = np.clip(C[i, j], -0.9999, 0.9999)
            z = np.arctanh(r) / se
            p_values[idx] = 2.0 * (1.0 - sp_norm.cdf(abs(z)))
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1

    # Benjamini-Hochberg correction
    sorted_order = np.argsort(p_values)
    bh_thresholds = np.arange(1, n_pairs + 1) / n_pairs * alpha
    rejected = p_values[sorted_order] <= bh_thresholds
    significant = np.zeros(n_pairs, dtype=bool)
    if np.any(rejected):
        max_reject = np.where(rejected)[0][-1]
        significant[sorted_order[: max_reject + 1]] = True

    # Build adjacency lists and find connected components via BFS
    adj = [[] for _ in range(d_active)]
    for idx in range(n_pairs):
        if significant[idx]:
            adj[pair_i[idx]].append(pair_j[idx])
            adj[pair_j[idx]].append(pair_i[idx])

    visited = np.zeros(d_active, dtype=bool)
    n_entangled_groups = 0
    for start in range(d_active):
        if visited[start] or len(adj[start]) == 0:
            continue
        # BFS
        component_size = 0
        queue = [start]
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            component_size += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
        if component_size >= 2:
            n_entangled_groups += 1

    return max(n_entangled_groups, 1)


# ── Helpers ──────────────────────────────────────────────────────────────────


def true_signal_rank(tree, node_id, labels, label_to_idx):
    """Number of distinct true clusters below node_id, minus 1.

    This is the theoretical rank of the between-cluster covariance matrix:
    K_v distinct clusters → rank K_v−1 between-cluster signal.
    Returns (K_v, K_v - 1) for display.
    """
    leaf_labels = tree.get_leaves(node=node_id, return_labels=True)
    leaf_indices = [label_to_idx[lbl] for lbl in leaf_labels if lbl in label_to_idx]
    cluster_ids = set(labels[leaf_indices])
    k_v = len(cluster_ids)
    return k_v, max(k_v - 1, 0)


def eigendecompose_covariance(data_sub: np.ndarray):
    """Eigendecompose the sample covariance matrix (not correlation).

    Returns (eigenvalues_desc, eigenvectors_cols_desc, d_active) or None.
    Eigenvectors are columns of the returned matrix, sorted by descending eigenvalue.
    """
    data_sub = np.asarray(data_sub, dtype=np.float64)
    col_var = np.var(data_sub, axis=0)
    active = col_var > 0
    d_active = int(np.sum(active))
    if d_active < 2:
        return None

    X = data_sub[:, active]
    X_centered = X - X.mean(axis=0)
    n = X_centered.shape[0]

    # Use dual form when beneficial
    if n < d_active:
        gram = X_centered @ X_centered.T / n
        evals, evecs_gram = np.linalg.eigh(gram)
        evals = evals[::-1]
        evecs_gram = evecs_gram[:, ::-1]
        evals = np.maximum(evals, 0.0)
        # Recover d-space eigenvectors
        k = int(np.sum(evals > 1e-12))
        if k == 0:
            return None
        top_evals = np.maximum(evals[:k], 1e-12)
        evecs_d = X_centered.T @ evecs_gram[:, :k] / (np.sqrt(top_evals) * np.sqrt(n))
        norms = np.linalg.norm(evecs_d, axis=0)
        norms[norms == 0] = 1.0
        evecs_d = evecs_d / norms
    else:
        cov = np.cov(X.T, ddof=0)
        evals, evecs_d = np.linalg.eigh(cov)
        evals = evals[::-1]
        evecs_d = evecs_d[:, ::-1]
        evals = np.maximum(evals, 0.0)
        k = evecs_d.shape[1]

    # Embed back into full d-space
    d = data_sub.shape[1]
    full_evecs = np.zeros((d, k), dtype=np.float64)
    full_evecs[active, :] = evecs_d[:, :k]

    return evals[:k], full_evecs, d_active, active


def between_within_residual(X_parent, X_left, X_right, matrix_type="covariance"):
    """Compute ||C_P - (w_L C_L + w_R C_R + between)||_F.

    For covariance: should be ~0 (machine epsilon).
    For correlation: should be > 0 (approximate).
    """
    n_p, n_l, n_r = X_parent.shape[0], X_left.shape[0], X_right.shape[0]
    w_l, w_r = n_l / n_p, n_r / n_p

    if matrix_type == "covariance":

        def mat(X):
            Xc = X - X.mean(axis=0)
            return Xc.T @ Xc / X.shape[0]

    else:  # correlation

        def mat(X):
            col_var = np.var(X, axis=0)
            active = col_var > 0
            d = X.shape[1]
            C = np.zeros((d, d))
            if np.sum(active) < 2:
                return C
            corr = np.corrcoef(X[:, active].T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            # Embed back
            idx = np.where(active)[0]
            C[np.ix_(idx, idx)] = corr
            return C

    C_P = mat(X_parent)
    C_L = mat(X_left)
    C_R = mat(X_right)

    mu_l = X_left.mean(axis=0)
    mu_r = X_right.mean(axis=0)
    delta = mu_l - mu_r

    between = (n_l * n_r / n_p**2) * np.outer(delta, delta)
    reconstructed = w_l * C_L + w_r * C_R + between

    residual = np.linalg.norm(C_P - reconstructed, "fro")
    norm_P = np.linalg.norm(C_P, "fro")
    relative = residual / norm_P if norm_P > 0 else 0.0

    return residual, relative


def fmt_vec(v, n=8):
    """Format a vector for display, showing top n values."""
    v = np.asarray(v)
    entries = [f"{x:.4f}" for x in v[:n]]
    if len(v) > n:
        entries.append("...")
    return "[" + ", ".join(entries) + "]"


def fmt_matrix(M, row_labels, col_labels, width=8):
    """Format a small matrix as a text table."""
    header = " " * (width + 2) + "".join(f"{c:>{width}s}" for c in col_labels)
    lines = [header]
    for i, rl in enumerate(row_labels):
        row = f"{rl:>{width}s}  " + "".join(f"{M[i, j]:{width}.3f}" for j in range(M.shape[1]))
        lines.append(row)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Study spectral structure along the tree")
    parser.add_argument("--clusters", "-k", type=int, default=4, help="Number of clusters")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--d", type=int, default=50, help="Number of features")
    parser.add_argument(
        "--entropy", type=float, default=0.1, help="Noise level (0=pure, 1=no structure)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=6, help="Number of eigenvectors to display")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth to display")
    args = parser.parse_args()

    K = args.clusters
    TOP_K = args.top_k

    # ── Generate data ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(
        f"Generating {K}-cluster block-diagonal data: n={args.n}, d={args.d}, "
        f"entropy={args.entropy}, seed={args.seed}"
    )
    print(f"{'='*70}\n")

    data_dict, cluster_assigns = generate_random_feature_matrix(
        n_rows=args.n,
        n_cols=args.d,
        entropy_param=args.entropy,
        n_clusters=K,
        random_seed=args.seed,
        balanced_clusters=True,
        feature_sparsity=0.05,
    )
    names = sorted(data_dict.keys())
    matrix = np.array([data_dict[name] for name in names], dtype=np.float64)
    labels = np.array([cluster_assigns[name] for name in names])
    data = pd.DataFrame(matrix, index=names, columns=[f"F{j}" for j in range(args.d)])

    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # ── Build tree ───────────────────────────────────────────────────────
    dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(leaf_data=data)

    root = tree.root()
    label_to_idx = {label: i for i, label in enumerate(data.index)}

    # ── BFS through the tree ─────────────────────────────────────────────
    # Store per-node results for parent-child comparisons
    node_results = {}  # node_id → { "corr_evals", "corr_evecs", "cov_evals", "cov_evecs", ... }

    queue = deque()
    queue.append((root, 0))  # (node_id, depth)

    while queue:
        node_id, depth = queue.popleft()

        if args.max_depth is not None and depth > args.max_depth:
            continue

        # Skip leaves
        children = list(tree.successors(node_id))
        if len(children) == 0:
            continue

        # Collect descendant leaf data
        leaf_labels = tree.get_leaves(node=node_id, return_labels=True)
        n_leaves = len(leaf_labels)
        if n_leaves < 2:
            continue

        leaf_indices = [label_to_idx[lbl] for lbl in leaf_labels if lbl in label_to_idx]
        X_node = matrix[leaf_indices, :]

        # ── True signal rank ──────────────────────────────────────────
        k_v, sig_rank = true_signal_rank(tree, node_id, labels, label_to_idx)

        # ── Count active features ─────────────────────────────────────
        d_active_node = count_active_features(X_node)

        # ── Correlation eigendecomposition ────────────────────────────
        corr_result = eigendecompose_correlation_backend(X_node, compute_eigenvectors=True)
        corr_evals, corr_evecs_full = None, None
        corr_erank = 0
        k_mp_corr = 1
        if corr_result is not None:
            k_avail = min(TOP_K, len(corr_result.eigenvalues))
            proj, evals = build_pca_projection_backend(
                corr_result, projection_dimension=len(corr_result.eigenvalues), n_features_total=args.d
            )
            if proj is not None and evals is not None:
                corr_evals = evals
                corr_evecs_full = proj  # (k × d), rows = eigenvectors
                corr_erank = effective_rank(corr_evals)
                k_mp_corr = marchenko_pastur_signal_count(
                    corr_evals, n_samples=n_leaves, n_features=d_active_node
                )

        # ── Covariance eigendecomposition ─────────────────────────────
        cov_result = eigendecompose_covariance(X_node)
        cov_evals, cov_evecs_full = None, None
        cov_erank = 0
        k_mp_cov = 1
        if cov_result is not None:
            cov_evals, cov_evecs_raw, _, _ = cov_result
            cov_evecs_full = cov_evecs_raw.T  # Convert to (k × d) row format
            cov_erank = effective_rank(cov_evals)
            k_mp_cov = marchenko_pastur_signal_count(
                cov_evals, n_samples=n_leaves, n_features=d_active_node
            )

        # ── KAK-inspired estimators ─────────────────────────────────
        k_mp2_corr = 1
        k_mp2_cov = 1
        if corr_evals is not None:
            k_mp2_corr = mp_two_sided_signal_count(
                corr_evals,
                n_desc=n_leaves,
                d_active=d_active_node,
                is_correlation=True,
            )
        if cov_evals is not None:
            k_mp2_cov = mp_two_sided_signal_count(
                cov_evals,
                n_desc=n_leaves,
                d_active=d_active_node,
                is_correlation=False,
            )
        k_entangle = correlation_entanglement_rank(X_node, alpha=0.05)

        node_results[node_id] = {
            "corr_evals": corr_evals,
            "corr_evecs": corr_evecs_full,  # (k × d)
            "cov_evals": cov_evals,
            "cov_evecs": cov_evecs_full,  # (k × d)
            "n_leaves": n_leaves,
            "depth": depth,
            "k_v": k_v,
            "sig_rank": sig_rank,
            "d_active": d_active_node,
            "corr_erank": corr_erank,
            "cov_erank": cov_erank,
            "k_mp_corr": k_mp_corr,
            "k_mp_cov": k_mp_cov,
            "k_mp2_corr": k_mp2_corr,
            "k_mp2_cov": k_mp2_cov,
            "k_entangle": k_entangle,
        }

        # ── Print node header ─────────────────────────────────────────
        is_root = node_id == root
        label = "root" if is_root else f"depth {depth}"
        print(f"\n{'='*70}")
        print(f"Node {node_id} ({label}, {n_leaves} leaves, {len(children)} children)")
        print(f"  True clusters below: K_v={k_v}, signal rank = {sig_rank}")
        print(f"  Active features: {d_active_node}")
        print(f"{'='*70}")

        if corr_evals is not None:
            print(f"\n  Eigenvalues (correlation): {fmt_vec(corr_evals, TOP_K)}")
            print(
                f"  Effective rank (corr):     {corr_erank:.2f}  →  k = {int(np.round(corr_erank))}"
            )
            print(f"  Marchenko-Pastur (corr):   k_MP = {k_mp_corr}")
        else:
            print("\n  Correlation eigendecomp: skipped (< 2 active features)")

        if cov_evals is not None:
            print(f"  Eigenvalues (covariance):  {fmt_vec(cov_evals, TOP_K)}")
            print(
                f"  Effective rank (cov):      {cov_erank:.2f}  →  k = {int(np.round(cov_erank))}"
            )
            print(f"  Marchenko-Pastur (cov):    k_MP = {k_mp_cov}")
        else:
            print("  Covariance eigendecomp:  skipped")

        print(f"\n  ┌─ k ESTIMATOR COMPARISON ({'corr' if corr_evals is not None else 'n/a'}) ─┐")
        print(f"  │  true signal rank     = {sig_rank:>4d}  (K_v={k_v})          │")
        if corr_evals is not None:
            print(
                f"  │  erank (corr)         = {int(np.round(corr_erank)):>4d}  (raw={corr_erank:.1f})       │"
            )
            print(f"  │  MP signal (corr)     = {k_mp_corr:>4d}                     │")
        if cov_evals is not None:
            print(
                f"  │  erank (cov)          = {int(np.round(cov_erank)):>4d}  (raw={cov_erank:.1f})       │"
            )
            print(f"  │  MP signal (cov)      = {k_mp_cov:>4d}                     │")
        print(f"  │  active features       = {d_active_node:>4d}                     │")
        print("  │  ── KAK-inspired ──────────────────────────── │")
        if corr_evals is not None:
            print(f"  │  MP 2-sided (corr)     = {k_mp2_corr:>4d}                     │")
        if cov_evals is not None:
            print(f"  │  MP 2-sided (cov)      = {k_mp2_cov:>4d}                     │")
        print(f"  │  corr-graph entangle   = {k_entangle:>4d}                     │")
        print("  └──────────────────────────────────────────────┘")

        # ── Per-child analysis ────────────────────────────────────────
        if len(children) == 2:
            left_id, right_id = children[0], children[1]

            # Get distributions
            dist_l = tree.nodes[left_id].get("distribution")
            dist_r = tree.nodes[right_id].get("distribution")

            if dist_l is not None and dist_r is not None:
                delta = np.asarray(dist_l, dtype=np.float64) - np.asarray(dist_r, dtype=np.float64)
                delta_norm = np.linalg.norm(delta)

                n_left = tree.nodes[left_id].get("leaf_count", 0)
                n_right = tree.nodes[right_id].get("leaf_count", 0)

                print(f"\n  Split → L={left_id} ({n_left} leaves), R={right_id} ({n_right} leaves)")
                print(f"  δ = dist_L − dist_R,  ||δ|| = {delta_norm:.4f}")

                # δ loading on parent eigenvectors
                if corr_evecs_full is not None and delta_norm > 1e-12:
                    delta_hat = delta / delta_norm
                    k_show = min(TOP_K, corr_evecs_full.shape[0])
                    loadings_corr = np.array(
                        [abs(np.dot(corr_evecs_full[i], delta_hat)) for i in range(k_show)]
                    )
                    print(
                        f"\n  δ loading on parent CORR eigenvecs: {fmt_vec(loadings_corr, TOP_K)}"
                    )
                    top_idx = np.argmax(loadings_corr)
                    print(
                        f"    → δ is mostly in eigenvector {top_idx+1} "
                        f"(loading = {loadings_corr[top_idx]:.3f})"
                    )

                if cov_evecs_full is not None and delta_norm > 1e-12:
                    delta_hat = delta / delta_norm
                    k_show = min(TOP_K, cov_evecs_full.shape[0])
                    loadings_cov = np.array(
                        [abs(np.dot(cov_evecs_full[i], delta_hat)) for i in range(k_show)]
                    )
                    print(f"  δ loading on parent COV eigenvecs:  {fmt_vec(loadings_cov, TOP_K)}")

                # Between-within decomposition residual
                left_labels = tree.get_leaves(node=left_id, return_labels=True)
                right_labels = tree.get_leaves(node=right_id, return_labels=True)
                left_idx = [label_to_idx[l] for l in left_labels if l in label_to_idx]
                right_idx = [label_to_idx[l] for l in right_labels if l in label_to_idx]
                X_left = matrix[left_idx, :]
                X_right = matrix[right_idx, :]

                res_cov, rel_cov = between_within_residual(X_node, X_left, X_right, "covariance")
                res_corr, rel_corr = between_within_residual(X_node, X_left, X_right, "correlation")

                print("\n  Between-within decomposition check:")
                print(
                    f"    Covariance:   residual = {res_cov:.2e}  (relative = {rel_cov:.2e})  "
                    f"{'✓ exact' if res_cov < 1e-10 else '✗ NOT exact'}"
                )
                print(
                    f"    Correlation:  residual = {res_corr:.2e}  (relative = {rel_corr:.2e})  "
                    f"{'≈ exact' if rel_corr < 0.01 else '✗ approximate'}"
                )

            # Enqueue children for further descent
            for child_id in children:
                queue.append((child_id, depth + 1))

        elif len(children) > 2:
            print(f"\n  Non-binary node ({len(children)} children) — skipping split analysis")
            for child_id in children:
                queue.append((child_id, depth + 1))

    # ── Parent ↔ Child eigenvector alignment ─────────────────────────────
    print(f"\n\n{'='*70}")
    print("EIGENVECTOR ALIGNMENT (Parent ↔ Child)")
    print(f"{'='*70}")

    for node_id, nr in node_results.items():
        children = list(tree.successors(node_id))
        if len(children) != 2:
            continue

        parent_evecs = nr["corr_evecs"]
        if parent_evecs is None:
            continue

        for child_id in children:
            if child_id not in node_results:
                continue
            child_evecs = node_results[child_id]["corr_evecs"]
            if child_evecs is None:
                continue

            k_p = min(TOP_K, parent_evecs.shape[0])
            k_c = min(TOP_K, child_evecs.shape[0])

            # Cosine similarity matrix
            alignment = np.zeros((k_p, k_c))
            for i in range(k_p):
                for j in range(k_c):
                    alignment[i, j] = abs(np.dot(parent_evecs[i], child_evecs[j]))

            n_child = node_results[child_id]["n_leaves"]
            print(f"\n  {node_id} → {child_id} ({n_child} leaves)")
            print("  |cos(parent_vi, child_vj)| matrix (correlation eigenvecs):")
            row_labels = [f"P_v{i+1}" for i in range(k_p)]
            col_labels = [f"C_v{i+1}" for i in range(k_c)]
            print("  " + fmt_matrix(alignment, row_labels, col_labels).replace("\n", "\n  "))

            # Find the parent eigenvector most aligned with δ and show it vanishes in child
            dist_l = tree.nodes[children[0]].get("distribution")
            dist_r = tree.nodes[children[1]].get("distribution")
            if dist_l is not None and dist_r is not None:
                delta = np.asarray(dist_l) - np.asarray(dist_r)
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-12:
                    delta_hat = delta / delta_norm
                    parent_delta_loadings = [
                        abs(np.dot(parent_evecs[i], delta_hat)) for i in range(k_p)
                    ]
                    delta_idx = int(np.argmax(parent_delta_loadings))
                    child_alignment_with_delta_vec = [alignment[delta_idx, j] for j in range(k_c)]
                    print(
                        f"  Parent's δ-carrying eigenvec (v{delta_idx+1}, "
                        f"load={parent_delta_loadings[delta_idx]:.3f}) "
                        f"aligns with child as: {fmt_vec(child_alignment_with_delta_vec, k_c)}"
                    )

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("SUMMARY: k Estimators Along Tree Depth")
    print(f"{'='*90}")
    print(
        f"\n  {'Node':<10s} {'Dp':>3s} {'n':>5s} {'K_v':>3s} "
        f"{'Sig':>3s} {'ER_c':>5s} {'MP_c':>5s} {'MP2c':>5s} "
        f"{'ER_v':>5s} {'MP_v':>5s} {'MP2v':>5s} {'Ent':>4s} {'d':>4s}"
    )
    print(
        f"  {'-'*10} {'-'*3} {'-'*5} {'-'*3} "
        f"{'-'*3} {'-'*5} {'-'*5} {'-'*5} "
        f"{'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*4}"
    )

    for node_id in sorted(node_results, key=lambda n: (node_results[n]["depth"], n)):
        nr = node_results[node_id]
        er_corr = nr["corr_erank"]
        er_cov = nr["cov_erank"]
        print(
            f"  {node_id:<10s} {nr['depth']:>3d} {nr['n_leaves']:>5d} "
            f"{nr['k_v']:>3d} {nr['sig_rank']:>3d} "
            f"{int(np.round(er_corr)):>5d} {nr['k_mp_corr']:>5d} {nr['k_mp2_corr']:>5d} "
            f"{int(np.round(er_cov)):>5d} {nr['k_mp_cov']:>5d} {nr['k_mp2_cov']:>5d} "
            f"{nr['k_entangle']:>4d} {nr['d_active']:>4d}"
        )

    # ── Error analysis ───────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("ERROR ANALYSIS: |k_estimated − signal_rank| per method")
    print(f"{'='*90}")

    methods = {
        "erank_corr": lambda nr: int(np.round(nr["corr_erank"])),
        "MP_corr": lambda nr: nr["k_mp_corr"],
        "MP2_corr": lambda nr: nr["k_mp2_corr"],
        "erank_cov": lambda nr: int(np.round(nr["cov_erank"])),
        "MP_cov": lambda nr: nr["k_mp_cov"],
        "MP2_cov": lambda nr: nr["k_mp2_cov"],
        "entangle": lambda nr: nr["k_entangle"],
    }

    errors = {m: [] for m in methods}
    for nr in node_results.values():
        truth = nr["sig_rank"]
        for m, fn in methods.items():
            errors[m].append(abs(fn(nr) - truth))

    print(f"\n  {'Method':<15s} {'MAE':>6s} {'Max':>6s} {'Median':>7s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7}")
    for m in methods:
        e = np.array(errors[m])
        print(f"  {m:<15s} {np.mean(e):>6.1f} {np.max(e):>6d} {np.median(e):>7.1f}")

    print()


if __name__ == "__main__":
    main()
