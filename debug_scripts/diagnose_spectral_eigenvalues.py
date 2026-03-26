#!/usr/bin/env python3
"""
Diagnose spectral decomposition (eigenvalues & eigenvectors) for a benchmark case.

For a given test case, this script:
  1. Generates the binarized data and builds the tree.
  2. Runs Gate 2 (edge) and Gate 3 (sibling) spectral decomposition.
  3. At the ROOT node, prints:
     - Eigenvalue spectrum (top-20)
     - Effective rank
     - Variance explained (cumulative)
     - Primal vs dual form used
     - PCA eigenvector loadings (top features)
  4. At the ROOT node, prints Gate 3 (sibling) spectral:
     - Pooled within-cluster eigenvalue spectrum
     - Within-cluster effective rank vs overall effective rank
  5. Compares eigenspectra per cluster (do eigenvalues differ by cluster?)
  6. Shows the z-vector components at root and projected Wald components.

Usage:
    python debug_scripts/diagnose_spectral_eigenvalues.py [case_name]

    Default case: gauss_clear_small
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.sibling_spectral_dimension import (
    compute_sibling_spectral_dimensions,
)

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    build_pca_projection_backend,
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.eigen_result import (
    EigenResult,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.chi2_pvalue import (
    compute_projected_pvalue,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    effective_rank,
    estimate_k_marchenko_pastur,
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_helpers import (
    precompute_descendants,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_eigenvalue_table(eigenvalues: np.ndarray, label: str, max_rows: int = 20) -> None:
    """Print eigenvalue spectrum with cumulative variance explained."""
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    total = eigs.sum()
    print(f"\n  {label} — top {min(len(eigs), max_rows)} eigenvalues (total={total:.4f}):")
    print(f"  {'#':>4}  {'λ':>10}  {'%var':>7}  {'cum%':>7}  {'bar'}")
    cum = 0.0
    for i, lam in enumerate(eigs[:max_rows]):
        pct = 100 * lam / total if total > 0 else 0.0
        cum += pct
        bar_len = int(pct / 2)
        print(f"  {i+1:>4}  {lam:>10.4f}  {pct:>6.2f}%  {cum:>6.2f}%  {'█' * bar_len}")
    if len(eigs) > max_rows:
        rest_var = 100 * eigs[max_rows:].sum() / total if total > 0 else 0.0
        print(f"  ... ({len(eigs) - max_rows} more, remaining {rest_var:.2f}% variance)")


def print_eigenvector_loadings(
    eig: EigenResult,
    feature_names: list[str] | None = None,
    top_k_vectors: int = 3,
    top_n_features: int = 8,
) -> None:
    """Show top feature loadings for leading eigenvectors."""
    if eig.use_dual:
        if eig.dual_sample_eigenvectors is None or eig.standardized_data_active is None:
            print("  (Dual form — eigenvectors not available)")
            return
        # Recover d-space eigenvectors from dual form
        k_avail = min(top_k_vectors, eig.dual_sample_eigenvectors.shape[1])
        top_eigs = np.maximum(eig.eigenvalues[:k_avail], 1e-12)
        vecs_active = (
            eig.standardized_data_active.T
            @ eig.dual_sample_eigenvectors[:, :k_avail]
            / (np.sqrt(top_eigs) * np.sqrt(eig.active_feature_count))
        )
        norms = np.linalg.norm(vecs_active, axis=0)
        norms[norms == 0] = 1.0
        vecs_active = vecs_active / norms
    else:
        if eig.eigenvectors_active is None:
            print("  (Primal form — eigenvectors not computed)")
            return
        k_avail = min(top_k_vectors, eig.eigenvectors_active.shape[1])
        vecs_active = eig.eigenvectors_active[:, :k_avail]

    # Map back to feature names via active_mask
    active_indices = np.where(eig.is_active_feature)[0]

    print(f"\n  Top-{top_n_features} feature loadings for leading {k_avail} PCs:")
    for pc_idx in range(k_avail):
        vec = vecs_active[:, pc_idx]
        # Sort by absolute loading
        sorted_idx = np.argsort(np.abs(vec))[::-1][:top_n_features]
        print(f"\n    PC{pc_idx+1} (λ={eig.eigenvalues[pc_idx]:.4f}):")
        for j in sorted_idx:
            feat_idx = active_indices[j]
            fname = feature_names[feat_idx] if feature_names else f"F{feat_idx}"
            print(f"      {fname:>12s}: {vec[j]:+.4f}")


def analyze_root_gate2(
    tree: PosetTree,
    leaf_data,
    root: str,
    children: list[str],
    X: np.ndarray,
    label_to_idx: dict,
    feature_names: list[str],
) -> None:
    """Analyze Gate 2 spectral decomposition at the root."""
    print_header("Gate 2 — Edge Test Spectral Decomposition (Root)")

    # Get descendant leaf indices for root
    desc_indices, _ = precompute_descendants(tree, label_to_idx)
    root_idx = desc_indices.get(root, [])
    data_sub = X[root_idx, :]

    print(f"  Root: {root}")
    print(f"  Descendants: n={len(root_idx)}, d={data_sub.shape[1]}")

    # Eigendecompose
    eig = eigendecompose_correlation_backend(data_sub, compute_eigenvectors=True)
    if eig is None:
        print("  ERROR: Eigendecomposition returned None (< 2 active features)")
        return

    print(f"  Form: {'DUAL (n×n Gram)' if eig.use_dual else 'PRIMAL (d×d correlation)'}")
    print(f"  Active features: {eig.active_feature_count} / {data_sub.shape[1]}")

    # Effective rank
    erank = effective_rank(eig.eigenvalues)
    mp_count = marchenko_pastur_signal_count(
        eig.eigenvalues, len(root_idx), eig.active_feature_count
    )  # positional: eigenvalues, n_samples, n_features
    k_used = estimate_k_marchenko_pastur(
        eig.eigenvalues,
        n_samples=len(root_idx),
        n_features=eig.active_feature_count,
        minimum_projection_dimension=(
            config.PROJECTION_MIN_K if isinstance(config.PROJECTION_MIN_K, int) else 4
        ),
    )

    print(f"  Effective rank: {erank:.2f}")
    print(f"  Marchenko-Pastur signal count: {mp_count}")
    print(f"  k used (with floor): {k_used}")

    print_eigenvalue_table(eig.eigenvalues, "Root overall correlation")
    print_eigenvector_loadings(eig, feature_names=feature_names)

    # Per-child spectral decomposition
    for child in children:
        child_idx = desc_indices.get(child, [])
        if len(child_idx) < 2:
            continue
        child_data = X[child_idx, :]
        child_eig = eigendecompose_correlation_backend(child_data, compute_eigenvectors=False)
        if child_eig is None:
            continue
        child_erank = effective_rank(child_eig.eigenvalues)
        print(
            f"\n  Child {child}: n={len(child_idx)}, d_active={child_eig.active_feature_count}, "
            f"erank={child_erank:.2f}, form={'dual' if child_eig.use_dual else 'primal'}"
        )
        print_eigenvalue_table(child_eig.eigenvalues, f"Child {child} correlation")


def analyze_root_gate3(
    tree: PosetTree,
    leaf_data,
    root: str,
    children: list[str],
    X: np.ndarray,
    label_to_idx: dict,
) -> None:
    """Analyze Gate 3 sibling spectral decomposition at the root."""
    print_header("Gate 3 — Sibling Test Spectral Decomposition (Root)")

    desc_indices, _ = precompute_descendants(tree, label_to_idx)

    if len(children) != 2:
        print(f"  Root has {len(children)} children (non-binary), skipping.")
        return

    left, right = children[0], children[1]
    left_idx = desc_indices.get(left, [])
    right_idx = desc_indices.get(right, [])

    print(f"  Left child: {left} (n={len(left_idx)})")
    print(f"  Right child: {right} (n={len(right_idx)})")

    if len(left_idx) < 2 or len(right_idx) < 2:
        print("  Too few samples in one child, skipping pooled within-cluster analysis.")
        return

    # Replicate the pooled within-cluster computation
    left_rows = X[left_idx, :]
    right_rows = X[right_idx, :]
    resid_left = left_rows - left_rows.mean(axis=0)
    resid_right = right_rows - right_rows.mean(axis=0)
    pooled_resid = np.vstack([resid_left, resid_right])

    print(f"  Pooled residuals shape: {pooled_resid.shape}")

    eig_pooled = eigendecompose_correlation_backend(pooled_resid, compute_eigenvectors=True)
    if eig_pooled is None:
        print("  ERROR: Pooled eigendecomposition returned None")
        return

    erank_pooled = effective_rank(eig_pooled.eigenvalues)
    k_pooled = estimate_k_marchenko_pastur(
        eig_pooled.eigenvalues,
        n_samples=pooled_resid.shape[0],
        n_features=eig_pooled.active_feature_count,
        minimum_projection_dimension=(
            config.PROJECTION_MIN_K if isinstance(config.PROJECTION_MIN_K, int) else 4
        ),
    )

    print(f"  Form: {'DUAL' if eig_pooled.use_dual else 'PRIMAL'}")
    print(f"  Active features (pooled): {eig_pooled.active_feature_count}")
    print(f"  Pooled within-cluster effective rank: {erank_pooled:.2f}")
    print(f"  k_pooled (with floor): {k_pooled}")

    print_eigenvalue_table(eig_pooled.eigenvalues, "Pooled within-cluster correlation")

    # Compare: overall root eig vs pooled
    root_idx = desc_indices.get(root, [])
    root_data = X[root_idx, :]
    eig_root = eigendecompose_correlation_backend(root_data, compute_eigenvectors=False)
    if eig_root is not None:
        erank_root = effective_rank(eig_root.eigenvalues)
        print("\n  Comparison:")
        print(f"    Overall root effective rank: {erank_root:.2f}")
        print(f"    Pooled within-cluster erank: {erank_pooled:.2f}")
        print(f"    Ratio (pooled/overall):      {erank_pooled / erank_root:.2f}")


def analyze_z_vector_and_wald(
    tree: PosetTree,
    leaf_data,
    root: str,
    children: list[str],
    X: np.ndarray,
    label_to_idx: dict,
) -> None:
    """Show the z-vector at root and the projected Wald test components."""
    print_header("Root z-vector and Projected Wald Components")

    desc_indices, _ = precompute_descendants(tree, label_to_idx)

    if len(children) != 2:
        return

    left, right = children[0], children[1]
    left_idx = desc_indices.get(left, [])
    right_idx = desc_indices.get(right, [])

    # Compute sibling z-vector: (θ_L - θ_R) / sqrt(Var)
    theta_left = X[left_idx, :].mean(axis=0)
    theta_right = X[right_idx, :].mean(axis=0)
    n_left = len(left_idx)
    n_right = len(right_idx)

    # Pooled proportion for variance
    theta_pool = (n_left * theta_left + n_right * theta_right) / (n_left + n_right)
    var = theta_pool * (1 - theta_pool) * (1.0 / n_left + 1.0 / n_right)
    var_safe = np.where(var > 0, var, 1.0)

    z = (theta_left - theta_right) / np.sqrt(var_safe)
    z[var <= 0] = 0.0  # features with no variance

    print(f"  z-vector (sibling diff, d={len(z)}):")
    print(f"    Non-zero entries: {np.count_nonzero(z)} / {len(z)}")
    print(f"    ||z||² = {np.sum(z**2):.4f}")
    print(f"    max|z| = {np.max(np.abs(z)):.4f}")
    print(f"    mean|z| = {np.mean(np.abs(z)):.4f}")

    # Show top-10 z components
    top_z_idx = np.argsort(np.abs(z))[::-1][:10]
    print("\n  Top-10 |z| components:")
    for j in top_z_idx:
        print(
            f"    F{j:>4d}: z={z[j]:+.4f}  θ_L={theta_left[j]:.3f}  θ_R={theta_right[j]:.3f}  θ_pool={theta_pool[j]:.3f}"
        )

    # Project z-vector via the same path used by the pipeline
    # Gate 3 uses pooled within-cluster spectral dimension
    left_rows = X[left_idx, :]
    right_rows = X[right_idx, :]
    resid_left = left_rows - left_rows.mean(axis=0)
    resid_right = right_rows - right_rows.mean(axis=0)
    pooled_resid = np.vstack([resid_left, resid_right])

    eig_pooled = eigendecompose_correlation_backend(pooled_resid, compute_eigenvectors=True)
    if eig_pooled is None:
        print("  Cannot compute pooled eigendecomposition for projection.")
        return

    erank_pooled = effective_rank(eig_pooled.eigenvalues)
    k = max(int(np.round(erank_pooled)), 4)
    k = min(k, eig_pooled.active_feature_count)

    proj, ev = build_pca_projection_backend(
        eig_pooled, projection_dimension=k, n_features_total=len(z)
    )
    if proj is None:
        print("  PCA projection matrix unavailable.")
        return

    # Project z
    projected = proj @ z
    print(f"\n  Projection: k={k}, proj.shape={proj.shape}")
    print(f"  Projected vector w = R·z (length {len(projected)}):")
    for i in range(min(k, 10)):
        wt = "whitened" if config.EIGENVALUE_WHITENING else "raw"
        w_sq_lam = (
            projected[i] ** 2 / ev[i] if ev is not None and i < len(ev) else projected[i] ** 2
        )
        print(
            f"    w[{i}] = {projected[i]:+.4f}  w²={projected[i]**2:.4f}  "
            f"λ={ev[i]:.4f}  w²/λ={w_sq_lam:.4f}"
        )

    # Compute test stat and p-value
    stat, eff_df, pval = compute_projected_pvalue(projected, k, eigenvalues=ev)
    print(
        f"\n  Projected Wald test ({('whitened' if config.EIGENVALUE_WHITENING else 'Satterthwaite')}):"
    )
    print(f"    T = {stat:.4f}")
    print(f"    df = {eff_df:.1f}")
    print(f"    p = {pval:.6f}")
    print(f"    Significant at α=0.05? {'YES' if pval < 0.05 else 'NO'}")


def analyze_cluster_eigenspectra(
    X: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Compare eigenvalue spectra across true clusters."""
    print_header("Per-Cluster Eigenvalue Spectra (True Labels)")

    labels = sorted(np.unique(y_true))
    for label in labels:
        mask = y_true == label
        cluster_data = X[mask, :]
        n_c = cluster_data.shape[0]

        if n_c < 3:
            print(f"\n  Cluster {int(label)}: n={n_c} (too few for eigendecomposition)")
            continue

        eig = eigendecompose_correlation_backend(cluster_data, compute_eigenvectors=False)
        if eig is None:
            print(f"\n  Cluster {int(label)}: n={n_c}, eigendecomposition failed")
            continue

        erank = effective_rank(eig.eigenvalues)
        top5 = eig.eigenvalues[:5]
        print(
            f"\n  Cluster {int(label)}: n={n_c}, d_active={eig.active_feature_count}, erank={erank:.2f}, "
            f"form={'dual' if eig.use_dual else 'primal'}"
        )
        print(f"    Top-5 λ: [{', '.join(f'{v:.3f}' for v in top5)}]")
        print(f"    Top-5 cumvar: {100 * top5.sum() / eig.eigenvalues.sum():.1f}%")


def analyze_all_nodes_spectral(
    tree: PosetTree,
    leaf_data,
    stats,
    X: np.ndarray,
    label_to_idx: dict,
) -> None:
    """Show spectral k vs Gate 2/3 outcomes for all internal nodes (summary table)."""
    print_header("Spectral Dimensions vs Gate Outcomes (All Internal Nodes)")

    # Get the spectral dimensions from the tree decomposition
    dims, _, _ = compute_spectral_decomposition(
        tree,
        leaf_data,
        method="effective_rank",
        min_k=config.PROJECTION_MIN_K if isinstance(config.PROJECTION_MIN_K, int) else 4,
        compute_projections=False,
    )
    sib_dims = compute_sibling_spectral_dimensions(
        tree,
        leaf_data,
        method="effective_rank",
        min_k=config.PROJECTION_MIN_K if isinstance(config.PROJECTION_MIN_K, int) else 4,
    )

    internal = sorted([n for n in tree.nodes() if tree.out_degree(n) > 0])
    desc_indices, _ = precompute_descendants(tree, label_to_idx)

    print(
        f"\n  {'Node':>8} {'n':>5} {'d_act':>5} {'k_G2':>5} {'k_G3':>5} "
        f"{'G2_sig':>6} {'G3_diff':>7} {'G3_p':>10} {'G3_stat':>8}"
    )
    print(f"  {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*10} {'-'*8}")

    for n in internal[:30]:  # cap at 30 rows
        ch = list(tree.successors(n))
        n_desc = len(desc_indices.get(n, []))
        k_g2 = dims.get(n, -1)
        k_g3 = sib_dims.get(n, -1)

        # Gate 2: either child passes
        g2_sig = False
        for c in ch:
            if c in stats.index and "Child_Parent_Divergence_Significant" in stats.columns:
                if stats.loc[c, "Child_Parent_Divergence_Significant"]:
                    g2_sig = True
                    break

        # Gate 3
        g3_diff = ""
        g3_p = ""
        g3_stat = ""
        if n in stats.index:
            skipped = (
                stats.loc[n, "Sibling_Divergence_Skipped"]
                if "Sibling_Divergence_Skipped" in stats.columns
                else True
            )
            if not skipped:
                g3_diff = str(bool(stats.loc[n, "Sibling_BH_Different"]))
                g3_p = (
                    f"{stats.loc[n, 'Sibling_Divergence_P_Value_Corrected']:.6f}"
                    if "Sibling_Divergence_P_Value_Corrected" in stats.columns
                    else "?"
                )
                g3_stat = (
                    f"{stats.loc[n, 'Sibling_Test_Statistic']:.2f}"
                    if "Sibling_Test_Statistic" in stats.columns
                    else "?"
                )
            else:
                g3_diff = "skip"

        # d_active
        data_sub = X[desc_indices.get(n, []), :]
        d_active = int(np.sum(np.var(data_sub, axis=0) > 0)) if data_sub.shape[0] > 1 else 0

        print(
            f"  {n:>8} {n_desc:>5} {d_active:>5} {k_g2:>5} {k_g3:>5} "
            f"{'T' if g2_sig else 'F':>6} {g3_diff:>7} {g3_p:>10} {g3_stat:>8}"
        )

    if len(internal) > 30:
        print(f"  ... ({len(internal) - 30} more nodes omitted)")


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else "gauss_clear_small"

    # ── Find case ──
    all_cases = get_default_test_cases()
    matches = [c for c in all_cases if c["name"] == case_name]
    if not matches:
        print(f"Case '{case_name}' not found. Available:")
        for c in sorted(all_cases, key=lambda c: c["name"]):
            print(f"  {c['name']}")
        sys.exit(1)
    tc = matches[0]

    print_header(f"Spectral Eigenvalue Diagnosis: {case_name}")
    print(
        f"  Config: EIGENVALUE_WHITENING={config.EIGENVALUE_WHITENING}, "
        "SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)"
    )
    print(
        f"  Config: PROJECTION_MIN_K={config.PROJECTION_MIN_K}, "
        f"PROJECTION_EPS={config.PROJECTION_EPS}"
    )

    for k, v in tc.items():
        if k != "generator":
            print(f"  {k}: {v}")

    # ── Generate data ──
    data_t, y_t, x_original, meta = generate_case_data(tc)
    X = data_t.values.astype(np.float64)
    feature_names = list(data_t.columns)
    label_to_idx = {label: i for i, label in enumerate(data_t.index)}

    print(f"\n  Data: {data_t.shape[0]} samples × {data_t.shape[1]} features")
    print(f"  True K: {len(np.unique(y_t))}")
    print(f"  Sparsity: {1 - X.mean():.3f}")

    # ── Build tree ──
    dist = pdist(X, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())

    # ── Run decomposition to populate stats ──
    decomp = tree.decompose(
        leaf_data=data_t,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
        passthrough=False,
    )
    stats = tree.annotations_df
    print(f"\n  Decomposition result: K={decomp['num_clusters']}")

    # ── Root info ──
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children = list(tree.successors(root))

    # === Section 1: Gate 2 spectral at root ===
    analyze_root_gate2(tree, data_t, root, children, X, label_to_idx, feature_names)

    # === Section 2: Gate 3 sibling spectral at root ===
    analyze_root_gate3(tree, data_t, root, children, X, label_to_idx)

    # === Section 3: z-vector and projected Wald components ===
    analyze_z_vector_and_wald(tree, data_t, root, children, X, label_to_idx)

    # === Section 4: Per-cluster eigenspectra ===
    analyze_cluster_eigenspectra(X, y_t)

    # === Section 5: All-nodes summary table ===
    analyze_all_nodes_spectral(tree, data_t, stats, X, label_to_idx)

    print_header("END")


if __name__ == "__main__":
    main()
