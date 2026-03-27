"""Anatomy of the ĉ deflation: why a single global constant fails.

For each sibling pair, computes:
  1. T_obs / k  — the raw ratio (what the Wald sees)
  2. mean(T_perm) / k  — the per-node inflation factor from permutation null
  3. ĉ (global)  — the weighted-mean inflation the pipeline applies

If inflation is heterogeneous (varies by node), a single ĉ cannot work:
  - Nodes with low inflation get over-deflated  → lost power (Wald misses)
  - Nodes with high inflation get under-deflated → false positives

Usage:
    python debug_scripts/enhancement_lab/exp_inflation_anatomy.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from lab_helpers import build_tree_and_data, compute_ari, run_decomposition  # noqa: E402

from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald import (  # noqa: E402
    run_projected_wald_kernel,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.pooled_variance import (  # noqa: E402
    standardize_proportion_difference,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
    get_sibling_data,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_child_pca_projections,
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 4999

# Focus on cases where ĉ matters (signal cases with K > 1)
DIAGNOSTIC_CASES: list[str] = [
    "gauss_moderate_3c",  # K=3, BH agree=40% — worst disagreement
    "gauss_noisy_3c",  # K=3, BH agree=60%
    "gauss_clear_small",  # K=3, BH agree=67%
    "binary_perfect_8c",  # K=8, Wald K=1 — total collapse
    "binary_hard_4c",  # K=4, some disagreement
    "gauss_overlap_4c_med",  # K=4, 44 nodes — large sample
    "gauss_null_small",  # K=1, perm over-rejects — null calibration
    "binary_balanced_low_noise",  # K=4, Wald K=1
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeInflation:
    """Per-node inflation anatomy."""

    parent: str
    n_left: int
    n_right: int
    n_parent: int
    spectral_k: int
    tree_depth: int  # depth from root
    t_obs: float  # observed test statistic
    ratio_obs: float  # T_obs / k
    c_perm_mean: float  # mean(T_perm) / k — per-node inflation
    c_perm_median: float  # median(T_perm) / k
    c_perm_std: float  # std(T_perm / k) — variability of inflation
    p_perm: float  # permutation p-value
    p_wald_raw: float  # raw Wald p-value (from pipeline)
    p_wald_bh: float  # BH-corrected Wald p-value
    edge_weight: float  # min(p_edge_L, p_edge_R) — calibration weight


@dataclass
class CaseInflation:
    """Inflation analysis for one case."""

    case_name: str
    true_k: int | None
    found_k: int
    ari: float
    c_hat_global: float  # the pipeline's ĉ
    nodes: list[NodeInflation] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core: permutation with distribution capture
# ---------------------------------------------------------------------------


def _node_seed(parent: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(parent.encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def _compute_t_from_leaf_data(
    leaf_data_left: np.ndarray,
    leaf_data_right: np.ndarray,
    mean_branch_length: float | None,
    branch_length_sum: float | None,
    spectral_k: int,
    pca_projection: np.ndarray | None,
    pca_eigenvalues: np.ndarray | None,
    child_pca_projections: list[np.ndarray] | None,
    whitening: str,
) -> float:
    n_left = leaf_data_left.shape[0]
    n_right = leaf_data_right.shape[0]
    if n_left < 1 or n_right < 1:
        return float("nan")

    theta_left = leaf_data_left.mean(axis=0)
    theta_right = leaf_data_right.mean(axis=0)

    z_scores, _ = standardize_proportion_difference(
        theta_left,
        theta_right,
        float(n_left),
        float(n_right),
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
    )
    if not np.isfinite(z_scores).all():
        return float("nan")

    t_stat, _k, _df, _p = run_projected_wald_kernel(
        z_scores.astype(np.float64),
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        whitening=whitening,
    )
    return t_stat


def permutation_null_distribution(
    tree,
    parent: str,
    left: str,
    right: str,
    leaf_data: pd.DataFrame,
    *,
    mean_branch_length: float | None,
    branch_length_sum: float | None,
    spectral_k: int,
    pca_projection: np.ndarray | None,
    pca_eigenvalues: np.ndarray | None,
    child_pca_projections: list[np.ndarray] | None,
    whitening: str = "per_component",
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, np.ndarray]:
    """Return (t_obs, array of T_perm values) for full distribution analysis."""
    left_labels = tree.get_leaves(left, return_labels=True)
    right_labels = tree.get_leaves(right, return_labels=True)

    left_data = leaf_data.loc[left_labels].values
    right_data = leaf_data.loc[right_labels].values
    n_left = left_data.shape[0]
    pooled = np.vstack([left_data, right_data])

    t_obs = _compute_t_from_leaf_data(
        left_data,
        right_data,
        mean_branch_length,
        branch_length_sum,
        spectral_k,
        pca_projection,
        pca_eigenvalues,
        child_pca_projections,
        whitening,
    )

    rng = np.random.default_rng(_node_seed(parent))
    n_total = pooled.shape[0]
    t_perms = np.empty(n_permutations)

    for i in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_left = pooled[perm_idx[:n_left]]
        perm_right = pooled[perm_idx[n_left:]]

        t_perms[i] = _compute_t_from_leaf_data(
            perm_left,
            perm_right,
            mean_branch_length,
            branch_length_sum,
            spectral_k,
            pca_projection,
            pca_eigenvalues,
            child_pca_projections,
            whitening,
        )

    return t_obs, t_perms


def _extract_branch_length_sum(tree, parent: str, left: str, right: str) -> float | None:
    bl_left = tree.edges[parent, left].get("branch_length")
    bl_right = tree.edges[parent, right].get("branch_length")
    if bl_left is not None and bl_right is not None:
        s = float(bl_left) + float(bl_right)
        return s if s > 0 else None
    return None


def _node_depth(tree, node: str) -> int:
    """Depth from root (root = 0)."""
    root = tree.root
    try:
        import networkx as nx

        path = nx.shortest_path(tree, root, node)
        return len(path) - 1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------


def run_case(case_name: str) -> CaseInflation:
    t0 = time.time()

    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

    true_k = tc.get("n_clusters")
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")

    # Extract global ĉ from the calibration audit
    audit = annotations_df.attrs.get("sibling_divergence_audit", {})
    c_hat_global = float(audit.get("global_inflation_factor", 1.0))

    # Gate 2 config for sibling test
    sibling_dims = derive_sibling_spectral_dims(tree, annotations_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(annotations_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, annotations_df, sibling_dims)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # Identify tested sibling pairs
    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length=mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca,
        pca_eigenvalues=sibling_eig,
    )

    nodes: list[NodeInflation] = []

    for rec in records:
        parent, left, right = rec.parent, rec.left, rec.right
        # Pipeline results
        p_wald_raw = float(annotations_df.at[parent, "Sibling_Divergence_P_Value"])
        p_wald_bh = float(annotations_df.at[parent, "Sibling_Divergence_P_Value_Corrected"])
        t_obs_pipeline = float(annotations_df.at[parent, "Sibling_Test_Statistic"])

        _, _, n_left, n_right, _, _ = get_sibling_data(tree, parent, left, right)

        spec_k = sibling_dims.get(parent, 0) if sibling_dims else 0
        pca_proj = sibling_pca.get(parent) if sibling_pca else None
        pca_eig = sibling_eig.get(parent) if sibling_eig else None
        child_pca = sibling_child_pca.get(parent) if sibling_child_pca else None

        if spec_k <= 0 or pca_proj is None:
            continue

        bl_sum = (
            _extract_branch_length_sum(tree, parent, left, right) if mean_bl is not None else None
        )

        # Extract edge weight for this parent
        edge_p_l = float(annotations_df.at[left, "Child_Parent_Divergence_P_Value_BH"])
        edge_p_r = float(annotations_df.at[right, "Child_Parent_Divergence_P_Value_BH"])
        edge_weight = min(edge_p_l, edge_p_r)

        # Full permutation null distribution
        t_obs, t_perms = permutation_null_distribution(
            tree,
            parent,
            left,
            right,
            data_df,
            mean_branch_length=mean_bl,
            branch_length_sum=bl_sum,
            spectral_k=spec_k,
            pca_projection=pca_proj,
            pca_eigenvalues=pca_eig,
            whitening=config.SIBLING_WHITENING,
            n_permutations=N_PERMUTATIONS,
        )

        finite_perms = t_perms[np.isfinite(t_perms)]
        if len(finite_perms) == 0:
            continue

        c_perm_mean = float(np.mean(finite_perms)) / spec_k
        c_perm_median = float(np.median(finite_perms)) / spec_k
        c_perm_std = float(np.std(finite_perms / spec_k))

        count_ge = np.sum(finite_perms >= t_obs) if np.isfinite(t_obs) else 0
        p_perm = (1 + count_ge) / (1 + N_PERMUTATIONS)

        depth = _node_depth(tree, parent)

        nodes.append(
            NodeInflation(
                parent=parent,
                n_left=n_left,
                n_right=n_right,
                n_parent=n_left + n_right,
                spectral_k=spec_k,
                tree_depth=depth,
                t_obs=t_obs_pipeline,
                ratio_obs=t_obs_pipeline / spec_k,
                c_perm_mean=c_perm_mean,
                c_perm_median=c_perm_median,
                c_perm_std=c_perm_std,
                p_perm=p_perm,
                p_wald_raw=p_wald_raw,
                p_wald_bh=p_wald_bh,
                edge_weight=edge_weight,
            )
        )

    result = CaseInflation(
        case_name=case_name,
        true_k=true_k,
        found_k=found_k,
        ari=ari,
        c_hat_global=c_hat_global,
        nodes=nodes,
        elapsed_seconds=time.time() - t0,
    )
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_inflation_table(result: CaseInflation) -> None:
    """Print per-node inflation anatomy."""
    hdr = (
        f"{'Parent':<10} {'nP':>4} {'k':>3} {'dep':>3} "
        f"{'T/k':>7} {'ĉ_perm':>7} {'ĉ_med':>7} {'σ_c':>6} "
        f"{'ĉ_glob':>7} {'Δ':>7} "
        f"{'p_wald':>8} {'p_perm':>8} {'ew':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in sorted(result.nodes, key=lambda x: x.n_parent):
        delta = result.c_hat_global - n.c_perm_mean  # positive = over-deflated
        print(
            f"{n.parent:<10} {n.n_parent:>4} {n.spectral_k:>3} {n.tree_depth:>3} "
            f"{n.ratio_obs:>7.2f} {n.c_perm_mean:>7.3f} {n.c_perm_median:>7.3f} {n.c_perm_std:>6.3f} "
            f"{result.c_hat_global:>7.3f} {delta:>+7.3f} "
            f"{n.p_wald_raw:>8.4f} {n.p_perm:>8.4f} {n.edge_weight:>6.3f}"
        )


def print_analysis(results: list[CaseInflation]) -> None:
    """Print cross-case inflation heterogeneity analysis."""
    print("\n" + "=" * 90)
    print("INFLATION HETEROGENEITY ANALYSIS")
    print("=" * 90)

    all_c_perm = []
    all_c_global = []
    all_deltas = []
    all_depths = []
    all_n_parents = []
    all_edge_weights = []
    over_deflated = 0  # ĉ > c_perm (killed power)
    under_deflated = 0  # ĉ < c_perm (missed inflation)

    for r in results:
        for n in r.nodes:
            c_perm = n.c_perm_mean
            delta = r.c_hat_global - c_perm
            all_c_perm.append(c_perm)
            all_c_global.append(r.c_hat_global)
            all_deltas.append(delta)
            all_depths.append(n.tree_depth)
            all_n_parents.append(n.n_parent)
            all_edge_weights.append(n.edge_weight)
            if delta > 0.1:
                over_deflated += 1
            elif delta < -0.1:
                under_deflated += 1

    n_total = len(all_c_perm)
    if n_total == 0:
        print("No nodes to analyze.")
        return

    all_c_perm = np.array(all_c_perm)
    all_deltas = np.array(all_deltas)
    all_depths = np.array(all_depths)
    all_n_parents = np.array(all_n_parents)
    all_edge_weights = np.array(all_edge_weights)

    print(f"\nTotal nodes: {n_total}")
    print(
        f"Per-node c_perm:  mean={np.mean(all_c_perm):.3f}, "
        f"std={np.std(all_c_perm):.3f}, "
        f"range=[{np.min(all_c_perm):.3f}, {np.max(all_c_perm):.3f}]"
    )
    print(f"Delta (ĉ - c_perm): mean={np.mean(all_deltas):+.3f}, " f"std={np.std(all_deltas):.3f}")
    print(
        f"Over-deflated (ĉ > c_perm + 0.1): {over_deflated}/{n_total} ({over_deflated/n_total:.1%})"
    )
    print(
        f"Under-deflated (ĉ < c_perm - 0.1): {under_deflated}/{n_total} ({under_deflated/n_total:.1%})"
    )

    # Correlation: does c_perm depend on node properties?
    print("\n--- Correlations with per-node inflation (c_perm_mean) ---")

    for name, values in [
        ("tree_depth", all_depths),
        ("n_parent", all_n_parents),
        ("log(n_parent)", np.log(all_n_parents + 1)),
        ("edge_weight", all_edge_weights),
    ]:
        finite_mask = np.isfinite(values) & np.isfinite(all_c_perm)
        if finite_mask.sum() >= 3:
            rho, p = spearmanr(values[finite_mask], all_c_perm[finite_mask])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<20}: rho={rho:+.3f}, p={p:.4f} {sig}")

    # Per-case summary
    print("\n--- Per-case inflation summary ---")
    hdr = f"{'Case':<30} {'ĉ_glob':>7} {'c̄_perm':>7} {'σ_perm':>7} {'range':>14} {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if not r.nodes:
            continue
        c_perms = [n.c_perm_mean for n in r.nodes]
        print(
            f"{r.case_name:<30} {r.c_hat_global:>7.3f} "
            f"{np.mean(c_perms):>7.3f} {np.std(c_perms):>7.3f} "
            f"[{np.min(c_perms):.2f}, {np.max(c_perms):.2f}] "
            f"{len(c_perms):>4}"
        )

    # The key question: within each case, is c_perm constant?
    print("\n--- Intra-case heterogeneity (c_perm coefficient of variation) ---")
    for r in results:
        if len(r.nodes) < 3:
            continue
        c_perms = np.array([n.c_perm_mean for n in r.nodes])
        cv = np.std(c_perms) / np.mean(c_perms)
        rng = np.max(c_perms) - np.min(c_perms)
        print(
            f"  {r.case_name:<30}: CV={cv:.3f}, range={rng:.3f}, "
            f"min={np.min(c_perms):.3f}, max={np.max(c_perms):.3f}"
        )

    # Breakdown: small vs large parent nodes
    print("\n--- Inflation by parent sample size ---")
    for label, lo, hi in [
        ("n≤5", 0, 5),
        ("5<n≤20", 6, 20),
        ("20<n≤50", 21, 50),
        ("n>50", 51, 10000),
    ]:
        mask = (all_n_parents >= lo) & (all_n_parents <= hi)
        if mask.sum() > 0:
            c_sub = all_c_perm[mask]
            print(
                f"  {label:<10}: n={mask.sum():>3}, c̄_perm={np.mean(c_sub):.3f}, "
                f"std={np.std(c_sub):.3f}, range=[{np.min(c_sub):.3f}, {np.max(c_sub):.3f}]"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(
        f"Config: METHOD={config.SIBLING_TEST_METHOD}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, "
        f"EDGE_ALPHA={config.EDGE_ALPHA}"
    )
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"Cases: {len(DIAGNOSTIC_CASES)}\n")

    results: list[CaseInflation] = []

    for i, case_name in enumerate(DIAGNOSTIC_CASES, 1):
        print(f"\n[{i}/{len(DIAGNOSTIC_CASES)}] {case_name}")
        result = run_case(case_name)
        results.append(result)

        print(
            f"  K={result.found_k}/{result.true_k}, ARI={result.ari:.3f}, "
            f"ĉ_global={result.c_hat_global:.3f}, nodes={len(result.nodes)}, "
            f"time={result.elapsed_seconds:.1f}s"
        )

        if result.nodes:
            print()
            print_inflation_table(result)

    print_analysis(results)


if __name__ == "__main__":
    main()
