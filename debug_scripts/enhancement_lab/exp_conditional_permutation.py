"""Conditional permutation test vs cousin-weighted Wald: diagnostic comparison.

For each sentinel case, runs the existing pipeline to get Wald p-values,
then re-tests every eligible sibling pair with a conditional permutation
test (4999 shuffles).  The permutation conditions on tree structure,
parent distribution, child sizes, and PCA projection space — only the
leaf assignment between L and R is shuffled.

If the Wald + calibration is well-calibrated, the two sets of p-values
should be rank-correlated and agree on BH decisions.  If they diverge,
the permutation test provides ground-truth p-values free of distributional
assumptions.

Usage:
    python debug_scripts/enhancement_lab/exp_conditional_permutation.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup (same pattern as other lab experiments)
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
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald import (  # noqa: E402
    run_projected_wald_kernel,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.pooled_variance import (  # noqa: E402
    standardize_proportion_difference,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_significant_sibling_pairs,
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

# Sentinel cases: mix of regression guards, intermediate, and failure cases
SENTINEL_CASES: list[str] = [
    # Regression guards (ARI = 1.0)
    "binary_low_noise_4c",
    "binary_perfect_4c",
    "binary_perfect_8c",
    "gauss_clear_large",
    "gauss_moderate_3c",
    "gauss_moderate_5c",
    "gauss_noisy_3c",
    "gauss_overlap_4c_med",
    # Intermediate cases (0.3 < ARI < 1.0)
    "binary_hard_4c",
    "binary_balanced_low_noise",
    "binary_low_noise_12c",
    "gauss_clear_small",
    # Null cases (K=1, should NOT reject)
    "binary_null_small",
    "gauss_null_small",
    # Failure case
    "binary_balanced_low_noise__2",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeComparison:
    """Per-node comparison between Wald and permutation p-values."""

    parent: str
    n_left: int
    n_right: int
    spectral_k: int
    t_obs: float
    p_wald_raw: float
    p_wald_bh: float
    p_perm: float


@dataclass
class CaseResult:
    """Aggregate result for one benchmark case."""

    case_name: str
    true_k: int | None
    found_k: int
    ari: float
    n_tested_nodes: int
    node_comparisons: list[NodeComparison] = field(default_factory=list)
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    bh_agreement_rate: float = float("nan")
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core: conditional permutation p-value
# ---------------------------------------------------------------------------


def _node_seed(parent: str, base_seed: int = 42) -> int:
    """Deterministic per-node seed from parent node ID."""
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
    """Compute the projected Wald T from raw binary leaf rows.

    Recomputes distributions from leaf data, then follows the standard
    pooled-variance → z-score → projected-Wald pipeline.
    """
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


def conditional_permutation_pvalue(
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
) -> tuple[float, float]:
    """Conditional permutation test for a sibling pair.

    Shuffles leaf assignments between L and R while keeping tree
    structure, parent distribution, child sizes, and PCA basis fixed.

    Returns (t_observed, permutation_p_value).
    """
    # Collect leaf labels for each child subtree
    left_labels = tree.get_leaves(left, return_labels=True)
    right_labels = tree.get_leaves(right, return_labels=True)

    left_data = leaf_data.loc[left_labels].values
    right_data = leaf_data.loc[right_labels].values

    n_left = left_data.shape[0]
    pooled = np.vstack([left_data, right_data])

    # Observed test statistic
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

    if not np.isfinite(t_obs):
        return float("nan"), float("nan")

    # Permutation loop
    rng = np.random.default_rng(_node_seed(parent))
    n_total = pooled.shape[0]
    count_ge = 0

    for _ in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_left = pooled[perm_idx[:n_left]]
        perm_right = pooled[perm_idx[n_left:]]

        t_perm = _compute_t_from_leaf_data(
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
        if np.isfinite(t_perm) and t_perm >= t_obs:
            count_ge += 1

    p_perm = (1 + count_ge) / (1 + n_permutations)
    return t_obs, p_perm


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------


def _extract_branch_length_sum(tree, parent: str, left: str, right: str) -> float | None:
    """Extract branch-length sum for a sibling pair."""
    bl_left = tree.edges[parent, left].get("branch_length")
    bl_right = tree.edges[parent, right].get("branch_length")
    if bl_left is not None and bl_right is not None:
        s = float(bl_left) + float(bl_right)
        return s if s > 0 else None
    return None


def run_case(case_name: str) -> CaseResult:
    """Run Wald pipeline + permutation test for one benchmark case."""
    t0 = time.time()

    # 1. Build tree, run full pipeline
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

    true_k = tc.get("n_clusters")
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")

    # 2. Extract Gate 2 output for sibling config
    sibling_dims = derive_sibling_spectral_dims(tree, annotations_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(annotations_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, annotations_df, sibling_dims)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    # 3. Identify tested sibling pairs from the pipeline
    parents, child_pairs, skipped, non_binary = collect_significant_sibling_pairs(
        tree,
        annotations_df,
    )

    # 4. Run permutation test for each eligible pair
    comparisons: list[NodeComparison] = []

    for parent, (left, right) in zip(parents, child_pairs, strict=False):
        # Wald results from the pipeline
        p_wald_raw = float(annotations_df.at[parent, "Sibling_Divergence_P_Value"])
        p_wald_bh = float(annotations_df.at[parent, "Sibling_Divergence_P_Value_Corrected"])
        t_obs_pipeline = float(annotations_df.at[parent, "Sibling_Test_Statistic"])

        _, _, n_left, n_right, _, _ = get_sibling_data(tree, parent, left, right)

        # Spectral config for this parent
        spec_k = sibling_dims.get(parent, 0) if sibling_dims else 0
        pca_proj = sibling_pca.get(parent) if sibling_pca else None
        pca_eig = sibling_eig.get(parent) if sibling_eig else None
        child_pca = sibling_child_pca.get(parent) if sibling_child_pca else None

        if spec_k <= 0 or pca_proj is None:
            continue

        bl_sum = (
            _extract_branch_length_sum(tree, parent, left, right) if mean_bl is not None else None
        )

        # Conditional permutation
        _, p_perm = conditional_permutation_pvalue(
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

        comparisons.append(
            NodeComparison(
                parent=parent,
                n_left=n_left,
                n_right=n_right,
                spectral_k=spec_k,
                t_obs=t_obs_pipeline,
                p_wald_raw=p_wald_raw,
                p_wald_bh=p_wald_bh,
                p_perm=p_perm,
            )
        )

    # 5. Aggregate statistics
    result = CaseResult(
        case_name=case_name,
        true_k=true_k,
        found_k=found_k,
        ari=ari,
        n_tested_nodes=len(comparisons),
        node_comparisons=comparisons,
        elapsed_seconds=time.time() - t0,
    )

    if len(comparisons) >= 2:
        raw_wald = [c.p_wald_raw for c in comparisons]
        perm = [c.p_perm for c in comparisons]
        finite_mask = [np.isfinite(w) and np.isfinite(p) for w, p in zip(raw_wald, perm)]
        finite_wald = [w for w, m in zip(raw_wald, finite_mask) if m]
        finite_perm = [p for p, m in zip(perm, finite_mask) if m]
        if len(finite_wald) >= 3:
            rho, sp = spearmanr(finite_wald, finite_perm)
            result.spearman_rho = float(rho)
            result.spearman_p = float(sp)

    # BH on permutation p-values for fair comparison
    if comparisons:
        perm_ps = np.array([c.p_perm for c in comparisons])
        perm_ps_safe = np.where(np.isfinite(perm_ps), perm_ps, 1.0)
        perm_reject, _, _ = benjamini_hochberg_correction(perm_ps_safe, alpha=config.SIBLING_ALPHA)

        wald_reject = np.array([c.p_wald_bh < config.SIBLING_ALPHA for c in comparisons])
        agree = np.sum(perm_reject == wald_reject)
        result.bh_agreement_rate = float(agree) / len(comparisons)

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_case_table(result: CaseResult) -> None:
    """Print per-node comparison table for one case."""
    hdr = (
        f"{'Parent':<12} {'nL':>4} {'nR':>4} {'k':>3} "
        f"{'T_obs':>8} {'p_wald':>8} {'p_w_bh':>8} "
        f"{'p_perm':>8} {'agree':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for c in result.node_comparisons:
        agree = (
            "Y"
            if ((c.p_wald_bh < config.SIBLING_ALPHA) == (c.p_perm < config.SIBLING_ALPHA))
            else "N"
        )
        print(
            f"{c.parent:<12} {c.n_left:>4} {c.n_right:>4} {c.spectral_k:>3} "
            f"{c.t_obs:>8.2f} {c.p_wald_raw:>8.4f} {c.p_wald_bh:>8.4f} "
            f"{c.p_perm:>8.4f} {agree:>6}"
        )

    print()


def print_summary_table(results: Sequence[CaseResult]) -> None:
    """Print aggregate summary across all cases."""
    hdr = (
        f"{'Case':<35} {'TK':>3} {'FK':>3} {'ARI':>6} "
        f"{'Nodes':>5} {'Spearman':>8} {'BH_agr':>7} {'Time':>6}"
    )
    print("\n" + "=" * len(hdr))
    print("SUMMARY: Conditional Permutation vs Cousin-Weighted Wald")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"Sibling method: {config.SIBLING_TEST_METHOD}")
    print(f"Sibling alpha: {config.SIBLING_ALPHA}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    total_nodes = 0
    total_agree = 0
    all_rhos: list[float] = []

    for r in results:
        rho_str = f"{r.spearman_rho:>8.3f}" if np.isfinite(r.spearman_rho) else "     n/a"
        bh_str = f"{r.bh_agreement_rate:>7.1%}" if np.isfinite(r.bh_agreement_rate) else "    n/a"
        print(
            f"{r.case_name:<35} {r.true_k or 0:>3} {r.found_k:>3} {r.ari:>6.3f} "
            f"{r.n_tested_nodes:>5} {rho_str} {bh_str} {r.elapsed_seconds:>5.1f}s"
        )
        total_nodes += r.n_tested_nodes
        if np.isfinite(r.bh_agreement_rate) and r.n_tested_nodes > 0:
            total_agree += int(round(r.bh_agreement_rate * r.n_tested_nodes))
        if np.isfinite(r.spearman_rho):
            all_rhos.append(r.spearman_rho)

    print("-" * len(hdr))
    overall_agree = total_agree / total_nodes if total_nodes > 0 else float("nan")
    mean_rho = float(np.mean(all_rhos)) if all_rhos else float("nan")
    total_time = sum(r.elapsed_seconds for r in results)
    print(
        f"{'TOTAL':<35} {'':>3} {'':>3} {'':>6} "
        f"{total_nodes:>5} {mean_rho:>8.3f} {overall_agree:>7.1%} {total_time:>5.1f}s"
    )

    # Pooled scatter: all (p_wald_raw, p_perm) pairs
    all_wald = []
    all_perm = []
    for r in results:
        for c in r.node_comparisons:
            if np.isfinite(c.p_wald_raw) and np.isfinite(c.p_perm):
                all_wald.append(c.p_wald_raw)
                all_perm.append(c.p_perm)

    if len(all_wald) >= 3:
        rho_pool, sp_pool = spearmanr(all_wald, all_perm)
        print(f"\nPooled Spearman (n={len(all_wald)}): rho={rho_pool:.3f}, p={sp_pool:.4f}")

    # Type I / II error analysis
    all_comparisons = [
        c
        for r in results
        for c in r.node_comparisons
        if np.isfinite(c.p_perm) and np.isfinite(c.p_wald_bh)
    ]
    if all_comparisons:
        wald_rej = sum(1 for c in all_comparisons if c.p_wald_bh < config.SIBLING_ALPHA)
        perm_rej = sum(1 for c in all_comparisons if c.p_perm < config.SIBLING_ALPHA)
        both_rej = sum(
            1
            for c in all_comparisons
            if c.p_wald_bh < config.SIBLING_ALPHA and c.p_perm < config.SIBLING_ALPHA
        )
        n = len(all_comparisons)
        print(f"\nDecision concordance (n={n} nodes):")
        print(f"  Wald rejects:       {wald_rej:>4} ({wald_rej/n:.1%})")
        print(f"  Permutation rejects: {perm_rej:>4} ({perm_rej/n:.1%})")
        print(f"  Both reject:         {both_rej:>4} ({both_rej/n:.1%})")
        print(f"  Either rejects:      {wald_rej + perm_rej - both_rej:>4}")
        if wald_rej > 0:
            print(f"  Wald-only rejects:   {wald_rej - both_rej:>4} (potential false positives)")
        if perm_rej > 0:
            print(f"  Perm-only rejects:   {perm_rej - both_rej:>4} (potential Wald misses)")

    print()


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
    print(f"Cases: {len(SENTINEL_CASES)}\n")

    results: list[CaseResult] = []

    for i, case_name in enumerate(SENTINEL_CASES, 1):
        print(f"[{i}/{len(SENTINEL_CASES)}] {case_name}")
        try:
            result = run_case(case_name)
            results.append(result)
            print(
                f"  K={result.found_k}/{result.true_k}, ARI={result.ari:.3f}, "
                f"nodes={result.n_tested_nodes}, "
                f"rho={result.spearman_rho:.3f}, "
                f"BH_agree={result.bh_agreement_rate:.1%}, "
                f"time={result.elapsed_seconds:.1f}s"
            )

            if result.node_comparisons:
                print()
                print_case_table(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
