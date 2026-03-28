"""Parametric inflation model: can c(n) replace permutations?

Hypothesis: post-selection inflation c scales as a power law with n_parent:
    c(n) = α · n^(-β)

If this holds, we can predict per-node c analytically from node properties
instead of using a single global ĉ or expensive permutations.

This experiment:
1. Uses the inflation anatomy data (permutation ground truth) to validate
2. Fits c = α · n^(-β) from null-like pairs within each case
3. Predicts c for focal pairs and computes corrected p-values
4. Compares against: (a) global ĉ pipeline, (b) permutation ground truth

Usage:
    python debug_scripts/enhancement_lab/exp_parametric_inflation.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2, spearmanr

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
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 4999

DIAGNOSTIC_CASES: list[str] = [
    "gauss_moderate_3c",
    "gauss_noisy_3c",
    "gauss_clear_small",
    "binary_perfect_8c",
    "binary_hard_4c",
    "gauss_overlap_4c_med",
    "gauss_null_small",
    "binary_balanced_low_noise",
]


# ---------------------------------------------------------------------------
# Permutation infrastructure (reused from exp_inflation_anatomy)
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


def permutation_c(
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
) -> tuple[float, float, float]:
    """Return (c_perm_mean, c_perm_median, p_perm) for one node."""
    left_labels = tree.get_leaves(left, return_labels=True)
    right_labels = tree.get_leaves(right, return_labels=True)

    left_data = leaf_data.loc[left_labels].values
    right_data = leaf_data.loc[right_labels].values
    n_left = left_data.shape[0]
    pooled = np.vstack([left_data, right_data])
    n_total = pooled.shape[0]

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
    t_perms = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        t_perms[i] = _compute_t_from_leaf_data(
            pooled[perm_idx[:n_left]],
            pooled[perm_idx[n_left:]],
            mean_branch_length,
            branch_length_sum,
            spectral_k,
            pca_projection,
            pca_eigenvalues,
            child_pca_projections,
            whitening,
        )

    finite = t_perms[np.isfinite(t_perms)]
    if len(finite) == 0:
        return 1.0, 1.0, 1.0

    c_mean = float(np.mean(finite)) / spectral_k
    c_median = float(np.median(finite)) / spectral_k
    count_ge = np.sum(finite >= t_obs) if np.isfinite(t_obs) else 0
    p_perm = (1 + count_ge) / (1 + n_permutations)
    return c_mean, c_median, p_perm


# ---------------------------------------------------------------------------
# Parametric model: c(n) = alpha * n^(-beta)
# ---------------------------------------------------------------------------
def _power_law(n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha * np.power(n, -beta)


@dataclass
class ParametricModel:
    alpha: float
    beta: float
    r_squared: float
    n_calibration: int

    def predict(self, n_parent: int) -> float:
        return self.alpha * n_parent ** (-self.beta)


def fit_power_law(n_parents: np.ndarray, c_values: np.ndarray) -> ParametricModel:
    """Fit c = alpha * n^(-beta) via log-linear OLS."""
    mask = (n_parents > 0) & (c_values > 0) & np.isfinite(c_values) & np.isfinite(n_parents)
    n_fit = n_parents[mask]
    c_fit = c_values[mask]

    if len(n_fit) < 2:
        return ParametricModel(
            alpha=float(np.median(c_fit)) if len(c_fit) > 0 else 1.0,
            beta=0.0,
            r_squared=0.0,
            n_calibration=len(n_fit),
        )

    # Log-linear: log(c) = log(alpha) - beta * log(n)
    log_n = np.log(n_fit)
    log_c = np.log(c_fit)
    # OLS
    A = np.column_stack([np.ones(len(log_n)), log_n])
    params, residuals, _, _ = np.linalg.lstsq(A, log_c, rcond=None)
    log_alpha, neg_beta = params
    alpha = np.exp(log_alpha)
    beta = -neg_beta

    # Also try scipy curve_fit for robustness
    try:
        popt, _ = curve_fit(_power_law, n_fit, c_fit, p0=[alpha, beta], maxfev=5000)
        alpha_cf, beta_cf = popt
        # Use curve_fit result if it converged to positive params
        if alpha_cf > 0 and beta_cf > 0:
            alpha, beta = alpha_cf, beta_cf
    except (RuntimeError, ValueError):
        pass

    # R-squared
    c_pred = alpha * n_fit ** (-beta)
    ss_res = np.sum((c_fit - c_pred) ** 2)
    ss_tot = np.sum((c_fit - np.mean(c_fit)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ParametricModel(
        alpha=float(alpha), beta=float(beta), r_squared=float(r2), n_calibration=len(n_fit)
    )


# ---------------------------------------------------------------------------
# Per-node result
# ---------------------------------------------------------------------------
@dataclass
class NodeResult:
    parent: str
    n_parent: int
    k: int
    t_obs: float
    ratio_obs: float  # T/k
    # Ground truth from permutation
    c_perm: float
    p_perm: float
    # Global ĉ method (current pipeline)
    c_global: float
    p_global: float
    # Parametric c(n) method (proposed)
    c_parametric: float
    p_parametric: float
    # Is this a null-like pair?
    is_null_like: bool
    edge_weight: float


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
def run_case(case_name: str) -> tuple[str, list[NodeResult], ParametricModel | None, float]:
    t0 = time.time()

    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

    true_k = tc.get("n_clusters")
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")

    # Global ĉ from pipeline
    audit = annotations_df.attrs.get("sibling_divergence_audit", {})
    c_hat_global = float(audit.get("global_inflation_factor", 1.0))

    # Collect all pair records (same as pipeline does)
    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    sibling_dims = derive_sibling_spectral_dims(tree, annotations_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(annotations_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, annotations_df, sibling_dims)

    records, _ = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca,
        pca_eigenvalues=sibling_eig,
        whitening=config.SIBLING_WHITENING,
    )

    # ─── Phase 1: Compute permutation ground truth for ALL valid pairs ───
    node_results: list[NodeResult] = []
    null_n_parents: list[float] = []
    null_c_perms: list[float] = []

    for rec in records:
        if rec.degrees_of_freedom <= 0 or not np.isfinite(rec.stat):
            continue

        k = rec.degrees_of_freedom
        children = list(tree.successors(rec.parent))
        if len(children) != 2:
            continue
        left, right = children

        pca_proj = sibling_pca.get(rec.parent) if sibling_pca else None
        pca_eig = sibling_eig.get(rec.parent) if sibling_eig else None
        child_pca = sibling_child_pca.get(rec.parent) if sibling_child_pca else None

        bl_left = tree.edges[rec.parent, left].get("branch_length")
        bl_right = tree.edges[rec.parent, right].get("branch_length")
        bl_sum = None
        if mean_bl is not None and bl_left is not None and bl_right is not None:
            s = float(bl_left) + float(bl_right)
            bl_sum = s if s > 0 else None

        c_perm_mean, c_perm_median, p_perm = permutation_c(
            tree,
            rec.parent,
            left,
            right,
            data_df,
            mean_branch_length=mean_bl,
            branch_length_sum=bl_sum,
            spectral_k=k,
            pca_projection=pca_proj,
            pca_eigenvalues=pca_eig,
            whitening=config.SIBLING_WHITENING,
        )

        # Global deflation p-value
        t_adj_global = rec.stat / c_hat_global if c_hat_global > 0 else rec.stat
        p_global = float(chi2.sf(t_adj_global, df=k))

        # Store null-like pairs for parametric fit
        if rec.is_null_like:
            null_n_parents.append(float(rec.n_parent))
            null_c_perms.append(c_perm_mean)

        node_results.append(
            NodeResult(
                parent=rec.parent,
                n_parent=rec.n_parent,
                k=k,
                t_obs=rec.stat,
                ratio_obs=rec.stat / k,
                c_perm=c_perm_mean,
                p_perm=p_perm,
                c_global=c_hat_global,
                p_global=p_global,
                is_null_like=rec.is_null_like,
                edge_weight=rec.edge_weight,
                # Placeholders — filled after parametric fit
                c_parametric=1.0,
                p_parametric=1.0,
            )
        )

    # ─── Phase 2: Fit parametric model from null-like pairs ───
    model = None
    if len(null_n_parents) >= 2:
        model = fit_power_law(np.array(null_n_parents), np.array(null_c_perms))

    # ─── Phase 3: Apply parametric model to ALL pairs ───
    for nr in node_results:
        if model is not None and model.alpha > 0:
            c_pred = model.predict(nr.n_parent)
            c_pred = max(c_pred, 1.0)  # never inflate
        else:
            # No model available — fall back to global
            c_pred = c_hat_global

        nr.c_parametric = c_pred
        t_adj_para = nr.t_obs / c_pred
        nr.p_parametric = float(chi2.sf(t_adj_para, df=nr.k))

    elapsed = time.time() - t0
    print(
        f"  K={found_k}/{true_k}, ARI={ari:.3f}, ĉ_global={c_hat_global:.3f}, "
        f"nodes={len(node_results)}, null-like={len(null_n_parents)}, "
        f"time={elapsed:.1f}s"
    )

    if model:
        print(
            f"  Model: c(n) = {model.alpha:.2f} * n^(-{model.beta:.3f}), "
            f"R²={model.r_squared:.3f}, calibration={model.n_calibration}"
        )

    return case_name, node_results, model, ari


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_results(all_results: list[tuple[str, list[NodeResult], ParametricModel | None, float]]):
    print("\n" + "=" * 100)
    print("COMPARISON: Global ĉ vs Parametric c(n) vs Permutation Ground Truth")
    print("=" * 100)

    # Per-case detail
    for case_name, nodes, model, ari in all_results:
        focal = [n for n in nodes if not n.is_null_like]
        if not focal:
            print(f"\n{case_name}: no focal pairs")
            continue

        print(f"\n{'─'*90}")
        print(f"Case: {case_name}  (ARI={ari:.3f})")
        if model:
            print(
                f"Model: c(n) = {model.alpha:.2f} × n^(-{model.beta:.3f}), R²={model.r_squared:.3f}"
            )
        else:
            print("Model: NO FIT (insufficient null-like pairs)")

        hdr = (
            f"{'Parent':<8} {'nP':>4} {'k':>3} {'T/k':>7} "
            f"{'c_perm':>7} {'c_glob':>7} {'c_para':>7} "
            f"{'p_perm':>8} {'p_glob':>8} {'p_para':>8}"
        )
        print(hdr)
        print("-" * len(hdr))

        for n in sorted(focal, key=lambda x: x.n_parent):
            print(
                f"{n.parent:<8} {n.n_parent:>4} {n.k:>3} {n.ratio_obs:>7.2f} "
                f"{n.c_perm:>7.3f} {n.c_global:>7.3f} {n.c_parametric:>7.3f} "
                f"{n.p_perm:>8.4f} {n.p_global:>8.4f} {n.p_parametric:>8.4f}"
            )

    # Aggregate: rejection agreement
    print(f"\n{'='*100}")
    print("AGGREGATE REJECTION ANALYSIS (focal pairs only, BH not applied)")
    print(f"{'='*100}")

    all_focal: list[NodeResult] = []
    for _, nodes, _, _ in all_results:
        all_focal.extend(n for n in nodes if not n.is_null_like)

    if not all_focal:
        print("No focal pairs to analyze.")
        return

    alpha = config.SIBLING_ALPHA

    perm_reject = np.array([n.p_perm < alpha for n in all_focal])
    glob_reject = np.array([n.p_global < alpha for n in all_focal])
    para_reject = np.array([n.p_parametric < alpha for n in all_focal])

    n_total = len(all_focal)
    print(f"\nTotal focal nodes: {n_total}, alpha = {alpha}")
    print(f"Permutation rejects: {perm_reject.sum()}/{n_total} ({perm_reject.mean():.1%})")
    print(f"Global ĉ rejects:    {glob_reject.sum()}/{n_total} ({glob_reject.mean():.1%})")
    print(f"Parametric rejects:  {para_reject.sum()}/{n_total} ({para_reject.mean():.1%})")

    # Agreement with permutation
    agree_glob = np.mean(perm_reject == glob_reject)
    agree_para = np.mean(perm_reject == para_reject)
    print("\nAgreement with permutation:")
    print(f"  Global ĉ:    {agree_glob:.1%}")
    print(f"  Parametric:  {agree_para:.1%}")

    # Breakdown: false negatives (perm rejects but method doesn't)
    fn_glob = np.sum(perm_reject & ~glob_reject)
    fn_para = np.sum(perm_reject & ~para_reject)
    fp_glob = np.sum(~perm_reject & glob_reject)
    fp_para = np.sum(~perm_reject & para_reject)
    print("\nFalse negatives (perm rejects, method misses):")
    print(f"  Global ĉ:    {fn_glob}/{n_total}")
    print(f"  Parametric:  {fn_para}/{n_total}")
    print("\nFalse positives (method rejects, perm does not):")
    print(f"  Global ĉ:    {fp_glob}/{n_total}")
    print(f"  Parametric:  {fp_para}/{n_total}")

    # c prediction accuracy
    c_perms = np.array([n.c_perm for n in all_focal])
    c_globs = np.array([n.c_global for n in all_focal])
    c_paras = np.array([n.c_parametric for n in all_focal])

    mae_glob = float(np.mean(np.abs(c_globs - c_perms)))
    mae_para = float(np.mean(np.abs(c_paras - c_perms)))
    rmse_glob = float(np.sqrt(np.mean((c_globs - c_perms) ** 2)))
    rmse_para = float(np.sqrt(np.mean((c_paras - c_perms) ** 2)))

    rho_glob, p_glob = spearmanr(c_globs, c_perms)
    rho_para, p_para = spearmanr(c_paras, c_perms)

    print("\nc prediction accuracy (focal pairs):")
    print(f"  {'Metric':<20} {'Global ĉ':>12} {'Parametric':>12}")
    print(f"  {'MAE':<20} {mae_glob:>12.3f} {mae_para:>12.3f}")
    print(f"  {'RMSE':<20} {rmse_glob:>12.3f} {rmse_para:>12.3f}")
    print(f"  {'Spearman ρ':<20} {rho_glob:>12.3f} {rho_para:>12.3f}")
    print(f"  {'Spearman p':<20} {p_glob:>12.4f} {p_para:>12.4f}")

    # Ratio analysis: how close is c_pred / c_perm to 1.0?
    ratio_glob = c_globs / c_perms
    ratio_para = c_paras / c_perms

    print("\nc_predicted / c_true ratio (1.0 = perfect):")
    print(f"  {'Metric':<20} {'Global ĉ':>12} {'Parametric':>12}")
    print(f"  {'mean ratio':<20} {np.mean(ratio_glob):>12.3f} {np.mean(ratio_para):>12.3f}")
    print(f"  {'median ratio':<20} {np.median(ratio_glob):>12.3f} {np.median(ratio_para):>12.3f}")
    print(f"  {'std ratio':<20} {np.std(ratio_glob):>12.3f} {np.std(ratio_para):>12.3f}")
    print(f"  {'max ratio':<20} {np.max(ratio_glob):>12.3f} {np.max(ratio_para):>12.3f}")
    print(f"  {'min ratio':<20} {np.min(ratio_glob):>12.3f} {np.min(ratio_para):>12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(
        f"Config: METHOD={config.SIBLING_TEST_METHOD}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, EDGE_ALPHA={config.EDGE_ALPHA}"
    )
    print(f"Permutations: {N_PERMUTATIONS}")
    print("Parametric model: c(n) = α × n^(-β) (fitted per-case from null-like pairs)")
    print(f"Cases: {len(DIAGNOSTIC_CASES)}\n")

    all_results = []
    for i, case_name in enumerate(DIAGNOSTIC_CASES, 1):
        print(f"[{i}/{len(DIAGNOSTIC_CASES)}] {case_name}")
        result = run_case(case_name)
        all_results.append(result)

    analyze_results(all_results)


if __name__ == "__main__":
    main()
