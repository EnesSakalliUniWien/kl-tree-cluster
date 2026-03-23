#!/usr/bin/env python3
"""Experiment 33: K-Adjusted Deflation — Full Benchmark Evaluation.

Runs each benchmark case under four deflation strategies:
  1. **intercept_only** — Current cousin_adjusted_wald (global ĉ for all pairs).
  2. **k_adj_all** — Per-node deflation c_i = ĉ · (k_i / k̄)^γ, where γ is
     inferred from ALL sibling pair T/k ratios via within-tree OLS.
  3. **k_adj_null** — Same formula but γ inferred from **null-like pairs only**
     (neither child edge-significant).  These have signal ≈ 0 so T/k ≈ c(k).
  4. **k_adj_weighted** — Same formula but γ from **edge-weight-weighted** OLS,
     where null-like pairs naturally dominate.

All methods use the SAME tree, SAME data, and SAME Gate 2 annotations.
Only the Gate 3 deflation step differs.

The experiment monkey-patches ``_deflate_and_test`` inside
``adjusted_wald_annotation.py`` to inject the k-adjusted calibration resolver,
avoiding any production code changes.

Output: CSV of (case, method, K_true, K_found, ARI, gamma_hat, c_hat) and a
summary markdown report.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    CalibrationModel,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    SiblingPairRecord,
    deflate_focal_pairs,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================


@dataclass
class InferredGamma:
    """Result of within-tree γ inference from sibling pair T/k ratios."""

    gamma: float
    se: float
    n_pairs: int
    r_squared: float
    log_k_mean: float  # geometric-mean anchor for centering


@dataclass
class CaseResult:
    """One (case × method) benchmark result."""

    case_name: str
    case_category: str
    n_clusters_true: int
    n_samples: int
    method: str
    n_clusters_found: int
    ari: float
    c_hat_global: float
    gamma_hat: float
    gamma_se: float
    gamma_r2: float
    gamma_n_pairs: int


@dataclass
class ExperimentSummary:
    """Aggregate statistics over all cases for one method."""

    method: str
    n_cases: int
    mean_ari: float
    median_ari: float
    exact_k_count: int
    k1_count: int
    mean_abs_k_error: float


# ============================================================================
# Gamma inference (no magic numbers — everything from data)
# ============================================================================


def infer_gamma_from_pairs(
    records: List[SiblingPairRecord],
    *,
    use_null_like_only: bool = False,
    use_edge_weights: bool = False,
) -> InferredGamma:
    """Infer the k-exponent γ from within-tree sibling pair T/k ratios.

    Fits the within-tree regression::

        log(T_i / k_i) = α + γ · log(k_i) + ε

    Parameters
    ----------
    records : list of SiblingPairRecord
        All sibling pair records from the tree.
    use_null_like_only : bool
        If True, only use null-like pairs (neither child edge-significant)
        for fit.  These pairs have signal ≈ 0, so T/k ≈ c purely, giving
        a cleaner estimate of the inflation k-slope without signal contamination.
    use_edge_weights : bool
        If True, weight each observation by edge_weight (= min(p_edge_L, p_edge_R)).
        Null-like pairs have high weights, signal pairs have low weights.
        This continuously attenuates the signal contribution.

    Returns InferredGamma with SE, R², and the geometric-mean log(k) anchor.
    When fewer than 3 valid pairs, returns γ=0 (no correction).
    """
    valid = [r for r in records if np.isfinite(r.stat) and r.degrees_of_freedom > 0 and r.stat > 0]

    if use_null_like_only:
        valid = [r for r in valid if r.is_null_like]

    if len(valid) < 3:
        return InferredGamma(
            gamma=0.0,
            se=math.nan,
            n_pairs=len(valid),
            r_squared=0.0,
            log_k_mean=0.0,
        )

    log_k = np.array([math.log(r.degrees_of_freedom) for r in valid])
    log_tk = np.array([math.log(r.stat / r.degrees_of_freedom) for r in valid])

    if use_edge_weights:
        w = np.array([max(r.edge_weight, 1e-12) for r in valid])
    else:
        w = np.ones(len(valid))

    # Weighted OLS: log(T/k) = α + γ·log(k)
    n = len(log_k)
    w_sum = w.sum()
    log_k_mean = float(np.average(log_k, weights=w))
    log_tk_mean = float(np.average(log_tk, weights=w))

    dk = log_k - log_k_mean
    dtk = log_tk - log_tk_mean

    ss_dk_w = float(np.sum(w * dk**2))

    if ss_dk_w < 1e-12:
        return InferredGamma(
            gamma=0.0,
            se=math.nan,
            n_pairs=n,
            r_squared=0.0,
            log_k_mean=log_k_mean,
        )

    gamma = float(np.sum(w * dk * dtk) / ss_dk_w)
    alpha = log_tk_mean - gamma * log_k_mean
    fitted = alpha + gamma * log_k
    residuals = log_tk - fitted
    ss_res_w = float(np.sum(w * residuals**2))
    ss_tot_w = float(np.sum(w * dtk**2))
    r_squared = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0
    # Effective sample size for weighted regression
    n_eff = float(w_sum**2 / np.sum(w**2)) if np.sum(w**2) > 0 else n
    mse = ss_res_w / max(n_eff - 2, 1)
    se = float(math.sqrt(mse / ss_dk_w)) if ss_dk_w > 0 else math.nan

    return InferredGamma(
        gamma=gamma,
        se=se,
        n_pairs=n,
        r_squared=r_squared,
        log_k_mean=log_k_mean,
    )


# ============================================================================
# K-adjusted calibration resolver
# ============================================================================


def make_k_adjusted_resolver(
    model: CalibrationModel,
    gamma_info: InferredGamma,
) -> Callable[[SiblingPairRecord], tuple[float, str]]:
    """Build a calibration resolver that applies per-node k-adjustment.

    For each focal pair, the resolved inflation factor is::

        c_i = ĉ_global · exp(γ · (log(k_i) - log(k̄)))

    which equals::

        c_i = ĉ_global · (k_i / k̄)^γ

    where γ and k̄ are inferred from the tree's own pair data.
    """
    c_global = model.global_inflation_factor
    gamma = gamma_info.gamma
    log_k_mean = gamma_info.log_k_mean

    def resolver(rec: SiblingPairRecord) -> tuple[float, str]:
        if rec.degrees_of_freedom <= 0:
            return c_global, "k_adjusted_fallback"

        log_k_i = math.log(rec.degrees_of_freedom)
        # Per-node inflation: c_global · (k_i / k̄)^γ
        c_i = c_global * math.exp(gamma * (log_k_i - log_k_mean))
        # Clamp: at least 1.0 (never inflate stat), at most max_observed
        c_i = max(c_i, 1.0)
        c_i = min(c_i, model.max_observed_ratio)
        return c_i, "k_adjusted"

    return resolver


# ============================================================================
# K-adjusted inference method specifications
# ============================================================================

# Each k-adjusted variant specifies how to call infer_gamma_from_pairs()
_K_ADJ_METHODS: dict[str, dict[str, bool]] = {
    "k_adj_all": {"use_null_like_only": False, "use_edge_weights": False},
    "k_adj_null": {"use_null_like_only": True, "use_edge_weights": False},
    "k_adj_weighted": {"use_null_like_only": False, "use_edge_weights": True},
}


# ============================================================================
# Run one case under all methods
# ============================================================================


def run_case(
    tc: dict,
    significance_level: float,
) -> list[CaseResult]:
    """Run one benchmark case under intercept_only and three k_adjusted variants.

    Builds the tree once, computes Gate 2 once for intercept_only, then applies
    each k-adjusted deflation strategy to fresh trees with the same linkage.
    """
    case_name = tc.get("name", "unnamed")
    true_k = int(tc.get("n_clusters", 0))
    category = tc.get("category", "")

    # Generate data
    data_df, y_true, _x_original, meta = generate_case_data(tc)
    distance_condensed = meta.get("precomputed_distance_condensed")
    if distance_condensed is not None:
        distance_condensed = np.asarray(distance_condensed, dtype=float)

    n_samples = len(data_df)

    # Build linkage (shared across all methods)
    metric = tc.get("tree_distance_metric", config.TREE_DISTANCE_METRIC)
    requires_precomputed = bool(
        tc.get("requires_precomputed_kl_distance", False)
        or meta.get("requires_precomputed_kl_distance", False)
    )

    if distance_condensed is not None:
        dist = distance_condensed
    elif requires_precomputed:
        logger.warning(
            "Case '%s' requires precomputed distance but none provided; skipping.", case_name
        )
        return []
    else:
        dist = pdist(data_df.values, metric=metric)

    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)

    results: list[CaseResult] = []

    # --- Method 1: intercept_only (current production) ---
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )
    labels = np.asarray(_labels_from_decomposition(decomp, data_df.index.tolist()))
    k_found = int(decomp.get("num_clusters", 0))
    ari = float(adjusted_rand_score(y_true, labels))

    audit = tree.stats_df.attrs.get("sibling_divergence_audit", {})
    c_hat = float(audit.get("global_inflation_factor", 1.0))

    results.append(
        CaseResult(
            case_name=case_name,
            case_category=category,
            n_clusters_true=true_k,
            n_samples=n_samples,
            method="intercept_only",
            n_clusters_found=k_found,
            ari=ari,
            c_hat_global=c_hat,
            gamma_hat=0.0,
            gamma_se=math.nan,
            gamma_r2=0.0,
            gamma_n_pairs=0,
        )
    )

    # --- Methods 2-4: k_adjusted variants ---
    import kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation as awa_mod

    original_deflate_and_test = awa_mod._deflate_and_test

    for method_name, gamma_kwargs in _K_ADJ_METHODS.items():
        tree_k = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

        captured_gamma_info: list[InferredGamma] = []

        def _make_patch(gkw: dict) -> Callable:
            """Create a patched _deflate_and_test for the given gamma kwargs."""

            def _patched(
                records: List[SiblingPairRecord],
                model: CalibrationModel,
            ) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
                gamma_info = infer_gamma_from_pairs(records, **gkw)
                captured_gamma_info.append(gamma_info)
                resolver = make_k_adjusted_resolver(model, gamma_info)
                return deflate_focal_pairs(records, calibration_resolver=resolver)

            return _patched

        try:
            awa_mod._deflate_and_test = _make_patch(gamma_kwargs)
            decomp_k = tree_k.decompose(
                leaf_data=data_df,
                alpha_local=significance_level,
                sibling_alpha=significance_level,
            )
        finally:
            awa_mod._deflate_and_test = original_deflate_and_test

        labels_k = np.asarray(_labels_from_decomposition(decomp_k, data_df.index.tolist()))
        k_found_k = int(decomp_k.get("num_clusters", 0))
        ari_k = float(adjusted_rand_score(y_true, labels_k))

        audit_k = tree_k.stats_df.attrs.get("sibling_divergence_audit", {})
        c_hat_k = float(audit_k.get("global_inflation_factor", 1.0))

        gi = (
            captured_gamma_info[0]
            if captured_gamma_info
            else InferredGamma(gamma=0.0, se=math.nan, n_pairs=0, r_squared=0.0, log_k_mean=0.0)
        )

        results.append(
            CaseResult(
                case_name=case_name,
                case_category=category,
                n_clusters_true=true_k,
                n_samples=n_samples,
                method=method_name,
                n_clusters_found=k_found_k,
                ari=ari_k,
                c_hat_global=c_hat_k,
                gamma_hat=gi.gamma,
                gamma_se=gi.se,
                gamma_r2=gi.r_squared,
                gamma_n_pairs=gi.n_pairs,
            )
        )

    return results


# ============================================================================
# Summarization
# ============================================================================


def summarize_results(df: pd.DataFrame) -> list[ExperimentSummary]:
    """Compute per-method aggregate statistics."""
    summaries: list[ExperimentSummary] = []
    for method, grp in df.groupby("method"):
        abs_k_err = (grp["n_clusters_found"] - grp["n_clusters_true"]).abs()
        summaries.append(
            ExperimentSummary(
                method=str(method),
                n_cases=len(grp),
                mean_ari=float(grp["ari"].mean()),
                median_ari=float(grp["ari"].median()),
                exact_k_count=int((grp["n_clusters_found"] == grp["n_clusters_true"]).sum()),
                k1_count=int((grp["n_clusters_found"] == 1).sum()),
                mean_abs_k_error=float(abs_k_err.mean()),
            )
        )
    return summaries


def write_report(
    df: pd.DataFrame,
    summaries: list[ExperimentSummary],
    output_path: Path,
) -> None:
    """Write a markdown summary report."""
    all_methods = sorted(df["method"].unique())
    k_adj_methods = [m for m in all_methods if m != "intercept_only"]

    lines: list[str] = [
        "# Experiment 33: K-Adjusted Deflation Benchmark",
        "",
        "## Method",
        "",
        "The k-adjusted deflation replaces the flat global ĉ with a per-node",
        "correction that accounts for k-dependence of post-selection inflation:",
        "",
        "$$",
        r"c_i = \hat{c}_{\text{global}} \cdot \left(\frac{k_i}{\bar{k}}\right)^{\gamma}",
        "$$",
        "",
        "where γ is **inferred from the tree's own sibling pair T/k ratios**",
        "via OLS regression of log(T/k) on log(k).  No hardcoded constants.",
        "",
        "### Variants",
        "",
        "| Method | γ Inference |",
        "|--------|-------------|",
        "| intercept_only | No k-adjustment (global ĉ, current production) |",
        "| k_adj_all | OLS on ALL sibling pairs |",
        "| k_adj_null | OLS on **null-like pairs only** (signal ≈ 0) |",
        "| k_adj_weighted | **Edge-weight-weighted** OLS (null-like pairs dominate) |",
        "",
    ]

    # Aggregate summary
    lines.extend(["## Aggregate Results", ""])
    lines.append("| Method | N | Mean ARI | Median ARI | Exact K | K=1 | MAE(K) |")
    lines.append("|--------|---|----------|------------|---------|-----|--------|")
    for s in sorted(summaries, key=lambda x: x.method):
        lines.append(
            f"| {s.method} | {s.n_cases} | {s.mean_ari:.4f} "
            f"| {s.median_ari:.4f} | {s.exact_k_count}/{s.n_cases} "
            f"| {s.k1_count} | {s.mean_abs_k_error:.2f} |"
        )
    lines.append("")

    # Inferred gamma distributions
    for method in k_adj_methods:
        k_adj = df[df["method"] == method]
        if k_adj.empty:
            continue
        lines.extend([f"## Inferred γ Distribution ({method})", ""])
        lines.append(f"- Mean γ: {k_adj['gamma_hat'].mean():.4f}")
        lines.append(f"- Median γ: {k_adj['gamma_hat'].median():.4f}")
        lines.append(f"- Std γ: {k_adj['gamma_hat'].std():.4f}")
        lines.append(f"- Range: [{k_adj['gamma_hat'].min():.4f}, {k_adj['gamma_hat'].max():.4f}]")
        lines.append(f"- Mean within-tree R²: {k_adj['gamma_r2'].mean():.4f}")
        lines.append(f"- Median pairs per tree: {k_adj['gamma_n_pairs'].median():.0f}")
        lines.append("")

    # Per-case comparison table
    lines.extend(["## Per-Case Comparison", ""])
    header = "| Case | True K | intercept K/ARI"
    for m in k_adj_methods:
        short = m.replace("k_adj_", "")
        header += f" | {short} K/ARI/γ"
    header += " |"
    lines.append(header)
    sep = "|------|--------|----------------"
    for _ in k_adj_methods:
        sep += "|---------------"
    sep += "|"
    lines.append(sep)

    case_names = df[df["method"] == "intercept_only"]["case_name"].tolist()
    for case_name in case_names:
        ri = df[(df["case_name"] == case_name) & (df["method"] == "intercept_only")]
        if ri.empty:
            continue
        ri = ri.iloc[0]
        row = f"| {case_name} | {ri['n_clusters_true']} | {ri['n_clusters_found']}/{ri['ari']:.3f}"
        for m in k_adj_methods:
            rk = df[(df["case_name"] == case_name) & (df["method"] == m)]
            if rk.empty:
                row += " | —"
            else:
                rk = rk.iloc[0]
                ari_diff = rk["ari"] - ri["ari"]
                marker = ""
                if abs(ari_diff) > 0.01:
                    marker = "↑" if ari_diff > 0 else "↓"
                row += f" | {rk['n_clusters_found']}/{rk['ari']:.3f}{marker}/{rk['gamma_hat']:.2f}"
        row += " |"
        lines.append(row)
    lines.append("")

    # Disagreements — cases where any k_adj method differs from intercept_only
    lines.extend(["## Cases Where Any K-Adj Disagrees (|ΔARI| > 0.01)", ""])
    for case_name in case_names:
        ri = df[(df["case_name"] == case_name) & (df["method"] == "intercept_only")]
        if ri.empty:
            continue
        ri = ri.iloc[0]
        diffs = []
        for m in k_adj_methods:
            rk = df[(df["case_name"] == case_name) & (df["method"] == m)]
            if rk.empty:
                continue
            rk = rk.iloc[0]
            delta = rk["ari"] - ri["ari"]
            if abs(delta) > 0.01:
                direction = "+" if delta > 0 else ""
                short = m.replace("k_adj_", "")
                diffs.append(
                    f"{short}: {rk['ari']:.3f} ({direction}{delta:.3f}, γ={rk['gamma_hat']:.2f})"
                )
        if diffs:
            lines.append(f"- **{case_name}** (intercept ARI={ri['ari']:.3f}): {'; '.join(diffs)}")
    lines.append("")

    # Best method per case summary
    lines.extend(["## Winner Summary", ""])
    win_counts: dict[str, int] = {m: 0 for m in all_methods}
    for case_name in case_names:
        best_ari = -1.0
        best_method = ""
        for m in all_methods:
            r = df[(df["case_name"] == case_name) & (df["method"] == m)]
            if not r.empty and r.iloc[0]["ari"] > best_ari:
                best_ari = r.iloc[0]["ari"]
                best_method = m
        if best_method:
            win_counts[best_method] += 1
    for m in all_methods:
        lines.append(f"- **{m}**: best ARI in {win_counts[m]}/{len(case_names)} cases")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp33: k-adjusted deflation benchmark")
    parser.add_argument(
        "--output-prefix",
        default="_exp33_k_adjusted_benchmark",
        help="Output file prefix (relative to script dir)",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=config.SIBLING_ALPHA,
        help="Significance level for both gates",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Run only these case categories (e.g. gaussian binary)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = _ROOT / args.output_prefix

    # Get cases
    all_cases = get_default_test_cases()
    if args.categories:
        cats = set(c.lower() for c in args.categories)
        all_cases = [
            tc for tc in all_cases if any(cat in tc.get("category", "").lower() for cat in cats)
        ]

    print(f"Running {len(all_cases)} cases × 4 methods")
    print(f"Significance level: {args.significance_level}")

    all_results: list[CaseResult] = []

    for i, tc in enumerate(all_cases, 1):
        case_name = tc.get("name", "?")
        print(f"\n[{i}/{len(all_cases)}] {case_name}...", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                case_results = run_case(tc, args.significance_level)
                all_results.extend(case_results)

                # Brief status — one line per method
                for r in case_results:
                    tag = f"  {r.method:20s}"
                    if r.method == "intercept_only":
                        print(
                            f"{tag}: K={r.n_clusters_found} ARI={r.ari:.3f} "
                            f"(true K={r.n_clusters_true})"
                        )
                    else:
                        print(
                            f"{tag}: K={r.n_clusters_found} ARI={r.ari:.3f} "
                            f"γ={r.gamma_hat:+.2f} (n={r.gamma_n_pairs})"
                        )
            except Exception as e:
                print(f"  FAILED: {e}")
                logger.exception("Case %s failed", case_name)

    if not all_results:
        print("\nNo results collected.")
        return

    df = pd.DataFrame([vars(r) for r in all_results])
    summaries = summarize_results(df)

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    for s in summaries:
        print(
            f"  {s.method:20s}: Mean ARI={s.mean_ari:.4f}, "
            f"Median={s.median_ari:.4f}, Exact K={s.exact_k_count}/{s.n_cases}, "
            f"K=1 count={s.k1_count}, MAE(K)={s.mean_abs_k_error:.2f}"
        )

    # Write outputs
    csv_path = Path(f"{prefix}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    md_path = Path(f"{prefix}_report.md")
    write_report(df, summaries, md_path)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
