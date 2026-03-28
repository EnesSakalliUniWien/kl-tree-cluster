"""Compare trace vs. pipeline interpolated values for gauss_clear_medium.

Runs BOTH paths — manual trace (Steps 1-8) and full pipeline decompose() —
on the same case (gauss_clear_medium: n=60, p=40, K=4, noise=0.6), then
compares every intermediate value that matters:

  - Pre/post interpolation sibling null priors for blocked pairs
  - T statistics, degrees of freedom, and sibling scale for ALL pairs
  - Calibrator summary (center, spread, record_count)
  - Per-focal-pair local c, T_adj, p_adj
  - Final BH outcomes (Sibling_BH_Different / Sibling_BH_Same)

Since SiblingPairRecord fields are transient (not stored in pipeline output),
the only way to compare is to re-run collection + interpolation independently
for both paths and diff the results.

Usage:
    python debug_scripts/sibling_calibration/compare_interpolation.py
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    resolve_minimum_projection_dimension_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.conditional_deflation import (
    SiblingLocalGaussianInflationCalibrator,
    fit_sibling_inflation_calibrator,
    predict_sibling_adjustment,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    fit_inflation_model,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_null_prior_interpolation import (
    interpolate_sibling_null_priors,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
    count_null_focal_pairs,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

CASE = {
    "generator": "blobs",
    "n_samples": 60,
    "n_features": 40,
    "n_clusters": 4,
    "cluster_std": 0.6,
    "seed": 42,
    "name": "gauss_clear_medium",
}
ALPHA = 0.05


def _build_tree(data_bin):
    """Build PosetTree from binary data (shared by both paths)."""
    Z = linkage(
        pdist(data_bin.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_bin.index.tolist())
    tree.populate_node_divergences(leaf_data=data_bin)
    return tree, Z


def _collect_records(tree, data_bin):
    """Run Gate 2 + collect sibling pairs (production args)."""
    min_proj_dim = resolve_minimum_projection_dimension_backend(
        config.PROJECTION_MINIMUM_DIMENSION,
        leaf_data=data_bin,
    )
    annotations_df = annotate_child_parent_divergence(
        tree,
        tree.annotations_df,
        significance_level_alpha=ALPHA,
        leaf_data=data_bin,
        minimum_projection_dimension=min_proj_dim,
    )
    spectral_dims = derive_sibling_spectral_dims(tree, annotations_df)
    pca_projections, pca_eigenvalues = derive_sibling_pca_projections(annotations_df, spectral_dims)
    child_pca_projections = derive_sibling_child_pca_projections(
        tree, annotations_df, spectral_dims
    )
    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca_projections,
        whitening=config.SIBLING_WHITENING,
    )
    return records, annotations_df


def _print_pair_table(records: list[SiblingPairRecord], title: str) -> None:
    """Print a summary table for a list of sibling pair records."""
    print(f"\n{title}")
    print(
        f"  {'Parent':>8} {'T':>10} {'df':>8} {'scale':>8} "
        f"{'raw_p':>10} {'T/df':>8} {'null':>5} {'blocked':>7} "
        f"{'prior':>8} {'smooth':>8} {'anc_sup':>8} {'lambda':>8}"
    )
    print("  " + "-" * 115)
    for r in sorted(records, key=lambda x: x.n_parent, reverse=True):
        ratio = r.stat / r.degrees_of_freedom if r.degrees_of_freedom > 0 else float("nan")
        smooth = (
            f"{r.smoothed_sibling_null_prior:.4f}"
            if r.smoothed_sibling_null_prior is not None
            else "None"
        )
        anc = f"{r.ancestor_support:.4f}" if r.ancestor_support is not None else "None"
        lam = f"{r.neighborhood_reliance:.4f}" if r.neighborhood_reliance is not None else "None"
        print(
            f"  {r.parent:>8} {r.stat:10.4f} {r.degrees_of_freedom:8.1f} "
            f"{r.sibling_scale:8.2f} {r.p_value:10.6f} {ratio:8.4f} "
            f"{str(r.is_null_like):>5} {str(r.is_gate2_blocked):>7} "
            f"{r.sibling_null_prior_from_edge_pvalue:8.4f} {smooth:>8} "
            f"{anc:>8} {lam:>8}"
        )


def _print_calibrator_summary(
    calibrator: SiblingLocalGaussianInflationCalibrator, label: str
) -> None:
    """Print calibrator summary."""
    print(f"\n  {label}:")
    print(f"    global_adjustment:  {calibrator.global_adjustment:.6f}")
    print(f"    center:             {calibrator.center:.6f}")
    print(f"    spread:             {calibrator.spread:.6f}")
    print(f"    spread_status:      {calibrator.spread_status}")
    print(f"    max_adjustment:     {calibrator.max_adjustment:.6f}")
    print(f"    record_count:       {calibrator.record_count}")


def _deflate_records(records, _model, calibrator):
    """Deflate focal pairs using the local calibrator and return results dict."""
    results = {}
    for r in records:
        if r.is_null_like:
            continue
        c_local = predict_sibling_adjustment(calibrator, r.sibling_scale)
        t_adj = r.stat / c_local
        p_adj = (
            float(chi2.sf(t_adj, df=r.degrees_of_freedom))
            if r.degrees_of_freedom > 0
            else float("nan")
        )
        results[r.parent] = {
            "T": r.stat,
            "df": r.degrees_of_freedom,
            "scale": r.sibling_scale,
            "c_local": c_local,
            "T_adj": t_adj,
            "p_adj": p_adj,
            "prior": r.sibling_null_prior_from_edge_pvalue,
        }
    return results


def main() -> None:
    data_bin, labels, _, _ = generate_case_data(CASE)
    print("=" * 100)
    print("INTERPOLATION COMPARISON: gauss_clear_medium (n=60, p=40, K=4, noise=0.6)")
    print(f"  Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")
    print(f"  Config: SIBLING_WHITENING={config.SIBLING_WHITENING}")
    print(f"  Config: FELSENSTEIN_SCALING={config.FELSENSTEIN_SCALING}")
    print("=" * 100)

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │  Shared tree (single PosetTree for both paths)                      │
    # └──────────────────────────────────────────────────────────────────────┘
    tree, Z = _build_tree(data_bin)

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │  PATH A: Manual trace (same steps as trace_calibration_production)  │
    # └──────────────────────────────────────────────────────────────────────┘
    print("\n" + "─" * 50)
    print("PATH A: Manual trace (production steps)")
    print("─" * 50)

    records_a, ann_a = _collect_records(tree, data_bin)
    n_null_a, n_focal_a, n_blocked_a = count_null_focal_pairs(records_a)
    print(
        f"Total pairs: {len(records_a)}  (null: {n_null_a}, focal: {n_focal_a}, blocked: {n_blocked_a})"
    )

    # Snapshot pre-interpolation priors
    pre_interp_a = {r.parent: r.sibling_null_prior_from_edge_pvalue for r in records_a}
    pre_smooth_a = {r.parent: r.smoothed_sibling_null_prior for r in records_a}

    _print_pair_table(records_a, "--- Before interpolation ---")

    if n_blocked_a > 0:
        records_a = interpolate_sibling_null_priors(records_a, tree, ann_a)
        _print_pair_table(records_a, "--- After interpolation ---")

    model_a = fit_inflation_model(records_a)
    calibrator_a = fit_sibling_inflation_calibrator(records_a, model_a)
    _print_calibrator_summary(calibrator_a, "Calibrator summary (Path A)")

    print(
        f"\n  Model: method={model_a.method}, global_adjustment={model_a.global_inflation_factor:.6f}, "
        f"max_adjustment={model_a.max_observed_ratio:.6f}, n_cal={model_a.n_calibration}"
    )

    deflation_a = _deflate_records(records_a, model_a, calibrator_a)

    print("\n  --- Focal pair deflation (Path A) ---")
    print(
        f"  {'Parent':>8} {'T':>10} {'df':>8} {'scale':>8} {'c_local':>8} {'T_adj':>10} {'p_adj':>10}"
    )
    print("  " + "-" * 70)
    for parent in sorted(deflation_a, key=lambda p: deflation_a[p]["T"], reverse=True):
        d = deflation_a[parent]
        print(
            f"  {parent:>8} {d['T']:10.4f} {d['df']:8.1f} {d['scale']:8.2f} "
            f"{d['c_local']:8.4f} {d['T_adj']:10.4f} {d['p_adj']:10.6f}"
        )

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │  PATH B: Full pipeline decompose() — with record capture           │
    # └──────────────────────────────────────────────────────────────────────┘
    print("\n" + "─" * 50)
    print("PATH B: Full pipeline decompose() [with record capture]")
    print("─" * 50)

    # Monkey-patch fit_sibling_inflation_calibrator at the USAGE site (adjusted_wald_annotation)
    import kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation as _awa_mod

    _captured_records_b = []
    _captured_model_b = []
    _orig_fit_calibrator = _awa_mod.fit_sibling_inflation_calibrator

    def _capturing_fit_calibrator(records, model):
        _captured_records_b.append(list(records))
        _captured_model_b.append(model)
        return _orig_fit_calibrator(records, model)

    _awa_mod.fit_sibling_inflation_calibrator = _capturing_fit_calibrator

    result = tree.decompose(leaf_data=data_bin, alpha_local=ALPHA, sibling_alpha=ALPHA)

    _awa_mod.fit_sibling_inflation_calibrator = _orig_fit_calibrator  # restore

    print(f"Found K = {result['num_clusters']}")
    print(f"  Captured {len(_captured_records_b)} fit_sibling_inflation_calibrator call(s)")

    if _captured_records_b:
        records_b = _captured_records_b[0]
        model_b = _captured_model_b[0]
        print(f"  Pipeline records: {len(records_b)}")
        print(f"  Pipeline global_adjustment: {model_b.global_inflation_factor:.6f}")

        # — Direct per-record comparison: prior and scale —
        print("\n--- Per-record prior comparison (trace vs pipeline) ---")
        print(
            f"  {'Parent':>8} {'prior_A':>10} {'prior_B':>10} {'delta':>12} "
            f"{'scale_A':>8} {'scale_B':>8} {'match':>6}"
        )
        print("  " + "-" * 74)
        rec_a_map = {r.parent: r for r in records_a}
        rec_b_map = {r.parent: r for r in records_b}
        prior_mismatches = 0
        for parent in sorted(set(rec_a_map) | set(rec_b_map)):
            ra = rec_a_map.get(parent)
            rb = rec_b_map.get(parent)
            if ra and rb:
                delta = (
                    ra.sibling_null_prior_from_edge_pvalue - rb.sibling_null_prior_from_edge_pvalue
                )
                match = abs(delta) < 1e-9
                if not match:
                    prior_mismatches += 1
                flag = "  OK" if match else " DIFF"
                print(
                    f"  {parent:>8} {ra.sibling_null_prior_from_edge_pvalue:10.6f} "
                    f"{rb.sibling_null_prior_from_edge_pvalue:10.6f} {delta:+12.6e} "
                    f"{ra.sibling_scale:6.1f} {rb.sibling_scale:6.1f} {flag:>6}"
                )
            elif ra:
                print(
                    f"  {parent:>8} {ra.sibling_null_prior_from_edge_pvalue:10.6f} {'N/A':>10} {'':>12} {'A':>6}"
                )
            else:
                print(
                    f"  {parent:>8} {'N/A':>10} {rb.sibling_null_prior_from_edge_pvalue:10.6f} {'':>12} {'B':>6}"
                )
        print(f"\n  Prior mismatches: {prior_mismatches}/{len(rec_a_map)}")

        # Re-fit the calibrator with the pipeline's records for comparison.
        calibrator_b = _orig_fit_calibrator(records_b, model_b)
        _print_calibrator_summary(
            calibrator_b,
            "Calibrator summary (re-fit from captured pipeline records)",
        )
    else:
        records_b = None

    sdf = tree.annotations_df
    audit = sdf.attrs.get("sibling_divergence_audit", {})
    if audit:
        print("\n  Pipeline audit:")
        for k, v in sorted(audit.items()):
            if k != "diagnostics":
                print(f"    {k}: {v}")
        diag = audit.get("diagnostics", {})
        if diag:
            for k, v in sorted(diag.items()):
                print(f"    diag.{k}: {v}")

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │  COMPARISON: Pair-by-pair diff                                      │
    # └──────────────────────────────────────────────────────────────────────┘
    print("\n" + "─" * 50)
    print("COMPARISON")
    print("─" * 50)

    # Compare deflated T_adj (trace local-kernel deflation vs pipeline Sibling_Test_Statistic)
    print("\n--- T_adj comparison (trace vs pipeline) ---")
    print(f"  {'Parent':>8} {'T_adj_tr':>12} {'T_adj_pipe':>12} {'delta':>12} {'match':>6}")
    print("  " + "-" * 56)
    n_match = 0
    n_total = 0
    for parent in sorted(deflation_a.keys()):
        t_adj_trace = deflation_a[parent]["T_adj"]
        t_adj_pipe = (
            sdf.loc[parent, "Sibling_Test_Statistic"]
            if parent in sdf.index and "Sibling_Test_Statistic" in sdf.columns
            else None
        )
        if t_adj_pipe is not None and np.isfinite(t_adj_pipe):
            delta = t_adj_trace - t_adj_pipe
            match = abs(delta) < 1e-6
            n_match += int(match)
            n_total += 1
            flag = "  OK" if match else " DIFF"
            print(f"  {parent:>8} {t_adj_trace:12.6f} {t_adj_pipe:12.6f} {delta:+12.6e} {flag:>6}")
        else:
            print(f"  {parent:>8} {t_adj_trace:12.6f} {'N/A':>12} {'':>12} {'skip':>6}")

    print(f"\n  T_adj match: {n_match}/{n_total}")

    # Compare deflated p-values
    print("\n--- Deflated p-value comparison ---")
    print(f"  {'Parent':>8} {'p_trace':>12} {'p_pipe_raw':>12} {'p_pipe_BH':>12} {'diff_same':>10}")
    print("  " + "-" * 62)
    for parent in sorted(deflation_a.keys()):
        p_trace = deflation_a[parent]["p_adj"]
        p_raw = (
            sdf.loc[parent, "Sibling_Divergence_P_Value"]
            if parent in sdf.index and "Sibling_Divergence_P_Value" in sdf.columns
            else None
        )
        p_bh = (
            sdf.loc[parent, "Sibling_Divergence_P_Value_Corrected"]
            if parent in sdf.index and "Sibling_Divergence_P_Value_Corrected" in sdf.columns
            else None
        )
        bh_diff = (
            sdf.loc[parent, "Sibling_BH_Different"]
            if parent in sdf.index and "Sibling_BH_Different" in sdf.columns
            else None
        )
        bh_same = (
            sdf.loc[parent, "Sibling_BH_Same"]
            if parent in sdf.index and "Sibling_BH_Same" in sdf.columns
            else None
        )

        p_raw_str = f"{p_raw:12.6f}" if p_raw is not None and np.isfinite(p_raw) else f"{'N/A':>12}"
        p_bh_str = f"{p_bh:12.6f}" if p_bh is not None and np.isfinite(p_bh) else f"{'N/A':>12}"
        outcome = f"D={bh_diff},S={bh_same}" if bh_diff is not None else "N/A"

        print(f"  {parent:>8} {p_trace:12.6f} {p_raw_str} {p_bh_str} {outcome:>10}")

    # Compare interpolation priors (blocked pairs only)
    blocked_records = [r for r in records_a if r.is_gate2_blocked]
    if blocked_records:
        print(f"\n--- Blocked pair interpolation details ({len(blocked_records)} pairs) ---")
        print(
            f"  {'Parent':>8} {'raw_prior':>10} {'post_prior':>10} {'smooth':>10} "
            f"{'anc_sup':>10} {'lambda':>10} {'delta_prior':>12}"
        )
        print("  " + "-" * 78)
        for r in sorted(blocked_records, key=lambda x: x.parent):
            raw = pre_interp_a.get(r.parent, float("nan"))
            post = r.sibling_null_prior_from_edge_pvalue
            smooth = (
                f"{r.smoothed_sibling_null_prior:.6f}"
                if r.smoothed_sibling_null_prior is not None
                else "None"
            )
            anc = f"{r.ancestor_support:.6f}" if r.ancestor_support is not None else "None"
            lam = f"{r.neighborhood_reliance:.6f}" if r.neighborhood_reliance is not None else "None"
            delta = post - raw
            print(
                f"  {r.parent:>8} {raw:10.6f} {post:10.6f} {smooth:>10} "
                f"{anc:>10} {lam:>10} {delta:+12.6e}"
            )
    else:
        print("\n  No blocked pairs — interpolation was not invoked.")

    # Compare calibrator-level summary vs pipeline audit.
    print("\n--- Calibrator summary vs pipeline audit ---")
    pipe_c = audit.get("global_inflation_factor")
    pipe_spread = audit.get("local_adjuster_spread")
    pipe_center = audit.get("local_adjuster_center")
    print(f"  {'Metric':<45} {'Trace':>12} {'Pipeline':>12} {'Match':>6}")
    print("  " + "-" * 78)

    def _cmp(label, trace_val, pipe_val, tol=1e-6):
        t_str = f"{trace_val:.6f}" if trace_val is not None else "N/A"
        p_str = f"{pipe_val:.6f}" if pipe_val is not None else "N/A"
        if trace_val is not None and pipe_val is not None:
            match = "OK" if abs(trace_val - pipe_val) < tol else "DIFF"
        else:
            match = "skip"
        print(f"  {label:<45} {t_str:>12} {p_str:>12} {match:>6}")

    _cmp("global_inflation_factor (global_adjustment)", calibrator_a.global_adjustment, pipe_c)
    _cmp("center", calibrator_a.center, pipe_center)
    _cmp("spread", calibrator_a.spread, pipe_spread)
    _cmp("max_adjustment", calibrator_a.max_adjustment, audit.get("max_observed_ratio"))
    _cmp("n_calibration", float(model_a.n_calibration), float(audit.get("calibration_n", 0)))

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("  Case:           gauss_clear_medium (n=60, p=40, K=4, noise=0.6)")
    print("  True K:         4")
    print(f"  Pipeline K:     {result['num_clusters']}")
    print(f"  T_adj match rate: {n_match}/{n_total}")
    print(f"  Blocked pairs:  {n_blocked_a}")
    if blocked_records:
        max_delta = max(
            abs(r.sibling_null_prior_from_edge_pvalue - pre_interp_a.get(r.parent, 0))
            for r in blocked_records
        )
        print(f"  Max prior shift: {max_delta:.6f}")
    print()


if __name__ == "__main__":
    main()
