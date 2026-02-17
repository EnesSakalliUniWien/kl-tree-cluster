#!/usr/bin/env python
"""Diagnostic script for Gamma GLM calibration on SMALL_TEST_CASES.

Patches _fit_weighted_inflation_model to capture the fitted model and
records, then prints a summary table showing:
- Calibration method (gamma_glm / weighted_regression / weighted_median / none)
- Number of calibration pairs, null-like vs focal
- Global c_hat, max_observed_ratio, null-like max, all-pairs max
- Beta coefficients
- Predictions at various (bl_sum, n_parent) combinations
- Pipeline output: K found, K true, ARI

Usage:
    python debug_scripts/diagnostics/diagnose_gamma_glm_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    WeightedCalibrationModel,
    predict_weighted_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _fit_weighted_inflation_model,
)
from tests.test_cases_config import SMALL_TEST_CASES


def main():
    models: list[WeightedCalibrationModel] = []
    records_list: list[list] = []

    original_fit = _fit_weighted_inflation_model

    def capturing_fit(records):
        model = original_fit(records)
        models.append(model)
        records_list.append(list(records))
        return model

    results = []

    with patch(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence"
        ".cousin_weighted_wald._fit_weighted_inflation_model",
        side_effect=capturing_fit,
    ):
        for case_name in ("clear", "moderate", "noisy"):
            case = next(c for c in SMALL_TEST_CASES if c["name"] == case_name)
            df, _ = benchmark_cluster_algorithm(
                test_cases=[case.copy()],
                verbose=False,
                plot_umap=False,
                methods=["kl"],
            )
            kl = df[df["Method"] == "KL Divergence"].iloc[0]
            model = models[-1]
            recs = records_list[-1]
            results.append((case_name, kl, model, recs))

    # ── Summary table ──
    print()
    print("=" * 100)
    print("  Gamma GLM Calibration Diagnostic  —  SMALL_TEST_CASES")
    print("=" * 100)

    for case_name, kl, model, recs in results:
        n_null = sum(1 for r in recs if r.is_null_like)
        n_focal = sum(1 for r in recs if not r.is_null_like)
        null_ratios = [r.stat / r.df for r in recs if r.is_null_like and r.stat > 0 and r.df > 0]
        all_ratios = [r.stat / r.df for r in recs if r.stat > 0 and r.df > 0 and r.weight > 0]
        null_max = max(null_ratios) if null_ratios else 0.0
        all_max = max(all_ratios) if all_ratios else 0.0

        print(f"\n  Case: {case_name}")
        print(f"    Method:           {model.method}")
        print(f"    n_calibration:    {model.n_calibration}")
        print(f"    null-like pairs:  {n_null}")
        print(f"    focal pairs:      {n_focal}")
        print(f"    global_c_hat:     {model.global_c_hat:.4f}")
        print(f"    max_observed_ratio (null-like): {model.max_observed_ratio:.4f}")
        print(f"    max T/k (null-like):            {null_max:.4f}")
        print(f"    max T/k (all pairs):            {all_max:.4f}")
        if model.beta is not None:
            print(f"    β = [{model.beta[0]:.4f}, {model.beta[1]:.4f}, {model.beta[2]:.4f}]")
        else:
            print("    β = None")

        if model.diagnostics:
            for key in (
                "r_squared",
                "deviance",
                "null_deviance",
                "aic",
                "scale",
                "effective_n",
                "n_null_like",
                "converged",
            ):
                if key in model.diagnostics:
                    print(f"    {key}: {model.diagnostics[key]}")

        # Predictions at different (bl_sum, n_parent)
        print("    Predictions:")
        for n_par in [5, 10, 20, 50]:
            for bl in [0.1, 0.3, 0.5]:
                c = predict_weighted_inflation_factor(model, bl, n_par)
                print(f"      bl={bl:.1f}, n={n_par:3d}  →  ĉ = {c:.4f}")

        print(f"    Pipeline result:  K={kl['Found']}/{kl['True']}  ARI={kl['ARI']:.3f}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
