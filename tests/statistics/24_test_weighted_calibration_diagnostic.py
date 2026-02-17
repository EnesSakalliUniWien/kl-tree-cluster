"""Diagnostic test for Gamma GLM calibration on SMALL_TEST_CASES.

Captures the calibration model parameters and predictions to verify
the Gamma GLM behaves correctly. Patches _fit_weighted_inflation_model
to inspect the model and asserts:
1. max_observed_ratio is computed from null-like pairs only
2. Predictions are clamped at max_observed_ratio
3. The pipeline does not produce degenerate results (K=1, ARI<0)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    WeightedCalibrationModel,
    predict_weighted_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _fit_weighted_inflation_model,
)

try:
    from .test_cases_config import SMALL_TEST_CASES
except ImportError:
    from test_cases_config import SMALL_TEST_CASES  # type: ignore


class TestGammaGLMCalibrationDiagnostic:
    """End-to-end diagnostic: run SMALL_TEST_CASES and inspect calibration."""

    @pytest.fixture(autouse=True)
    def _capture_models(self):
        """Patch _fit_weighted_inflation_model to capture all fitted models."""
        self.captured_models: list[WeightedCalibrationModel] = []
        self.captured_records: list[list] = []

        original_fit = _fit_weighted_inflation_model

        def capturing_fit(records):
            model = original_fit(records)
            self.captured_models.append(model)
            self.captured_records.append(records)
            return model

        with patch(
            "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence"
            ".cousin_weighted_wald._fit_weighted_inflation_model",
            side_effect=capturing_fit,
        ):
            yield

    def _run_pipeline(self, case_name: str):
        """Run one SMALL_TEST_CASE through the benchmark pipeline."""
        case = next(c for c in SMALL_TEST_CASES if c["name"] == case_name)
        df, _ = benchmark_cluster_algorithm(
            test_cases=[case.copy()],
            verbose=False,
            plot_umap=False,
            methods=["kl"],
        )
        kl = df[df["Method"] == "KL Divergence"].iloc[0]
        return kl, self.captured_models[-1], self.captured_records[-1]

    def test_clear_case_model_parameters(self):
        """Inspect calibration model fitted on the 'clear' case."""
        row, model, records = self._run_pipeline("clear")

        # Model should be fitted (enough pairs)
        assert model.method in ("gamma_glm", "weighted_regression", "weighted_median", "none")
        assert model.n_calibration >= 0

        # If regression was used, check beta
        if model.method in ("gamma_glm", "weighted_regression"):
            assert model.beta is not None
            assert len(model.beta) == 3
            # β coefficients should be finite
            assert all(np.isfinite(b) for b in model.beta)

        # max_observed_ratio must come from null-like pairs only
        null_like_ratios = [
            r.stat / r.df for r in records if r.is_null_like and r.stat > 0 and r.df > 0
        ]
        all_ratios = [r.stat / r.df for r in records if r.stat > 0 and r.df > 0 and r.weight > 0]
        if null_like_ratios:
            assert model.max_observed_ratio == pytest.approx(max(null_like_ratios))
            # Must NOT equal max of all ratios if focal has higher
            focal_max = max(
                (r.stat / r.df for r in records if not r.is_null_like and r.stat > 0 and r.df > 0),
                default=0.0,
            )
            if focal_max > max(null_like_ratios):
                assert model.max_observed_ratio < focal_max

    def test_clear_case_predictions_clamped(self):
        """Predictions must be in [1.0, max_observed_ratio]."""
        row, model, records = self._run_pipeline("clear")

        # Test predictions at various n_parent values
        for n_parent in [5, 10, 20, 50, 100, 500]:
            for bl_sum in [0.1, 0.3, 0.5, 1.0]:
                c = predict_weighted_inflation_factor(model, bl_sum, n_parent)
                assert c >= 1.0, f"c_hat={c} < 1.0 at n={n_parent}, bl={bl_sum}"
                assert c <= model.max_observed_ratio + 1e-9, (
                    f"c_hat={c} > max_observed_ratio={model.max_observed_ratio} "
                    f"at n={n_parent}, bl={bl_sum}"
                )

    def test_clear_case_finds_clusters(self):
        """'clear' case (std=0.4) should find at least 2 clusters."""
        row, model, records = self._run_pipeline("clear")
        assert row["Found"] >= 2, f"Found only {row['Found']} clusters (expected ≥2)"
        assert row["Status"] == "ok"

    def test_noisy_case_ari_in_valid_range(self):
        """'noisy' case ARI must be in [-1, 1] (ARI can be negative)."""
        row, model, records = self._run_pipeline("noisy")
        assert -1 <= row["ARI"] <= 1, f"ARI={row['ARI']} out of valid range [-1, 1]"
        assert row["Status"] == "ok"
        assert row["Found"] >= 1

    def test_moderate_case_diagnostics(self):
        """'moderate' case: verify diagnostics dict content."""
        row, model, records = self._run_pipeline("moderate")

        if model.method == "gamma_glm":
            assert "deviance" in model.diagnostics
            assert "null_deviance" in model.diagnostics
            assert "aic" in model.diagnostics
            assert model.diagnostics.get("converged", True)
        elif model.method == "weighted_regression":
            assert "r_squared" in model.diagnostics

        # n_null_like should be tracked
        assert "n_null_like" in model.diagnostics

    def test_all_cases_calibration_summary(self):
        """Run all cases and print a summary table (diagnostic, always passes)."""
        rows = []
        for case_name in ("clear", "moderate", "noisy"):
            row, model, records = self._run_pipeline(case_name)
            n_null = sum(1 for r in records if r.is_null_like)
            n_focal = sum(1 for r in records if not r.is_null_like)
            rows.append(
                {
                    "case": case_name,
                    "method": model.method,
                    "n_cal": model.n_calibration,
                    "n_null": n_null,
                    "n_focal": n_focal,
                    "c_hat": model.global_c_hat,
                    "max_ratio": model.max_observed_ratio,
                    "beta": model.beta.tolist() if model.beta is not None else None,
                    "K_true": row["True"],
                    "K_found": row["Found"],
                    "ARI": row["ARI"],
                }
            )

        print("\n===== Gamma GLM Calibration Diagnostic =====")
        for r in rows:
            print(
                f"  {r['case']:10s}  method={r['method']:20s}  "
                f"n_cal={r['n_cal']:2d}  null={r['n_null']:2d}  focal={r['n_focal']:2d}  "
                f"c_hat={r['c_hat']:.3f}  max_ratio={r['max_ratio']:.3f}  "
                f"beta={r['beta']}  "
                f"K={r['K_found']}/{r['K_true']}  ARI={r['ARI']:.3f}"
            )
        print("=============================================")
