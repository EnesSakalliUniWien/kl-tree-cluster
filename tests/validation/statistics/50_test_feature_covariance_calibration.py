from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.validation.statistics.feature_covariance_calibration import (
    TARGET_IDS,
    CategoricalCalibrationSetting,
    ContinuousCalibrationSetting,
    create_missing_manifest,
    run_feature_covariance_calibration,
    validate_feature_covariance_report,
    write_feature_covariance_report,
    write_feature_covariance_summary_csv,
)


class FeatureCovarianceCalibrationTests(unittest.TestCase):
    def test_missing_manifest_lists_covariance_targets_without_evidence(self) -> None:
        manifest = create_missing_manifest(created_utc="2026-06-01T00:00:00Z")

        self.assertEqual(
            manifest["manifest_schema_version"],
            "feature_covariance_calibration/v1",
        )
        self.assertEqual(
            [entry["target_id"] for entry in manifest["targets"]],
            list(TARGET_IDS),
        )
        self.assertEqual(validate_feature_covariance_report(manifest), [])
        for entry in manifest["targets"]:
            self.assertEqual(entry["evidence_status"], "missing")
            self.assertEqual(entry["evidence"]["source_path"], None)
            self.assertEqual(entry["evidence"]["metrics"], {})
            self.assertGreater(len(entry["evidence"]["missing_required_fields"]), 0)

    def test_smoke_run_reports_categorical_and_continuous_calibration_metrics(self) -> None:
        report = run_feature_covariance_calibration(
            n_replicates=8,
            alpha=0.05,
            base_seed=17,
            code_commit="test-commit",
            git_worktree_status=[],
            run_command="uv run pytest smoke",
            categorical_settings=(
                CategoricalCalibrationSetting(
                    n_features=2,
                    n_categories=5,
                    n_left=30,
                    n_right=30,
                    probability_profile="uniform",
                ),
            ),
            continuous_settings=(
                ContinuousCalibrationSetting(
                    dimension=3,
                    n_left=20,
                    n_right=20,
                    covariance_profile="ar1_0.6",
                ),
            ),
            created_utc="2026-06-01T00:00:00Z",
        )

        self.assertEqual(validate_feature_covariance_report(report), [])
        self.assertEqual(
            [entry["target_id"] for entry in report["targets"]],
            list(TARGET_IDS),
        )
        for entry in report["targets"]:
            self.assertEqual(entry["evidence_status"], "complete")
            results = entry["evidence"]["metrics"]["results"]
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result["n_replicates"], 8)
            self.assertGreaterEqual(result["rejection_rate"], 0.0)
            self.assertLessEqual(result["rejection_rate"], 1.0)
            interval = result["confidence_interval"]
            self.assertGreaterEqual(interval["low"], 0.0)
            self.assertLessEqual(interval["high"], 1.0)
            self.assertIn("ks_p_value", result["p_value_uniformity_summary"])

    def test_report_writes_json_and_csv(self) -> None:
        report = run_feature_covariance_calibration(
            n_replicates=4,
            alpha=0.05,
            base_seed=23,
            code_commit="test-commit",
            git_worktree_status=[],
            run_command="uv run pytest smoke",
            categorical_settings=(
                CategoricalCalibrationSetting(
                    n_features=1,
                    n_categories=4,
                    n_left=20,
                    n_right=20,
                    probability_profile="rare_tail",
                ),
            ),
            continuous_settings=(
                ContinuousCalibrationSetting(
                    dimension=2,
                    n_left=12,
                    n_right=12,
                    covariance_profile="identity",
                ),
            ),
            created_utc="2026-06-01T00:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "feature_covariance.json"
            csv_path = Path(tmp) / "feature_covariance.csv"

            write_feature_covariance_report(report, json_path)
            write_feature_covariance_summary_csv(report, csv_path)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_feature_covariance_report(decoded), [])
            csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                csv_lines[0],
                "target_id,setting_id,n_replicates,alpha,rejection_count,"
                "rejection_rate,ci_low,ci_high,effect_estimate,ks_statistic,ks_p_value",
            )
            csv_text = "\n".join(csv_lines)
            self.assertIn("categorical_multinomial_drop_last_covariance", csv_text)
            self.assertIn("continuous_empirical_gaussian_covariance", csv_text)


if __name__ == "__main__":
    unittest.main()
