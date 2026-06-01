from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.validation.selected_pca_projected_wald_calibration import (
    TARGET_ID,
    SelectedPcaCalibrationSetting,
    create_missing_manifest,
    run_selected_pca_projected_wald_calibration,
    validate_selected_pca_projected_wald_report,
    write_selected_pca_projected_wald_report,
    write_selected_pca_projected_wald_summary_csv,
)


class SelectedPcaProjectedWaldCalibrationTests(unittest.TestCase):
    def test_missing_manifest_lists_selected_pca_target_without_evidence(self) -> None:
        manifest = create_missing_manifest(created_utc="2026-06-01T00:00:00Z")

        self.assertEqual(
            manifest["manifest_schema_version"],
            "selected_pca_projected_wald_calibration/v1",
        )
        self.assertEqual(manifest["targets"][0]["target_id"], TARGET_ID)
        self.assertEqual(validate_selected_pca_projected_wald_report(manifest), [])
        evidence = manifest["targets"][0]["evidence"]
        self.assertEqual(manifest["targets"][0]["evidence_status"], "missing")
        self.assertEqual(evidence["source_path"], None)
        self.assertEqual(evidence["metrics"], {})
        self.assertGreater(len(evidence["missing_required_fields"]), 0)

    def test_smoke_run_reports_selected_pca_calibration_metrics(self) -> None:
        report = run_selected_pca_projected_wald_calibration(
            n_replicates=5,
            alpha=0.05,
            base_seed=17,
            code_commit="test-commit",
            git_worktree_status=[],
            run_command="uv run pytest smoke",
            settings=(
                SelectedPcaCalibrationSetting(
                    dimension=3,
                    n_left=12,
                    n_right=12,
                    covariance_profile="identity",
                    minimum_projection_dimension=2,
                    include_child_mean_rows=True,
                ),
            ),
            created_utc="2026-06-01T00:00:00Z",
        )

        self.assertEqual(validate_selected_pca_projected_wald_report(report), [])
        target = report["targets"][0]
        self.assertEqual(target["target_id"], TARGET_ID)
        self.assertEqual(target["evidence_status"], "complete")
        result = target["evidence"]["metrics"]["results"][0]
        self.assertEqual(result["n_replicates"], 5)
        self.assertGreaterEqual(result["rejection_rate"], 0.0)
        self.assertLessEqual(result["rejection_rate"], 1.0)
        self.assertIn("ks_p_value", result["p_value_uniformity_summary"])
        self.assertIn("mean", result["projection_dimension_summary"])
        self.assertIn("mean", result["raw_mp_signal_count_summary"])

    def test_report_writes_json_and_csv(self) -> None:
        report = run_selected_pca_projected_wald_calibration(
            n_replicates=4,
            alpha=0.05,
            base_seed=23,
            code_commit="test-commit",
            git_worktree_status=[],
            run_command="uv run pytest smoke",
            settings=(
                SelectedPcaCalibrationSetting(
                    dimension=2,
                    n_left=10,
                    n_right=10,
                    covariance_profile="ar1_0.6",
                    minimum_projection_dimension=1,
                    include_child_mean_rows=False,
                ),
            ),
            created_utc="2026-06-01T00:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "selected_pca.json"
            csv_path = Path(tmp) / "selected_pca.csv"

            write_selected_pca_projected_wald_report(report, json_path)
            write_selected_pca_projected_wald_summary_csv(report, csv_path)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_selected_pca_projected_wald_report(decoded), [])
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("target_id,setting_id,n_replicates", csv_text)
            self.assertIn(TARGET_ID, csv_text)


if __name__ == "__main__":
    unittest.main()
