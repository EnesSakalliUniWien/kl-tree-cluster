from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.validation.contracts.method_constants_manifest import (
    CONSTANT_IDS,
    REQUIRED_OUTPUT_FIELDS,
    create_manifest,
    validate_manifest,
)


class MethodConstantsManifestTests(unittest.TestCase):
    def test_create_manifest_lists_constants_and_marks_evidence_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_001"
            run_dir.mkdir()
            (run_dir / "full_benchmark_comparison.csv").write_text(
                "case_id,method,ari\n",
                encoding="utf-8",
            )

            manifest = create_manifest(
                [run_dir],
                created_utc="2026-05-24T00:00:00Z",
            )

        self.assertEqual(
            manifest["manifest_schema_version"],
            "method_constant_validation_manifest/v1",
        )
        self.assertEqual(
            [entry["constant_id"] for entry in manifest["constants"]],
            list(CONSTANT_IDS),
        )
        self.assertIn("sibling_gate_profile", CONSTANT_IDS)
        self.assertIn("fixed_sibling_gate_alpha_penalty", CONSTANT_IDS)
        self.assertIn("root_stability_guard_threshold", CONSTANT_IDS)
        self.assertIn("root_stability_subsample_replicates", CONSTANT_IDS)
        self.assertIn("root_stability_feature_fraction", CONSTANT_IDS)
        self.assertIn("root_selective_permutation_guard_replicates", CONSTANT_IDS)
        self.assertIn("root_selective_permutation_guard_alpha", CONSTANT_IDS)
        self.assertIn("root_selective_permutation_guard_scope", CONSTANT_IDS)
        self.assertTrue(manifest["source_paths"][0]["exists"])
        self.assertEqual(manifest["source_paths"][0]["kind"], "directory")
        self.assertIn(
            "full_benchmark_comparison.csv",
            manifest["source_paths"][0]["recognized_outputs"],
        )

        for entry in manifest["constants"]:
            constant_id = entry["constant_id"]
            self.assertEqual(entry["evidence_status"], "missing")
            self.assertEqual(entry["evidence"]["metrics"], {})
            self.assertEqual(entry["evidence"]["source_path"], None)
            self.assertEqual(
                entry["evidence"]["missing_required_fields"],
                REQUIRED_OUTPUT_FIELDS[constant_id],
            )

        serialized_metrics = json.dumps(
            [entry["evidence"]["metrics"] for entry in manifest["constants"]],
            sort_keys=True,
        )
        self.assertNotIn("mean_ari", serialized_metrics)
        self.assertNotIn("p_value", serialized_metrics)

    def test_validate_manifest_rejects_missing_constant(self) -> None:
        manifest = create_manifest([], created_utc="2026-05-24T00:00:00Z")
        manifest["constants"] = manifest["constants"][:-1]

        errors = validate_manifest(manifest)

        self.assertTrue(
            any("constant_id set" in error for error in errors),
            errors,
        )

    def test_create_manifest_json_round_trips(self) -> None:
        manifest = create_manifest([], created_utc="2026-05-24T00:00:00Z")

        decoded = json.loads(json.dumps(manifest))

        self.assertEqual(validate_manifest(decoded), [])


if __name__ == "__main__":
    unittest.main()
