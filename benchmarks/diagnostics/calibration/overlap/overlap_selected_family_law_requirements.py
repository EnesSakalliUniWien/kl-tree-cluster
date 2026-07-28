"""Selected-family law requirements for residual overlap traversal failures.

This diagnostic converts residual family covariates and the threshold
stability contract into a concrete requirements envelope for the missing
selected-family structural recovery law. It is diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_selected_family_law_requirements_not_calibration"
SCHEMA_VERSION = "overlap_selected_family_law_requirements/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_selected_family_law_requirements"

CONDITIONING_METRICS = (
    "residual_family_size",
    "residual_neg_log10_min_sibling_p_value",
    "residual_max_homogeneity_gain_min",
    "residual_max_continuous_context_margin",
    "residual_max_subspace_consensus_jaccard_topk",
    "residual_max_depth",
    "residual_median_parent_size",
    "residual_max_barycentric_balance",
    "residual_min_fragment_risk_proxy_score",
    "residual_max_balanced_recovery_proxy_score",
    "residual_min_size_balance",
    "residual_min_edge_norm_balance",
)

ENVELOPE_COLUMNS = (
    "schema_version",
    "study_role",
    "residual_family_truth_role",
    "family_count",
    "metric",
    "min_value",
    "median_value",
    "max_value",
)

REQUIREMENT_COLUMNS = (
    "schema_version",
    "study_role",
    "requirement_id",
    "requirement_type",
    "conditioning_variables",
    "evidence_from_current_diagnostics",
    "minimum_validation_obligation",
    "current_status",
)


@dataclass(frozen=True)
class OverlapSelectedFamilyLawRequirementsConfig:
    """Runtime contract for selected-family law requirement extraction."""

    residual_family_rows_path: Path
    stability_contract_rows_path: Path
    output_dir: Path

    @property
    def conditioning_envelope_path(self) -> Path:
        return self.output_dir / "overlap_selected_family_conditioning_envelope.csv"

    @property
    def requirements_path(self) -> Path:
        return self.output_dir / "overlap_selected_family_law_requirements.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_inputs(
    residual_family_rows: pd.DataFrame,
    stability_rows: pd.DataFrame,
) -> None:
    missing_family = sorted(
        {"residual_family_truth_role", *CONDITIONING_METRICS} - set(residual_family_rows.columns)
    )
    if missing_family:
        raise ValueError(f"Residual family rows are missing columns: {missing_family!r}")
    missing_stability = sorted(
        {
            "stage_name",
            "threshold_stability_status",
            "requires_selected_family_law",
        }
        - set(stability_rows.columns)
    )
    if missing_stability:
        raise ValueError(f"Stability rows are missing columns: {missing_stability!r}")


def build_conditioning_envelope(residual_family_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-family conditioning covariates by diagnostic role."""
    records: list[dict[str, object]] = []
    for role, group in residual_family_rows.groupby("residual_family_truth_role", sort=True):
        for metric in CONDITIONING_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "residual_family_truth_role": str(role),
                    "family_count": int(group.shape[0]),
                    "metric": metric,
                    "min_value": float(values.min()) if not values.empty else math.nan,
                    "median_value": (float(values.median()) if not values.empty else math.nan),
                    "max_value": float(values.max()) if not values.empty else math.nan,
                }
            )
    return pd.DataFrame.from_records(records, columns=ENVELOPE_COLUMNS)


def build_selected_family_law_requirements(
    residual_family_rows: pd.DataFrame,
    stability_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return concrete requirements for the missing selected-family law."""
    _validate_inputs(residual_family_rows, stability_rows)
    roles = residual_family_rows["residual_family_truth_role"].astype(str)
    counts = roles.value_counts()
    law_required_stages = stability_rows[
        stability_rows["requires_selected_family_law"].astype(bool)
    ]["stage_name"].astype(str)
    nontransferable_stages = stability_rows[
        stability_rows["threshold_stability_status"]
        .astype(str)
        .eq("nontransferable_focused_cutpoint")
    ]["stage_name"].astype(str)
    conditioning = ",".join(CONDITIONING_METRICS)
    records = [
        {
            "requirement_id": "selected_family_null_evidence_law",
            "requirement_type": "null_law",
            "conditioning_variables": conditioning,
            "evidence_from_current_diagnostics": (
                "Residual p-value evidence separates recovery from selected null "
                "in the focused panel, but transfer leaks selected-null and "
                "non-recovery families."
            ),
            "minimum_validation_obligation": (
                "Control held-out selected-null leakage while retaining recovery "
                "families across case and replicate splits."
            ),
            "current_status": "law_required_not_transferable_cutpoint",
        },
        {
            "requirement_id": "selected_family_structural_recovery_condition",
            "requirement_type": "structural_recovery",
            "conditioning_variables": conditioning,
            "evidence_from_current_diagnostics": (
                f"Residual roles: truth_recovery={int(counts.get('residual_truth_recovery_family', 0))}, "
                f"nonrecovery={int(counts.get('residual_nonrecovery_family', 0))}, "
                f"selected_null={int(counts.get('residual_null_like_family', 0))}. "
                "Non-recovery selected signal can be more p-value-extreme than recovery."
            ),
            "minimum_validation_obligation": (
                "Separate structural recovery from diffuse/wrong-granularity "
                "selected signal without relying only on p-value extremeness."
            ),
            "current_status": "law_required_structural_target_missing",
        },
        {
            "requirement_id": "threshold_transfer_validation",
            "requirement_type": "validation",
            "conditioning_variables": "case_id,replicate," + conditioning,
            "evidence_from_current_diagnostics": (
                "Focused max-negative residual cutpoints are non-transferable: "
                + ";".join(nontransferable_stages)
            ),
            "minimum_validation_obligation": (
                "Any proposed recovery rule must be predeclared and pass "
                "held-out case/replicate transfer before production consideration."
            ),
            "current_status": "fail_closed_transfer_not_satisfied",
        },
        {
            "requirement_id": "multiscale_unstable_zone_reporting",
            "requirement_type": "reporting",
            "conditioning_variables": "diagnostic_traversal_action," + conditioning,
            "evidence_from_current_diagnostics": (
                "Law-required stages: " + ";".join(law_required_stages)
            ),
            "minimum_validation_obligation": (
                "Until the selected-family law is validated, residual weak "
                "families must remain unstable multi-scale output."
            ),
            "current_status": "stable_reporting_only_no_production_promotion",
        },
    ]
    output = pd.DataFrame.from_records(records)
    output.insert(0, "schema_version", SCHEMA_VERSION)
    output.insert(1, "study_role", STUDY_ROLE)
    return output.loc[:, REQUIREMENT_COLUMNS]


def run_overlap_selected_family_law_requirements(
    config: OverlapSelectedFamilyLawRequirementsConfig,
) -> dict[str, Path]:
    """Run selected-family law requirement extraction and write outputs."""
    residual_family_rows = pd.read_csv(config.residual_family_rows_path)
    stability_rows = pd.read_csv(config.stability_contract_rows_path)
    envelope = build_conditioning_envelope(residual_family_rows)
    requirements = build_selected_family_law_requirements(
        residual_family_rows,
        stability_rows,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    envelope.to_csv(config.conditioning_envelope_path, index=False)
    requirements.to_csv(config.requirements_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "residual_family_rows_path": str(config.residual_family_rows_path),
        "stability_contract_rows_path": str(config.stability_contract_rows_path),
        "outputs": {
            "conditioning_envelope": str(config.conditioning_envelope_path),
            "requirements": str(config.requirements_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "conditioning_envelope": config.conditioning_envelope_path,
        "requirements": config.requirements_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-family-rows-path", required=True, type=Path)
    parser.add_argument("--stability-contract-rows-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_selected_family_law_requirements(
        OverlapSelectedFamilyLawRequirementsConfig(
            residual_family_rows_path=args.residual_family_rows_path,
            stability_contract_rows_path=args.stability_contract_rows_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
