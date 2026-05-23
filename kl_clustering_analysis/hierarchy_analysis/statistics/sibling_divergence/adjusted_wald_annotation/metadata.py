"""Annotation and audit metadata helpers for adjusted sibling Wald tests."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ..scale_correction.types.calibration_model import EmpiricalNullScaleModel
from ..pair_testing.types.sibling_pair_record import SiblingPairRecord


def write_record_projection_metadata(
    annotations_dataframe: pd.DataFrame,
    records: list[SiblingPairRecord],
) -> None:
    """Write record-level projection metadata into the annotations frame."""
    record_parents = [record.parent for record in records]
    annotations_dataframe.loc[record_parents, "Sibling_Projection_Dimension_Source"] = [
        record.projection_dimension_source for record in records
    ]
    annotations_dataframe.loc[record_parents, "Sibling_Resolved_Projection_Dimension"] = [
        record.resolved_projection_dimension for record in records
    ]
    annotations_dataframe.loc[record_parents, "Sibling_Used_Parent_Principal_Component_Basis"] = [
        record.used_parent_principal_component_basis for record in records
    ]


def _projection_dimension_source_counts(records: list[SiblingPairRecord]) -> dict[str, int]:
    return {
        str(source): int(count)
        for source, count in sorted(
            Counter(record.projection_dimension_source for record in records).items()
        )
    }


def build_sibling_divergence_audit(
    *,
    records: list[SiblingPairRecord],
    model: EmpiricalNullScaleModel,
    n_null: int,
    n_focal: int,
    n_blocked: int,
) -> dict[str, object]:
    """Build the audit metadata attached to the annotations frame."""
    projection_dimension_source_counts = _projection_dimension_source_counts(records)
    tested_projection_dimension_source_counts = _projection_dimension_source_counts(
        [record for record in records if not record.is_null_like]
    )

    return {
        "total_pairs": len(records),
        "null_like_pairs": n_null,
        "focal_pairs": n_focal,
        "gate2_blocked_pairs": n_blocked,
        "calibration_method": model.method,
        "calibration_n": model.n_calibration,
        "baseline_scale_factor": model.baseline_scale_factor,
        "calibration_mode": "context_weighted_empirical_null_scale",
        "projection_dimension_source_counts": projection_dimension_source_counts,
        "tested_projection_dimension_source_counts": tested_projection_dimension_source_counts,
        "diagnostics": model.diagnostics,
        "test_method": "calibrated_projected_wald",
    }


def build_no_focal_sibling_divergence_audit(
    *,
    records: list[SiblingPairRecord],
    n_null: int,
    n_focal: int,
    n_blocked: int,
) -> dict[str, object]:
    """Build audit metadata when no focal sibling tests require calibration."""
    if n_focal != 0:
        raise ValueError("No-focal sibling audit requires n_focal == 0.")
    return {
        "total_pairs": len(records),
        "null_like_pairs": n_null,
        "focal_pairs": n_focal,
        "gate2_blocked_pairs": n_blocked,
        "calibration_method": "not_required_no_focal_tests",
        "calibration_n": 0,
        "calibration_mode": "not_required_no_focal_tests",
        "projection_dimension_source_counts": _projection_dimension_source_counts(records),
        "tested_projection_dimension_source_counts": {},
        "diagnostics": {"fit_status": "not_required_no_focal_tests"},
        "test_method": "calibrated_projected_wald",
    }


__all__ = [
    "build_no_focal_sibling_divergence_audit",
    "build_sibling_divergence_audit",
    "write_record_projection_metadata",
]
