"""Recognized scientific-support outcomes for TBS benchmark runners."""

from __future__ import annotations

import pandas as pd

from benchmarks.shared.types import (
    UnsupportedEvidence,
    UnsupportedReason,
    UnsupportedReasonCode,
)

EMPIRICAL_NULL_GATE_METHOD = "projected_wald_inflation"
UNDEFINED_INTERNAL_SUPPORT = "undefined_no_internal_support"


def _require_columns(annotations: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in annotations.columns]
    if missing:
        raise ValueError(
            "TBS unsupported-outcome detection requires production-stamped "
            f"annotation columns; missing={missing}."
        )


def unsupported_empirical_null_reason(
    annotations: pd.DataFrame,
    *,
    sibling_gate_method: str,
) -> UnsupportedReason | None:
    """Return the typed no-support reason stamped by the empirical-null gate."""
    if str(sibling_gate_method) != EMPIRICAL_NULL_GATE_METHOD:
        return None

    calibration_column = "Sibling_Gate_P_Value_Calibration"
    _require_columns(annotations, (calibration_column,))
    focal_mask = annotations[calibration_column].eq(UNDEFINED_INTERNAL_SUPPORT)
    focal_record_count = int(focal_mask.sum())
    if focal_record_count == 0:
        return None

    required = (
        "Sibling_Role_Supported",
        "Sibling_Divergence_Invalid",
        "Child_Parent_Divergence_Tested",
        "Child_Parent_Divergence_Significant",
    )
    _require_columns(annotations, required)
    supported_roles = annotations["Sibling_Role_Supported"].fillna(False).astype(bool)
    focal_supported_count = int(supported_roles.loc[focal_mask].sum())
    if focal_supported_count != 0:
        raise ValueError(
            "TBS annotations are inconsistent: undefined_no_internal_support focal rows "
            f"were stamped with {focal_supported_count} supported roles."
        )
    admissible_support_count = int(supported_roles.sum())
    if admissible_support_count != 0:
        return None

    invalid_record_count = int(
        annotations.loc[focal_mask, "Sibling_Divergence_Invalid"]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    upstream_tested_count = int(
        annotations["Child_Parent_Divergence_Tested"].fillna(False).astype(bool).sum()
    )
    upstream_rejected_count = int(
        annotations["Child_Parent_Divergence_Significant"]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    return UnsupportedReason(
        code=UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT,
        stage="sibling_calibration",
        message=(
            "The selected hierarchy contains focal sibling tests but no "
            "admissible internal empirical-null calibration support."
        ),
        evidence=UnsupportedEvidence(
            focal_record_count=focal_record_count,
            admissible_support_count=admissible_support_count,
            invalid_record_count=invalid_record_count,
            upstream_tested_count=upstream_tested_count,
            upstream_rejected_count=upstream_rejected_count,
        ),
    )


__all__ = ["unsupported_empirical_null_reason"]
