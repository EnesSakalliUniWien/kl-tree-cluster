"""Projection-dimension annotations for inflated projected-Wald sibling tests."""

from __future__ import annotations

import pandas as pd

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord


def write_record_projection_dimensions(
    annotations_dataframe: pd.DataFrame,
    records: list[SiblingPairRecord],
) -> None:
    """Write sibling projection dimensions into the annotations frame."""
    record_parents = [record.parent for record in records]
    annotations_dataframe.loc[record_parents, "Sibling_Projection_Dimension"] = [
        record.sibling_projection_dimension for record in records
    ]


def write_record_calibration_evidence(
    annotations_dataframe: pd.DataFrame,
    records: list[SiblingPairRecord],
) -> None:
    """Write per-record calibration-support evidence for every sibling record.

    Skipped parents are never re-annotated by the focal test path, so their
    degrees of freedom would otherwise be absent from the audit frame and the
    calibration-support population could not be reconstructed.
    """
    record_parents = [record.parent for record in records]
    annotations_dataframe.loc[record_parents, "Sibling_Degrees_of_Freedom"] = [
        float(record.degrees_of_freedom) for record in records
    ]
    annotations_dataframe.loc[record_parents, "Sibling_Role_Supported"] = [
        record.has_empirical_null_support for record in records
    ]
    annotations_dataframe.loc[record_parents, "Sibling_Null_Weight"] = [
        float(record.sibling_null_weight) for record in records
    ]
    annotations_dataframe.loc[
        record_parents, "Sibling_Parent_Positive_Eigenvalue_Count"
    ] = [float(record.parent_positive_eigenvalue_count) for record in records]


__all__ = [
    "write_record_calibration_evidence",
    "write_record_projection_dimensions",
]
