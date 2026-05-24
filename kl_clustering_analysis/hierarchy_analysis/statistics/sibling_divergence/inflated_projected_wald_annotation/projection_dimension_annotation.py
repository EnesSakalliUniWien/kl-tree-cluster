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


__all__ = ["write_record_projection_dimensions"]
