"""Explicit typed feature-space blocks for tree feature data."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FeatureFamily = Literal["bernoulli", "categorical", "continuous"]
FeatureChart = Literal["identity", "simplex_drop_last"]
CovarianceModel = Literal["bernoulli", "multinomial_drop_last", "empirical_gaussian"]

_CATEGORICAL_COLUMN_PATTERN = re.compile(r"^F(?P<feature>\d+)_c(?P<category>\d+)$")


@dataclass(frozen=True)
class FeatureBlock:
    """One typed block in the raw feature matrix."""

    name: str
    family: FeatureFamily
    column_indices: tuple[int, ...]
    chart: FeatureChart
    covariance: CovarianceModel
    contrast_dimension: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureBlock.name must be non-empty.")
        if not self.column_indices:
            raise ValueError(f"Feature block {self.name!r} must contain columns.")
        if len(set(self.column_indices)) != len(self.column_indices):
            raise ValueError(
                f"Feature block {self.name!r} contains duplicate column indices."
            )
        if min(self.column_indices) < 0:
            raise ValueError(
                f"Feature block {self.name!r} contains a negative column index."
            )

        expected_chart: FeatureChart
        expected_covariance: CovarianceModel
        expected_contrast_dimension: int
        if self.family == "bernoulli":
            if len(self.column_indices) != 1:
                raise ValueError(
                    f"Bernoulli block {self.name!r} must contain exactly one column."
                )
            expected_chart = "identity"
            expected_covariance = "bernoulli"
            expected_contrast_dimension = 1
        elif self.family == "categorical":
            if len(self.column_indices) < 2:
                raise ValueError(
                    f"Categorical block {self.name!r} must contain at least two categories."
                )
            expected_chart = "simplex_drop_last"
            expected_covariance = "multinomial_drop_last"
            expected_contrast_dimension = len(self.column_indices) - 1
        elif self.family == "continuous":
            expected_chart = "identity"
            expected_covariance = "empirical_gaussian"
            expected_contrast_dimension = len(self.column_indices)
        else:
            raise ValueError(f"Unknown feature family: {self.family!r}.")

        if self.chart != expected_chart:
            raise ValueError(
                f"Feature block {self.name!r} uses chart {self.chart!r}; "
                f"{self.family!r} requires {expected_chart!r}."
            )
        if self.covariance != expected_covariance:
            raise ValueError(
                f"Feature block {self.name!r} uses covariance {self.covariance!r}; "
                f"{self.family!r} requires {expected_covariance!r}."
            )
        if self.contrast_dimension != expected_contrast_dimension:
            raise ValueError(
                f"Feature block {self.name!r} has contrast_dimension="
                f"{self.contrast_dimension}; expected {expected_contrast_dimension}."
            )

    @property
    def raw_dimension(self) -> int:
        return len(self.column_indices)

    @property
    def signature(self) -> tuple[str, FeatureFamily, tuple[int, ...], FeatureChart, CovarianceModel, int]:
        return (
            self.name,
            self.family,
            self.column_indices,
            self.chart,
            self.covariance,
            self.contrast_dimension,
        )


@dataclass(frozen=True)
class FeatureSpace:
    """Canonical feature-space contract for raw matrix columns."""

    column_names: tuple[str, ...]
    blocks: tuple[FeatureBlock, ...]

    def __post_init__(self) -> None:
        if not self.column_names:
            raise ValueError("FeatureSpace requires at least one column.")
        if not self.blocks:
            raise ValueError("FeatureSpace requires at least one block.")
        if len(set(self.column_names)) != len(self.column_names):
            raise ValueError("FeatureSpace column_names must be unique.")
        block_names = [block.name for block in self.blocks]
        if len(set(block_names)) != len(block_names):
            raise ValueError("FeatureSpace block names must be unique.")

        covered_indices: list[int] = []
        for block in self.blocks:
            covered_indices.extend(block.column_indices)
        if len(set(covered_indices)) != len(covered_indices):
            raise ValueError("FeatureSpace blocks must not overlap.")
        expected_indices = tuple(range(len(self.column_names)))
        if tuple(sorted(covered_indices)) != expected_indices:
            raise ValueError(
                "FeatureSpace blocks must cover every column exactly once. "
                f"Expected indices {expected_indices!r}; got {tuple(sorted(covered_indices))!r}."
            )

    @property
    def raw_dimension(self) -> int:
        return len(self.column_names)

    @property
    def contrast_dimension(self) -> int:
        return sum(block.contrast_dimension for block in self.blocks)

    @property
    def has_categorical_blocks(self) -> bool:
        return any(block.family == "categorical" for block in self.blocks)

    @property
    def has_continuous_blocks(self) -> bool:
        return any(block.family == "continuous" for block in self.blocks)

    @property
    def continuous_blocks(self) -> tuple[FeatureBlock, ...]:
        return tuple(block for block in self.blocks if block.family == "continuous")

    @property
    def family_label(self) -> str:
        families = {block.family for block in self.blocks}
        if len(families) == 1:
            return next(iter(families))
        return "mixed"

    @property
    def signature(
        self,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, FeatureFamily, tuple[int, ...], FeatureChart, CovarianceModel, int], ...]]:
        return self.column_names, tuple(block.signature for block in self.blocks)


def bernoulli_feature_space_from_columns(columns: Sequence[object]) -> FeatureSpace:
    """Build one Bernoulli block per column."""
    column_names = tuple(str(column) for column in columns)
    blocks = tuple(
        FeatureBlock(
            name=column_name,
            family="bernoulli",
            column_indices=(column_index,),
            chart="identity",
            covariance="bernoulli",
            contrast_dimension=1,
        )
        for column_index, column_name in enumerate(column_names)
    )
    return FeatureSpace(column_names=column_names, blocks=blocks)


def continuous_feature_space_from_columns(columns: Sequence[object]) -> FeatureSpace:
    """Build one empirical-Gaussian block spanning all continuous columns."""
    column_names = tuple(str(column) for column in columns)
    block = FeatureBlock(
        name="continuous",
        family="continuous",
        column_indices=tuple(range(len(column_names))),
        chart="identity",
        covariance="empirical_gaussian",
        contrast_dimension=len(column_names),
    )
    return FeatureSpace(column_names=column_names, blocks=(block,))


def contains_categorical_feature_columns(columns: Sequence[object]) -> bool:
    """Return whether any column uses the ``F{i}_c{k}`` categorical schema."""
    return any(_CATEGORICAL_COLUMN_PATTERN.fullmatch(str(column)) for column in columns)


def infer_feature_space_from_columns(columns: Sequence[object]) -> FeatureSpace:
    """Infer Bernoulli and ``F{i}_c{k}`` categorical blocks from column names.

    Non-categorical columns become one-dimensional Bernoulli blocks. Categorical
    blocks may have different category counts; each block must be contiguous and
    ordered by category id within the matrix.
    """
    column_names = tuple(str(column) for column in columns)
    if not column_names:
        raise ValueError("Cannot infer a feature space from an empty column list.")

    categorical_entries_by_feature: dict[int, list[tuple[int, int]]] = {}
    categorical_column_indices: set[int] = set()
    for column_index, column_name in enumerate(column_names):
        match = _CATEGORICAL_COLUMN_PATTERN.fullmatch(column_name)
        if match is None:
            continue
        feature_id = int(match.group("feature"))
        category_id = int(match.group("category"))
        categorical_entries_by_feature.setdefault(feature_id, []).append(
            (category_id, column_index)
        )
        categorical_column_indices.add(column_index)

    blocks_by_start_index: dict[int, FeatureBlock] = {}
    for feature_id, entries in sorted(categorical_entries_by_feature.items()):
        sorted_entries = sorted(entries)
        category_ids = [category_id for category_id, _column_index in sorted_entries]
        expected_category_ids = list(range(len(sorted_entries)))
        if category_ids != expected_category_ids:
            raise ValueError(
                "Categorical one-hot columns must have contiguous category ids "
                f"starting at 0. Feature F{feature_id} has {category_ids!r}."
            )

        column_indices = tuple(column_index for _category_id, column_index in sorted_entries)
        expected_column_indices = tuple(range(min(column_indices), max(column_indices) + 1))
        if column_indices != expected_column_indices:
            raise ValueError(
                "Categorical one-hot columns for each feature must be contiguous "
                "and ordered by category id. "
                f"Feature F{feature_id} uses column indices {column_indices!r}."
            )

        blocks_by_start_index[column_indices[0]] = FeatureBlock(
            name=f"F{feature_id}",
            family="categorical",
            column_indices=column_indices,
            chart="simplex_drop_last",
            covariance="multinomial_drop_last",
            contrast_dimension=len(column_indices) - 1,
        )

    blocks: list[FeatureBlock] = []
    column_index = 0
    while column_index < len(column_names):
        categorical_block = blocks_by_start_index.get(column_index)
        if categorical_block is not None:
            blocks.append(categorical_block)
            column_index += categorical_block.raw_dimension
            continue
        if column_index in categorical_column_indices:
            raise ValueError(
                "Internal categorical feature-space parser error: categorical column "
                f"{column_names[column_index]!r} was not assigned to a block."
            )
        blocks.append(
            FeatureBlock(
                name=column_names[column_index],
                family="bernoulli",
                column_indices=(column_index,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            )
        )
        column_index += 1

    return FeatureSpace(column_names=column_names, blocks=tuple(blocks))


def resolve_feature_space(
    columns: Sequence[object],
    feature_space: FeatureSpace | None,
) -> FeatureSpace:
    """Resolve the active feature space for leaf data columns."""
    if feature_space is None:
        if contains_categorical_feature_columns(columns):
            raise ValueError(
                "leaf_data contains categorical one-hot columns. Pass an explicit "
                "feature_space so multinomial blocks are part of the statistical contract."
            )
        return bernoulli_feature_space_from_columns(columns)

    validate_feature_space(columns, feature_space)
    return feature_space


def validate_feature_space(columns: Sequence[object], feature_space: FeatureSpace) -> None:
    """Validate that columns exactly match an explicit feature-space contract."""
    column_names = tuple(str(column) for column in columns)
    if column_names != feature_space.column_names:
        raise ValueError(
            "feature_space does not match leaf_data columns. "
            f"Expected {feature_space.column_names!r}; got {column_names!r}."
        )


def validate_feature_vector(
    row: npt.ArrayLike,
    feature_space: FeatureSpace,
    *,
    value_name: str = "feature row",
) -> npt.NDArray[np.float64]:
    """Validate one raw-coordinate distribution vector."""
    array = np.asarray(row, dtype=np.float64).reshape(-1)
    if array.shape != (feature_space.raw_dimension,):
        raise ValueError(
            f"{value_name} has shape {array.shape}; expected "
            f"{(feature_space.raw_dimension,)}."
        )
    validate_feature_matrix(array[None, :], feature_space, value_name=value_name)
    return array


def validate_feature_matrix(
    matrix: npt.ArrayLike,
    feature_space: FeatureSpace,
    *,
    value_name: str = "feature matrix",
) -> npt.NDArray[np.float64]:
    """Validate a raw-coordinate feature matrix against typed blocks."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != feature_space.raw_dimension:
        raise ValueError(
            f"{value_name} has shape {array.shape}; expected "
            f"(n_samples, {feature_space.raw_dimension})."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{value_name} must contain only finite values.")

    _validate_bernoulli_blocks(array, feature_space)
    _validate_categorical_blocks(array, feature_space)

    return array


def _validate_bernoulli_blocks(
    array: npt.NDArray[np.float64],
    feature_space: FeatureSpace,
) -> None:
    bernoulli_blocks = [
        block for block in feature_space.blocks if block.family == "bernoulli"
    ]
    if not bernoulli_blocks:
        return

    column_indices = np.asarray(
        [block.column_indices[0] for block in bernoulli_blocks],
        dtype=np.int64,
    )
    block_values = array[:, column_indices]
    if np.any(block_values < 0.0) or np.any(block_values > 1.0):
        invalid_mask = (block_values < 0.0) | (block_values > 1.0)
        _row_index, block_offset = np.argwhere(invalid_mask)[0]
        block = bernoulli_blocks[int(block_offset)]
        _raise_unit_interval_error(
            array[:, block.column_indices],
            value_name=f"Bernoulli block {block.name!r}",
        )


def _validate_categorical_blocks(
    array: npt.NDArray[np.float64],
    feature_space: FeatureSpace,
) -> None:
    categorical_blocks_by_dimension: dict[int, list[FeatureBlock]] = {}
    for block in feature_space.blocks:
        if block.family == "continuous":
            continue
        if block.family == "categorical":
            categorical_blocks_by_dimension.setdefault(block.raw_dimension, []).append(block)
            continue

    for blocks in categorical_blocks_by_dimension.values():
        column_indices = np.asarray(
            [block.column_indices for block in blocks],
            dtype=np.int64,
        )
        block_values = array[:, column_indices]
        if np.any(block_values < 0.0) or np.any(block_values > 1.0):
            invalid_mask = (block_values < 0.0) | (block_values > 1.0)
            _row_index, block_offset, _category_offset = np.argwhere(invalid_mask)[0]
            block = blocks[int(block_offset)]
            _raise_unit_interval_error(
                array[:, block.column_indices],
                value_name=f"Categorical block {block.name!r}",
            )

        row_sums = block_values.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0):
            invalid_mask = ~np.isclose(row_sums, 1.0, atol=1e-8, rtol=0.0)
            _row_index, block_offset = np.argwhere(invalid_mask)[0]
            block = blocks[int(block_offset)]
            block_row_sums = array[:, block.column_indices].sum(axis=1)
            raise ValueError(
                f"Categorical block {block.name!r} rows must sum to 1. "
                f"Got row-sum range "
                f"[{float(np.min(block_row_sums)):.6g}, "
                f"{float(np.max(block_row_sums)):.6g}]."
            )


def _raise_unit_interval_error(
    values: npt.NDArray[np.float64],
    *,
    value_name: str,
) -> None:
    raise ValueError(
        f"{value_name} values must lie in [0, 1]. "
        f"Range=[{float(np.min(values)):.6g}, {float(np.max(values)):.6g}]."
    )


__all__ = [
    "CovarianceModel",
    "FeatureBlock",
    "FeatureChart",
    "FeatureFamily",
    "FeatureSpace",
    "bernoulli_feature_space_from_columns",
    "contains_categorical_feature_columns",
    "continuous_feature_space_from_columns",
    "infer_feature_space_from_columns",
    "resolve_feature_space",
    "validate_feature_matrix",
    "validate_feature_space",
    "validate_feature_vector",
]
