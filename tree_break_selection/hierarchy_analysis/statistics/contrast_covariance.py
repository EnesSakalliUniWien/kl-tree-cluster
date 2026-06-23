"""Contrast covariance construction for projected Wald tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from tree_break_selection.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    bernoulli_feature_space_from_columns,
    validate_feature_matrix,
    validate_feature_vector,
)

ComparisonKind = Literal["sibling", "child_parent"]
FeatureSpaceLabel = Literal["bernoulli", "categorical", "continuous", "mixed"]


@dataclass(frozen=True)
class _ResolvedContrastInputs:
    first: NDArray[np.float64]
    second: NDArray[np.float64]
    covariance_distribution: NDArray[np.float64]
    feature_space: FeatureSpace
    variance_scale: float
    continuous_covariance_by_block: dict[str, NDArray[np.float64]]
    ridge: float


@dataclass(frozen=True)
class ContrastCovariance:
    """A Wald contrast and its block covariance in independent coordinates."""

    contrast_vector: NDArray[np.float64]
    covariance_blocks: tuple[NDArray[np.float64], ...]
    feature_family: FeatureSpaceLabel
    comparison: ComparisonKind

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.contrast_vector.shape[0])

    def whitened_vector(self) -> NDArray[np.float64]:
        """Return ``V^{-1/2} contrast`` using the explicit covariance blocks."""
        whitened_blocks: list[NDArray[np.float64]] = []
        offset = 0
        for covariance_block in self.covariance_blocks:
            block_width = int(covariance_block.shape[0])
            contrast_block = self.contrast_vector[offset : offset + block_width]
            cho, lower = linalg.cho_factor(
                covariance_block,
                lower=True,
                check_finite=False,
            )
            whitened_blocks.append(
                linalg.solve_triangular(
                    cho,
                    contrast_block,
                    lower=lower,
                    check_finite=False,
                )
            )
            offset += block_width
        if offset != self.contrast_vector.shape[0]:
            raise ValueError(
                "Contrast covariance blocks do not cover the contrast vector: "
                f"covered={offset}, contrast_dimension={self.contrast_vector.shape[0]}."
            )
        return np.concatenate(whitened_blocks, axis=0)


def build_contrast_covariance(
    first_distribution: NDArray[np.floating],
    second_distribution: NDArray[np.floating],
    first_sample_size: float,
    second_sample_size: float,
    *,
    comparison: ComparisonKind,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
    tree_time: float | None = None,
    tree_time_normalizer: float | None = None,
    ridge: float = 1e-12,
) -> ContrastCovariance:
    """Build the canonical contrast-covariance object.

    Distributions are flat raw-coordinate vectors. The feature-space blocks
    define the contrast chart and covariance model for each coordinate block.
    """
    resolved = _resolve_contrast_inputs(
        first_distribution,
        second_distribution,
        first_sample_size,
        second_sample_size,
        comparison=comparison,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        tree_time=tree_time,
        tree_time_normalizer=tree_time_normalizer,
        ridge=ridge,
    )

    return _feature_space_contrast_covariance(
        resolved.first,
        resolved.second,
        resolved.covariance_distribution,
        feature_space=resolved.feature_space,
        variance_scale=resolved.variance_scale,
        comparison=comparison,
        continuous_covariance_by_block=resolved.continuous_covariance_by_block,
        ridge=resolved.ridge,
    )


def _resolve_contrast_inputs(
    first_distribution: NDArray[np.floating],
    second_distribution: NDArray[np.floating],
    first_sample_size: float,
    second_sample_size: float,
    *,
    comparison: ComparisonKind,
    feature_space: FeatureSpace | None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None,
    tree_time: float | None,
    tree_time_normalizer: float | None,
    ridge: float,
) -> _ResolvedContrastInputs:
    """Validate and resolve the objects shared by all contrast computations."""
    first = _as_flat_distribution_array(
        first_distribution,
        value_name="first_distribution",
    )
    second = _as_flat_distribution_array(
        second_distribution,
        value_name="second_distribution",
    )
    if first.shape != second.shape:
        raise ValueError(
            "Compared distributions must have identical shape. "
            f"Got {first.shape} vs {second.shape}."
        )

    active_feature_space = _resolve_distribution_feature_space(first, feature_space)
    first = validate_feature_vector(
        first,
        active_feature_space,
        value_name="first_distribution",
    )
    second = validate_feature_vector(
        second,
        active_feature_space,
        value_name="second_distribution",
    )
    _validate_sample_sizes(first_sample_size, second_sample_size)
    ridge_value = _validate_ridge(ridge)
    continuous_covariance_blocks = _validate_continuous_covariance_by_block(
        active_feature_space,
        continuous_covariance_by_block,
    )

    if comparison == "sibling":
        variance_scale = _sibling_variance_scale(first_sample_size, second_sample_size)
        covariance_distribution = (
            first_sample_size * first + second_sample_size * second
        ) / (first_sample_size + second_sample_size)
    elif comparison == "child_parent":
        variance_scale = _child_parent_variance_scale(first_sample_size, second_sample_size)
        covariance_distribution = second
    else:
        raise ValueError(f"Unknown comparison kind: {comparison!r}.")

    variance_scale *= _tree_time_variance_multiplier(
        tree_time,
        tree_time_normalizer=tree_time_normalizer,
    )

    return _ResolvedContrastInputs(
        first=first,
        second=second,
        covariance_distribution=covariance_distribution,
        feature_space=active_feature_space,
        variance_scale=variance_scale,
        continuous_covariance_by_block=continuous_covariance_blocks,
        ridge=ridge_value,
    )


def compute_whitened_wald_contrast(
    first_distribution: NDArray[np.floating],
    second_distribution: NDArray[np.floating],
    first_sample_size: float,
    second_sample_size: float,
    *,
    comparison: ComparisonKind,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
    tree_time: float | None = None,
    tree_time_normalizer: float | None = None,
    ridge: float = 1e-12,
) -> NDArray[np.float64]:
    """Return the covariance-whitened Wald contrast vector."""
    resolved = _resolve_contrast_inputs(
        first_distribution,
        second_distribution,
        first_sample_size,
        second_sample_size,
        comparison=comparison,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        tree_time=tree_time,
        tree_time_normalizer=tree_time_normalizer,
        ridge=ridge,
    )
    fast_z = _compute_vectorized_whitened_wald_contrast(resolved)
    if fast_z is not None:
        return fast_z

    return _feature_space_contrast_covariance(
        resolved.first,
        resolved.second,
        resolved.covariance_distribution,
        feature_space=resolved.feature_space,
        variance_scale=resolved.variance_scale,
        comparison=comparison,
        continuous_covariance_by_block=resolved.continuous_covariance_by_block,
        ridge=resolved.ridge,
    ).whitened_vector()


def build_null_whitened_tangent_matrix(
    distributions: NDArray[np.floating],
    null_distribution: NDArray[np.floating],
    *,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
    ridge: float = 1e-12,
) -> NDArray[np.float64]:
    """Map distributions into the null-whitened tangent coordinates used by Wald."""
    distribution_matrix = np.asarray(distributions, dtype=np.float64)
    if distribution_matrix.ndim != 2:
        raise ValueError(
            "Tangent data must be a raw-coordinate 2-D matrix. "
            f"Got shape {distribution_matrix.shape}."
        )
    null = _as_flat_distribution_array(
        null_distribution,
        value_name="null_distribution",
    )
    active_feature_space = _resolve_distribution_feature_space(null, feature_space)
    ridge_value = _validate_ridge(ridge)
    if _uses_bernoulli_null_whitening(active_feature_space):
        distribution_matrix, null = _validate_bernoulli_tangent_inputs(
            distribution_matrix,
            null,
            active_feature_space,
        )
        if continuous_covariance_by_block is not None:
            raise ValueError(
                "continuous_covariance_by_block was provided, but feature_space has "
                "no continuous blocks."
            )
        return _build_bernoulli_null_whitened_tangent_matrix(
            distribution_matrix,
            null,
            active_feature_space,
            ridge=ridge_value,
        )

    if _uses_grouped_categorical_null_whitening(active_feature_space):
        distribution_matrix, null = _validate_grouped_categorical_tangent_inputs(
            distribution_matrix,
            null,
            active_feature_space,
            continuous_covariance_by_block=continuous_covariance_by_block,
        )
        return _build_trusted_null_whitened_tangent_matrix(
            distribution_matrix,
            null,
            active_feature_space,
            {},
            ridge=ridge_value,
        )

    distribution_matrix = validate_feature_matrix(
        distribution_matrix,
        active_feature_space,
        value_name="distributions",
    )
    null = validate_feature_vector(
        null,
        active_feature_space,
        value_name="null_distribution",
    )
    continuous_covariance_blocks = _validate_continuous_covariance_by_block(
        active_feature_space,
        continuous_covariance_by_block,
    )

    return _build_trusted_null_whitened_tangent_matrix(
        distribution_matrix,
        null,
        active_feature_space,
        continuous_covariance_blocks,
        ridge=ridge_value,
    )


def _build_trusted_null_whitened_tangent_matrix(
    distribution_matrix: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    feature_space: FeatureSpace,
    continuous_covariance_by_block: Mapping[str, NDArray[np.float64]],
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Map already-validated distributions into Wald tangent coordinates."""
    if _uses_bernoulli_null_whitening(feature_space):
        return _build_bernoulli_null_whitened_tangent_matrix(
            distribution_matrix,
            null_distribution,
            feature_space,
            ridge=ridge,
        )

    if _uses_grouped_categorical_null_whitening(feature_space):
        return _build_grouped_categorical_null_whitened_tangent_matrix(
            distribution_matrix,
            null_distribution,
            feature_space,
            ridge=ridge,
        )

    tangent_blocks = [
        _block_null_whitened_tangent_matrix(
            distribution_matrix,
            null_distribution,
            block,
            continuous_covariance_by_block=continuous_covariance_by_block,
            ridge=ridge,
        )
        for block in feature_space.blocks
    ]
    return np.column_stack(tangent_blocks)


def _compute_vectorized_whitened_wald_contrast(
    resolved: _ResolvedContrastInputs,
) -> NDArray[np.float64] | None:
    """Return a fast whitened contrast for common feature-space contracts."""
    if _uses_bernoulli_null_whitening(resolved.feature_space):
        return _build_bernoulli_whitened_wald_contrast(resolved)

    if _uses_grouped_categorical_null_whitening(resolved.feature_space):
        return _build_grouped_categorical_whitened_wald_contrast(resolved)

    if (
        resolved.feature_space.family_label == "continuous"
        and all(
            resolved.continuous_covariance_by_block[block.name].ndim == 1
            for block in resolved.feature_space.continuous_blocks
        )
    ):
        return _build_diagonal_continuous_whitened_wald_contrast(resolved)

    return None


def _uses_bernoulli_null_whitening(feature_space: FeatureSpace) -> bool:
    return feature_space.family_label == "bernoulli" and all(
        block.raw_dimension == 1 for block in feature_space.blocks
    )


def _uses_grouped_categorical_null_whitening(feature_space: FeatureSpace) -> bool:
    return feature_space.family_label == "categorical"


def _validate_bernoulli_tangent_inputs(
    distributions: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    feature_space: FeatureSpace,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if distributions.shape[1] != feature_space.raw_dimension:
        raise ValueError(
            f"distributions has shape {distributions.shape}; expected "
            f"(n_samples, {feature_space.raw_dimension})."
        )
    if not np.isfinite(distributions).all():
        raise ValueError("distributions must contain only finite values.")
    if null_distribution.shape != (feature_space.raw_dimension,):
        raise ValueError(
            f"null_distribution has shape {null_distribution.shape}; expected "
            f"{(feature_space.raw_dimension,)}."
        )

    if not _uses_bernoulli_null_whitening(feature_space):
        raise ValueError("Bernoulli null whitening requires pure Bernoulli blocks.")
    if np.any(distributions < 0.0) or np.any(distributions > 1.0):
        raise ValueError(
            "Bernoulli tangent distributions must lie in [0, 1]. "
            f"Range=[{float(np.min(distributions)):.6g}, "
            f"{float(np.max(distributions)):.6g}]."
        )
    if np.any(null_distribution < 0.0) or np.any(null_distribution > 1.0):
        raise ValueError(
            "Bernoulli null_distribution must lie in [0, 1]. "
            f"Range=[{float(np.min(null_distribution)):.6g}, "
            f"{float(np.max(null_distribution)):.6g}]."
        )

    return distributions, null_distribution


def _build_bernoulli_null_whitened_tangent_matrix(
    distributions: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    feature_space: FeatureSpace,
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Vectorize the exact block whitening map for pure Bernoulli feature spaces."""
    if not _uses_bernoulli_null_whitening(feature_space):
        raise ValueError("Bernoulli null whitening requires pure Bernoulli blocks.")

    column_indices = np.asarray(
        [block.column_indices[0] for block in feature_space.blocks],
        dtype=np.int64,
    )
    block_values = distributions[:, column_indices]
    block_null = null_distribution[column_indices]

    variances = block_null * (1.0 - block_null) + ridge
    return (block_values - block_null) / np.sqrt(variances)


def _build_bernoulli_whitened_wald_contrast(
    resolved: _ResolvedContrastInputs,
) -> NDArray[np.float64]:
    """Vectorize Wald whitening for pure Bernoulli feature spaces."""
    if not _uses_bernoulli_null_whitening(resolved.feature_space):
        raise ValueError("Bernoulli Wald whitening requires pure Bernoulli blocks.")

    column_indices = np.asarray(
        [block.column_indices[0] for block in resolved.feature_space.blocks],
        dtype=np.int64,
    )
    contrast = resolved.first[column_indices] - resolved.second[column_indices]
    probabilities = resolved.covariance_distribution[column_indices]
    variances = probabilities * (1.0 - probabilities) * resolved.variance_scale
    variances = variances + resolved.ridge
    return contrast / np.sqrt(variances)


def _validate_grouped_categorical_tangent_inputs(
    distributions: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    feature_space: FeatureSpace,
    *,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if distributions.shape[1] != feature_space.raw_dimension:
        raise ValueError(
            f"distributions has shape {distributions.shape}; expected "
            f"(n_samples, {feature_space.raw_dimension})."
        )
    if not np.isfinite(distributions).all():
        raise ValueError("distributions must contain only finite values.")
    if null_distribution.shape != (feature_space.raw_dimension,):
        raise ValueError(
            f"null_distribution has shape {null_distribution.shape}; expected "
            f"{(feature_space.raw_dimension,)}."
        )
    if continuous_covariance_by_block is not None:
        raise ValueError(
            "continuous_covariance_by_block was provided, but feature_space has "
            "no continuous blocks."
        )
    if np.any(distributions < 0.0) or np.any(distributions > 1.0):
        raise ValueError(
            "Categorical tangent distributions must lie in [0, 1]. "
            f"Range=[{float(np.min(distributions)):.6g}, "
            f"{float(np.max(distributions)):.6g}]."
        )
    if np.any(null_distribution < 0.0) or np.any(null_distribution > 1.0):
        raise ValueError(
            "Categorical null_distribution must lie in [0, 1]. "
            f"Range=[{float(np.min(null_distribution)):.6g}, "
            f"{float(np.max(null_distribution)):.6g}]."
        )

    for category_count, indexed_blocks in _categorical_blocks_by_category_count(
        feature_space
    ).items():
        column_indices = np.asarray(
            [block.column_indices for _block_index, block in indexed_blocks],
            dtype=np.int64,
        )
        block_values = distributions[:, column_indices]
        row_sums = block_values.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0):
            raise ValueError(
                "Categorical tangent distribution rows must sum to 1 within each "
                f"{category_count}-category block. Got row-sum range "
                f"[{float(np.min(row_sums)):.6g}, {float(np.max(row_sums)):.6g}]."
            )

        null_blocks = null_distribution[column_indices]
        null_sums = null_blocks.sum(axis=1)
        if not np.allclose(null_sums, 1.0, atol=1e-8, rtol=0.0):
            raise ValueError(
                "Categorical null_distribution blocks must sum to 1. "
                f"Got row-sum range "
                f"[{float(np.min(null_sums)):.6g}, {float(np.max(null_sums)):.6g}]."
            )

    return distributions, null_distribution


def _build_grouped_categorical_null_whitened_tangent_matrix(
    distributions: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    feature_space: FeatureSpace,
    *,
    ridge: float,
) -> NDArray[np.float64]:
    """Vectorize the exact simplex block whitening map by category count."""
    if not _uses_grouped_categorical_null_whitening(feature_space):
        raise ValueError(
            "Grouped categorical whitening requires a pure categorical feature space."
        )

    output = np.empty(
        (distributions.shape[0], feature_space.contrast_dimension),
        dtype=np.float64,
    )
    output_slices: list[slice] = []
    offset = 0
    for block in feature_space.blocks:
        next_offset = offset + block.contrast_dimension
        output_slices.append(slice(offset, next_offset))
        offset = next_offset

    for category_count, indexed_blocks in _categorical_blocks_by_category_count(
        feature_space
    ).items():
        contrast_dimension = category_count - 1
        column_indices = np.asarray(
            [block.column_indices for _block_index, block in indexed_blocks],
            dtype=np.int64,
        )
        block_values = distributions[:, column_indices]
        reduced_null = null_distribution[column_indices][:, :-1]

        covariance_blocks = -np.einsum(
            "bi,bj->bij",
            reduced_null,
            reduced_null,
            optimize=True,
        )
        diagonal = np.arange(contrast_dimension)
        covariance_blocks[:, diagonal, diagonal] += reduced_null
        covariance_blocks[:, diagonal, diagonal] += ridge

        cholesky_blocks = np.linalg.cholesky(covariance_blocks)
        tangent_blocks = block_values[:, :, :-1] - reduced_null[None, :, :]
        solved = np.linalg.solve(
            cholesky_blocks,
            np.transpose(tangent_blocks, (1, 2, 0)),
        )
        solved = np.transpose(solved, (2, 0, 1))

        for grouped_block_index, (block_index, _block) in enumerate(indexed_blocks):
            output[:, output_slices[block_index]] = solved[:, grouped_block_index, :]

    return output


def _build_grouped_categorical_whitened_wald_contrast(
    resolved: _ResolvedContrastInputs,
) -> NDArray[np.float64]:
    """Vectorize simplex-block Wald whitening by category count."""
    if not _uses_grouped_categorical_null_whitening(resolved.feature_space):
        raise ValueError(
            "Grouped categorical Wald whitening requires a pure categorical feature space."
        )

    output = np.empty(resolved.feature_space.contrast_dimension, dtype=np.float64)
    output_slices: list[slice] = []
    offset = 0
    for block in resolved.feature_space.blocks:
        next_offset = offset + block.contrast_dimension
        output_slices.append(slice(offset, next_offset))
        offset = next_offset

    for category_count, indexed_blocks in _categorical_blocks_by_category_count(
        resolved.feature_space
    ).items():
        contrast_dimension = category_count - 1
        column_indices = np.asarray(
            [block.column_indices for _block_index, block in indexed_blocks],
            dtype=np.int64,
        )
        first_blocks = resolved.first[column_indices]
        second_blocks = resolved.second[column_indices]
        reduced_probability = resolved.covariance_distribution[column_indices][:, :-1]

        covariance_blocks = -np.einsum(
            "bi,bj->bij",
            reduced_probability,
            reduced_probability,
            optimize=True,
        )
        diagonal = np.arange(contrast_dimension)
        covariance_blocks[:, diagonal, diagonal] += reduced_probability
        covariance_blocks *= resolved.variance_scale
        covariance_blocks[:, diagonal, diagonal] += resolved.ridge

        cholesky_blocks = np.linalg.cholesky(covariance_blocks)
        contrast_blocks = first_blocks[:, :-1] - second_blocks[:, :-1]
        solved = np.linalg.solve(cholesky_blocks, contrast_blocks[..., None])[:, :, 0]

        for grouped_block_index, (block_index, _block) in enumerate(indexed_blocks):
            output[output_slices[block_index]] = solved[grouped_block_index]

    return output


def _categorical_blocks_by_category_count(
    feature_space: FeatureSpace,
) -> dict[int, list[tuple[int, FeatureBlock]]]:
    blocks_by_category_count: dict[int, list[tuple[int, FeatureBlock]]] = {}
    for block_index, block in enumerate(feature_space.blocks):
        if block.family != "categorical":
            raise ValueError(
                "Categorical block grouping requires a pure categorical feature space."
            )
        blocks_by_category_count.setdefault(block.raw_dimension, []).append(
            (block_index, block)
        )
    return blocks_by_category_count


def _resolve_distribution_feature_space(
    distribution: NDArray[np.float64],
    feature_space: FeatureSpace | None,
) -> FeatureSpace:
    if feature_space is not None:
        if feature_space.raw_dimension != distribution.shape[0]:
            raise ValueError(
                "feature_space raw dimension does not match distribution width. "
                f"feature_space.raw_dimension={feature_space.raw_dimension}, "
                f"distribution_width={distribution.shape[0]}."
            )
        return feature_space
    return bernoulli_feature_space_from_columns(
        tuple(f"x{column_index}" for column_index in range(distribution.shape[0]))
    )


def _as_flat_distribution_array(
    distribution: NDArray[np.floating],
    *,
    value_name: str,
) -> NDArray[np.float64]:
    array = np.asarray(distribution, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"{value_name} must be a flat raw-coordinate distribution vector. "
            f"Got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{value_name} must contain only finite values.")
    return array


def _validate_sample_sizes(first_sample_size: float, second_sample_size: float) -> None:
    if (
        not np.isfinite(first_sample_size)
        or not np.isfinite(second_sample_size)
        or first_sample_size <= 0.0
        or second_sample_size <= 0.0
    ):
        raise ValueError(
            "Sample sizes must be finite positive values. "
            f"Got first_sample_size={first_sample_size}, "
            f"second_sample_size={second_sample_size}."
        )


def _validate_ridge(ridge: float) -> float:
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError(f"ridge must be finite and non-negative. Got ridge={ridge!r}.")
    return ridge_value


def _validate_continuous_covariance_by_block(
    feature_space: FeatureSpace,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None,
) -> dict[str, NDArray[np.float64]]:
    continuous_blocks = feature_space.continuous_blocks
    if not continuous_blocks:
        if continuous_covariance_by_block is not None:
            raise ValueError(
                "continuous_covariance_by_block was provided, but feature_space has "
                "no continuous blocks."
            )
        return {}
    if continuous_covariance_by_block is None:
        raise ValueError(
            "Continuous feature blocks require continuous_covariance_by_block."
        )

    expected_block_names = {block.name for block in continuous_blocks}
    actual_block_names = set(continuous_covariance_by_block)
    if actual_block_names != expected_block_names:
        raise ValueError(
            "continuous_covariance_by_block keys must exactly match continuous "
            "feature block names. "
            f"expected={sorted(expected_block_names)!r}, "
            f"actual={sorted(actual_block_names)!r}."
        )

    covariance_blocks: dict[str, NDArray[np.float64]] = {}
    for block in continuous_blocks:
        covariance = np.asarray(
            continuous_covariance_by_block[block.name],
            dtype=np.float64,
        )
        expected_matrix_shape = (block.raw_dimension, block.raw_dimension)
        expected_diagonal_shape = (block.raw_dimension,)
        if covariance.shape not in {expected_matrix_shape, expected_diagonal_shape}:
            raise ValueError(
                f"Continuous covariance block {block.name!r} has shape "
                f"{covariance.shape}; expected {expected_matrix_shape} "
                f"or diagonal shape {expected_diagonal_shape}."
            )
        if not np.isfinite(covariance).all():
            raise ValueError(
                f"Continuous covariance block {block.name!r} must contain only finite values."
            )
        if covariance.ndim == 1:
            if float(np.min(covariance)) < -1e-10:
                raise ValueError(
                    f"Continuous diagonal covariance block {block.name!r} must be "
                    "non-negative."
                )
            covariance_blocks[block.name] = covariance
            continue
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8):
            raise ValueError(
                f"Continuous covariance block {block.name!r} must be symmetric."
            )
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) < -1e-10:
            raise ValueError(
                f"Continuous covariance block {block.name!r} must be positive "
                "semidefinite."
            )
        covariance_blocks[block.name] = covariance
    return covariance_blocks


def _sibling_variance_scale(first_sample_size: float, second_sample_size: float) -> float:
    return 1.0 / first_sample_size + 1.0 / second_sample_size


def _child_parent_variance_scale(child_sample_size: float, parent_sample_size: float) -> float:
    nested_factor = 1.0 / child_sample_size - 1.0 / parent_sample_size
    if nested_factor <= 0.0:
        raise ValueError(
            "Invalid tree structure: child sample size must be strictly less than "
            "parent sample size. "
            f"Got child_sample_size={child_sample_size}, "
            f"parent_sample_size={parent_sample_size}, "
            f"nested_factor={nested_factor:.6f}."
        )
    return nested_factor


def _tree_time_variance_multiplier(
    tree_time: float | None,
    *,
    tree_time_normalizer: float | None,
) -> float:
    """Return variance inflation from normalized tree time.

    The multiplier keeps finite-sample variance as the baseline and allows
    longer tree time to explain more contrast under the null. Missing or zero
    time preserves the legacy sampling-only variance.
    """
    if tree_time is None:
        return 1.0
    tree_time_value = float(tree_time)
    if not np.isfinite(tree_time_value) or tree_time_value < 0.0:
        raise ValueError(
            f"tree_time must be a finite non-negative value; got {tree_time!r}."
        )
    if tree_time_value == 0.0:
        return 1.0
    if tree_time_normalizer is None:
        raise ValueError(
            "tree_time_normalizer is required when tree_time is positive; "
            f"got tree_time={tree_time!r}."
        )
    normalizer = float(tree_time_normalizer)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(
            "tree_time_normalizer must be a finite positive value when "
            f"tree_time is positive; got {tree_time_normalizer!r}."
        )
    return 1.0 + tree_time_value / normalizer


def _feature_space_contrast_covariance(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    covariance_distribution: NDArray[np.float64],
    *,
    feature_space: FeatureSpace,
    variance_scale: float,
    comparison: ComparisonKind,
    continuous_covariance_by_block: Mapping[str, NDArray[np.float64]],
    ridge: float,
) -> ContrastCovariance:
    contrast_blocks: list[NDArray[np.float64]] = []
    covariance_blocks: list[NDArray[np.float64]] = []
    for block in feature_space.blocks:
        block_contrast, block_covariance = _block_contrast_covariance(
            first,
            second,
            covariance_distribution,
            block,
            variance_scale=variance_scale,
            continuous_covariance_by_block=continuous_covariance_by_block,
            ridge=ridge,
        )
        contrast_blocks.append(block_contrast)
        covariance_blocks.append(block_covariance)

    return ContrastCovariance(
        contrast_vector=np.concatenate(contrast_blocks, axis=0),
        covariance_blocks=tuple(covariance_blocks),
        feature_family=_feature_family_label(feature_space),
        comparison=comparison,
    )


def _block_contrast_covariance(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    covariance_distribution: NDArray[np.float64],
    block: FeatureBlock,
    *,
    variance_scale: float,
    continuous_covariance_by_block: Mapping[str, NDArray[np.float64]],
    ridge: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    column_indices = list(block.column_indices)
    block_first = first[column_indices]
    block_second = second[column_indices]
    block_covariance_distribution = covariance_distribution[column_indices]

    if block.family == "bernoulli":
        probability = float(block_covariance_distribution[0])
        variance = probability * (1.0 - probability) * variance_scale + ridge
        return (
            np.array([float(block_first[0] - block_second[0])], dtype=np.float64),
            np.array([[variance]], dtype=np.float64),
        )

    if block.family == "categorical":
        reduced_first = block_first[:-1]
        reduced_second = block_second[:-1]
        reduced_probability = block_covariance_distribution[:-1]
        covariance = (
            np.diag(reduced_probability)
            - np.outer(reduced_probability, reduced_probability)
        ) * variance_scale + ridge * np.eye(block.contrast_dimension, dtype=np.float64)
        return (
            (reduced_first - reduced_second).astype(np.float64, copy=False),
            covariance,
        )

    if block.family == "continuous":
        covariance_block = continuous_covariance_by_block[block.name]
        if covariance_block.ndim == 1:
            covariance = np.diag(
                covariance_block * variance_scale + ridge,
            )
        else:
            covariance = (
                covariance_block * variance_scale
                + ridge * np.eye(block.raw_dimension, dtype=np.float64)
            )
        return (
            (block_first - block_second).astype(np.float64, copy=False),
            covariance,
        )

    raise ValueError(f"Unknown feature family: {block.family!r}.")


def _block_null_whitened_tangent_matrix(
    distributions: NDArray[np.float64],
    null_distribution: NDArray[np.float64],
    block: FeatureBlock,
    *,
    continuous_covariance_by_block: Mapping[str, NDArray[np.float64]],
    ridge: float,
) -> NDArray[np.float64]:
    column_indices = list(block.column_indices)
    block_values = distributions[:, column_indices]
    block_null = null_distribution[column_indices]

    if block.family == "bernoulli":
        probability = float(block_null[0])
        variance = probability * (1.0 - probability) + ridge
        return (block_values - probability) / np.sqrt(variance)

    if block.family == "categorical":
        reduced_null = block_null[:-1]
        covariance_block = (
            np.diag(reduced_null)
            - np.outer(reduced_null, reduced_null)
            + ridge * np.eye(block.contrast_dimension, dtype=np.float64)
        )
        cho, lower = linalg.cho_factor(
            covariance_block,
            lower=True,
            check_finite=False,
        )
        tangent_block = block_values[:, :-1] - reduced_null
        return linalg.solve_triangular(
            cho,
            tangent_block.T,
            lower=lower,
            check_finite=False,
        ).T

    if block.family == "continuous":
        covariance_block = continuous_covariance_by_block[block.name]
        tangent_block = block_values - block_null
        if covariance_block.ndim == 1:
            return tangent_block / np.sqrt(covariance_block + ridge)
        covariance_block = covariance_block + ridge * np.eye(
            block.raw_dimension,
            dtype=np.float64,
        )
        cho, lower = linalg.cho_factor(
            covariance_block,
            lower=True,
            check_finite=False,
        )
        return linalg.solve_triangular(
            cho,
            tangent_block.T,
            lower=lower,
            check_finite=False,
        ).T

    raise ValueError(f"Unknown feature family: {block.family!r}.")


def _build_diagonal_continuous_whitened_wald_contrast(
    resolved: _ResolvedContrastInputs,
) -> NDArray[np.float64]:
    output_blocks: list[NDArray[np.float64]] = []
    for block in resolved.feature_space.continuous_blocks:
        column_indices = list(block.column_indices)
        covariance = resolved.continuous_covariance_by_block[block.name]
        if covariance.ndim != 1:
            raise ValueError("Diagonal continuous whitening requires vector variances.")
        variances = covariance * resolved.variance_scale + resolved.ridge
        contrast = resolved.first[column_indices] - resolved.second[column_indices]
        output_blocks.append(contrast / np.sqrt(variances))
    return np.concatenate(output_blocks, axis=0)


def _feature_family_label(feature_space: FeatureSpace) -> FeatureSpaceLabel:
    family_label = feature_space.family_label
    if family_label in {"bernoulli", "categorical", "continuous", "mixed"}:
        return family_label
    raise ValueError(f"Unknown feature family label: {family_label!r}.")


__all__ = [
    "ComparisonKind",
    "ContrastCovariance",
    "FeatureSpaceLabel",
    "build_contrast_covariance",
    "build_null_whitened_tangent_matrix",
    "compute_whitened_wald_contrast",
]
