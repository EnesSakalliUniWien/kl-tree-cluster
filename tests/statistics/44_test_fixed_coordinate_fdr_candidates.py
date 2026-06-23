from __future__ import annotations

import numpy as np
import pytest
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    coordinate_fdr_adjusted_minimum,
    fixed_block_simes_bh_p_value,
    fixed_coordinate_fdr_p_value,
    fixed_subspace_sibling_p_value,
)
from tree_break_selection.tree.feature_space import infer_feature_space_from_columns


def _categorical_feature_space(*, n_blocks: int, n_categories: int):
    return infer_feature_space_from_columns(
        tuple(
            f"F{feature_index}_c{category_index}"
            for feature_index in range(n_blocks)
            for category_index in range(n_categories)
        )
    )


def test_bh_reacts_to_repeated_moderate_coordinate_evidence() -> None:
    p_values = np.ones(100, dtype=np.float64)
    p_values[:10] = 0.004

    assert coordinate_fdr_adjusted_minimum(p_values, method="bh") < 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="by") > 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="holm") > 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="bonferroni") > 0.05


def test_bonferroni_and_holm_only_react_to_very_strong_sparse_evidence() -> None:
    p_values = np.ones(100, dtype=np.float64)
    p_values[0] = 0.0004

    assert coordinate_fdr_adjusted_minimum(p_values, method="bh") < 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="holm") < 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="bonferroni") < 0.05
    assert coordinate_fdr_adjusted_minimum(p_values, method="by") > 0.05


def test_benjamini_yekutieli_applies_harmonic_dependence_penalty() -> None:
    p_values = np.ones(100, dtype=np.float64)
    p_values[0] = 0.0004
    harmonic = float(np.sum(1.0 / np.arange(1, p_values.size + 1, dtype=float)))

    bh_adjusted = coordinate_fdr_adjusted_minimum(p_values, method="bh")
    by_adjusted = coordinate_fdr_adjusted_minimum(p_values, method="by")

    assert by_adjusted == pytest.approx(harmonic * bh_adjusted)
    assert by_adjusted > 0.05

    p_values[0] = 0.00001
    assert coordinate_fdr_adjusted_minimum(p_values, method="by") < 0.05


def test_block_chi_square_reacts_to_dense_categorical_block_evidence() -> None:
    feature_space = _categorical_feature_space(n_blocks=10, n_categories=5)
    z = np.zeros(feature_space.contrast_dimension, dtype=np.float64)
    z[:4] = 2.0

    coordinate_bh = fixed_coordinate_fdr_p_value(z, method="bh")
    block_simes_bh = fixed_block_simes_bh_p_value(z, feature_space)
    block_chi_square_bh = fixed_subspace_sibling_p_value(
        z,
        feature_space,
        method="fixed_block_bh",
    )

    assert coordinate_bh > 0.05
    assert block_simes_bh > 0.05
    assert block_chi_square_bh < 0.05


def test_coordinate_fdr_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unknown coordinate FDR method"):
        coordinate_fdr_adjusted_minimum(
            np.array([0.01, 0.2], dtype=np.float64),
            method="adaptive",  # type: ignore[arg-type]
        )
