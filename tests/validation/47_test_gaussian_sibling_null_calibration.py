from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.gaussian_sibling_null_calibration import (
    CONTINUOUS_FIXED_SUBSPACE_SCOPE,
    NONCONTINUOUS_Z_PROXY_SCOPE,
    GaussianSiblingNullContext,
    append_external_gaussian_null_columns,
    simulate_fixed_subspace_gaussian_null,
)


def _context(**overrides) -> GaussianSiblingNullContext:
    values = {
        "case_id": "synthetic",
        "parent": "N0",
        "feature_family": "continuous",
        "standardized_contrast_dimension": 8,
        "projection_dimension": 4,
        "degrees_of_freedom": 4.0,
        "reference_scale": 1.0,
        "observed_statistic": 12.0,
        "observed_p_value": 0.017351265,
        "sibling_alpha": 0.01,
        "parent_sample_size": 20,
        "left_sample_size": 10,
        "right_sample_size": 10,
    }
    values.update(overrides)
    return GaussianSiblingNullContext(**values)


def test_fixed_subspace_gaussian_null_has_unit_mean_over_reference() -> None:
    result = simulate_fixed_subspace_gaussian_null(
        _context(),
        n_replicates=10_000,
        seed=123,
    )

    assert result.context.validity_scope == CONTINUOUS_FIXED_SUBSPACE_SCOPE
    assert 0.95 < result.mean_over_reference < 1.05
    assert 0.0 < result.empirical_tail_p_value < 1.0
    assert result.chi2_reference_p_value == pytest.approx(
        result.context.observed_p_value,
        rel=5e-7,
    )


def test_noncontinuous_feature_family_is_labeled_as_z_proxy() -> None:
    result = simulate_fixed_subspace_gaussian_null(
        _context(feature_family="bernoulli"),
        n_replicates=1_000,
        seed=456,
    )

    assert result.context.validity_scope == NONCONTINUOUS_Z_PROXY_SCOPE


def test_fixed_subspace_gaussian_null_rejects_mismatched_reference_law() -> None:
    with pytest.raises(ValueError, match="degrees_of_freedom to equal"):
        simulate_fixed_subspace_gaussian_null(
            _context(degrees_of_freedom=3.0),
            n_replicates=10,
            seed=1,
        )


def test_external_null_columns_must_cover_every_target() -> None:
    target_rows = pd.DataFrame.from_records(
        [
            {"case_id": "synthetic", "parent": "N0", "blocker_candidate": True},
            {"case_id": "synthetic", "parent": "N1", "blocker_candidate": True},
        ]
    )
    result = simulate_fixed_subspace_gaussian_null(
        _context(parent="N0"),
        n_replicates=10,
        seed=1,
    )

    with pytest.raises(ValueError, match="did not cover every target row"):
        append_external_gaussian_null_columns(target_rows, (result,))
