from __future__ import annotations

import pytest

from benchmarks.diagnostics.selection_conditioned_sibling_null import (
    CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE,
    NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE,
    SelectionConditionedSiblingNullContext,
    simulate_local_edge_selection_conditioned_null,
)


def _context(**overrides) -> SelectionConditionedSiblingNullContext:
    values = {
        "case_id": "synthetic",
        "parent": "N0",
        "feature_family": "continuous",
        "standardized_contrast_dimension": 8,
        "edge_projection_dimension": 4,
        "sibling_projection_dimension": 4,
        "degrees_of_freedom": 4.0,
        "reference_scale": 1.0,
        "observed_statistic": 18.0,
        "observed_p_value": 0.001234,
        "edge_alpha": 0.05,
        "sibling_alpha": 0.01,
        "parent_sample_size": 20,
        "left_sample_size": 10,
        "right_sample_size": 10,
    }
    values.update(overrides)
    return SelectionConditionedSiblingNullContext(**values)


def test_local_edge_selection_inflates_mean_above_unconditioned_reference() -> None:
    result = simulate_local_edge_selection_conditioned_null(
        _context(),
        n_candidates=100_000,
        seed=123,
        chunk_size=20_000,
    )

    assert result.context.conditioning_scope == CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE
    assert result.acceptance_rate == pytest.approx(0.05, rel=0.2)
    assert result.conditional_mean_statistic > result.context.reference_expectation
    assert result.conditional_mean_over_reference > 2.0
    assert 0.0 < result.empirical_tail_p_value < 1.0


def test_noncontinuous_context_is_labeled_as_z_proxy() -> None:
    result = simulate_local_edge_selection_conditioned_null(
        _context(feature_family="bernoulli"),
        n_candidates=10_000,
        seed=456,
        chunk_size=5_000,
    )

    assert result.context.conditioning_scope == NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE


def test_sibling_projection_must_fit_inside_edge_projection() -> None:
    with pytest.raises(ValueError, match="sibling_projection_dimension cannot exceed"):
        simulate_local_edge_selection_conditioned_null(
            _context(
                edge_projection_dimension=3,
                sibling_projection_dimension=4,
            ),
            n_candidates=1_000,
            seed=1,
        )


def test_zero_acceptance_fails_clearly() -> None:
    with pytest.raises(ValueError, match="accepted zero candidates"):
        simulate_local_edge_selection_conditioned_null(
            _context(edge_alpha=1e-12),
            n_candidates=100,
            seed=1,
        )
