from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.diagnostics.spectral.local_mp_identity_law_diagnostic import (
    centered_self_whitening_reference_eigenvalue,
    finite_identity_null_top_eigenvalue_quantile,
    identity_mp_positive_quantiles,
    identity_mp_support,
    summarize_categorical_extreme_nodes,
    summarize_node_spectrum,
    summarize_spectral_law_relationships,
)
from benchmarks.diagnostics.spectral.profile_spectral_backends import SpectralMatrixRecord


def test_identity_mp_support_matches_closed_form_edge() -> None:
    support = identity_mp_support(4.0)

    assert support.lower_edge == 1.0
    assert support.upper_edge == 9.0
    assert support.positive_mass == 0.25


def test_identity_mp_positive_quantiles_are_inside_support() -> None:
    support = identity_mp_support(1.0)
    quantiles = identity_mp_positive_quantiles(1.0, (0.5, 0.9), grid_size=1024)

    assert support.lower_edge <= quantiles[0.5] <= support.upper_edge
    assert support.lower_edge <= quantiles[0.9] <= support.upper_edge
    assert quantiles[0.5] < quantiles[0.9]


def test_summarize_node_spectrum_detects_strong_spike() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(80, 10))
    matrix[:, 0] *= 20.0
    record = SpectralMatrixRecord(
        node_id="u",
        matrix=matrix,
        descendant_leaf_rows=80,
        feature_count=10,
    )

    summary = summarize_node_spectrum(record, min_positive_eigenvalues=4, grid_size=1024)

    assert summary["diagnostic_status"] == "evaluated"
    assert summary["raw_mp_signal_count"] >= 1
    assert summary["top_eigenvalue_over_mp_upper"] > 1.0
    assert summary["above_finite_identity_null_top_quantile"] is True
    assert summary["top_eigenvalue_over_finite_identity_null_quantile"] > 1.0
    assert summary["top_eigenvalue_over_centered_self_whitening_reference"] > 1.0


def test_summarize_node_spectrum_reports_insufficient_rows() -> None:
    record = SpectralMatrixRecord(
        node_id="leaf",
        matrix=np.zeros((1, 4), dtype=np.float64),
        descendant_leaf_rows=1,
        feature_count=4,
    )

    summary = summarize_node_spectrum(record)

    assert summary["diagnostic_status"] == "insufficient_rows"


def test_centered_self_whitening_reference_uses_backend_covariance_scale() -> None:
    assert centered_self_whitening_reference_eigenvalue(10) == 0.9


def test_finite_identity_null_quantile_is_reproducible_and_above_mp_edge() -> None:
    support = identity_mp_support(10 / 80)
    first = finite_identity_null_top_eigenvalue_quantile(80, 10, 12, 0.95, 7)
    second = finite_identity_null_top_eigenvalue_quantile(80, 10, 12, 0.95, 7)

    assert first == second
    assert first > support.upper_edge


def test_spectral_law_relationships_report_explicit_statuses() -> None:
    node_spectrum = {
        "case_name": ["a", "a", "a", "a"],
        "source_spectral_family": ["bernoulli_discretized"] * 4,
        "diagnostic_status": ["evaluated"] * 4,
        "log_top_eigenvalue_over_finite_identity_null_quantile": [0.0, 0.2, 0.4, 0.8],
        "log_node_size_fraction": [0.0, -0.1, -0.3, -0.6],
        "log_aspect_ratio": [1.0, 1.0, 1.0, 1.0],
        "log_matrix_rows": [4.0, 3.0, 2.0, 1.0],
        "log_active_feature_count": [1.0, 2.0, 3.0, 4.0],
    }

    relationships = summarize_spectral_law_relationships(pd.DataFrame(node_spectrum))

    status_by_covariate = dict(
        zip(relationships["covariate"], relationships["relationship_status"], strict=False)
    )
    assert status_by_covariate["log_node_size_fraction"] == "evaluated"
    assert status_by_covariate["log_aspect_ratio"] == "constant_covariate"


def test_spectral_law_relationships_report_constant_target() -> None:
    node_spectrum = pd.DataFrame(
        {
            "case_name": ["a", "a", "a"],
            "source_spectral_family": ["categorical"] * 3,
            "diagnostic_status": ["evaluated"] * 3,
            "log_top_eigenvalue_over_finite_identity_null_quantile": [0.2, 0.2, 0.2],
            "log_node_size_fraction": [0.0, -0.2, -0.4],
            "log_aspect_ratio": [1.0, 1.2, 1.4],
            "log_matrix_rows": [4.0, 3.0, 2.0],
            "log_active_feature_count": [2.0, 2.2, 2.4],
        }
    )

    relationships = summarize_spectral_law_relationships(node_spectrum)

    assert set(relationships["relationship_status"]) == {"constant_target"}


def test_categorical_extreme_nodes_are_reported_separately() -> None:
    node_spectrum = pd.DataFrame(
        {
            "case_name": ["cat", "cat", "bin"],
            "node_id": ["u", "v", "w"],
            "source_spectral_family": ["categorical", "categorical", "bernoulli"],
            "matrix_rows": [50, 40, 30],
            "node_size_fraction": [1.0, 0.8, 0.6],
            "active_feature_count": [20, 20, 10],
            "aspect_ratio": [0.4, 0.5, 0.3],
            "raw_mp_signal_count": [1, 0, 2],
            "top_eigenvalue_over_mp_upper": [1.2, 0.8, 1.4],
            "top_eigenvalue_over_finite_identity_null_quantile": [1.1, 0.7, 1.3],
            "finite_identity_null_top_over_mp_upper": [1.04, 1.05, 1.03],
            "above_finite_identity_null_top_quantile": [True, False, True],
        }
    )

    extreme_nodes = summarize_categorical_extreme_nodes(node_spectrum)

    assert list(extreme_nodes["node_id"]) == ["u"]
