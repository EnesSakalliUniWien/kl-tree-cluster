from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.statistics.covariance_laplacian_panel import (
    STUDY_ROLE,
    analyze_covariance_laplacian,
    analyze_covariance_laplacian_panel,
    run_covariance_laplacian_panel,
    summarize_covariance_laplacian_rows,
)


def test_laplacian_panel_marks_diagonal_covariance_as_disconnected() -> None:
    row = analyze_covariance_laplacian(
        np.eye(4),
        matrix_id="diag",
        matrix_context_role="unit_test",
        feature_family="bernoulli",
    )

    assert row["laplacian_status"] == "laplacian_diagonal_covariance"
    assert row["n_connected_components"] == 4
    assert row["off_diagonal_abs_mass_fraction"] == pytest.approx(0.0)
    assert row["study_role"] == STUDY_ROLE


def test_laplacian_panel_marks_block_covariance_as_disconnected() -> None:
    covariance = np.array(
        [
            [1.0, 0.4, 0.0, 0.0],
            [0.4, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )
    row = analyze_covariance_laplacian(
        covariance,
        matrix_id="block",
        matrix_context_role="unit_test",
    )

    assert row["laplacian_status"] == "laplacian_disconnected_covariance"
    assert row["n_connected_components"] == 2
    assert row["largest_component_fraction"] == pytest.approx(0.5)


def test_laplacian_panel_marks_dense_covariance_as_connected() -> None:
    covariance = np.full((5, 5), 0.25)
    np.fill_diagonal(covariance, 1.0)

    row = analyze_covariance_laplacian(
        covariance,
        matrix_id="dense",
        matrix_context_role="unit_test",
        min_algebraic_connectivity=1e-3,
    )

    assert row["laplacian_status"] == "laplacian_connected_covariance"
    assert row["n_connected_components"] == 1
    assert row["normalized_laplacian_lambda2"] > 0.0


def test_laplacian_panel_summarizes_rows() -> None:
    rows = analyze_covariance_laplacian_panel(
        [
            ("diag", np.eye(3), {"feature_family": "bernoulli"}),
            (
                "dense",
                np.full((3, 3), 0.2) + 0.8 * np.eye(3),
                {"feature_family": "bernoulli"},
            ),
        ]
    )
    summary = summarize_covariance_laplacian_rows(rows)

    assert set(summary["laplacian_status"]) == {
        "laplacian_connected_covariance",
        "laplacian_diagonal_covariance",
    }


def test_laplacian_panel_rejects_invalid_matrix() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        analyze_covariance_laplacian(
            np.array([[1.0, 0.2], [0.4, 1.0]]),
            matrix_id="bad",
        )


def test_run_covariance_laplacian_panel_writes_outputs(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    pd.DataFrame(np.eye(3)).to_csv(matrix_dir / "identity.csv", header=False, index=False)
    output_dir = tmp_path / "out"

    outputs = run_covariance_laplacian_panel(
        matrix_dir=matrix_dir,
        output_dir=output_dir,
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert (output_dir / "covariance_laplacian_rows.csv").exists()
    assert (output_dir / "covariance_laplacian_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
