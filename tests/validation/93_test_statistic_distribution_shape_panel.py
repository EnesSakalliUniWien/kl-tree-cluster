from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.statistic_distribution_shape_panel import (
    STUDY_ROLE,
    infer_satterthwaite_reference_from_eigenvalues,
    normalize_statistic_distribution_records,
    run_statistic_distribution_shape_panel,
    summarize_statistic_distribution_shape,
)
from scipy.stats import chi2


def _chi_square_quantile_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    probabilities = (np.arange(1, 101, dtype=float) - 0.5) / 100.0
    for df in (1.0, 4.0):
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "record_id": f"df_{df:g}_row_{index}",
                    "test_family": "bernoulli",
                    "statistic_context_role": "strict_null",
                    "test_statistic": float(chi2.ppf(probability, df=df)),
                    "degrees_of_freedom": df,
                    "alternate_degrees_of_freedom": df,
                    "alternate_reference_scale": 1.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_satterthwaite_reference_matches_equal_weights() -> None:
    scale, degrees = infer_satterthwaite_reference_from_eigenvalues([2.0, 2.0, 2.0])

    assert scale == pytest.approx(2.0)
    assert degrees == pytest.approx(3.0)


def test_distribution_shape_panel_separates_df_skew_from_tail_shape() -> None:
    rows = normalize_statistic_distribution_records(_chi_square_quantile_records())
    summary = summarize_statistic_distribution_shape(
        rows,
        group_columns=("test_family", "statistic_context_role"),
        min_rows=30,
        skew_tolerance=0.6,
    )
    result = summary.iloc[0]

    assert rows["study_role"].eq(STUDY_ROLE).all()
    assert result["distribution_shape_status"] == "chi_square_shape_candidate"
    assert abs(result["chi_square_tail_delta_from_alpha"]) <= 0.02
    assert abs(result["skew_excess_over_df_reference"]) <= 0.6
    assert result["n_alternate_df_rows"] == 200


def test_distribution_shape_panel_reports_alternate_df_tail_difference() -> None:
    records = _chi_square_quantile_records()
    records["alternate_degrees_of_freedom"] = 1.0
    records["alternate_reference_scale"] = 1.0
    rows = normalize_statistic_distribution_records(records)
    summary = summarize_statistic_distribution_shape(
        rows,
        group_columns=("test_family", "statistic_context_role"),
        min_rows=30,
        skew_tolerance=0.6,
    )
    result = summary.iloc[0]

    assert result["df_difference_alternate_minus_current_mean"] < 0.0
    assert result["alternate_chi_square_tail_rate_at_alpha"] > (
        result["chi_square_tail_rate_at_alpha"]
    )
    assert result["alternate_chi_square_pit_ks_p_value"] < (
        result["chi_square_pit_ks_p_value"]
    )


def test_distribution_shape_panel_rejects_invalid_rows() -> None:
    records = _chi_square_quantile_records()
    records.loc[0, "degrees_of_freedom"] = 0.0

    with pytest.raises(ValueError, match="degrees_of_freedom"):
        normalize_statistic_distribution_records(records)


def test_run_statistic_distribution_shape_panel_writes_outputs(tmp_path: Path) -> None:
    records_path = tmp_path / "records.csv"
    output_dir = tmp_path / "out"
    _chi_square_quantile_records().to_csv(records_path, index=False)

    outputs = run_statistic_distribution_shape_panel(
        records_path=records_path,
        output_dir=output_dir,
        statistic_column="test_statistic",
        degrees_of_freedom_column="degrees_of_freedom",
        context_role_column="statistic_context_role",
        test_family_column="test_family",
        alternate_degrees_of_freedom_column="alternate_degrees_of_freedom",
        alternate_reference_scale_column="alternate_reference_scale",
        min_rows=30,
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert (output_dir / "statistic_distribution_shape_rows.csv").exists()
    assert (output_dir / "statistic_distribution_shape_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
