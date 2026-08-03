from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected.hierarchy.selected_quadratic_law_audit import (
    build_selected_quadratic_law_audit_tables,
    run_selected_quadratic_law_audit,
)


def _records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index, case_id in enumerate(("case_low", "case_mid", "case_high")):
        for branch_source in ("linkage_ultrametric_diagnostic", "fixed_topology_nnls"):
            for index in range(8):
                spectrum_level = 2.0 + group_index * 2.0
                stat = (index + 1.0) * (group_index + 1.0) * (1.0 + 0.05 * index)
                rows.append(
                    {
                        "source_case_id": case_id,
                        "branch_source": branch_source,
                        "spectral_context": "leaf_only",
                        "parent": f"parent_{index % 3}",
                        "stat": stat,
                        "degrees_of_freedom": 2.0,
                        "reference_scale": 1.0,
                        "branch_length_sum": 0.1 + 0.02 * index,
                        "n_parent": 4 + index,
                        "sibling_null_weight": 1.0,
                        "parent_spectral_log_pseudodeterminant": spectrum_level,
                        "parent_spectral_geometric_mean": float(np.exp(spectrum_level / 2.0)),
                        "parent_effective_rank": 1.0 + group_index,
                        "parent_eigenvalue_sum": 3.0 + group_index,
                        "parent_top_spectral_gap": 0.2 * group_index,
                    }
                )
    return pd.DataFrame(rows)


def _cells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index, case_id in enumerate(("case_low", "case_mid", "case_high")):
        for branch_source in ("linkage_ultrametric_diagnostic", "fixed_topology_nnls"):
            rows.append(
                {
                    "source_case_id": case_id,
                    "branch_source": branch_source,
                    "spectral_context": "leaf_only",
                    "n_samples": 24 + group_index * 8,
                    "n_features": 6 + group_index * 2,
                    "true_clusters": 2 + group_index,
                    "tree_n_leaves": 24 + group_index * 8,
                    "tree_n_internal_nodes": 23 + group_index * 8,
                    "tree_n_nodes": 47 + group_index * 16,
                    "spectral_total_descendant_leaf_rows": 80 + group_index * 12,
                    "spectral_total_internal_distribution_rows": 8 + group_index * 2,
                    "spectral_total_matrix_rows": 88 + group_index * 14,
                    "spectral_max_internal_distribution_rows": 3 + group_index,
                    "root_descendant_leaf_rows": 24 + group_index * 8,
                    "root_internal_distribution_rows": 2 + group_index,
                    "root_spectral_matrix_rows": 26 + group_index * 9,
                }
            )
    return pd.DataFrame(rows)


def _parent_eigenvalues() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_index, case_id in enumerate(("case_low", "case_mid", "case_high")):
        for branch_source in ("linkage_ultrametric_diagnostic", "fixed_topology_nnls"):
            for parent_index in range(3):
                for eigenvalue_index, eigenvalue in enumerate(
                    (3.0 + case_index, 1.5 + 0.2 * parent_index)
                ):
                    rows.append(
                        {
                            "source_case_id": case_id,
                            "branch_source": branch_source,
                            "spectral_context": "leaf_only",
                            "parent": f"parent_{parent_index}",
                            "eigenvalue_index": eigenvalue_index,
                            "eigenvalue": eigenvalue,
                            "parent_descendant_leaf_rows": 8 + parent_index,
                            "parent_internal_distribution_rows": parent_index,
                            "parent_spectral_matrix_rows": 8 + parent_index * 2,
                            "parent_active_feature_count": 4 + case_index,
                            "parent_mp_threshold_rows": 8 + parent_index,
                        }
                    )
    return pd.DataFrame(rows)


def test_selected_quadratic_law_audit_builds_df_scale_and_status_tables() -> None:
    tables = build_selected_quadratic_law_audit_tables(
        _records(),
        parent_eigenvalues=_parent_eigenvalues(),
        cells=_cells(),
    )

    assert set(tables) == {
        "prepared_records",
        "frame_dimensions",
        "parent_spectrum_moments",
        "group_fits",
        "factor_skew_summary",
        "factor_skew_bins",
        "predictor_fits",
        "calibration_comparison",
        "eigen_satterthwaite_comparison",
        "predicted_spectrum_model",
        "generalized_chi_square_status",
    }
    assert not tables["group_fits"].empty
    assert {
        "satterthwaite_df",
        "satterthwaite_scale",
        "mle_df",
        "mle_scale",
        "log_mle_scale",
        "median_n_samples",
        "median_n_features",
        "median_feature_sample_ratio",
        "median_parent_sample_fraction",
        "median_tree_n_leaves",
        "median_tree_n_internal_nodes",
        "median_tree_n_nodes",
        "median_spectral_total_internal_distribution_rows",
        "median_root_internal_row_fraction",
        "median_parent_internal_distribution_rows",
        "median_parent_internal_row_fraction",
        "mean_spectrum_positive_rank",
        "fraction_spectrum_rank_1",
        "fraction_spectrum_rank_2",
        "fraction_spectrum_rank_ge_3",
    }.issubset(tables["group_fits"].columns)
    assert not tables["frame_dimensions"].empty
    assert not tables["factor_skew_summary"].empty
    assert not tables["factor_skew_bins"].empty
    assert {
        "factor",
        "target",
        "spearman_correlation",
        "target_high_minus_low_decile",
    }.issubset(tables["factor_skew_summary"].columns)
    assert set(tables["predictor_fits"]["target"]) == {"mle_df", "log_mle_scale"}
    assert not tables["parent_spectrum_moments"].empty
    assert not tables["eigen_satterthwaite_comparison"].empty
    assert tables["generalized_chi_square_status"].loc[0, "status"] == (
        "available_parent_eigenvalue_vectors_exported"
    )


def test_selected_quadratic_law_audit_handles_single_group_without_predictors() -> None:
    records = _records()
    cells = _cells()
    parent_eigenvalues = _parent_eigenvalues()
    mask = (
        records["source_case_id"].eq("case_low")
        & records["branch_source"].eq("linkage_ultrametric_diagnostic")
    )
    group_records = records.loc[mask].copy()
    group_cells = cells[
        cells["source_case_id"].eq("case_low")
        & cells["branch_source"].eq("linkage_ultrametric_diagnostic")
    ].copy()
    group_parent_eigenvalues = parent_eigenvalues[
        parent_eigenvalues["source_case_id"].eq("case_low")
        & parent_eigenvalues["branch_source"].eq("linkage_ultrametric_diagnostic")
    ].copy()

    tables = build_selected_quadratic_law_audit_tables(
        group_records,
        parent_eigenvalues=group_parent_eigenvalues,
        cells=group_cells,
    )

    assert not tables["group_fits"].empty
    assert tables["predictor_fits"].empty
    assert tables["predicted_spectrum_model"].empty


def test_selected_quadratic_law_audit_writes_bundle_and_plots(tmp_path) -> None:
    records_csv = tmp_path / "records.csv"
    eigenvalues_csv = tmp_path / "parent_eigenvalues.csv"
    cells_csv = tmp_path / "cells.csv"
    _records().to_csv(records_csv, index=False)
    _parent_eigenvalues().to_csv(eigenvalues_csv, index=False)
    _cells().to_csv(cells_csv, index=False)

    paths = run_selected_quadratic_law_audit(
        records_csv=records_csv,
        parent_eigenvalues_csv=eigenvalues_csv,
        cells_csv=cells_csv,
        output_dir=tmp_path / "audit",
    )

    assert paths["manifest"].exists()
    assert paths["frame_dimensions"].exists()
    assert paths["group_fits"].exists()
    assert paths["factor_skew_summary"].exists()
    assert paths["factor_skew_bins"].exists()
    assert paths["predictor_fits"].exists()
    assert paths["eigen_satterthwaite_comparison"].exists()
    assert paths["factor_skew_spearman_overview_png"].exists()
    assert paths["ks_comparison_png"].exists()


def test_selected_quadratic_law_audit_requires_spectral_columns() -> None:
    records = _records().drop(columns=["parent_spectral_log_pseudodeterminant"])

    with pytest.raises(ValueError, match="missing columns"):
        build_selected_quadratic_law_audit_tables(records)
