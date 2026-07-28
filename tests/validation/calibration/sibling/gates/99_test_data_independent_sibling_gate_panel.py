from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_panel import (
    DataIndependentSiblingGateConfig,
    build_data_independent_gate_components,
    data_independent_coordinate_gate_p_value,
    data_independent_feature_block_gate_p_value,
    data_independent_gate_p_value,
    data_independent_global_gate_p_value,
    run_data_independent_sibling_gate_panel,
    summarize_data_independent_gate_penalty_transfer,
    summarize_data_independent_sibling_gate_rows,
)
from benchmarks.diagnostics.calibration.traversal.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    fixed_coordinate_bh_p_value,
    fixed_subspace_sibling_p_value,
)
from tree_break_selection.tree.feature_space import FeatureBlock, FeatureSpace


def _summary_rows(
    *,
    data_role: str,
    rejection_rate: float,
    n_rows: int = 100,
) -> pd.DataFrame:
    n_rejected = int(round(rejection_rate * n_rows))
    rejected = np.array([True] * n_rejected + [False] * (n_rows - n_rejected))
    return pd.DataFrame(
        {
            "case_id": "unit",
            "data_role": data_role,
            "topology_mode": "selected_topology",
            "candidate_method": "coordinate_bonferroni",
            "selected_topology_penalty": 10.0,
            "source_family": "binary_template",
            "data_independent_gate_p_value": np.where(rejected, 0.0005, 0.5),
            "rejected_at_sibling_alpha": rejected,
            "rejected_at_effective_alpha": rejected,
            "effective_selected_alpha": 0.001,
            "parent_sample_size": np.linspace(1.0, 100.0, n_rows),
            "sibling_alpha": 0.01,
        }
    )


def test_coordinate_gate_bonferroni_uses_fixed_coordinate_family() -> None:
    z = np.array([0.0, 3.0])

    p_value = data_independent_coordinate_gate_p_value(
        z,
        candidate_method="coordinate_bonferroni",
    )

    assert p_value == pytest.approx(min(1.0, 2.0 * chi2.sf(9.0, df=1.0)))


def test_global_gate_uses_full_fixed_subspace_chi_square() -> None:
    z = np.array([1.0, 2.0, 3.0])
    feature_space = FeatureSpace(
        column_names=("x0", "x1", "x2"),
        blocks=(
            FeatureBlock(
                name="x0",
                family="bernoulli",
                column_indices=(0,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
            FeatureBlock(
                name="x1",
                family="bernoulli",
                column_indices=(1,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
            FeatureBlock(
                name="x2",
                family="bernoulli",
                column_indices=(2,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
        ),
    )

    p_value = data_independent_global_gate_p_value(
        z,
        feature_space,
        candidate_method="global_chi_square",
    )

    assert p_value == pytest.approx(chi2.sf(14.0, df=3))
    assert data_independent_gate_p_value(
        z,
        feature_space,
        candidate_method="global_chi_square",
    ) == pytest.approx(
        fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_global_chi_square",
        )
    )


def test_coordinate_gate_bh_matches_statsmodels_adjusted_minimum() -> None:
    z = np.array([0.0, 2.0, 3.0])
    coordinate_p = chi2.sf(z * z, df=1.0)

    p_value = data_independent_coordinate_gate_p_value(
        z,
        candidate_method="coordinate_bh",
    )

    assert p_value == pytest.approx(float(np.min(multipletests(coordinate_p, method="fdr_bh")[1])))
    assert fixed_coordinate_bh_p_value(z) == pytest.approx(p_value)


def test_coordinate_gate_bh_delegates_to_production_fixed_subspace_gate() -> None:
    z = np.array([0.0, 2.0, 3.0])
    feature_space = FeatureSpace(
        column_names=("x0", "x1", "x2"),
        blocks=(
            FeatureBlock(
                name="x0",
                family="bernoulli",
                column_indices=(0,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
            FeatureBlock(
                name="x1",
                family="bernoulli",
                column_indices=(1,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
            FeatureBlock(
                name="x2",
                family="bernoulli",
                column_indices=(2,),
                chart="identity",
                covariance="bernoulli",
                contrast_dimension=1,
            ),
        ),
    )

    assert data_independent_coordinate_gate_p_value(
        z,
        candidate_method="coordinate_bh",
    ) == pytest.approx(
        fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_coordinate_bh",
        )
    )


def test_block_gate_uses_feature_block_degrees_of_freedom() -> None:
    feature_space = FeatureSpace(
        column_names=("F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2"),
        blocks=(
            FeatureBlock(
                name="F0",
                family="categorical",
                column_indices=(0, 1, 2),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=2,
            ),
            FeatureBlock(
                name="F1",
                family="categorical",
                column_indices=(3, 4, 5),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=2,
            ),
        ),
    )
    z = np.array([3.0, 4.0, 0.0, 0.0])

    p_value = data_independent_feature_block_gate_p_value(
        z,
        feature_space,
        candidate_method="block_bonferroni",
    )

    assert p_value == pytest.approx(min(1.0, 2.0 * chi2.sf(25.0, df=2)))


def test_block_gate_bh_delegates_to_production_fixed_subspace_gate() -> None:
    feature_space = FeatureSpace(
        column_names=("F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2"),
        blocks=(
            FeatureBlock(
                name="F0",
                family="categorical",
                column_indices=(0, 1, 2),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=2,
            ),
            FeatureBlock(
                name="F1",
                family="categorical",
                column_indices=(3, 4, 5),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=2,
            ),
        ),
    )
    z = np.array([3.0, 4.0, 0.0, 0.0])

    assert data_independent_feature_block_gate_p_value(
        z,
        feature_space,
        candidate_method="block_bh",
    ) == pytest.approx(
        fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_block_bh",
        )
    )


def test_summary_marks_null_candidate_and_inflated_statuses() -> None:
    candidate = summarize_data_independent_sibling_gate_rows(
        _summary_rows(data_role="null", rejection_rate=0.01),
        min_rows=50,
    )
    inflated = summarize_data_independent_sibling_gate_rows(
        _summary_rows(data_role="null", rejection_rate=0.02),
        min_rows=50,
    )

    assert (
        candidate.iloc[0]["data_independent_gate_status"] == "data_independent_gate_null_candidate"
    )
    assert inflated.iloc[0]["data_independent_gate_status"] == "data_independent_gate_null_inflated"


def test_summary_marks_signal_retained_and_weak_statuses() -> None:
    retained = summarize_data_independent_sibling_gate_rows(
        _summary_rows(data_role="signal", rejection_rate=0.16),
        min_rows=50,
        min_signal_rejection_rate=0.15,
    )
    weak = summarize_data_independent_sibling_gate_rows(
        _summary_rows(data_role="signal", rejection_rate=0.14),
        min_rows=50,
        min_signal_rejection_rate=0.15,
    )

    assert (
        retained.iloc[0]["data_independent_gate_status"] == "data_independent_gate_signal_retained"
    )
    assert weak.iloc[0]["data_independent_gate_status"] == "data_independent_gate_signal_weak"


def test_data_independent_gate_components_remain_diagnostic_only() -> None:
    summary = summarize_data_independent_sibling_gate_rows(
        pd.concat(
            [
                _summary_rows(data_role="null", rejection_rate=0.01),
                _summary_rows(data_role="signal", rejection_rate=0.16),
            ],
            ignore_index=True,
        ),
        min_rows=50,
        min_signal_rejection_rate=0.15,
    )
    transfer_summary = summarize_data_independent_gate_penalty_transfer(summary)

    components = build_data_independent_gate_components(summary, transfer_summary)
    rows = evaluate_production_admissibility_components(components)
    contract = summarize_production_admissibility_contracts(rows)

    assert contract.iloc[0]["production_decision"] == "diagnostic_only"
    assert (
        transfer_summary.iloc[0]["data_independent_gate_transfer_status"]
        == "data_independent_gate_penalty_transfer_candidate"
    )


def test_run_data_independent_sibling_gate_panel_writes_outputs(tmp_path: Path) -> None:
    outputs = run_data_independent_sibling_gate_panel(
        DataIndependentSiblingGateConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            data_roles=("null", "signal"),
            candidate_methods=("coordinate_bonferroni",),
            sibling_alpha=0.01,
            selected_topology_penalties=(10.0,),
            replicates=1,
            base_seed=20260613,
            min_rows=1,
            min_signal_rejection_rate=0.0,
        )
    )

    assert set(outputs) == {
        "rows",
        "summary",
        "transfer_summary",
        "production_components",
        "production_summary",
        "manifest",
    }
    rows = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_rows.csv",
        keep_default_na=False,
    )
    summary = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_summary.csv",
        keep_default_na=False,
    )
    transfer_summary = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_penalty_transfer_summary.csv",
        keep_default_na=False,
    )
    production_summary = pd.read_csv(tmp_path / "production_admissibility_summary.csv")

    assert not rows.empty
    assert set(rows["data_role"]) == {"null", "signal"}
    assert set(summary["candidate_method"]) == {"coordinate_bonferroni"}
    assert set(transfer_summary["selected_topology_penalty"]) == {10.0}
    assert production_summary.iloc[0]["production_decision"] in {
        "diagnostic_only",
        "fail_closed_undefined",
    }
