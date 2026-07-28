from __future__ import annotations

import math
import tempfile
from pathlib import Path

from benchmarks.diagnostics.calibration.selected.hierarchy.internal_vs_selected_hierarchy_inflation import (
    inflation_adjusted_p_value,
    required_inflation_to_block,
    run_internal_vs_selected_hierarchy_inflation_study,
)
from scipy.stats import chi2
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)


def _record(
    *,
    stat: float = 20.0,
    reference_scale: float = 1.0,
    degrees_of_freedom: float = 2.0,
) -> SiblingPairRecord:
    return SiblingPairRecord(
        parent="parent",
        left="left",
        right="right",
        stat=stat,
        reference_scale=reference_scale,
        degrees_of_freedom=degrees_of_freedom,
        p_value=0.001,
        branch_length_sum=0.0,
        n_parent=40,
        is_null_like=False,
        is_edge_blocked=False,
        sibling_null_weight=0.0,
        sibling_projection_dimension=2.0,
        feature_family="bernoulli",
    )


def test_required_inflation_to_block_matches_chi_square_boundary() -> None:
    record = _record(stat=32.0, degrees_of_freedom=2.0)

    required = required_inflation_to_block(record, alpha=0.01)
    p_value = inflation_adjusted_p_value(record, inflation_factor=required)

    assert required > 1.0
    assert math.isclose(p_value, 0.01)


def test_inflation_adjusted_p_value_uses_projected_wald_scale() -> None:
    record = _record(stat=10.0, reference_scale=2.0, degrees_of_freedom=3.0)

    p_value = inflation_adjusted_p_value(record, inflation_factor=5.0)

    assert math.isclose(p_value, chi2.sf(1.0, df=3.0))


def test_internal_vs_selected_hierarchy_study_smoke_writes_explicit_status() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_internal_vs_selected_hierarchy_inflation_study(
            case_names=["gauss_null_large"],
            output_dir=Path(tmpdir),
            n_replicates=1,
            seed=20260603,
            target_mode="root",
            context_match="projection",
        )

        assert (Path(tmpdir) / "internal_vs_selected_hierarchy_inflation.csv").exists()
        assert (Path(tmpdir) / "manifest.json").exists()
        assert summary.shape[0] == 1
        assert summary.iloc[0]["status"] in {"ok", "skip"}
        if summary.iloc[0]["status"] == "ok":
            assert "internal_support_status" in summary.columns
            assert "selected_hierarchy_support_status" in summary.columns
            assert "calibration_comparison_status" in summary.columns
