from __future__ import annotations

import math

import pandas as pd
import pytest

from benchmarks.shared.sibling_inflation_diagnostic import (
    build_sibling_inflation_diagnostic_tables,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)


def _record(
    parent: str,
    *,
    stat: float,
    degrees_of_freedom: float,
    sibling_null_weight: float,
    is_null_like: bool,
    is_edge_blocked: bool = False,
    n_parent: int = 40,
    sibling_projection_dimension: float = 2.0,
) -> SiblingPairRecord:
    return SiblingPairRecord(
        parent=parent,
        left=f"{parent}_left",
        right=f"{parent}_right",
        stat=stat,
        reference_scale=1.0,
        degrees_of_freedom=degrees_of_freedom,
        p_value=0.001,
        branch_length_sum=0.0,
        n_parent=n_parent,
        is_null_like=is_null_like,
        is_edge_blocked=is_edge_blocked,
        sibling_null_weight=sibling_null_weight,
        sibling_projection_dimension=sibling_projection_dimension,
        feature_family="bernoulli",
    )


def _trace(*node_ids: str) -> pd.DataFrame:
    if not node_ids:
        node_ids = ("target",)
    return pd.DataFrame(
        [
            {
                "case_id": "synthetic",
                "failure_class": "gate_under_split",
                "node_id": node_id,
                "actual_decision": "boundary",
                "trace_relation": "actual_stops_above_oracle_boundary",
                "actual_boundary": True,
                "oracle_true_k_boundary": False,
                "split_prerequisites_open": True,
                "sibling_gate_open": False,
                "has_descendant_split": False,
                "sibling_adjusted_p_value": 0.08,
                "sibling_corrected_p_value": 0.08,
                "left_edge_significant": True,
                "right_edge_significant": True,
                "left_edge_p_value_bh": 1e-12,
                "right_edge_p_value_bh": 1e-12,
            }
            for node_id in node_ids
        ]
    )


def test_sibling_inflation_diagnostic_exposes_blocking_inflation_threshold() -> None:
    records = (
        _record(
            "target",
            stat=32.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1e-12,
            is_null_like=False,
        ),
        _record(
            "calibration_high",
            stat=120.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=True,
        ),
        _record(
            "calibration_low",
            stat=2.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=0.1,
            is_null_like=True,
        ),
    )
    model = fit_empirical_null_inflation_model(list(records))

    tables = build_sibling_inflation_diagnostic_tables(
        records=records,
        model=model,
        trace_df=_trace(),
        sibling_alpha=0.01,
        max_contributors=2,
    )

    target = tables.targets.iloc[0]
    assert bool(target["blocker_candidate"])
    assert bool(target["raw_rejects_at_alpha"])
    assert bool(target["current_blocks_at_alpha"])
    assert bool(target["annotation_corrected_blocks_at_alpha"])
    assert bool(target["inflation_crosses_alpha"])
    assert target["blocking_stage"] == "inflation_adjusted_test"
    assert target["current_empirical_inflation_factor"] > target["inflation_factor_at_alpha"]
    assert target["inflation_excess_ratio"] > 1.0
    assert math.isclose(
        target["current_empirical_inflation_factor"],
        target["local_recomputed_inflation_factor"],
    )

    contributors = tables.contributors
    assert contributors["target_parent"].tolist() == ["target", "target"]
    assert contributors.iloc[0]["contributor_parent"] == "calibration_high"
    assert contributors.iloc[0]["local_weight_share"] > contributors.iloc[1][
        "local_weight_share"
    ]
    assert tables.summary.iloc[0]["n_blocker_candidates"] == 1


def test_sibling_inflation_diagnostic_reports_calibration_variants() -> None:
    records = (
        _record(
            "target",
            stat=1000.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=False,
        ),
        _record(
            "null_low",
            stat=2.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=True,
        ),
        _record(
            "edge_blocked_high",
            stat=20.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=True,
            is_edge_blocked=True,
        ),
    )
    model = fit_empirical_null_inflation_model(list(records))

    tables = build_sibling_inflation_diagnostic_tables(
        records=records,
        model=model,
        trace_df=_trace(),
        sibling_alpha=0.01,
        max_contributors=2,
    )

    target = tables.targets.iloc[0]
    assert target["leave_one_out_status"] == "ok"
    assert target["strict_null_like_status"] == "ok"
    assert target["edge_blocked_or_null_like_status"] == "ok"
    assert target["current_empirical_inflation_factor"] == target[
        "leave_one_out_inflation_factor"
    ]
    assert target["leave_one_out_inflation_factor"] == target[
        "strict_null_like_inflation_factor"
    ]
    assert target["edge_blocked_or_null_like_inflation_factor"] == target[
        "leave_one_out_inflation_factor"
    ]
    assert target["strict_null_like_calibration_records"] == 2
    assert target["edge_blocked_or_null_like_calibration_records"] == 2
    assert target["calibration_support_status"] == "strict_empirical_null_supported"
    assert bool(target["empirical_null_supported"])
    assert not bool(target["current_blocks_at_alpha"])
    assert not bool(target["leave_one_out_blocks_at_alpha"])
    assert tables.summary.iloc[0]["n_leave_one_out_blocks"] == 0
    assert tables.summary.iloc[0]["n_strict_empirical_null_supported"] == 1


def test_sibling_inflation_diagnostic_rejects_selected_nonnull_only_support() -> None:
    records = (
        _record(
            "target",
            stat=1000.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=False,
        ),
        _record(
            "selected_context",
            stat=100.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=False,
        ),
    )
    with pytest.raises(ValueError, match="selected non-null"):
        fit_empirical_null_inflation_model(list(records))


def test_sibling_inflation_diagnostic_rejects_target_only_support() -> None:
    records = (
        _record(
            "target",
            stat=1000.0,
            degrees_of_freedom=2.0,
            sibling_null_weight=1.0,
            is_null_like=False,
        ),
    )
    with pytest.raises(ValueError, match="selected non-null"):
        fit_empirical_null_inflation_model(list(records))
