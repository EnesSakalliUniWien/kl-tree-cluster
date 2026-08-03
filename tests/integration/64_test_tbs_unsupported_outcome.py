from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from benchmarks.shared.runners.tbs_support import unsupported_empirical_null_reason
from benchmarks.shared.types import UnsupportedReasonCode
from scipy.spatial.distance import pdist


def _unsupported_annotations() -> pd.DataFrame:
    annotations = pd.DataFrame(index=[f"N{idx}" for idx in range(78)])
    annotations["Sibling_Gate_P_Value_Calibration"] = ""
    annotations.loc[annotations.index[:39], "Sibling_Gate_P_Value_Calibration"] = (
        "undefined_no_internal_support"
    )
    annotations["Sibling_Role_Supported"] = False
    annotations["Sibling_Divergence_Invalid"] = False
    annotations.loc[annotations.index[:39], "Sibling_Divergence_Invalid"] = True
    annotations["Child_Parent_Divergence_Tested"] = True
    annotations["Child_Parent_Divergence_Significant"] = True
    return annotations


def test_unsupported_reason_uses_production_stamped_calibration_evidence():
    reason = unsupported_empirical_null_reason(
        _unsupported_annotations(),
        sibling_gate_method="projected_wald_inflation",
    )

    assert reason is not None
    assert reason.code is UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT
    assert reason.stage == "sibling_calibration"
    assert reason.evidence.focal_record_count == 39
    assert reason.evidence.admissible_support_count == 0
    assert reason.evidence.invalid_record_count == 39
    assert reason.evidence.upstream_tested_count == 78
    assert reason.evidence.upstream_rejected_count == 78


def test_non_empirical_and_supported_gates_do_not_report_unsupported():
    annotations = _unsupported_annotations()

    assert (
        unsupported_empirical_null_reason(
            annotations,
            sibling_gate_method="fixed_coordinate_bh",
        )
        is None
    )

    mixed_support = _unsupported_annotations()
    mixed_support.loc[mixed_support.index[-1], "Sibling_Role_Supported"] = True
    assert (
        unsupported_empirical_null_reason(
            mixed_support,
            sibling_gate_method="projected_wald_inflation",
        )
        is None
    )

    inconsistent_focal = _unsupported_annotations()
    inconsistent_focal.loc[inconsistent_focal.index[0], "Sibling_Role_Supported"] = True
    with pytest.raises(ValueError, match="focal rows"):
        unsupported_empirical_null_reason(
            inconsistent_focal,
            sibling_gate_method="projected_wald_inflation",
        )

    annotations.loc[annotations.index[0], "Sibling_Role_Supported"] = True
    annotations.loc[:, "Sibling_Gate_P_Value_Calibration"] = "empirical_null_inflation"
    assert (
        unsupported_empirical_null_reason(
            annotations,
            sibling_gate_method="projected_wald_inflation",
        )
        is None
    )


def test_dense_high_dimensional_empirical_gate_returns_unsupported():
    case = next(
        case
        for case in get_default_test_cases()
        if case["name"] == "gauss_dense_signal_highd"
    )
    data, _truth, _original, _metadata = generate_case_data(case)
    distances = pdist(data.to_numpy(), metric="hamming")

    result = run_tbs_on_distance(
        data,
        distances,
        0.01,
        edge_alpha=0.001,
        trace_level="full",
    )

    assert result.status == "unsupported"
    assert result.labels is None
    assert result.found_clusters == 0
    assert result.unsupported_reason is not None
    assert result.unsupported_reason.evidence.admissible_support_count == 0
    assert result.unsupported_reason.evidence.upstream_rejected_count == 78


def test_dense_high_dimensional_fixed_coordinate_control_remains_ok():
    case = next(
        case
        for case in get_default_test_cases()
        if case["name"] == "gauss_dense_signal_highd"
    )
    data, _truth, _original, _metadata = generate_case_data(case)
    distances = pdist(data.to_numpy(), metric="hamming")

    result = run_tbs_on_distance(
        data,
        distances,
        0.01,
        edge_alpha=0.001,
        sibling_gate_method="fixed_coordinate_bh",
        trace_level="full",
    )

    assert result.status == "ok"
    assert result.labels is not None
    assert len(np.unique(result.labels)) == 4
