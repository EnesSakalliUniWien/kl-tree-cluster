from __future__ import annotations

import pytest
from debug_scripts.enhancement_lab.exp_parametric_inflation import DIAGNOSTIC_CASES
from debug_scripts.enhancement_lab.exp_parametric_inflation_v3 import (
    build_rows,
    evaluate_case,
    select_v3_model,
)


@pytest.mark.slow
def test_v3_selector_does_not_increase_null_rejections_with_conservative_focal_tradeoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Enhancement-lab coverage for the pre-local-adjuster global baseline.
    # Production local Gaussian calibration is guarded separately.
    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence"
        ".adjusted_wald_annotation.calibration.predict_sibling_adjustment",
        lambda calibrator, sibling_test_calibration_scale: calibrator.global_adjustment,
    )
    _run_v3_assertions()


def _run_v3_assertions() -> None:
    rows = build_rows(DIAGNOSTIC_CASES)

    total_raw_null_power = 0
    total_raw_null_v3 = 0
    total_bh_null_power = 0
    total_bh_null_v3 = 0
    total_bh_focal_global = 0
    total_bh_focal_power = 0
    total_bh_focal_v3 = 0

    for case_name in DIAGNOSTIC_CASES:
        train_null_rows = [row for row in rows if row.case_name != case_name and row.is_null_like]
        train_eval_rows = [row for row in rows if row.case_name != case_name]
        if not train_null_rows:
            train_null_rows = [row for row in rows if row.case_name == case_name and row.is_null_like]
            train_eval_rows = [row for row in rows if row.case_name == case_name]

        model = select_v3_model(train_null_rows, train_eval_rows=train_eval_rows)
        evaluation = evaluate_case(case_name, rows, model)

        total_raw_null_power += evaluation.raw_null_power
        total_raw_null_v3 += evaluation.raw_null_v3
        total_bh_null_power += evaluation.bh_null_power
        total_bh_null_v3 += evaluation.bh_null_v3
        total_bh_focal_global += evaluation.bh_focal_global
        total_bh_focal_power += evaluation.bh_focal_power
        total_bh_focal_v3 += evaluation.bh_focal_v3

    # v3 is intentionally conservative relative to the pooled power-law model:
    # it may give up focal BH rejections in exchange for null control. The
    # current strict calibration contract can already drive BH null rejections
    # to zero before v3, so the live contract is non-increase rather than a
    # guaranteed strict reduction.
    assert total_bh_null_v3 <= total_bh_null_power
    assert total_raw_null_v3 <= total_raw_null_power
    assert total_bh_focal_v3 <= total_bh_focal_power
    assert total_bh_focal_v3 > 0
