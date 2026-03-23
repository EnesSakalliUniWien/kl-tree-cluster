from __future__ import annotations

import pytest

from kl_clustering_analysis import config
from debug_scripts.enhancement_lab.exp_parametric_inflation import DIAGNOSTIC_CASES
from debug_scripts.enhancement_lab.exp_parametric_inflation_v3 import (
    build_rows,
    evaluate_case,
    select_v3_model,
)


@pytest.mark.slow
def test_v3_selector_preserves_null_control_with_conservative_focal_tradeoff() -> None:
    # Pin to global-constant deflation: this test was calibrated against
    # the original intercept-only model and is independent of conditional
    # deflation improvements.
    saved = config.CONDITIONAL_DEFLATION_ALPHA
    config.CONDITIONAL_DEFLATION_ALPHA = None
    try:
        _run_v3_assertions()
    finally:
        config.CONDITIONAL_DEFLATION_ALPHA = saved


def _run_v3_assertions() -> None:
    rows = build_rows(DIAGNOSTIC_CASES)

    total_raw_null_power = 0
    total_raw_null_v3 = 0
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
        total_bh_null_v3 += evaluation.bh_null_v3
        total_bh_focal_global += evaluation.bh_focal_global
        total_bh_focal_power += evaluation.bh_focal_power
        total_bh_focal_v3 += evaluation.bh_focal_v3

    # v3 is intentionally conservative relative to the pooled power-law model:
    # it may give up focal BH rejections in exchange for tighter null control,
    # but it should still retain some focal discoveries.
    assert total_bh_null_v3 == 0
    assert total_raw_null_v3 < total_raw_null_power
    assert total_bh_focal_v3 <= total_bh_focal_power
    assert total_bh_focal_v3 > 0
