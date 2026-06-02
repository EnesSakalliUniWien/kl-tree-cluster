from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.selected_hierarchy_external_calibration_contract import (
    ExternalCalibrationContract,
    evaluate_external_calibration_contract,
)


def _records_from_ratios(
    ratios: list[float],
    *,
    records_per_replicate: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, ratio in enumerate(ratios):
        rows.append(
            {
                "case_id": "case",
                "feature_family": "bernoulli",
                "n_samples": 100,
                "feature_dimension": 40,
                "projection_dimension": 2,
                "parent_size_bin": "root_0.75_1",
                "replicate_index": index // records_per_replicate,
                "statistic": float(2.0 * ratio),
                "reference_scale": 1.0,
                "degrees_of_freedom": 2.0,
                "selected_hierarchy_ratio": float(ratio),
            }
        )
    return pd.DataFrame.from_records(rows)


def test_contract_marks_under_supported_stratum_as_not_production_admissible() -> None:
    records = _records_from_ratios([1.0, 1.2, 0.8, 1.1])

    summary = evaluate_external_calibration_contract(
        records,
        contract=ExternalCalibrationContract(
            min_matching_simulations=10,
            min_matched_records=10,
        ),
    )

    row = summary.iloc[0]
    assert not bool(row["production_external_calibration_admissible"])
    assert "matching_simulations_below_tail_resolution_contract" in str(
        row["admissibility_failure_reasons"]
    )
    assert row["estimator_family_decision"] == "undefined_not_production_admissible"


def test_contract_accepts_well_supported_scalar_c_shape() -> None:
    rng = np.random.default_rng(7)
    ratios = list(rng.chisquare(df=2, size=200) / 2.0)
    records = _records_from_ratios(ratios)

    summary = evaluate_external_calibration_contract(
        records,
        contract=ExternalCalibrationContract(
            min_matching_simulations=100,
            min_matched_records=100,
            max_relative_c_simulation_se=0.2,
            scalar_rejection_absolute_tolerance=0.05,
            scalar_tail_quantile_excess_limit=1.75,
            min_scalar_uniformity_ks_p_value=0.001,
        ),
    )

    row = summary.iloc[0]
    assert bool(row["production_external_calibration_admissible"])
    assert row["scalar_c_descriptive_status"] == "descriptive_scalar_c_plausible"
    assert row["estimator_family_decision"] == "scalar_c_external_model_candidate"


def test_contract_flags_heavy_selected_tail_as_tail_law_required() -> None:
    ratios = [1.0] * 180 + [80.0] * 20
    records = _records_from_ratios(ratios)

    summary = evaluate_external_calibration_contract(
        records,
        contract=ExternalCalibrationContract(
            min_matching_simulations=100,
            min_matched_records=100,
            max_relative_c_simulation_se=1.0,
            scalar_rejection_absolute_tolerance=0.005,
            scalar_tail_quantile_excess_limit=1.25,
        ),
    )

    row = summary.iloc[0]
    assert bool(row["production_external_calibration_admissible"])
    assert row["scalar_c_descriptive_status"] == "descriptive_full_tail_law_indicated"
    assert row["estimator_family_decision"] == "full_selected_ratio_tail_law_required"
