from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.root.root_tail_values import bandwidth_gap_status
from benchmarks.diagnostics.calibration.root.selected.root_tail_action_support import (
    prepare_root_tail_action_support,
)
from benchmarks.diagnostics.calibration.root.tie_rank.comparison import (
    partition_target_and_generated_rows,
    row_bandwidth_gap_status,
)
from benchmarks.diagnostics.calibration.root.tie_rank.proposal_generators import (
    ensure_nonempty_binary_feature_columns,
)


def _root_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "target",
                "data_role": "observed_target",
                "calibration_role": "observed_target_not_null_support",
                "proposal_family": "observed_target",
                "root_bandwidth_reopen_band": "",
                "root_sibling_selected_ratio": 10.0,
                "root_tie_rank_median_fraction": 0.8,
                "root_edge_path_statistic_margin": 20.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
            },
            {
                "case_id": "support",
                "data_role": "external_selected_null",
                "calibration_role": "external_null_support",
                "proposal_family": "probe",
                "root_bandwidth_reopen_band": "",
                "root_sibling_selected_ratio": 12.0,
                "root_tie_rank_median_fraction": 0.81,
                "root_edge_path_statistic_margin": 21.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 1.5,
            },
        ]
    )


def test_shared_root_action_preparation_fills_optional_coordinates() -> None:
    targets, support = prepare_root_tail_action_support(
        _root_rows(),
        h_u_population_law_status="identity",
    )

    assert targets["case_id"].tolist() == ["target"]
    assert support["case_id"].tolist() == ["support"]
    assert support["_h_u"].tolist() == ["identity"]
    assert support["_bandwidth"].tolist() == [""]


def test_shared_target_generated_partition_and_bandwidth_status() -> None:
    targets, generated = partition_target_and_generated_rows(_root_rows())

    assert targets["case_id"].tolist() == ["target"]
    assert generated["case_id"].tolist() == ["support"]
    assert row_bandwidth_gap_status(targets.iloc[0], generated.iloc[0]) == "bandwidth_band_match"
    assert bandwidth_gap_status(
        target_band="observed",
        generated_band="bandwidth_reopen_missing",
    ) == "generated_bandwidth_unmeasured"


def test_binary_generator_repair_is_deterministic_and_preserves_nonempty_columns() -> None:
    matrix = np.array([[0, 1], [0, 0]], dtype=int)

    repaired = ensure_nonempty_binary_feature_columns(
        matrix,
        rng=np.random.default_rng(7),
    )

    assert matrix[:, 0].sum() == 0
    assert np.all(repaired.sum(axis=0) > 0)
