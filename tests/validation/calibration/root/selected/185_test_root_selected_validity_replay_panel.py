from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_validity_replay_panel as panel,
)


def _tail(
    *,
    case_id: str,
    tail_status: str,
    support_count: int = 0,
    s_h_u: float = 0.5,
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "root_tail_inference_status": tail_status,
        "selected_null_support_count": support_count,
        "s_root_spectral_excess_log": s_h_u,
        "t_selected_tie_rank_fraction": 0.8,
        "a_selected_ratio_action_log1p": 4.0,
        "e_edge_margin_action_log1p": 5.0,
        "b_bandwidth_topology_status": "bandwidth_root_reopen_observed",
        "h_u_population_law_status": "deformed_mp_edge_measured_support_side",
    }


def _profile_replay(
    *,
    case_id: str,
    mean_ari: float,
    p_value: float = 0.005,
    blocked: bool = False,
    would_block: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "root_stability_subsample_mean_ari": mean_ari,
        "observed_root_stability_guard_threshold": 0.24,
        "root_stability_guard_blocked": blocked,
        "root_selective_permutation_p_value": p_value,
        "observed_root_selective_permutation_guard_alpha": 0.01,
        "root_selective_permutation_guard_blocked": False,
        "root_selective_permutation_guard_would_block": would_block,
    }


def test_valid_root_with_tail_support_is_usable() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="supported",
                    tail_status=panel.TAIL_CALIBRATED_STATUS,
                    support_count=4,
                )
            ]
        ),
        root_replay_rows=pd.DataFrame.from_records(
            [_profile_replay(case_id="supported", mean_ari=0.85)]
        ),
    )

    row = rows.iloc[0]
    assert row["root_validity_status"] == ("root_validity_supported_by_stability_and_selection")
    assert row["selected_root_usability_status"] == (
        "usable_selected_root_tail_after_validity_replay"
    )
    assert row["method_action"] == "use_selected_root_tail_with_validity_annotation"


def test_tail_calibrated_inside_unstable_root_fails_closed() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="unstable",
                    tail_status=panel.TAIL_CALIBRATED_STATUS,
                    support_count=3,
                )
            ]
        ),
        root_replay_rows=pd.DataFrame.from_records(
            [_profile_replay(case_id="unstable", mean_ari=0.12)]
        ),
    )

    row = rows.iloc[0]
    assert row["root_validity_status"] == ("root_validity_failed_feature_subsample_replay")
    assert row["selected_root_usability_status"] == ("fail_closed_root_validity_failed")
    assert row["method_action"] == "fail_closed_root_unstable_under_topology_replay"


def test_valid_root_still_fails_closed_when_tail_support_missing() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="tail_missing",
                    tail_status=("fail_closed_selected_root_spectral_tail_support_missing"),
                )
            ]
        ),
        root_replay_rows=pd.DataFrame.from_records(
            [_profile_replay(case_id="tail_missing", mean_ari=0.91)]
        ),
    )

    row = rows.iloc[0]
    assert row["root_validity_status"] == ("root_validity_supported_by_stability_and_selection")
    assert row["selected_root_usability_status"] == ("fail_closed_valid_root_tail_support_missing")
    assert row["method_action"] == "fail_closed_generate_tail_support_for_valid_root"


def test_selected_root_permutation_failure_overrides_stability() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="noise_root",
                    tail_status=panel.TAIL_CALIBRATED_STATUS,
                    support_count=3,
                )
            ]
        ),
        root_replay_rows=pd.DataFrame.from_records(
            [_profile_replay(case_id="noise_root", mean_ari=0.8, p_value=0.2)]
        ),
    )

    row = rows.iloc[0]
    assert row["root_validity_status"] == ("root_validity_failed_selected_root_permutation")
    assert row["method_action"] == "fail_closed_root_not_selectively_significant"


def test_missing_root_replay_evidence_fails_closed() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="unmeasured",
                    tail_status=panel.TAIL_CALIBRATED_STATUS,
                    support_count=2,
                )
            ]
        )
    )
    summary = panel.summarize_root_selected_validity_replay_rows(rows).iloc[0]

    row = rows.iloc[0]
    assert row["root_validity_status"] == "root_validity_unmeasured"
    assert row["selected_root_usability_status"] == ("fail_closed_root_validity_unmeasured")
    assert summary["root_validity_unmeasured_count"] == 1
    assert summary["tail_calibrated_but_root_invalid_or_unmeasured_count"] == 1


def test_explicit_topology_family_replay_counts_alternative_roots() -> None:
    rows = panel.build_root_selected_validity_replay_rows(
        tail_rows=pd.DataFrame.from_records(
            [
                _tail(
                    case_id="family",
                    tail_status=("fail_closed_selected_root_spectral_tail_support_missing"),
                )
            ]
        ),
        root_replay_rows=pd.DataFrame.from_records(
            [
                {
                    "case_id": "family",
                    "root_partition_ari_to_observed": 0.95,
                    "root_selective_permutation_p_value": 0.005,
                },
                {
                    "case_id": "family",
                    "root_partition_ari_to_observed": 0.1,
                    "root_selective_permutation_p_value": 0.005,
                },
            ]
        ),
    )

    row = rows.iloc[0]
    assert row["root_replay_source"] == "explicit_root_topology_family_replay"
    assert row["root_replay_count"] == 2
    assert row["plausible_alternative_root_count"] == 1
    assert row["root_validity_status"] == ("root_validity_supported_by_stability_and_selection")


def test_runner_writes_outputs(tmp_path: Path) -> None:
    tail_path = tmp_path / "tail.csv"
    replay_path = tmp_path / "replay.csv"
    pd.DataFrame.from_records(
        [
            _tail(
                case_id="supported",
                tail_status=panel.TAIL_CALIBRATED_STATUS,
                support_count=3,
            )
        ]
    ).to_csv(tail_path, index=False)
    pd.DataFrame.from_records([_profile_replay(case_id="supported", mean_ari=0.9)]).to_csv(
        replay_path, index=False
    )

    outputs = panel.run_root_selected_validity_replay_panel(
        panel.RootSelectedValidityReplayConfig(
            output_dir=tmp_path / "out",
            tail_rows_path=tail_path,
            root_replay_rows_path=replay_path,
        )
    )

    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    output_rows = pd.read_csv(outputs["rows"])
    assert output_rows.iloc[0]["selected_root_usability_status"] == (
        "usable_selected_root_tail_after_validity_replay"
    )
