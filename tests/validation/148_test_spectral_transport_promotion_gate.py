from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.spectral_transport_promotion_gate import (
    BASELINE_METHOD,
    BASELINE_PROFILE,
    CANDIDATE_METHOD,
    CANDIDATE_PROFILE,
    SpectralTransportPromotionGateConfig,
    evaluate_spectral_transport_promotion_gate,
    run_spectral_transport_promotion_gate,
)


def _dispatch_pairwise(delta_ari: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": f"case_{index}",
                "baseline_method": BASELINE_METHOD,
                "candidate_method": CANDIDATE_METHOD,
                "baseline_status": "ok",
                "candidate_status": "ok",
                "delta_ari_candidate_minus_baseline": delta_ari,
                "partition_ari_between_methods": 1.0,
            }
            for index in range(3)
        ]
    )


def _selected_family_rows(*, candidate_fixes_null: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.extend(
            [
                {
                    "case_id": f"null_{index}",
                    "data_role": "selected_null",
                    "method_id": BASELINE_PROFILE,
                    "replicate": 0,
                    "ari": 0.0,
                    "false_split": index == 0,
                },
                {
                    "case_id": f"null_{index}",
                    "data_role": "selected_null",
                    "method_id": CANDIDATE_PROFILE,
                    "replicate": 0,
                    "ari": 1.0 if index == 0 and candidate_fixes_null else 0.0,
                    "false_split": False if index == 0 and candidate_fixes_null else index == 0,
                },
                {
                    "case_id": f"signal_{index}",
                    "data_role": "signal",
                    "method_id": BASELINE_PROFILE,
                    "replicate": 0,
                    "ari": 0.8,
                    "false_split": False,
                },
                {
                    "case_id": f"signal_{index}",
                    "data_role": "signal",
                    "method_id": CANDIDATE_PROFILE,
                    "replicate": 0,
                    "ari": 0.8,
                    "false_split": False,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_promotion_gate_passes_when_signal_retained_and_null_false_split_reduced(
    tmp_path: Path,
) -> None:
    config = SpectralTransportPromotionGateConfig(output_dir=tmp_path)
    components, summary = evaluate_spectral_transport_promotion_gate(
        dispatch_pairwise=_dispatch_pairwise(),
        selected_family_rows=_selected_family_rows(candidate_fixes_null=True),
        config=config,
    )

    assert components["component_status"].eq("passes").all()
    assert summary.iloc[0]["promotion_decision"] == "promotion_admissible"


def test_promotion_gate_defaults_use_replicate_selected_family_evidence(
    tmp_path: Path,
) -> None:
    config = SpectralTransportPromotionGateConfig(output_dir=tmp_path)

    assert (
        "selected_family_traversal_spectral_transport_promoted_replicates"
        in str(config.selected_family_rows_path)
    )


def test_promotion_gate_blocks_when_null_false_split_is_not_reduced(
    tmp_path: Path,
) -> None:
    config = SpectralTransportPromotionGateConfig(output_dir=tmp_path)
    components, summary = evaluate_spectral_transport_promotion_gate(
        dispatch_pairwise=_dispatch_pairwise(),
        selected_family_rows=_selected_family_rows(candidate_fixes_null=False),
        config=config,
    )

    null_component = components[
        components["component_id"].eq("selected_family_null_false_split_reduction")
    ].iloc[0]
    assert null_component["component_status"] == "fails"
    assert null_component["false_split_reduction"] == 0
    assert summary.iloc[0]["promotion_decision"] == "diagnostic_only_not_promoted"
    assert summary.iloc[0]["blocking_component_ids"] == (
        "selected_family_null_false_split_reduction"
    )


def test_run_spectral_transport_promotion_gate_writes_outputs(tmp_path: Path) -> None:
    dispatch_path = tmp_path / "dispatch_pairwise.csv"
    selected_path = tmp_path / "selected_family.csv"
    _dispatch_pairwise().to_csv(dispatch_path, index=False)
    _selected_family_rows(candidate_fixes_null=False).to_csv(selected_path, index=False)

    outputs = run_spectral_transport_promotion_gate(
        SpectralTransportPromotionGateConfig(
            output_dir=tmp_path / "out",
            dispatch_pairwise_path=dispatch_path,
            selected_family_rows_path=selected_path,
        )
    )

    assert set(outputs) == {"components", "summary", "manifest"}
    assert outputs["components"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    summary = pd.read_csv(outputs["summary"])
    assert summary.iloc[0]["promotion_decision"] == "diagnostic_only_not_promoted"
