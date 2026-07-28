from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_weak_zone_separability import (
    OverlapWeakZoneSeparabilityConfig,
    build_weak_zone_separability,
    rank_auc,
    run_overlap_weak_zone_separability,
)


def _rows() -> pd.DataFrame:
    records = []
    for role, values in (
        ("signal_truth_aligned", (0.20, 0.40, 0.80)),
        ("null_like", (0.10, 0.30, 0.60)),
        ("signal_truth_misaligned", (0.05, 0.35, 0.70)),
    ):
        for idx, value in enumerate(values):
            records.append(
                {
                    "structural_decision_zone": "unstable_weak_homogeneity_zone",
                    "structural_truth_role": role,
                    "sibling_p_value": 0.001,
                    "homogeneity_gain_min": value,
                    "continuous_homogeneity_threshold": 0.25,
                    "depth": idx,
                    "n_parent": 100 + idx,
                    "barycentric_balance": 0.4,
                    "subspace_consensus_jaccard_topk": 0.5,
                }
            )
    records.append(
        {
            "structural_decision_zone": "stable_structural_accept",
            "structural_truth_role": "signal_truth_aligned",
            "sibling_p_value": 0.001,
            "homogeneity_gain_min": 1.0,
            "continuous_homogeneity_threshold": 0.25,
            "depth": 4,
            "n_parent": 80,
            "barycentric_balance": 0.5,
            "subspace_consensus_jaccard_topk": 0.5,
        }
    )
    return pd.DataFrame.from_records(records)


def test_rank_auc_uses_pairwise_order_and_ties() -> None:
    assert rank_auc([2.0, 3.0], [1.0, 2.0]) == 0.875
    assert rank_auc([1.0], [2.0]) == 0.0


def test_weak_zone_summary_reports_partial_zero_negative_retention() -> None:
    summary, scan = build_weak_zone_separability(
        _rows(),
        metrics=("homogeneity_gain_min",),
    )

    combined = summary[summary["comparison"].eq("aligned_vs_null_or_misaligned")].iloc[0]
    assert combined["best_direction"] == "greater_equal"
    assert int(combined["zero_negative_positive_count"]) == 1
    assert float(combined["zero_negative_positive_retention"]) == 1 / 3
    assert combined["zero_negative_status"] == ("zero_negative_partial_positive_retention")
    assert set(scan["direction"]) == {"greater_equal", "less_equal"}


def test_run_weak_zone_separability_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "zones.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_weak_zone_separability(
        OverlapWeakZoneSeparabilityConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
            metrics=("homogeneity_gain_min", "context_homogeneity_margin"),
        )
    )

    for path in outputs.values():
        assert path.exists()
