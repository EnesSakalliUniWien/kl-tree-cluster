import pandas as pd
from benchmarks.diagnostics.calibration.overlap.binary_threshold_scan import (
    THRESHOLD_SCAN_COLUMNS,
    scan_binary_role_thresholds,
)


def test_binary_threshold_scan_preserves_role_counts_and_both_directions() -> None:
    rows = pd.DataFrame(
        {
            "guard_truth_role": ["truth_recovery", "negative", "negative"],
            "score": [0.8, 0.2, 0.9],
        }
    )

    result = scan_binary_role_thresholds(
        rows,
        metric="score",
        schema_version="test/v1",
        study_role="diagnostic_test",
    )

    assert tuple(result.columns) == THRESHOLD_SCAN_COLUMNS
    assert set(result["direction"]) == {"greater_equal", "less_equal"}
    assert set(result["truth_total"]) == {1}
    assert set(result["negative_total"]) == {2}
    assert set(result["schema_version"]) == {"test/v1"}
    assert set(result["study_role"]) == {"diagnostic_test"}
