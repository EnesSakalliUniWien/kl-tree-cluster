from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.shared.util.decomposition import (
    _labels_and_report_from_decomposition,
    _ok_result_from_labels,
)


def test_labels_and_report_from_decomposition_share_sample_assignment_table() -> None:
    decomposition = {
        "cluster_assignments": {
            1: {"root_node": "left", "leaves": ["S1", "S3"], "size": 2},
            2: {"root_node": "right", "leaves": ["S2"], "size": 1},
        }
    }

    labels, report_df = _labels_and_report_from_decomposition(
        decomposition,
        ["S3", "S0", "S2", "S1"],
    )

    assert np.array_equal(labels, np.array([1, -1, 2, 1], dtype=int))
    assert report_df.to_dict(orient="index") == {
        "S1": {"cluster_id": 1, "cluster_size": 2},
        "S3": {"cluster_id": 1, "cluster_size": 2},
        "S2": {"cluster_id": 2, "cluster_size": 1},
    }
    assert report_df.index.name == "sample_id"


def test_ok_result_from_labels_counts_non_noise_clusters() -> None:
    result = _ok_result_from_labels(
        np.array([2, 2, -1, 4], dtype=int),
        pd.Index(["a", "b", "noise", "c"], name="sample_id"),
    )

    assert result.status == "ok"
    assert result.skip_reason is None
    assert result.found_clusters == 2
    assert np.array_equal(result.labels, np.array([2, 2, -1, 4], dtype=int))
    assert result.report_df is not None
    assert result.report_df.loc["noise", "cluster_size"] == 1
