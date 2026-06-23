from __future__ import annotations

import numpy as np
from benchmarks.experiments.branch_length.logic import (
    _one_hot_encode_sequence_matrix,
    run_branch_length_benchmark,
)
from tree_break_selection.tree.feature_space import infer_feature_space_from_columns


def test_branch_length_sequence_encoding_uses_categorical_feature_blocks() -> None:
    encoded, columns = _one_hot_encode_sequence_matrix(
        np.array([[0, 1], [2, 3]], dtype=int),
        n_categories=4,
    )

    assert encoded.shape == (2, 8)
    assert columns == [
        "F0_c0",
        "F0_c1",
        "F0_c2",
        "F0_c3",
        "F1_c0",
        "F1_c1",
        "F1_c2",
        "F1_c3",
    ]
    assert encoded[0].tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert encoded[1].tolist() == [0, 0, 1, 0, 0, 0, 0, 1]

    feature_space = infer_feature_space_from_columns(columns)
    assert feature_space.family_label == "categorical"
    assert feature_space.contrast_dimension == 6


def test_branch_length_benchmark_fixed_coordinate_method_runs_encoded_sites() -> None:
    rows = run_branch_length_benchmark(
        n_leaves=20,
        n_features=12,
        n_categories=4,
        branch_lengths=[0.2],
        random_seed=7,
        method="tbs_fixed_coordinate_bh",
        verbose=False,
    )

    assert rows.loc[0, "status"] == "ok"
    assert rows.loc[0, "skip_reason"] is None
