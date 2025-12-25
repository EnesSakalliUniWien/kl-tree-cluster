import numpy as np
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.mi_feature_selection import (
    compute_feature_scores,
    select_informative_features,
)


def test_compute_feature_scores_auto_disables_for_continuous_values():
    left = np.array([10.0, 0.0, -2.0])
    right = np.array([7.0, 1.0, -5.0])
    scores = compute_feature_scores(left, right, n_left=10, n_right=10, score_mode="auto")
    assert np.all(scores == 0.0)


def test_select_informative_features_auto_continuous_keeps_all_features():
    left = np.array([0.0, 10.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 9.0, 0.0])

    mask, scores, n_selected = select_informative_features(
        left,
        right,
        n_left=10,
        n_right=10,
        min_fraction=0.5,
        quantile_threshold=0.9,
        score_mode="auto",
    )
    assert n_selected == left.shape[0]
    assert mask.all()
    assert np.all(scores == 0.0)


def test_compute_feature_scores_unknown_mode_raises():
    with pytest.raises(ValueError):
        compute_feature_scores(
            np.array([0.2, 0.8]),
            np.array([0.3, 0.7]),
            n_left=10,
            n_right=10,
            score_mode="nope",
        )


def test_compute_feature_scores_sklearn_mi_is_nonnegative_for_bernoulli_means():
    left = np.array([0.1, 0.9, 0.5])
    right = np.array([0.9, 0.1, 0.5])
    scores = compute_feature_scores(left, right, n_left=50, n_right=50, score_mode="sklearn")
    assert scores.shape == left.shape
    assert np.all(scores >= 0.0)
