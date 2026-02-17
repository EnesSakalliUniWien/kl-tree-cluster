import numpy as np

from benchmarks.shared.runners.louvain_runner import _run_louvain_method
from benchmarks.shared.runners.optics_runner import _run_optics_method
from benchmarks.shared.runners.spectral_runner import _run_spectral_method


def _toy_distance_matrix() -> np.ndarray:
    # Two compact neighborhoods with larger between-group distances.
    return np.array(
        [
            [0.0, 0.12, 0.85, 0.91],
            [0.12, 0.0, 0.83, 0.89],
            [0.85, 0.83, 0.0, 0.18],
            [0.91, 0.89, 0.18, 0.0],
        ],
        dtype=float,
    )


def _toy_feature_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=float,
    )


def _assert_singleton_ok(result) -> None:
    assert result.status == "ok"
    assert result.skip_reason is None
    assert result.labels is not None
    assert len(result.labels) == 1
    assert int(result.found_clusters) == 1


def test_singleton_behavior_is_consistent_across_runners():
    singleton_distance = np.zeros((1, 1), dtype=float)
    singleton_features = np.zeros((1, 2), dtype=float)

    _assert_singleton_ok(_run_louvain_method(singleton_distance, {}, seed=7))
    _assert_singleton_ok(_run_optics_method(singleton_distance, {}, seed=7))
    _assert_singleton_ok(_run_spectral_method(singleton_features, {}, seed=7))


def test_failure_paths_use_skip_and_populate_skip_reason():
    distance = _toy_distance_matrix()
    features = _toy_feature_matrix()

    louvain_res = _run_louvain_method(distance, {"resolution": "bad"}, seed=42)
    optics_res = _run_optics_method(distance, {"min_samples": "bad"}, seed=42)
    spectral_res = _run_spectral_method(features, {"n_clusters": "bad"}, seed=42)

    for res in (louvain_res, optics_res, spectral_res):
        assert res.status == "skip"
        assert isinstance(res.skip_reason, str)
        assert res.skip_reason.strip()
        assert res.labels is None


def test_deterministic_seed_handling_is_consistent():
    distance = _toy_distance_matrix()
    features = _toy_feature_matrix()

    # Louvain
    l1 = _run_louvain_method(distance, {"n_neighbors": 2, "resolution": 1.0}, seed=123)
    l2 = _run_louvain_method(distance, {"n_neighbors": 2, "resolution": 1.0}, seed=123)
    assert l1.status == l2.status
    if l1.labels is not None and l2.labels is not None:
        assert np.array_equal(np.asarray(l1.labels), np.asarray(l2.labels))

    # OPTICS
    o1 = _run_optics_method(distance, {"min_samples": 2, "xi": 0.05}, seed=123)
    o2 = _run_optics_method(distance, {"min_samples": 2, "xi": 0.05}, seed=123)
    assert o1.status == o2.status == "ok"
    assert o1.skip_reason is None and o2.skip_reason is None
    assert np.array_equal(np.asarray(o1.labels), np.asarray(o2.labels))

    # Spectral
    s1 = _run_spectral_method(features, {"n_clusters": 2, "affinity": "nearest_neighbors"}, seed=123)
    s2 = _run_spectral_method(features, {"n_clusters": 2, "affinity": "nearest_neighbors"}, seed=123)
    assert s1.status == s2.status == "ok"
    assert s1.skip_reason is None and s2.skip_reason is None
    assert np.array_equal(np.asarray(s1.labels), np.asarray(s2.labels))
