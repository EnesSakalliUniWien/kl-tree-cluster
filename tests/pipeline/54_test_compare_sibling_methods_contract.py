"""Contract tests for compare_sibling_methods benchmark script."""

from __future__ import annotations

import numpy as np
import pandas as pd

import benchmarks.compare_sibling_methods as csm


def test_run_single_case_uses_precomputed_condensed_distance(monkeypatch):
    expected_dist = np.array([0.11, 0.22, 0.33], dtype=float)
    captured_dists: list[np.ndarray] = []

    def _fake_get_default_test_cases():
        return [{"name": "case_0", "n_clusters": 3}]

    def _fake_generate_case_data(_case):
        data_df = pd.DataFrame(
            [[0, 1], [1, 0], [1, 1]],
            index=["S0", "S1", "S2"],
            columns=["F0", "F1"],
        )
        true_labels = np.array([0, 1, 2], dtype=int)
        x_original = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        meta = {
            "precomputed_distance_condensed": expected_dist.copy(),
            "precomputed_distance_matrix": None,
        }
        return data_df, true_labels, x_original, meta

    def _fake_linkage(dist, method):  # noqa: ARG001
        captured_dists.append(np.asarray(dist, dtype=float))
        # Minimal valid linkage for n=3 leaves.
        return np.array(
            [
                [0.0, 1.0, 0.1, 2.0],
                [2.0, 3.0, 0.2, 3.0],
            ],
            dtype=float,
        )

    class _FakeTree:
        def decompose(self, leaf_data, alpha_local, sibling_alpha):  # noqa: ARG002
            return {
                "num_clusters": 1,
                "cluster_assignments": {
                    0: {"leaves": list(leaf_data.index), "size": len(leaf_data.index)}
                },
            }

    monkeypatch.setattr(csm, "get_default_test_cases", _fake_get_default_test_cases)
    monkeypatch.setattr(csm, "generate_case_data", _fake_generate_case_data)
    monkeypatch.setattr(csm, "linkage", _fake_linkage)
    monkeypatch.setattr(csm.PosetTree, "from_linkage", lambda Z, leaf_names: _FakeTree())  # noqa: ARG005

    rows = csm._run_single_case_index(0)

    assert len(rows) == len(csm.METHODS)
    assert len(captured_dists) == len(csm.METHODS)
    for observed in captured_dists:
        np.testing.assert_allclose(observed, expected_dist)

