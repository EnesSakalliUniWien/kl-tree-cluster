from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns


def _small_continuous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 0.1, 3.0, 3.1],
            "y": [0.0, 0.2, 2.9, 3.2],
        },
        index=["L0", "L1", "L2", "L3"],
    )


def test_linkage_branch_time_requires_recomputed_branch_lengths() -> None:
    data = _small_continuous_frame()

    with pytest.raises(ValueError, match="requires recomputed fixed-topology branch lengths"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            edge_branch_length_variance_policy="normalized_branch_length",
        )


def test_linkage_branch_time_raw_heights_are_explicit_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_continuous_frame()
    captured: dict[str, object] = {}

    def fake_fit(*args: object, **kwargs: object) -> None:
        captured["called"] = True
        raise AssertionError("raw diagnostic branch-time must not invoke NNLS")

    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_runner.fit_fixed_topology_nnls_branch_lengths",
        fake_fit,
    )
    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_runner.run_gate_annotation_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after branch contract")),
    )

    with pytest.raises(RuntimeError, match="stop after branch contract"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            edge_branch_length_variance_policy="normalized_branch_length",
            allow_linkage_ultrametric_branch_time=True,
        )

    assert "called" not in captured


def test_fixed_topology_nnls_accepts_nonmonotone_linkage_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_continuous_frame()
    captured: dict[str, object] = {}

    def fake_linkage(*args: object, **kwargs: object) -> np.ndarray:
        return np.array(
            [
                [0, 1, 2.0, 2],
                [2, 3, 6.0, 2],
                [4, 5, 5.0, 4],
            ],
            dtype=float,
        )

    def fake_fit(tree: object, *args: object, **kwargs: object) -> None:
        captured["edge_lengths"] = [attrs["branch_length"] for _, _, attrs in tree.edges(data=True)]
        raise RuntimeError("stop after topology-only tree construction")

    monkeypatch.setattr(
        "tree_break_selection.tree.construction.build.linkage",
        fake_linkage,
    )
    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_runner.fit_fixed_topology_nnls_branch_lengths",
        fake_fit,
    )

    with pytest.raises(RuntimeError, match="topology-only tree construction"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="centroid",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            edge_branch_length_variance_policy="normalized_branch_length",
            branch_length_optimization_method="fixed_topology_nnls",
            branch_length_data_df=data,
        )

    assert captured["edge_lengths"] == [1.0] * 6


def test_fixed_topology_nnls_rejects_implicit_branch_geometry() -> None:
    data = _small_continuous_frame()

    with pytest.raises(ValueError, match="requires explicit branch_length_data_df"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            branch_length_optimization_method="fixed_topology_nnls",
        )


def test_fixed_topology_nnls_accepts_separate_aligned_branch_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_continuous_frame()
    geometry = pd.DataFrame(
        {"geometry": [0.0, 0.1, 1.0, 1.1]},
        index=data.index,
    )
    captured: dict[str, object] = {}

    def fake_fit(_tree: object, branch_data: pd.DataFrame, **_kwargs: object) -> None:
        captured["branch_data"] = branch_data
        raise RuntimeError("stop after aligned branch geometry")

    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_runner.fit_fixed_topology_nnls_branch_lengths",
        fake_fit,
    )

    with pytest.raises(RuntimeError, match="aligned branch geometry"):
        run_tbs_on_distance(
            data,
            pdist(geometry.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            branch_length_data_df=geometry,
            branch_length_optimization_method="fixed_topology_nnls",
            branch_length_optimization_target_metric="squared_euclidean",
        )

    assert captured["branch_data"] is geometry


def test_fixed_topology_nnls_rejects_misaligned_branch_geometry() -> None:
    data = _small_continuous_frame()
    geometry = data.iloc[::-1]

    with pytest.raises(ValueError, match="index must exactly match"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            branch_length_data_df=geometry,
            branch_length_optimization_method="fixed_topology_nnls",
        )
