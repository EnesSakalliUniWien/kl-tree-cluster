from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.tbs_diffusion_runner import (
    _run_tbs_diffusion_graphtools_method,
)
from benchmarks.shared.types import MethodRunResult
from benchmarks.validation.sweeps.family_metric_nnls_grid import (
    TOPOLOGIES,
    build_family_geometry,
    run_family_metric_nnls_grid,
)
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import (
    FeatureBlock,
    FeatureSpace,
    bernoulli_feature_space_from_columns,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)


def test_family_geometry_distance_identities() -> None:
    binary = pd.DataFrame(
        [[1, 0, 1], [1, 1, 0], [0, 0, 0]],
        columns=["a", "b", "c"],
    )
    binary_space = bernoulli_feature_space_from_columns(binary.columns)
    balanced = build_family_geometry(
        {"generator": "binary", "category": "balanced", "name": "balanced"},
        binary,
        binary_space,
    )
    assert balanced.family == "balanced_binary"
    assert np.allclose(
        pdist(balanced.embedding, metric="sqeuclidean"),
        pdist(binary, metric="hamming"),
    )

    overlap = build_family_geometry(
        {"generator": "binary", "category": "overlapping_binary", "name": "overlap"},
        binary,
        binary_space,
    )
    expected_cosine = np.array([0.5, 1.0, 1.0])
    assert overlap.family == "sparse_overlap_binary"
    assert np.allclose(pdist(overlap.embedding, metric="sqeuclidean"), expected_cosine)

    categorical = pd.DataFrame(
        [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
        columns=["F0_c0", "F0_c1", "F1_c0", "F1_c1"],
    )
    categorical_geometry = build_family_geometry(
        {"generator": "categorical", "category": "categorical", "name": "cat"},
        categorical,
        infer_feature_space_from_columns(categorical.columns),
    )
    assert categorical_geometry.family == "categorical"
    assert np.allclose(
        pdist(categorical_geometry.embedding, metric="sqeuclidean"),
        [0.5, 1.0, 0.5],
    )


def test_continuous_and_sbm_geometries_are_finite_and_label_free() -> None:
    rng = np.random.default_rng(4)
    continuous = pd.DataFrame(rng.normal(size=(8, 3)), columns=["x", "y", "z"])
    standard = build_family_geometry(
        {"generator": "blobs_continuous", "category": "gaussian", "name": "gauss"},
        continuous,
        continuous_feature_space_from_columns(continuous.columns),
    )
    assert standard.family == "continuous_gaussian"
    assert np.isfinite(standard.embedding.to_numpy()).all()

    high_dimensional = pd.DataFrame(rng.normal(size=(5, 12)))
    high_dimensional.columns = [f"x{i}" for i in range(high_dimensional.shape[1])]
    shrinkage = build_family_geometry(
        {"generator": "continuous_low_rank_factor", "category": "proof", "name": "hd"},
        high_dimensional,
        continuous_feature_space_from_columns(high_dimensional.columns),
    )
    assert shrinkage.family == "high_dimensional_continuous"
    assert 1 <= shrinkage.embedding.shape[1] <= len(high_dimensional)
    assert np.isfinite(shrinkage.embedding.to_numpy()).all()

    adjacency = pd.DataFrame(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )
    spectral = build_family_geometry(
        {"generator": "sbm", "category": "graph", "name": "sbm"},
        adjacency,
        None,
    )
    assert spectral.family == "sbm"
    assert spectral.embedding.shape == (4, 2)
    assert np.isfinite(spectral.embedding.to_numpy()).all()


def test_mixed_feature_geometry_fails_closed_until_gower_is_defined() -> None:
    data = pd.DataFrame(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        columns=["continuous", "category_a", "category_b"],
    )
    feature_space = FeatureSpace(
        column_names=tuple(data.columns),
        blocks=(
            FeatureBlock(
                name="continuous",
                family="continuous",
                column_indices=(0,),
                chart="identity",
                covariance="empirical_gaussian",
                contrast_dimension=1,
            ),
            FeatureBlock(
                name="category",
                family="categorical",
                column_indices=(1, 2),
                chart="simplex_drop_last",
                covariance="multinomial_drop_last",
                contrast_dimension=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="mixed continuous/categorical Gower geometry"):
        build_family_geometry(
            {"generator": "mixed", "category": "mixed", "name": "mixed"},
            data,
            feature_space,
        )


def test_grid_writes_all_evidence_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_runner(data: pd.DataFrame, *_args: object, **_kwargs: object) -> MethodRunResult:
        labels = np.arange(len(data)) % 2
        report = pd.DataFrame({"cluster_id": labels}, index=data.index)
        return MethodRunResult(
            labels=labels,
            found_clusters=2,
            report_df=report,
            status="ok",
            skip_reason=None,
            extra={"full_edge_traversal_trace": []},
        )

    monkeypatch.setattr(
        "benchmarks.validation.sweeps.family_metric_nnls_grid._run_tbs_diffusion_graphtools_method",
        fake_runner,
    )
    outputs = run_family_metric_nnls_grid(
        output_dir=tmp_path,
        case_names=("binary_low_noise_4c",),
        topologies=TOPOLOGIES,
        branch_modes=("none",),
    )

    assert set(outputs) == {
        "cells",
        "labels",
        "pairwise_agreement",
        "stability",
        "rankings",
        "selection",
        "branch_time_pairs",
        "case_summary",
        "method_summary",
        "report",
        "manifest",
    }
    assert all(path.exists() for path in outputs.values())
    cells = pd.read_csv(outputs["cells"])
    assert len(cells) == 8
    assert set(cells["tree_inference"]) == set(TOPOLOGIES)
    assert cells["status"].eq("ok").all()
    assert cells["nnls_target"].eq("squared_euclidean").all()


def test_graphtools_runner_forwards_aligned_graph_and_branch_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame([[0, 1], [1, 0], [1, 1]], index=["a", "b", "c"])
    geometry = pd.DataFrame([[0.0], [1.0], [2.0]], index=data.index)
    captured: dict[str, object] = {}

    def fake_distance(graph_data: pd.DataFrame, **_kwargs: object):
        captured["graph_data"] = graph_data
        return np.array([1.0, 2.0, 1.0]), {"backend": "fake"}

    def fake_tbs(*_args: object, **kwargs: object) -> MethodRunResult:
        captured["branch_data"] = kwargs["branch_length_data_df"]
        return MethodRunResult(None, 0, None, "skip", "test", {})

    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_diffusion_runner._build_graphtools_diffusion_distance",
        fake_distance,
    )
    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_diffusion_runner.run_tbs_on_distance",
        fake_tbs,
    )
    _run_tbs_diffusion_graphtools_method(
        data,
        0.01,
        2,
        3,
        2,
        "euclidean",
        40,
        0.0,
        "+",
        0,
        graph_data_df=geometry,
        branch_length_data_df=geometry,
    )

    assert captured == {"graph_data": geometry, "branch_data": geometry}

    with pytest.raises(ValueError, match="index must exactly match"):
        _run_tbs_diffusion_graphtools_method(
            data,
            0.01,
            2,
            3,
            2,
            "euclidean",
            40,
            0.0,
            "+",
            0,
            graph_data_df=geometry.iloc[::-1],
        )
