from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns
from tree_break_selection.tree.optimized_branch_lengths import BranchLengthOptimizationResult


def _small_continuous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 0.1, 3.0, 3.1],
            "y": [0.0, 0.2, 2.9, 3.2],
        },
        index=["L0", "L1", "L2", "L3"],
    )


def _branch_length_result(*, status: str, applied_to_tree: bool) -> BranchLengthOptimizationResult:
    return BranchLengthOptimizationResult(
        method="fixed_topology_nnls",
        target_metric="squared_standardized_euclidean",
        status=status,
        n_leaves=4,
        n_edges=6,
        n_pairs_total=6,
        n_pairs_used=6,
        design_nnz=20,
        design_density=20.0 / 36.0,
        n_zero_design_columns=0,
        zero_design_column_fraction=0.0,
        pair_sample_size=None,
        random_state=0,
        applied_to_tree=applied_to_tree,
        apply_nonconverged=applied_to_tree and status != "ok",
        elapsed_sec=0.01,
        cost=1.0,
        optimality=1.0,
        iterations=1,
        target_mean=1.0,
        target_median=1.0,
        fitted_mean=1.0,
        fitted_median=1.0,
        residual_rmse=1.0,
        residual_mae=1.0,
        residual_rmse_to_target_mean=1.0,
        residual_mae_to_target_mean=1.0,
        branch_length_mean=1.0,
        branch_length_median=1.0,
        branch_length_min=1.0,
        branch_length_max=1.0,
        solver_message="test",
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


def test_fixed_topology_nnls_rejects_nonmonotone_linkage_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_continuous_frame()

    def fake_linkage(*args: object, **kwargs: object) -> np.ndarray:
        return np.array(
            [
                [0, 1, 2.0, 2],
                [2, 3, 6.0, 2],
                [4, 5, 5.0, 4],
            ],
            dtype=float,
        )

    monkeypatch.setattr(
        "tree_break_selection.tree.construction.build.linkage",
        fake_linkage,
    )

    with pytest.raises(ValueError, match="nondecreasing"):
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


def test_fixed_topology_nnls_runner_rejects_non_applied_solver_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_continuous_frame()

    monkeypatch.setattr(
        "benchmarks.shared.runners.tbs_runner.fit_fixed_topology_nnls_branch_lengths",
        lambda *_args, **_kwargs: _branch_length_result(
            status="solver_not_converged",
            applied_to_tree=False,
        ),
    )

    with pytest.raises(ValueError, match="did not apply branch lengths"):
        run_tbs_on_distance(
            data,
            pdist(data.to_numpy(), metric="euclidean"),
            0.01,
            tree_linkage_method="average",
            feature_space=continuous_feature_space_from_columns(tuple(data.columns)),
            branch_length_data_df=data,
            branch_length_optimization_method="fixed_topology_nnls",
        )
