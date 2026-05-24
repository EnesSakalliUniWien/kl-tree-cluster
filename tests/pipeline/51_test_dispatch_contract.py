import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types.method_spec import MethodSpec
from scipy.spatial.distance import pdist, squareform


def _toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
            ],
            dtype=float,
        ),
        columns=["f0", "f1"],
    )


def test_dispatch_result_rejects_invalid_runner_params():
    df = _toy_dataframe()
    with pytest.raises(ValueError):
        run_clustering_result(
            data_df=df,
            method_id="kmeans",
            params={"n_clusters": "bad"},
            seed=42,
        )


def test_dispatch_result_rejects_invalid_spectral_params():
    df = _toy_dataframe()
    with pytest.raises(ValueError):
        run_clustering_result(
            data_df=df,
            method_id="spectral",
            params={"n_clusters": "bad"},
            seed=42,
        )


def test_dispatch_result_propagates_unexpected_exception(monkeypatch):
    def _raise_runner(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        METHOD_SPECS,
        "kmeans",
        MethodSpec(
            name="K-Means",
            runner=_raise_runner,
            param_grid=[{"n_clusters": 2}],
        ),
    )

    df = _toy_dataframe()
    with pytest.raises(RuntimeError, match="boom"):
        run_clustering_result(
            data_df=df,
            method_id="kmeans",
            params={"n_clusters": 2},
            seed=42,
        )


def test_run_clustering_result_uses_provided_kl_distance_condensed():
    df = _toy_dataframe()
    dist_condensed = pdist(df.values, metric="euclidean")
    result = run_clustering_result(
        data_df=df,
        method_id="kl",
        params={"tree_distance_metric": "euclidean", "tree_linkage_method": "average"},
        seed=42,
        distance_condensed=dist_condensed,
    )

    assert result.status in {"ok", "skip"}
    if result.status == "ok":
        assert result.labels is not None
        assert len(result.labels) == len(df)
        assert result.skip_reason is None
    else:
        assert result.labels is None
        assert isinstance(result.skip_reason, str)
        assert result.skip_reason.strip()


def test_run_clustering_result_uses_provided_graph_distance_matrix():
    df = _toy_dataframe()
    dist_condensed = pdist(df.values, metric="euclidean")
    dist_matrix = squareform(dist_condensed)
    result = run_clustering_result(
        data_df=df,
        method_id="leiden",
        params={"n_neighbors": 2, "resolution": 1.0},
        seed=42,
        distance_matrix=dist_matrix,
    )

    assert result.status in {"ok", "skip"}
    if result.status == "ok":
        assert result.labels is not None
        assert len(result.labels) == len(df)
        assert result.skip_reason is None
    else:
        assert result.labels is None
        assert isinstance(result.skip_reason, str)
        assert result.skip_reason.strip()
