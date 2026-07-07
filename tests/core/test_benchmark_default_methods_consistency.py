import pytest
from benchmarks.shared.benchmark_grid import benchmark_grid, strip_benchmark_metadata
from benchmarks.shared.config import DEFAULT_METHODS
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.method_selection import (
    resolve_methods_from_env,
    resolve_selected_methods_and_param_sets,
)


def test_full_runner_default_method_selection_matches_shared_defaults(monkeypatch):
    monkeypatch.delenv("TBS_METHODS", raising=False)

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)
    assert "tbs_diffusion_adaptive_nnls" in DEFAULT_METHODS


def test_full_runner_all_keyword_expands_to_shared_defaults(monkeypatch):
    monkeypatch.setenv("TBS_METHODS", "all")

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)


def test_explicit_empty_method_param_grid_is_invalid():
    with pytest.raises(ValueError, match="empty parameter grid"):
        resolve_selected_methods_and_param_sets(
            methods=["tbs"],
            method_params={"tbs": []},
            default_methods=DEFAULT_METHODS,
            method_specs=METHOD_SPECS,
        )


def test_benchmark_grid_expands_cartesian_axes_and_repeats():
    grid = benchmark_grid(
        benchmark_class="validation",
        grid_name="toy",
        base_params={"alpha": 0.01},
        axes={"linkage": ("average", "complete"), "k": (5, 10)},
        repeats=2,
    )

    assert len(grid) == 8
    assert {row["benchmark_class"] for row in grid} == {"validation"}
    assert {row["benchmark_grid"] for row in grid} == {"toy"}
    assert {row["benchmark_repeat"] for row in grid} == {0, 1}
    assert len({row["benchmark_run_id"] for row in grid}) == 8
    assert strip_benchmark_metadata(grid[0])["alpha"] == 0.01
    assert "benchmark_run_id" not in strip_benchmark_metadata(grid[0])


def test_duplicate_benchmark_run_ids_are_invalid():
    duplicate_params = [
        {"benchmark_run_id": "same", "n_clusters": "true"},
        {"benchmark_run_id": "same", "n_clusters": "true"},
    ]

    with pytest.raises(ValueError, match="Duplicate benchmark run_id"):
        resolve_selected_methods_and_param_sets(
            methods=["kmeans"],
            method_params={"kmeans": duplicate_params},
            default_methods=DEFAULT_METHODS,
            method_specs=METHOD_SPECS,
        )
