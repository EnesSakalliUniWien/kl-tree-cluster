from __future__ import annotations

import pandas as pd
from benchmarks.full.run import (
    _compute_resume_coverage,
    _filter_methods_and_params_by_run_ids,
    _run_ids_for_params,
)
from benchmarks.shared.benchmark_grid import benchmark_grid, benchmark_run_id


def test_full_runner_resume_tracks_grid_run_ids():
    param_sets = {
        "kmeans": benchmark_grid(
            benchmark_class="canonical",
            grid_name="toy_grid",
            base_params={"n_clusters": "true", "n_init": 10},
            axes={"variant": ("a", "b")},
        )
    }
    expected_run_ids = _run_ids_for_params(["kmeans"], param_sets)
    existing_results = pd.DataFrame(
        [
            {"test_case": 1, "method": "kmeans", "run_id": expected_run_ids[0]},
            {"test_case": 1, "method": "kmeans", "run_id": expected_run_ids[1]},
            {"test_case": 2, "method": "kmeans", "run_id": expected_run_ids[0]},
        ]
    )

    completed, missing_by_case, tracked_count = _compute_resume_coverage(
        existing_results,
        expected_run_ids,
    )

    assert completed == {1}
    assert missing_by_case == {2: [expected_run_ids[1]]}
    assert tracked_count == 2

    methods, filtered_params = _filter_methods_and_params_by_run_ids(
        methods=["kmeans"],
        param_sets=param_sets,
        run_ids={expected_run_ids[1]},
    )

    assert methods == ["kmeans"]
    assert filtered_params == {"kmeans": [param_sets["kmeans"][1]]}
    assert benchmark_run_id("kmeans", filtered_params["kmeans"][0]) == expected_run_ids[1]
