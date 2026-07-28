"""Package import and default-ownership contracts."""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys

from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from tree_break_selection.tree.construction import (
    DEFAULT_BINARY_TREE_DISTANCE_METRIC,
    DEFAULT_TREE_LINKAGE_METHOD,
)


def test_root_package_import_does_not_initialize_plotting() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import tree_break_selection; print('matplotlib.pyplot' in sys.modules)"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_plot_package_selects_file_safe_backend_before_pyplot() -> None:
    environment = {key: value for key, value in os.environ.items() if key != "MPLBACKEND"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tree_break_selection.plot.multiscale_umap import plt; "
                "fig, _ = plt.subplots(); "
                "print(plt.get_backend()); "
                "plt.close(fig)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip().lower() == "agg"


def test_runtime_config_modules_are_removed() -> None:
    assert importlib.util.find_spec("tree_break_selection.config") is None
    assert (
        importlib.util.find_spec("tree_break_selection.hierarchy_analysis.decomposition.config")
        is None
    )


def test_owned_defaults_are_explicit_and_runner_trace_is_compact() -> None:
    assert DEFAULT_BINARY_TREE_DISTANCE_METRIC == "hamming"
    assert DEFAULT_TREE_LINKAGE_METHOD == "average"
    trace_default = inspect.signature(run_tbs_on_distance).parameters["trace_level"].default
    assert trace_default == "compact"
