"""Small helpers for benchmark parameter grids and row identity."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

from benchmarks.shared.util.params import format_params_for_display

BENCHMARK_CLASS_KEY = "benchmark_class"
BENCHMARK_GRID_KEY = "benchmark_grid"
BENCHMARK_REPEAT_KEY = "benchmark_repeat"
BENCHMARK_RUN_ID_KEY = "benchmark_run_id"
BENCHMARK_METADATA_KEYS = frozenset(
    {
        BENCHMARK_CLASS_KEY,
        BENCHMARK_GRID_KEY,
        BENCHMARK_REPEAT_KEY,
        BENCHMARK_RUN_ID_KEY,
    }
)


def _slug(value: object) -> str:
    raw = str(value)
    slug = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_").lower()
    return slug or "empty"


def benchmark_grid(
    *,
    benchmark_class: str,
    grid_name: str,
    base_params: Mapping[str, object],
    axes: Mapping[str, Iterable[object]] | None = None,
    repeats: int = 1,
) -> list[dict[str, object]]:
    """Build an explicit Cartesian benchmark grid with repeat annotations."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    axis_items = tuple((name, tuple(values)) for name, values in (axes or {}).items())
    combinations = product(*(values for _name, values in axis_items)) if axis_items else [()]

    rows: list[dict[str, object]] = []
    for combination in combinations:
        params = dict(base_params)
        cell_parts: list[str] = []
        for (axis_name, _values), axis_value in zip(axis_items, combination):
            params[axis_name] = axis_value
            cell_parts.append(f"{axis_name}_{_slug(axis_value)}")
        grid_cell = "__".join(cell_parts) if cell_parts else "default"
        for repeat in range(repeats):
            row = dict(params)
            row[BENCHMARK_CLASS_KEY] = benchmark_class
            row[BENCHMARK_GRID_KEY] = grid_name
            row[BENCHMARK_REPEAT_KEY] = repeat
            row[BENCHMARK_RUN_ID_KEY] = f"{grid_name}__{grid_cell}__r{repeat}"
            rows.append(row)
    return rows


def strip_benchmark_metadata(params: Mapping[str, object]) -> dict[str, object]:
    """Return runner parameters without benchmark bookkeeping keys."""
    return {key: value for key, value in params.items() if key not in BENCHMARK_METADATA_KEYS}


def benchmark_run_id(method_id: str, params: Mapping[str, object]) -> str:
    """Return a stable method-qualified run id for one method parameter row."""
    configured = params.get(BENCHMARK_RUN_ID_KEY)
    if configured is not None and str(configured).strip():
        return f"{method_id}::{configured}"

    display = format_params_for_display(strip_benchmark_metadata(params))
    suffix = _slug(display) if display else "default"
    repeat = params.get(BENCHMARK_REPEAT_KEY)
    if repeat is not None:
        suffix = f"{suffix}__r{int(repeat)}"
    return f"{method_id}::{suffix}"


def benchmark_param_metadata(
    method_id: str,
    params: Mapping[str, object],
    *,
    default_class: str = "unclassified",
    default_grid: str = "default",
) -> dict[str, object]:
    """Return normalized benchmark annotations for one method parameter row."""
    return {
        "benchmark_class": str(params.get(BENCHMARK_CLASS_KEY, default_class)),
        "benchmark_grid": str(params.get(BENCHMARK_GRID_KEY, default_grid)),
        "benchmark_repeat": int(params.get(BENCHMARK_REPEAT_KEY, 0)),
        "run_id": benchmark_run_id(method_id, params),
    }


__all__ = [
    "BENCHMARK_CLASS_KEY",
    "BENCHMARK_GRID_KEY",
    "BENCHMARK_METADATA_KEYS",
    "BENCHMARK_REPEAT_KEY",
    "BENCHMARK_RUN_ID_KEY",
    "benchmark_grid",
    "benchmark_param_metadata",
    "benchmark_run_id",
    "strip_benchmark_metadata",
]
