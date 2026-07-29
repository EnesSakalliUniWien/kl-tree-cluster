"""Parameter sweep for root selected spectral-lift proposal families.

The selected spectral generator target panel shows that current dense/action
proposal families occupy the high action-edge/tie root stratum but do not have
enough selected spectral excess. This diagnostic varies the proposal generator
knobs and asks a narrower pre-replay question: can any generated root reach the
observed spectral excess while also satisfying the high action-edge/tie root
conditions?

Rows that pass this root-metric screen still require generated-neighborhood
replay before they can be used in measured-bandwidth calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.root.root_tail_values import (
    finite_float,
    require_columns,
    safe_log1p,
    spectral_excess_log,
)
from benchmarks.diagnostics.calibration.root.selected.root_selected_region_margins import (
    collect_observed_root_selected_region_row,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_null_proposal_frontier import (
    COUPLED_EDGE_SPECTRAL_PROPOSAL,
    TWO_BLOCK_TILT_PROPOSAL,
    build_proposal_mixed_rows,
    generate_binary_proposal_matrix,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_selected_null_simulation_pilot import (
    DEFAULT_OBSERVED_MIXED_ROWS,
    binary_null_probability_for_case,
)
from benchmarks.shared.cases import get_test_cases_by_suite

SCHEMA_VERSION = "root_tie_rank_spectral_lift_parameter_sweep/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_spectral_lift_parameter_sweep_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_spectral_lift_parameter_sweep"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FAMILIES = (
    COUPLED_EDGE_SPECTRAL_PROPOSAL,
    TWO_BLOCK_TILT_PROPOSAL,
)
COHERENT_RANK_ONE_SPIKE_PROPOSAL = "coherent_rank_one_spike_proposal"
CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL = "conditioned_coherent_rank_one_spike_proposal"
DEFAULT_TWO_BLOCK_GRID = (0.35, 0.45, 0.55)
DEFAULT_SPIKE_FRACTION_GRID = (0.10, 0.20)
DEFAULT_SPIKE_DELTA_GRID = (0.45, 0.65)
DEFAULT_REPLICATES_PER_SETTING = 1
DEFAULT_SEED_OFFSET = 1_430_000

GENERATED_ROWS_OUTPUT = "root_tie_rank_spectral_lift_sweep_generated_rows.csv"
TARGET_ROWS_OUTPUT = "root_tie_rank_spectral_lift_sweep_target_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_spectral_lift_sweep_summary.csv"
FAILURES_OUTPUT = "root_tie_rank_spectral_lift_sweep_failures.csv"
MANIFEST_OUTPUT = "manifest.json"

SETTING_COLUMNS = (
    "sweep_setting_id",
    "proposal_family",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "conditioning_target_case_id",
    "conditioning_action_edge_bottleneck",
    "conditioning_tie_fraction",
    "conditioning_action_edge_score",
    "conditioning_geometry_score",
)

TARGET_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "sweep_setting_id",
    "proposal_family",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "target_spectral_excess_log",
    "target_action_edge_bottleneck",
    "target_tie_fraction",
    "eligible_generated_count",
    "best_generated_case_id",
    "best_generated_base_case_id",
    "best_generated_spectral_excess_log",
    "best_generated_action_edge_bottleneck",
    "best_generated_tie_fraction",
    "spectral_lift_log_required",
    "spectral_lift_multiplier_required",
    "spectral_reach_after_setting",
    "root_metric_screen_status",
    "next_replay_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "sweep_setting_id",
    "proposal_family",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "generated_root_count",
    "target_count",
    "conditioning_target_count",
    "spectral_reach_target_count",
    "median_spectral_lift_multiplier_required",
    "max_spectral_lift_multiplier_required",
    "median_best_generated_spectral_excess_log",
    "max_best_generated_spectral_excess_log",
    "summary_status",
)

FAILURE_COLUMNS = (
    "schema_version",
    "study_role",
    "base_case_id",
    "sweep_setting_id",
    "proposal_family",
    "replicate",
    "simulation_case_id",
    "seed",
    "failure_type",
    "failure_message",
)


@dataclass(frozen=True)
class RootTieRankSpectralLiftParameterSweepConfig:
    """Configuration for root spectral-lift parameter sweeps."""

    output_dir: Path
    observed_mixed_region_rows_path: Path = DEFAULT_OBSERVED_MIXED_ROWS
    suite: str = "full"
    case_names: tuple[str, ...] | None = None
    proposal_families: tuple[str, ...] = DEFAULT_PROPOSAL_FAMILIES
    two_block_delta_grid: tuple[float, ...] = DEFAULT_TWO_BLOCK_GRID
    spike_feature_fraction_grid: tuple[float, ...] = DEFAULT_SPIKE_FRACTION_GRID
    spike_delta_grid: tuple[float, ...] = DEFAULT_SPIKE_DELTA_GRID
    replicates_per_setting: int = DEFAULT_REPLICATES_PER_SETTING
    seed_offset: int = DEFAULT_SEED_OFFSET
    min_action_edge_fraction: float = 0.95
    min_tie_fraction_floor: float = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-mixed-region-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_MIXED_ROWS,
    )
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", default=None)
    parser.add_argument(
        "--proposal-families",
        default=",".join(DEFAULT_PROPOSAL_FAMILIES),
    )
    parser.add_argument(
        "--two-block-delta-grid",
        default=",".join(str(value) for value in DEFAULT_TWO_BLOCK_GRID),
    )
    parser.add_argument(
        "--spike-feature-fraction-grid",
        default=",".join(str(value) for value in DEFAULT_SPIKE_FRACTION_GRID),
    )
    parser.add_argument(
        "--spike-delta-grid",
        default=",".join(str(value) for value in DEFAULT_SPIKE_DELTA_GRID),
    )
    parser.add_argument("--replicates-per-setting", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument("--min-action-edge-fraction", type=float, default=0.95)
    parser.add_argument("--min-tie-fraction-floor", type=float, default=0.70)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankSpectralLiftParameterSweepConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_csv_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or None


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def _action_edge_bottleneck(row: pd.Series) -> float:
    tie = finite_float(row.get("root_tie_rank_median_fraction", math.nan))
    action = safe_log1p(row.get("root_sibling_selected_ratio", math.nan))
    edge = safe_log1p(row.get("root_edge_path_statistic_margin", math.nan))
    if not (math.isfinite(tie) and math.isfinite(action) and math.isfinite(edge)):
        return math.nan
    return float(tie * min(action, edge))


def _safe_float_token(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "_").replace("-", "m")


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace("/", "_").replace(".", "_").replace("=", "_")


def _select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing}.")
    return [by_name[name].copy() for name in case_names]


def _setting_records(
    *,
    proposal_families: Sequence[str],
    two_block_delta_grid: Sequence[float],
    spike_feature_fraction_grid: Sequence[float],
    spike_delta_grid: Sequence[float],
) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []

    def default_conditioning_fields() -> dict[str, object]:
        return {
            "conditioning_target_case_id": "",
            "conditioning_action_edge_bottleneck": math.nan,
            "conditioning_tie_fraction": math.nan,
            "conditioning_action_edge_score": math.nan,
            "conditioning_geometry_score": math.nan,
        }

    for family in proposal_families:
        family = str(family)
        if family == TWO_BLOCK_TILT_PROPOSAL:
            for two_block_delta in two_block_delta_grid:
                setting_id = f"{family}__td{_safe_float_token(float(two_block_delta))}"
                settings.append(
                    {
                        "sweep_setting_id": setting_id,
                        "proposal_family": family,
                        "proposal_two_block_delta": float(two_block_delta),
                        "proposal_spike_feature_fraction": math.nan,
                        "proposal_spike_delta": math.nan,
                        **default_conditioning_fields(),
                    }
                )
        elif family == COUPLED_EDGE_SPECTRAL_PROPOSAL:
            for two_block_delta in two_block_delta_grid:
                for spike_fraction in spike_feature_fraction_grid:
                    for spike_delta in spike_delta_grid:
                        setting_id = (
                            f"{family}__td{_safe_float_token(float(two_block_delta))}"
                            f"__sf{_safe_float_token(float(spike_fraction))}"
                            f"__sd{_safe_float_token(float(spike_delta))}"
                        )
                        settings.append(
                            {
                                "sweep_setting_id": setting_id,
                                "proposal_family": family,
                                "proposal_two_block_delta": float(two_block_delta),
                                "proposal_spike_feature_fraction": float(spike_fraction),
                                "proposal_spike_delta": float(spike_delta),
                                **default_conditioning_fields(),
                            }
                        )
        elif family == COHERENT_RANK_ONE_SPIKE_PROPOSAL:
            for spike_fraction in spike_feature_fraction_grid:
                for spike_delta in spike_delta_grid:
                    setting_id = (
                        f"{family}"
                        f"__sf{_safe_float_token(float(spike_fraction))}"
                        f"__sd{_safe_float_token(float(spike_delta))}"
                    )
                    settings.append(
                        {
                            "sweep_setting_id": setting_id,
                            "proposal_family": family,
                            "proposal_two_block_delta": math.nan,
                            "proposal_spike_feature_fraction": float(spike_fraction),
                            "proposal_spike_delta": float(spike_delta),
                            **default_conditioning_fields(),
                        }
                    )
        else:
            raise ValueError(f"Unsupported sweep proposal family: {family!r}.")
    return settings


def _scale_to_unit_interval(value: float, minimum: float, maximum: float) -> float:
    if not (math.isfinite(value) and math.isfinite(minimum) and math.isfinite(maximum)):
        return 0.0
    if maximum <= minimum:
        return 0.5
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def _conditioned_coherent_spike_setting_records(
    observed_mixed_rows: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build target-conditioned coherent spike settings.

    The conditioning uses only selected tie-rank/action-edge geometry. The
    target spectral excess is deliberately excluded so this remains a geometry
    conditioner rather than a target-fitting spectral rescue.
    """
    require_columns(
        observed_mixed_rows,
        {
            "case_id",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
        },
        "observed mixed rows",
    )
    targets = observed_mixed_rows.copy()
    targets["_action_edge_bottleneck"] = targets.apply(
        _action_edge_bottleneck,
        axis=1,
    )
    finite_bottlenecks = pd.to_numeric(
        targets["_action_edge_bottleneck"],
        errors="coerce",
    )
    finite_bottlenecks = finite_bottlenecks[np.isfinite(finite_bottlenecks)]
    minimum = float(finite_bottlenecks.min()) if not finite_bottlenecks.empty else 0.0
    maximum = float(finite_bottlenecks.max()) if not finite_bottlenecks.empty else 1.0
    settings: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        target_id = str(target["case_id"])
        action_edge = finite_float(target.get("_action_edge_bottleneck", math.nan))
        tie_fraction = finite_float(target.get("root_tie_rank_median_fraction", math.nan))
        action_score = _scale_to_unit_interval(action_edge, minimum, maximum)
        tie_score = float(np.clip(tie_fraction, 0.0, 1.0)) if math.isfinite(tie_fraction) else 0.0
        geometry_score = float(math.sqrt(max(action_score * tie_score, 0.0)))
        spike_fraction = float(np.clip(0.50 - 0.30 * geometry_score, 0.15, 0.50))
        spike_delta = float(np.clip(0.45 + 0.20 * geometry_score, 0.45, 0.65))
        setting_id = (
            f"{CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL}"
            f"__target_{_safe_id(target_id)}"
            f"__sf{_safe_float_token(spike_fraction)}"
            f"__sd{_safe_float_token(spike_delta)}"
        )
        settings.append(
            {
                "sweep_setting_id": setting_id,
                "proposal_family": CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL,
                "proposal_two_block_delta": math.nan,
                "proposal_spike_feature_fraction": spike_fraction,
                "proposal_spike_delta": spike_delta,
                "conditioning_target_case_id": target_id,
                "conditioning_action_edge_bottleneck": action_edge,
                "conditioning_tie_fraction": tie_fraction,
                "conditioning_action_edge_score": action_score,
                "conditioning_geometry_score": geometry_score,
            }
        )
    return settings


def generate_coherent_rank_one_spike_matrix(
    *,
    base_case: dict[str, object],
    seed: int,
    spike_feature_fraction: float,
    spike_delta: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate a binary matrix with one coherent rank-one block spike.

    Unlike the coupled dense-plus-sparse diagnostic, this proposal puts the
    active feature shifts in the same direction. It is designed to test whether
    a coherent population mode separates from the local MP bulk; it is not a
    selected-null calibration generator.
    """
    if not (0.0 < float(spike_feature_fraction) <= 1.0):
        raise ValueError("spike_feature_fraction must be in (0, 1].")
    if float(spike_delta) < 0.0:
        raise ValueError("spike_delta must be nonnegative.")
    n_samples = int(base_case["n_samples"])
    n_features = int(base_case["n_features"])
    base_probability = binary_null_probability_for_case(base_case)
    rng = np.random.default_rng(seed)
    active_count = max(
        1,
        int(math.ceil(float(spike_feature_fraction) * float(n_features))),
    )
    active_features = rng.choice(n_features, size=active_count, replace=False)
    left_probabilities = np.full(n_features, base_probability, dtype=float)
    right_probabilities = np.full(n_features, base_probability, dtype=float)
    left_probabilities[active_features] = np.clip(
        base_probability + float(spike_delta),
        1e-3,
        1.0 - 1e-3,
    )
    right_probabilities[active_features] = np.clip(
        base_probability - float(spike_delta),
        1e-3,
        1.0 - 1e-3,
    )
    block_labels = np.zeros(n_samples, dtype=int)
    block_labels[n_samples // 2 :] = 1
    rng.shuffle(block_labels)
    probability_matrix = np.where(
        block_labels[:, None] == 0,
        left_probabilities[None, :],
        right_probabilities[None, :],
    )
    matrix = rng.binomial(1, probability_matrix).astype(int)
    feature_probabilities = np.mean(probability_matrix, axis=0)
    for feature_index in np.where(matrix.sum(axis=0) == 0)[0]:
        matrix[int(rng.integers(0, n_samples)), int(feature_index)] = 1
    metadata = {
        "null_feature_probability": float(base_probability),
        "proposal_beta_concentration": math.nan,
        "proposal_two_block_delta": math.nan,
        "proposal_spike_feature_fraction": float(spike_feature_fraction),
        "proposal_spike_delta": float(spike_delta),
        "proposal_spike_feature_count": int(active_count),
        "generated_feature_probability_mean": float(np.mean(feature_probabilities)),
        "generated_feature_probability_min": float(np.min(feature_probabilities)),
        "generated_feature_probability_max": float(np.max(feature_probabilities)),
        "generated_matrix_density": float(np.mean(matrix)),
    }
    return matrix, metadata


def _write_generated_case(
    *,
    base_case: dict[str, object],
    simulation_case_id: str,
    matrix_dir: Path,
    seed: int,
    setting: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    family = str(setting["proposal_family"])
    spike_fraction = finite_float(setting["proposal_spike_feature_fraction"])
    spike_delta = finite_float(setting["proposal_spike_delta"])
    if family in {
        COHERENT_RANK_ONE_SPIKE_PROPOSAL,
        CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL,
    }:
        matrix, metadata = generate_coherent_rank_one_spike_matrix(
            base_case=base_case,
            seed=int(seed),
            spike_feature_fraction=(spike_fraction if math.isfinite(spike_fraction) else 0.1),
            spike_delta=spike_delta if math.isfinite(spike_delta) else 0.45,
        )
    else:
        matrix, metadata = generate_binary_proposal_matrix(
            base_case=base_case,
            proposal_family=family,
            seed=int(seed),
            two_block_delta=float(setting["proposal_two_block_delta"]),
            spike_feature_fraction=(spike_fraction if math.isfinite(spike_fraction) else 0.1),
            spike_delta=spike_delta if math.isfinite(spike_delta) else 0.45,
        )
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{simulation_case_id}.csv"
    sample_names = [f"L{i + 1}" for i in range(matrix.shape[0])]
    feature_names = [f"F{j}" for j in range(matrix.shape[1])]
    pd.DataFrame(matrix, index=sample_names, columns=feature_names).to_csv(matrix_path)
    case_metadata = {
        "name": simulation_case_id,
        "generator": "preloaded",
        "file_path": str(matrix_path),
        "sep": ",",
        "n_clusters": 1,
        "category": "root_tie_rank_spectral_lift_parameter_sweep",
        "baseline_case_name": str(base_case["name"]),
        "null_generation": family,
        "proposal_family": family,
        "proposal_calibration_status": "diagnostic_proposal_not_calibration",
        "seed": int(seed),
    }
    case_metadata.update(metadata)
    return case_metadata, metadata


def collect_spectral_lift_sweep_mixed_rows(
    *,
    base_cases: Sequence[dict[str, object]],
    settings: Sequence[dict[str, object]],
    output_dir: Path,
    replicates_per_setting: int = DEFAULT_REPLICATES_PER_SETTING,
    seed_offset: int = DEFAULT_SEED_OFFSET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sweep roots and return mixed-law rows plus failures."""
    if int(replicates_per_setting) <= 0:
        raise ValueError("replicates_per_setting must be positive.")
    root_records: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    matrix_dir = Path(output_dir) / "generated_spectral_lift_matrices"
    for setting_index, setting in enumerate(settings):
        setting_id = str(setting["sweep_setting_id"])
        family = str(setting["proposal_family"])
        safe_setting = _safe_id(setting_id)
        for base_index, base_case in enumerate(base_cases):
            base_case_id = str(base_case["name"])
            for replicate in range(int(replicates_per_setting)):
                seed = (
                    int(seed_offset) + setting_index * 1_000_000 + base_index * 10_000 + replicate
                )
                simulation_case_id = (
                    f"{base_case_id}__spectral_lift_{safe_setting}_r{replicate:04d}"
                )
                try:
                    proposal_case, proposal_metadata = _write_generated_case(
                        base_case=base_case,
                        simulation_case_id=simulation_case_id,
                        matrix_dir=matrix_dir,
                        seed=seed,
                        setting=setting,
                    )
                    row, margins = collect_observed_root_selected_region_row(proposal_case)
                    row.update(
                        {
                            "base_case_id": base_case_id,
                            "data_role": "diagnostic_proposal",
                            "calibration_role": ("diagnostic_proposal_not_null_support"),
                            "replicate": int(replicate),
                            "proposal_family": family,
                            "proposal_calibration_status": ("diagnostic_proposal_not_calibration"),
                            "null_seed": int(seed),
                            **{column: setting[column] for column in SETTING_COLUMNS},
                        }
                    )
                    row.update(proposal_metadata)
                    root_records.append(row)
                    case_margins = margins.copy()
                    case_margins.insert(0, "case_id", row["case_id"])
                    margin_tables.append(case_margins)
                except Exception as exc:  # pragma: no cover - integration safeguard
                    failures.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "study_role": STUDY_ROLE,
                            "base_case_id": base_case_id,
                            "sweep_setting_id": setting_id,
                            "proposal_family": family,
                            "replicate": int(replicate),
                            "simulation_case_id": simulation_case_id,
                            "seed": int(seed),
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    )
    root_rows = pd.DataFrame.from_records(root_records)
    merge_margins = pd.concat(margin_tables, ignore_index=True) if margin_tables else pd.DataFrame()
    if root_rows.empty or merge_margins.empty:
        mixed_rows = pd.DataFrame()
    else:
        _, mixed_rows = build_proposal_mixed_rows(
            root_rows=root_rows,
            merge_margins=merge_margins,
        )
        metadata_columns = [
            "case_id",
            "sweep_setting_id",
            "proposal_two_block_delta",
            "proposal_spike_feature_fraction",
            "proposal_spike_delta",
            "conditioning_target_case_id",
            "conditioning_action_edge_bottleneck",
            "conditioning_tie_fraction",
            "conditioning_action_edge_score",
            "conditioning_geometry_score",
        ]
        missing_metadata_columns = [
            column
            for column in metadata_columns
            if column == "case_id" or column not in mixed_rows.columns
        ]
        metadata = root_rows[missing_metadata_columns].drop_duplicates("case_id")
        mixed_rows = mixed_rows.merge(metadata, on="case_id", how="left")
    failures_frame = pd.DataFrame.from_records(failures, columns=FAILURE_COLUMNS)
    return mixed_rows, failures_frame


def _best_generated_row(eligible: pd.DataFrame) -> pd.Series | None:
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        [
            "spectral_excess_log",
            "_action_edge_bottleneck",
            "_tie_fraction",
            "case_id",
        ],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def build_spectral_lift_sweep_target_rows(
    *,
    observed_mixed_rows: pd.DataFrame,
    generated_mixed_rows: pd.DataFrame,
    min_action_edge_fraction: float = 0.95,
    min_tie_fraction_floor: float = 0.70,
) -> pd.DataFrame:
    """Compare each sweep setting against each observed target."""
    require_columns(
        observed_mixed_rows,
        {
            "case_id",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "observed mixed rows",
    )
    if generated_mixed_rows.empty:
        return pd.DataFrame(columns=TARGET_ROW_COLUMNS)
    require_columns(
        generated_mixed_rows,
        {
            "case_id",
            "base_case_id",
            "sweep_setting_id",
            "proposal_family",
            "proposal_two_block_delta",
            "proposal_spike_feature_fraction",
            "proposal_spike_delta",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "generated mixed rows",
    )
    generated = generated_mixed_rows.copy()
    generated["spectral_excess_log"] = generated[
        "root_selected_eigenvalue_over_mp_upper_bound"
    ].map(spectral_excess_log)
    generated["_action_edge_bottleneck"] = generated.apply(
        _action_edge_bottleneck,
        axis=1,
    )
    generated["_tie_fraction"] = pd.to_numeric(
        generated["root_tie_rank_median_fraction"],
        errors="coerce",
    )
    records: list[dict[str, object]] = []
    setting_keys = [
        "sweep_setting_id",
        "proposal_family",
        "proposal_two_block_delta",
        "proposal_spike_feature_fraction",
        "proposal_spike_delta",
    ]
    for _, target in observed_mixed_rows.sort_values("case_id").iterrows():
        target_id = str(target["case_id"])
        target_spectral = spectral_excess_log(
            target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        )
        target_bottleneck = _action_edge_bottleneck(target)
        target_tie = finite_float(target.get("root_tie_rank_median_fraction", math.nan))
        target_generated = generated
        if "conditioning_target_case_id" in generated.columns:
            conditioning_target = generated["conditioning_target_case_id"].fillna("")
            target_generated = generated.loc[
                conditioning_target.eq("") | conditioning_target.eq(target_id)
            ].copy()
        for keys, group in target_generated.groupby(
            setting_keys,
            dropna=False,
            sort=True,
        ):
            threshold = float(min_action_edge_fraction) * target_bottleneck
            eligible = group.loc[
                group["_action_edge_bottleneck"].ge(threshold)
                & group["_tie_fraction"].ge(float(min_tie_fraction_floor))
            ].copy()
            best = _best_generated_row(eligible)
            if best is None:
                best_case_id = ""
                best_base_case_id = ""
                best_spectral = math.nan
                best_bottleneck = math.nan
                best_tie = math.nan
            else:
                best_case_id = str(best["case_id"])
                best_base_case_id = str(best.get("base_case_id", ""))
                best_spectral = finite_float(best["spectral_excess_log"])
                best_bottleneck = finite_float(best["_action_edge_bottleneck"])
                best_tie = finite_float(best["_tie_fraction"])
            if math.isfinite(target_spectral) and math.isfinite(best_spectral):
                lift_log = float(max(target_spectral - best_spectral, 0.0))
            else:
                lift_log = math.nan
            reached = math.isfinite(lift_log) and lift_log <= 0.0
            if eligible.empty:
                status = "root_metric_conditioning_missing"
                next_step = "increase_action_edge_tie_before_neighborhood_replay"
            elif reached:
                status = "root_metric_spectral_reach_replay_needed"
                next_step = "replay_setting_through_selected_neighborhood"
            else:
                status = "root_metric_spectral_lift_still_required"
                next_step = "increase_spectral_lift_before_neighborhood_replay"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "target_case_id": target_id,
                    "sweep_setting_id": keys[0],
                    "proposal_family": keys[1],
                    "proposal_two_block_delta": keys[2],
                    "proposal_spike_feature_fraction": keys[3],
                    "proposal_spike_delta": keys[4],
                    "target_spectral_excess_log": target_spectral,
                    "target_action_edge_bottleneck": target_bottleneck,
                    "target_tie_fraction": target_tie,
                    "eligible_generated_count": int(eligible.shape[0]),
                    "best_generated_case_id": best_case_id,
                    "best_generated_base_case_id": best_base_case_id,
                    "best_generated_spectral_excess_log": best_spectral,
                    "best_generated_action_edge_bottleneck": best_bottleneck,
                    "best_generated_tie_fraction": best_tie,
                    "spectral_lift_log_required": lift_log,
                    "spectral_lift_multiplier_required": float(math.exp(lift_log))
                    if math.isfinite(lift_log)
                    else math.nan,
                    "spectral_reach_after_setting": bool(reached),
                    "root_metric_screen_status": status,
                    "next_replay_step": next_step,
                }
            )
    return pd.DataFrame.from_records(records, columns=TARGET_ROW_COLUMNS)


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.median()) if not finite.empty else math.nan


def _safe_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.max()) if not finite.empty else math.nan


def _summary_status(group: pd.DataFrame) -> str:
    covered = int(group["eligible_generated_count"].gt(0).sum())
    reached = int(group["spectral_reach_after_setting"].sum())
    if covered <= 0:
        return "setting_misses_action_edge_tie_conditioning"
    if reached == int(group.shape[0]):
        return "setting_reaches_all_targets_root_metrics_replay_needed"
    if reached > 0:
        return "setting_reaches_some_targets_root_metrics_replay_needed"
    return "setting_requires_more_spectral_lift"


def summarize_spectral_lift_sweep_targets(
    *,
    target_rows: pd.DataFrame,
    generated_mixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target coverage by sweep setting."""
    if target_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    generated_counts = (
        generated_mixed_rows.groupby("sweep_setting_id", dropna=False)
        .size()
        .rename("generated_root_count")
        if not generated_mixed_rows.empty and "sweep_setting_id" in generated_mixed_rows
        else pd.Series(dtype=int)
    )
    records: list[dict[str, object]] = []
    setting_keys = [
        "sweep_setting_id",
        "proposal_family",
        "proposal_two_block_delta",
        "proposal_spike_feature_fraction",
        "proposal_spike_delta",
    ]
    for keys, group in target_rows.groupby(setting_keys, dropna=False, sort=True):
        covered = int(group["eligible_generated_count"].gt(0).sum())
        reached = int(group["spectral_reach_after_setting"].sum())
        setting_id = str(keys[0])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "sweep_setting_id": setting_id,
                "proposal_family": keys[1],
                "proposal_two_block_delta": keys[2],
                "proposal_spike_feature_fraction": keys[3],
                "proposal_spike_delta": keys[4],
                "generated_root_count": int(generated_counts.get(setting_id, 0)),
                "target_count": int(group.shape[0]),
                "conditioning_target_count": covered,
                "spectral_reach_target_count": reached,
                "median_spectral_lift_multiplier_required": _safe_median(
                    group.loc[
                        group["eligible_generated_count"].gt(0),
                        "spectral_lift_multiplier_required",
                    ]
                ),
                "max_spectral_lift_multiplier_required": _safe_max(
                    group.loc[
                        group["eligible_generated_count"].gt(0),
                        "spectral_lift_multiplier_required",
                    ]
                ),
                "median_best_generated_spectral_excess_log": _safe_median(
                    group["best_generated_spectral_excess_log"]
                ),
                "max_best_generated_spectral_excess_log": _safe_max(
                    group["best_generated_spectral_excess_log"]
                ),
                "summary_status": _summary_status(group),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_spectral_lift_parameter_sweep(
    config: RootTieRankSpectralLiftParameterSweepConfig,
) -> dict[str, pd.DataFrame]:
    """Run the sweep and return output tables."""
    observed_mixed = pd.read_csv(config.observed_mixed_region_rows_path)
    require_columns(observed_mixed, {"case_id"}, "observed mixed rows")
    case_names = (
        tuple(config.case_names)
        if config.case_names
        else tuple(observed_mixed["case_id"].astype(str).tolist())
    )
    base_cases = _select_cases(suite=config.suite, case_names=case_names)
    conditioned_requested = CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL in config.proposal_families
    static_families = tuple(
        family
        for family in config.proposal_families
        if family != CONDITIONED_COHERENT_RANK_ONE_SPIKE_PROPOSAL
    )
    settings = (
        _setting_records(
            proposal_families=static_families,
            two_block_delta_grid=config.two_block_delta_grid,
            spike_feature_fraction_grid=config.spike_feature_fraction_grid,
            spike_delta_grid=config.spike_delta_grid,
        )
        if static_families
        else []
    )
    if conditioned_requested:
        settings.extend(_conditioned_coherent_spike_setting_records(observed_mixed))
    generated_rows, failures = collect_spectral_lift_sweep_mixed_rows(
        base_cases=base_cases,
        settings=settings,
        output_dir=config.output_dir,
        replicates_per_setting=int(config.replicates_per_setting),
        seed_offset=int(config.seed_offset),
    )
    target_rows = build_spectral_lift_sweep_target_rows(
        observed_mixed_rows=observed_mixed,
        generated_mixed_rows=generated_rows,
        min_action_edge_fraction=float(config.min_action_edge_fraction),
        min_tie_fraction_floor=float(config.min_tie_fraction_floor),
    )
    summary = summarize_spectral_lift_sweep_targets(
        target_rows=target_rows,
        generated_mixed_rows=generated_rows,
    )
    return {
        "generated_rows": generated_rows,
        "target_rows": target_rows,
        "summary": summary,
        "failures": failures,
    }


def run_spectral_lift_parameter_sweep(
    config: RootTieRankSpectralLiftParameterSweepConfig,
) -> dict[str, Path]:
    """Run the spectral-lift parameter sweep and write outputs."""
    start = perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_spectral_lift_parameter_sweep(config)
    paths = {
        "generated_rows": output_dir / GENERATED_ROWS_OUTPUT,
        "target_rows": output_dir / TARGET_ROWS_OUTPUT,
        "summary": output_dir / SUMMARY_OUTPUT,
        "failures": output_dir / FAILURES_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": float(perf_counter() - start),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    outputs = run_spectral_lift_parameter_sweep(
        RootTieRankSpectralLiftParameterSweepConfig(
            output_dir=args.output_dir,
            observed_mixed_region_rows_path=args.observed_mixed_region_rows_path,
            suite=str(args.suite),
            case_names=_parse_csv_list(args.case_names),
            proposal_families=_parse_csv_list(args.proposal_families) or DEFAULT_PROPOSAL_FAMILIES,
            two_block_delta_grid=_parse_float_grid(args.two_block_delta_grid),
            spike_feature_fraction_grid=_parse_float_grid(args.spike_feature_fraction_grid),
            spike_delta_grid=_parse_float_grid(args.spike_delta_grid),
            replicates_per_setting=int(args.replicates_per_setting),
            seed_offset=int(args.seed_offset),
            min_action_edge_fraction=float(args.min_action_edge_fraction),
            min_tie_fraction_floor=float(args.min_tie_fraction_floor),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
