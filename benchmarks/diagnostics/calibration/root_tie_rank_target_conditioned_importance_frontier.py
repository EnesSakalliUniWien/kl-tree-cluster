"""Target-conditioned importance frontier for selected-root spectral tails.

Blind scalar tilts populate some selected-root strata but miss mixed
action-edge bands. This diagnostic searches proposal settings per unsupported
observed target while preserving external-null likelihood-ratio semantics. Rows
remain diagnostic until generated-neighborhood topology replay measures B and
the root-tail panel finds same-stratum support.
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

from benchmarks.diagnostics.calibration.root_selected_region_margins import (
    collect_observed_root_selected_region_row,
)
from benchmarks.diagnostics.calibration.root_tie_rank_null_proposal_frontier import (
    IMPORTANCE_COUPLED_EXTERNAL_NULL,
    IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL,
    _bernoulli_log_probability,
    build_proposal_mixed_rows,
    generate_binary_proposal_matrix,
)
from benchmarks.diagnostics.calibration.root_tie_rank_selected_null_simulation_pilot import (
    DEFAULT_OBSERVED_MIXED_ROWS,
    binary_null_probability_for_case,
)
from benchmarks.shared.cases import get_test_cases_by_suite

SCHEMA_VERSION = "root_tie_rank_target_conditioned_importance_frontier/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_target_conditioned_importance_frontier"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "root_tie_rank_target_conditioned_importance_frontier"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_importance_external_mild_accumulated"
    / "root_selected_spectral_tail_law_rows.csv"
)
DEFAULT_PROPOSAL_FAMILIES = (
    IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL,
    IMPORTANCE_COUPLED_EXTERNAL_NULL,
    "importance_unbalanced_two_block_external_null",
)
IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL = (
    "importance_correlated_two_factor_external_null"
)
DEFAULT_TWO_BLOCK_GRID = (0.125, 0.135, 0.145)
DEFAULT_BLOCK_FRACTION_GRID = (0.35, 0.45, 0.55, 0.65)
DEFAULT_SPIKE_FRACTION_GRID = (0.08, 0.12)
DEFAULT_SPIKE_DELTA_GRID = (0.20, 0.30)
DEFAULT_REPLICATES_PER_SETTING = 1
DEFAULT_SEED_OFFSET = 1_860_000

GENERATED_ROWS_OUTPUT = "target_conditioned_importance_generated_rows.csv"
TARGET_ROWS_OUTPUT = "target_conditioned_importance_target_rows.csv"
SUMMARY_OUTPUT = "target_conditioned_importance_summary.csv"
FAILURES_OUTPUT = "target_conditioned_importance_failures.csv"
MANIFEST_OUTPUT = "manifest.json"

TARGET_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_pre_topology_stratum_key",
    "proposal_family",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "proposal_block_fraction",
    "candidate_count",
    "pre_topology_stratum_hit_count",
    "best_candidate_case_id",
    "best_candidate_spectral_ratio",
    "best_candidate_importance_log_weight",
    "target_conditioning_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "generated_count",
    "pre_topology_supported_target_count",
    "summary_status",
)

FAILURE_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "proposal_family",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "proposal_block_fraction",
    "replicate",
    "proposal_attempt",
    "simulation_case_id",
    "seed",
    "failure_type",
    "failure_message",
)

GENERATED_EMPTY_COLUMNS = (
    "case_id",
    "conditioning_target_case_id",
    "conditioning_target_pre_topology_stratum_key",
    "target_conditioning_setting_id",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "proposal_block_fraction",
    "proposal_attempt",
    "proposal_attempts_per_setting",
    "proposal_acceptance_status",
    "candidate_pre_topology_stratum_key",
    "importance_log_weight",
    "importance_law_status",
)


@dataclass(frozen=True)
class TargetConditionedImportanceFrontierConfig:
    """Input/output contract for target-conditioned importance search."""

    output_dir: Path
    observed_mixed_region_rows_path: Path = DEFAULT_OBSERVED_MIXED_ROWS
    tail_rows_path: Path | None = DEFAULT_TAIL_ROWS
    suite: str = "full"
    target_case_ids: tuple[str, ...] = ()
    proposal_families: tuple[str, ...] = DEFAULT_PROPOSAL_FAMILIES
    two_block_delta_grid: tuple[float, ...] = DEFAULT_TWO_BLOCK_GRID
    spike_feature_fraction_grid: tuple[float, ...] = DEFAULT_SPIKE_FRACTION_GRID
    spike_delta_grid: tuple[float, ...] = DEFAULT_SPIKE_DELTA_GRID
    block_fraction_grid: tuple[float, ...] = DEFAULT_BLOCK_FRACTION_GRID
    replicates_per_setting: int = DEFAULT_REPLICATES_PER_SETTING
    attempts_per_setting: int = 1
    accept_target_pre_topology_stratum: bool = False
    seed_offset: int = DEFAULT_SEED_OFFSET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-mixed-region-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_MIXED_ROWS,
    )
    parser.add_argument("--tail-rows-path", type=Path, default=DEFAULT_TAIL_ROWS)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--target-case-ids", default=None)
    parser.add_argument("--proposal-families", default=",".join(DEFAULT_PROPOSAL_FAMILIES))
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
    parser.add_argument(
        "--block-fraction-grid",
        default=",".join(str(value) for value in DEFAULT_BLOCK_FRACTION_GRID),
    )
    parser.add_argument("--replicates-per-setting", type=int, default=1)
    parser.add_argument("--attempts-per-setting", type=int, default=1)
    parser.add_argument("--accept-target-pre-topology-stratum", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, TargetConditionedImportanceFrontierConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_csv_list(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("grid must contain at least one value.")
    return values


def _safe_float_token(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "_").replace("-", "m")


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace("/", "_").replace(".", "_")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _tie_band(value: object) -> str:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return "tie_missing"
    if numeric < 0.70:
        return "tie_low_lt_0_70"
    if numeric < 0.85:
        return "tie_mid_0_70_0_85"
    return "tie_high_ge_0_85"


def _action_band(value: object) -> str:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return "action_missing"
    log_value = math.log1p(max(numeric, 0.0))
    if log_value < 5.0:
        return "action_log_low_lt_5"
    if log_value < 7.0:
        return "action_log_mid_5_7"
    return "action_log_high_ge_7"


def _pre_topology_stratum_key(row: pd.Series | dict[str, object]) -> str:
    return "|".join(
        [
            str(row.get("root_mixed_region_component", "root_component_missing")),
            _tie_band(row.get("root_tie_rank_median_fraction", math.nan)),
            _action_band(row.get("root_sibling_selected_ratio", math.nan)),
            _action_band(row.get("root_edge_path_statistic_margin", math.nan)),
        ]
    )


def _select_cases(*, suite: str, case_names: Sequence[str]) -> dict[str, dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case.copy() for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing}.")
    return {name: by_name[name] for name in case_names}


def _target_case_ids(
    *,
    observed_mixed: pd.DataFrame,
    tail_rows_path: Path | None,
    explicit: tuple[str, ...],
) -> tuple[str, ...]:
    if explicit:
        return explicit
    if tail_rows_path is not None and Path(tail_rows_path).exists():
        tail_rows = pd.read_csv(tail_rows_path)
        _require_columns(
            tail_rows,
            {"target_case_id", "root_tail_inference_status"},
            "tail rows",
        )
        missing = tail_rows.loc[
            tail_rows["root_tail_inference_status"]
            .astype(str)
            .eq("fail_closed_selected_root_spectral_tail_support_missing"),
            "target_case_id",
        ]
        return tuple(missing.astype(str).tolist())
    return tuple(observed_mixed["case_id"].astype(str).tolist())


def _setting_records(
    config: TargetConditionedImportanceFrontierConfig,
) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    for family in config.proposal_families:
        for two_block_delta in config.two_block_delta_grid:
            if family == IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL:
                settings.append(
                    {
                        "proposal_family": family,
                        "proposal_two_block_delta": float(two_block_delta),
                        "proposal_spike_feature_fraction": math.nan,
                        "proposal_spike_delta": math.nan,
                        "proposal_block_fraction": math.nan,
                    }
                )
                continue
            if family == "importance_unbalanced_two_block_external_null":
                for block_fraction in config.block_fraction_grid:
                    settings.append(
                        {
                            "proposal_family": family,
                            "proposal_two_block_delta": float(two_block_delta),
                            "proposal_spike_feature_fraction": math.nan,
                            "proposal_spike_delta": math.nan,
                            "proposal_block_fraction": float(block_fraction),
                        }
                    )
                continue
            if family == IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL:
                for block_fraction in config.block_fraction_grid:
                    for spike_fraction in config.spike_feature_fraction_grid:
                        for spike_delta in config.spike_delta_grid:
                            settings.append(
                                {
                                    "proposal_family": family,
                                    "proposal_two_block_delta": float(two_block_delta),
                                    "proposal_spike_feature_fraction": float(
                                        spike_fraction
                                    ),
                                    "proposal_spike_delta": float(spike_delta),
                                    "proposal_block_fraction": float(block_fraction),
                                }
                            )
                continue
            for spike_fraction in config.spike_feature_fraction_grid:
                for spike_delta in config.spike_delta_grid:
                    settings.append(
                        {
                            "proposal_family": family,
                            "proposal_two_block_delta": float(two_block_delta),
                            "proposal_spike_feature_fraction": float(spike_fraction),
                            "proposal_spike_delta": float(spike_delta),
                            "proposal_block_fraction": math.nan,
                        }
                    )
    return settings


def _ensure_nonempty_feature_columns(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    fixed = np.asarray(matrix, dtype=int).copy()
    n_samples = fixed.shape[0]
    for feature_index in np.where(fixed.sum(axis=0) == 0)[0]:
        fixed[int(rng.integers(0, n_samples)), int(feature_index)] = 1
    return fixed


def generate_unbalanced_two_block_importance_matrix(
    *,
    base_case: dict[str, object],
    seed: int,
    two_block_delta: float,
    block_fraction: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate a weighted external-null proposal with an unbalanced root block."""
    if float(two_block_delta) < 0.0:
        raise ValueError("two_block_delta must be nonnegative.")
    if not (0.0 < float(block_fraction) < 1.0):
        raise ValueError("block_fraction must be in (0, 1).")
    n_samples = int(base_case["n_samples"])
    n_features = int(base_case["n_features"])
    base_probability = binary_null_probability_for_case(base_case)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_features)
    left_probabilities = np.clip(
        base_probability + float(two_block_delta) * signs,
        1e-3,
        1.0 - 1e-3,
    )
    right_probabilities = np.clip(
        base_probability - float(two_block_delta) * signs,
        1e-3,
        1.0 - 1e-3,
    )
    left_count = int(np.clip(round(float(block_fraction) * n_samples), 1, n_samples - 1))
    block_labels = np.ones(n_samples, dtype=int)
    block_labels[:left_count] = 0
    rng.shuffle(block_labels)
    probability_matrix = np.where(
        block_labels[:, None] == 0,
        left_probabilities[None, :],
        right_probabilities[None, :],
    )
    matrix = rng.binomial(1, probability_matrix).astype(int)
    matrix = _ensure_nonempty_feature_columns(matrix, rng=rng)
    target_probability_matrix = np.full_like(
        np.asarray(matrix, dtype=float),
        base_probability,
        dtype=float,
    )
    target_log_probability = _bernoulli_log_probability(
        matrix,
        target_probability_matrix,
    )
    proposal_log_probability = _bernoulli_log_probability(matrix, probability_matrix)
    feature_probabilities = np.mean(probability_matrix, axis=0)
    metadata = {
        "null_feature_probability": float(base_probability),
        "proposal_beta_concentration": math.nan,
        "proposal_two_block_delta": float(two_block_delta),
        "proposal_spike_feature_fraction": math.nan,
        "proposal_spike_delta": math.nan,
        "proposal_spike_feature_count": 0,
        "proposal_block_fraction": float(block_fraction),
        "generated_feature_probability_mean": float(np.mean(feature_probabilities)),
        "generated_feature_probability_min": float(np.min(feature_probabilities)),
        "generated_feature_probability_max": float(np.max(feature_probabilities)),
        "generated_matrix_density": float(np.mean(matrix)),
        "target_null_log_probability": target_log_probability,
        "proposal_log_probability": proposal_log_probability,
        "importance_log_weight": target_log_probability - proposal_log_probability,
        "importance_law_status": "target_iid_bernoulli_over_unbalanced_tilted_proposal",
    }
    return matrix, metadata


def generate_correlated_two_factor_importance_matrix(
    *,
    base_case: dict[str, object],
    seed: int,
    two_block_delta: float,
    spike_feature_fraction: float,
    spike_delta: float,
    factor_correlation: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate a weighted proposal with primary and residual root factors."""
    if float(two_block_delta) < 0.0:
        raise ValueError("two_block_delta must be nonnegative.")
    if not (0.0 < float(spike_feature_fraction) <= 1.0):
        raise ValueError("spike_feature_fraction must be in (0, 1].")
    if float(spike_delta) < 0.0:
        raise ValueError("spike_delta must be nonnegative.")
    if not (0.0 <= float(factor_correlation) <= 1.0):
        raise ValueError("factor_correlation must be in [0, 1].")
    n_samples = int(base_case["n_samples"])
    n_features = int(base_case["n_features"])
    base_probability = binary_null_probability_for_case(base_case)
    rng = np.random.default_rng(seed)

    primary_labels = np.ones(n_samples, dtype=float)
    primary_labels[: n_samples // 2] = -1.0
    rng.shuffle(primary_labels)
    independent_labels = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
    correlated_mask = rng.random(n_samples) < float(factor_correlation)
    residual_labels = np.where(correlated_mask, primary_labels, independent_labels)

    primary_signs = rng.choice(np.array([-1.0, 1.0]), size=n_features)
    residual_signs = np.zeros(n_features, dtype=float)
    active_count = max(
        1,
        int(math.ceil(float(spike_feature_fraction) * float(n_features))),
    )
    active_features = rng.choice(n_features, size=active_count, replace=False)
    residual_signs[active_features] = rng.choice(
        np.array([-1.0, 1.0]),
        size=active_count,
    )

    shift = (
        float(two_block_delta) * primary_labels[:, None] * primary_signs[None, :]
        + float(spike_delta) * residual_labels[:, None] * residual_signs[None, :]
    )
    probability_matrix = np.clip(base_probability + shift, 1e-3, 1.0 - 1e-3)
    matrix = rng.binomial(1, probability_matrix).astype(int)
    matrix = _ensure_nonempty_feature_columns(matrix, rng=rng)
    target_probability_matrix = np.full_like(
        np.asarray(matrix, dtype=float),
        base_probability,
        dtype=float,
    )
    target_log_probability = _bernoulli_log_probability(
        matrix,
        target_probability_matrix,
    )
    proposal_log_probability = _bernoulli_log_probability(matrix, probability_matrix)
    feature_probabilities = np.mean(probability_matrix, axis=0)
    metadata = {
        "null_feature_probability": float(base_probability),
        "proposal_beta_concentration": math.nan,
        "proposal_two_block_delta": float(two_block_delta),
        "proposal_spike_feature_fraction": float(spike_feature_fraction),
        "proposal_spike_delta": float(spike_delta),
        "proposal_spike_feature_count": int(active_count),
        "proposal_block_fraction": float(factor_correlation),
        "generated_feature_probability_mean": float(np.mean(feature_probabilities)),
        "generated_feature_probability_min": float(np.min(feature_probabilities)),
        "generated_feature_probability_max": float(np.max(feature_probabilities)),
        "generated_matrix_density": float(np.mean(matrix)),
        "target_null_log_probability": target_log_probability,
        "proposal_log_probability": proposal_log_probability,
        "importance_log_weight": target_log_probability - proposal_log_probability,
        "importance_law_status": (
            "target_iid_bernoulli_over_correlated_two_factor_tilted_proposal"
        ),
    }
    return matrix, metadata


def collect_target_conditioned_importance_rows(
    *,
    observed_mixed: pd.DataFrame,
    base_cases_by_id: dict[str, dict[str, object]],
    target_case_ids: Sequence[str],
    config: TargetConditionedImportanceFrontierConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate target-conditioned rows and return root, margins, failures."""
    root_records: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    matrix_dir = Path(config.output_dir) / "generated_target_conditioned_matrices"
    target_lookup = {
        str(row["case_id"]): row
        for _, row in observed_mixed.set_index("case_id", drop=False).iterrows()
    }
    settings = _setting_records(config)
    for target_index, target_id in enumerate(target_case_ids):
        base_case = base_cases_by_id[str(target_id)]
        target_row = target_lookup[str(target_id)]
        target_key = _pre_topology_stratum_key(target_row)
        for setting_index, setting in enumerate(settings):
            family = str(setting["proposal_family"])
            two_block_delta = _finite_float(setting["proposal_two_block_delta"])
            spike_fraction = _finite_float(setting["proposal_spike_feature_fraction"])
            spike_delta = _finite_float(setting["proposal_spike_delta"])
            block_fraction = _finite_float(setting["proposal_block_fraction"])
            setting_id = (
                f"{family}"
                f"__td{_safe_float_token(two_block_delta)}"
                f"__sf{_safe_float_token(spike_fraction) if math.isfinite(spike_fraction) else 'nan'}"
                f"__sd{_safe_float_token(spike_delta) if math.isfinite(spike_delta) else 'nan'}"
                f"__bf{_safe_float_token(block_fraction) if math.isfinite(block_fraction) else 'nan'}"
            )
            attempts_per_setting = max(int(config.attempts_per_setting), 1)
            for replicate in range(int(config.replicates_per_setting)):
                for attempt in range(attempts_per_setting):
                    seed = (
                        int(config.seed_offset)
                        + target_index * 1_000_000
                        + setting_index * 10_000
                        + replicate * attempts_per_setting
                        + attempt
                    )
                    simulation_case_id = (
                        f"{target_id}__target_importance_{_safe_id(setting_id)}"
                        f"_r{replicate:04d}_a{attempt:04d}"
                    )
                    try:
                        if family == "importance_unbalanced_two_block_external_null":
                            matrix, metadata = (
                                generate_unbalanced_two_block_importance_matrix(
                                    base_case=base_case,
                                    seed=seed,
                                    two_block_delta=two_block_delta,
                                    block_fraction=block_fraction
                                    if math.isfinite(block_fraction)
                                    else 0.5,
                                )
                            )
                        elif family == IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL:
                            matrix, metadata = (
                                generate_correlated_two_factor_importance_matrix(
                                    base_case=base_case,
                                    seed=seed,
                                    two_block_delta=two_block_delta,
                                    spike_feature_fraction=spike_fraction
                                    if math.isfinite(spike_fraction)
                                    else 0.1,
                                    spike_delta=spike_delta
                                    if math.isfinite(spike_delta)
                                    else 0.25,
                                    factor_correlation=block_fraction
                                    if math.isfinite(block_fraction)
                                    else 0.5,
                                )
                            )
                        else:
                            matrix, metadata = generate_binary_proposal_matrix(
                                base_case=base_case,
                                proposal_family=family,
                                seed=seed,
                                two_block_delta=two_block_delta,
                                spike_feature_fraction=spike_fraction
                                if math.isfinite(spike_fraction)
                                else 0.1,
                                spike_delta=spike_delta
                                if math.isfinite(spike_delta)
                                else 0.25,
                            )
                        matrix_dir.mkdir(parents=True, exist_ok=True)
                        matrix_path = matrix_dir / f"{simulation_case_id}.csv"
                        sample_names = [f"L{i + 1}" for i in range(matrix.shape[0])]
                        feature_names = [f"F{j}" for j in range(matrix.shape[1])]
                        pd.DataFrame(
                            matrix, index=sample_names, columns=feature_names
                        ).to_csv(matrix_path)
                        proposal_case = {
                            "name": simulation_case_id,
                            "generator": "preloaded",
                            "file_path": str(matrix_path),
                            "sep": ",",
                            "n_clusters": 1,
                            "category": "target_conditioned_importance_frontier",
                            "baseline_case_name": str(target_id),
                            "null_generation": family,
                            "proposal_family": family,
                            "proposal_calibration_status": (
                                "importance_weighted_external_null_support"
                            ),
                            "seed": int(seed),
                        }
                        row, margins = collect_observed_root_selected_region_row(
                            proposal_case
                        )
                        candidate_key = _pre_topology_stratum_key(row)
                        accepted = candidate_key == target_key
                        if config.accept_target_pre_topology_stratum and not accepted:
                            continue
                        row.update(
                            {
                                "base_case_id": str(target_id),
                                "data_role": "external_selected_null",
                                "calibration_role": "external_null_support",
                                "replicate": int(replicate),
                                "proposal_attempt": int(attempt),
                                "proposal_attempts_per_setting": int(
                                    attempts_per_setting
                                ),
                                "proposal_acceptance_status": (
                                    "accepted_target_pre_topology_stratum"
                                    if accepted
                                    else "unfiltered_candidate"
                                ),
                                "candidate_pre_topology_stratum_key": candidate_key,
                                "proposal_family": family,
                                "proposal_calibration_status": (
                                    "importance_weighted_external_null_support"
                                ),
                                "conditioning_target_case_id": str(target_id),
                                "conditioning_target_pre_topology_stratum_key": target_key,
                                "target_conditioning_setting_id": setting_id,
                                "proposal_two_block_delta": two_block_delta,
                                "proposal_spike_feature_fraction": spike_fraction,
                                "proposal_spike_delta": spike_delta,
                                "proposal_block_fraction": block_fraction,
                                "null_seed": int(seed),
                            }
                        )
                        row.update(metadata)
                        root_records.append(row)
                        case_margins = margins.copy()
                        case_margins.insert(0, "case_id", row["case_id"])
                        margin_tables.append(case_margins)
                    except Exception as exc:  # pragma: no cover - integration guard
                        failures.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "study_role": STUDY_ROLE,
                                "target_case_id": str(target_id),
                                "proposal_family": family,
                                "proposal_two_block_delta": two_block_delta,
                                "proposal_spike_feature_fraction": spike_fraction,
                                "proposal_spike_delta": spike_delta,
                                "proposal_block_fraction": block_fraction,
                                "replicate": int(replicate),
                                "proposal_attempt": int(attempt),
                                "simulation_case_id": simulation_case_id,
                                "seed": int(seed),
                                "failure_type": type(exc).__name__,
                                "failure_message": str(exc),
                            }
                        )
    root_rows = pd.DataFrame.from_records(root_records)
    merge_margins = (
        pd.concat(margin_tables, ignore_index=True) if margin_tables else pd.DataFrame()
    )
    failure_rows = pd.DataFrame.from_records(failures, columns=FAILURE_COLUMNS)
    return root_rows, merge_margins, failure_rows


def build_generated_mixed_rows(
    *,
    root_rows: pd.DataFrame,
    merge_margins: pd.DataFrame,
) -> pd.DataFrame:
    if root_rows.empty or merge_margins.empty:
        return pd.DataFrame(columns=GENERATED_EMPTY_COLUMNS)
    _, mixed_rows = build_proposal_mixed_rows(
        root_rows=root_rows,
        merge_margins=merge_margins,
    )
    metadata_columns = [
        "case_id",
        "conditioning_target_case_id",
        "conditioning_target_pre_topology_stratum_key",
        "target_conditioning_setting_id",
        "proposal_two_block_delta",
        "proposal_spike_feature_fraction",
        "proposal_spike_delta",
        "proposal_block_fraction",
        "proposal_attempt",
        "proposal_attempts_per_setting",
        "proposal_acceptance_status",
        "candidate_pre_topology_stratum_key",
        "importance_log_weight",
        "importance_law_status",
    ]
    metadata = root_rows[
        [
            column
            for column in metadata_columns
            if column in root_rows.columns
            and (column == "case_id" or column not in mixed_rows.columns)
        ]
    ].drop_duplicates("case_id")
    if metadata.columns.tolist() == ["case_id"]:
        return mixed_rows
    return mixed_rows.merge(metadata, on="case_id", how="left")


def build_target_rows(
    *,
    observed_mixed: pd.DataFrame,
    generated_mixed_rows: pd.DataFrame,
    target_case_ids: Sequence[str],
) -> pd.DataFrame:
    if generated_mixed_rows.empty:
        return pd.DataFrame(columns=TARGET_ROW_COLUMNS)
    generated = generated_mixed_rows.copy()
    generated["_pre_topology_key"] = generated.apply(_pre_topology_stratum_key, axis=1)
    observed = observed_mixed.set_index("case_id", drop=False)
    records: list[dict[str, object]] = []
    setting_keys = [
        "proposal_family",
        "proposal_two_block_delta",
        "proposal_spike_feature_fraction",
        "proposal_spike_delta",
        "proposal_block_fraction",
    ]
    for target_id in target_case_ids:
        target = observed.loc[str(target_id)]
        target_key = _pre_topology_stratum_key(target)
        target_generated = generated.loc[
            generated["conditioning_target_case_id"].astype(str).eq(str(target_id))
        ].copy()
        for keys, group in target_generated.groupby(setting_keys, dropna=False, sort=True):
            hits = group.loc[group["_pre_topology_key"].eq(target_key)].copy()
            ranked = hits.sort_values(
                [
                    "root_selected_eigenvalue_over_mp_upper_bound",
                    "case_id",
                ],
                ascending=[False, True],
            )
            best = ranked.iloc[0] if not ranked.empty else None
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "target_case_id": str(target_id),
                    "target_pre_topology_stratum_key": target_key,
                    "proposal_family": keys[0],
                    "proposal_two_block_delta": keys[1],
                    "proposal_spike_feature_fraction": keys[2],
                    "proposal_spike_delta": keys[3],
                    "proposal_block_fraction": keys[4],
                    "candidate_count": int(group.shape[0]),
                    "pre_topology_stratum_hit_count": int(hits.shape[0]),
                    "best_candidate_case_id": str(best["case_id"])
                    if best is not None
                    else "",
                    "best_candidate_spectral_ratio": _finite_float(
                        best.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
                    )
                    if best is not None
                    else math.nan,
                    "best_candidate_importance_log_weight": _finite_float(
                        best.get("importance_log_weight", math.nan)
                    )
                    if best is not None
                    else math.nan,
                    "target_conditioning_status": (
                        "pre_topology_stratum_hit_replay_needed"
                        if not hits.empty
                        else "pre_topology_stratum_missing"
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=TARGET_ROW_COLUMNS)


def summarize_target_rows(
    *,
    target_rows: pd.DataFrame,
    generated_mixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    if target_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    supported = int(
        target_rows.groupby("target_case_id")["pre_topology_stratum_hit_count"]
        .max()
        .gt(0)
        .sum()
    )
    generated_count = int(generated_mixed_rows.shape[0])
    target_count = int(target_rows["target_case_id"].nunique())
    status = (
        "all_targets_have_pre_topology_candidates_replay_needed"
        if supported == target_count and target_count > 0
        else "partial_pre_topology_candidate_support"
        if supported > 0
        else "no_pre_topology_candidate_support"
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": target_count,
                "generated_count": generated_count,
                "pre_topology_supported_target_count": supported,
                "summary_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_target_conditioned_importance_frontier(
    config: TargetConditionedImportanceFrontierConfig,
) -> dict[str, pd.DataFrame]:
    observed = pd.read_csv(config.observed_mixed_region_rows_path)
    _require_columns(observed, {"case_id"}, "observed mixed rows")
    target_case_ids = _target_case_ids(
        observed_mixed=observed,
        tail_rows_path=config.tail_rows_path,
        explicit=config.target_case_ids,
    )
    base_cases_by_id = _select_cases(suite=config.suite, case_names=target_case_ids)
    root_rows, merge_margins, failures = collect_target_conditioned_importance_rows(
        observed_mixed=observed,
        base_cases_by_id=base_cases_by_id,
        target_case_ids=target_case_ids,
        config=config,
    )
    generated_rows = build_generated_mixed_rows(
        root_rows=root_rows,
        merge_margins=merge_margins,
    )
    target_rows = build_target_rows(
        observed_mixed=observed,
        generated_mixed_rows=generated_rows,
        target_case_ids=target_case_ids,
    )
    summary = summarize_target_rows(
        target_rows=target_rows,
        generated_mixed_rows=generated_rows,
    )
    return {
        "generated_rows": generated_rows,
        "target_rows": target_rows,
        "summary": summary,
        "failures": failures,
    }


def run_target_conditioned_importance_frontier(
    config: TargetConditionedImportanceFrontierConfig,
) -> dict[str, Path]:
    start = perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_target_conditioned_importance_frontier(config)
    paths = {
        "generated_rows": config.output_dir / GENERATED_ROWS_OUTPUT,
        "target_rows": config.output_dir / TARGET_ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
        "failures": config.output_dir / FAILURES_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
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
    outputs = run_target_conditioned_importance_frontier(
        TargetConditionedImportanceFrontierConfig(
            output_dir=args.output_dir,
            observed_mixed_region_rows_path=args.observed_mixed_region_rows_path,
            tail_rows_path=args.tail_rows_path,
            suite=str(args.suite),
            target_case_ids=_parse_csv_list(args.target_case_ids),
            proposal_families=_parse_csv_list(args.proposal_families)
            or DEFAULT_PROPOSAL_FAMILIES,
            two_block_delta_grid=_parse_float_grid(args.two_block_delta_grid),
            spike_feature_fraction_grid=_parse_float_grid(
                args.spike_feature_fraction_grid
            ),
            spike_delta_grid=_parse_float_grid(args.spike_delta_grid),
            block_fraction_grid=_parse_float_grid(args.block_fraction_grid),
            replicates_per_setting=int(args.replicates_per_setting),
            attempts_per_setting=int(args.attempts_per_setting),
            accept_target_pre_topology_stratum=bool(
                args.accept_target_pre_topology_stratum
            ),
            seed_offset=int(args.seed_offset),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
