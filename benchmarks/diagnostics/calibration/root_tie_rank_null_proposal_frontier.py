"""Proposal frontier for root tie-rank selected-null calibration.

The iid Bernoulli selected-null pilot is a valid first calibration candidate,
but the first overlap run did not hit the observed high-margin root strata.
This diagnostic adds proposal families that can explore those strata while
keeping calibration semantics explicit: proposal rows are frontier evidence,
not null support, unless their calibration role says so.
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

from benchmarks.diagnostics.calibration.root_selected_mixed_region_law import (
    build_root_selected_mixed_region_law_rows,
)
from benchmarks.diagnostics.calibration.root_selected_region_margins import (
    SCHEMA_VERSION as ROOT_MARGIN_SCHEMA_VERSION,
)
from benchmarks.diagnostics.calibration.root_selected_region_margins import (
    collect_observed_root_selected_region_row,
)
from benchmarks.diagnostics.calibration.root_selected_tie_cell_burden import (
    build_root_selected_tie_cell_burden_rows,
)
from benchmarks.diagnostics.calibration.root_tie_rank_calibration_feasibility import (
    build_root_tie_rank_calibration_feasibility_rows,
    summarize_root_tie_rank_calibration_feasibility,
    summarize_root_tie_rank_calibration_strata,
)
from benchmarks.diagnostics.calibration.root_tie_rank_selected_null_simulation_pilot import (
    DEFAULT_OBSERVED_MIXED_ROWS,
    DEFAULT_RELATIVE_SE_TARGET,
    DEFAULT_TARGET_ALPHA,
    binary_null_probability_for_case,
)
from benchmarks.diagnostics.calibration.root_tie_rank_selected_null_simulation_pilot import (
    build_target_support_rows as build_selected_null_target_support_rows,
)
from benchmarks.shared.cases import get_test_cases_by_suite

SCHEMA_VERSION = "root_tie_rank_null_proposal_frontier/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_null_proposal_frontier_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root_tie_rank_null_proposal_frontier"
)

IID_MARGINAL_BERNOULLI = "iid_marginal_bernoulli"
COLUMN_BETA_BERNOULLI = "column_beta_bernoulli"
TWO_BLOCK_TILT_PROPOSAL = "two_block_tilt_proposal"
SPARSE_BLOCK_SPIKE_PROPOSAL = "sparse_block_spike_proposal"
COUPLED_EDGE_SPECTRAL_PROPOSAL = "coupled_edge_spectral_proposal"
IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL = "importance_two_block_external_null"
IMPORTANCE_COUPLED_EXTERNAL_NULL = "importance_coupled_external_null"
DEFAULT_PROPOSAL_FAMILIES = (
    IID_MARGINAL_BERNOULLI,
    COLUMN_BETA_BERNOULLI,
    TWO_BLOCK_TILT_PROPOSAL,
    SPARSE_BLOCK_SPIKE_PROPOSAL,
    COUPLED_EDGE_SPECTRAL_PROPOSAL,
)
SUPPORTED_PROPOSAL_FAMILIES = (
    *DEFAULT_PROPOSAL_FAMILIES,
    IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL,
    IMPORTANCE_COUPLED_EXTERNAL_NULL,
)
IMPORTANCE_EXTERNAL_NULL_FAMILIES = {
    IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL,
    IMPORTANCE_COUPLED_EXTERNAL_NULL,
}
CALIBRATION_SUPPORT_ROLES = {
    "selected_null",
    "selected_null_candidate_support",
    "external_selected_null",
    "external_null_support",
    "calibration_null",
}

DEFAULT_REPLICATES_PER_CASE = 1
DEFAULT_SEED_OFFSET = 1_120_000
DEFAULT_BETA_CONCENTRATION = 8.0
DEFAULT_TWO_BLOCK_DELTA = 0.25
DEFAULT_SPIKE_FEATURE_FRACTION = 0.10
DEFAULT_SPIKE_DELTA = 0.45

ROOT_ROWS_OUTPUT = "root_tie_rank_null_proposal_root_rows.csv"
MERGE_MARGINS_OUTPUT = "root_tie_rank_null_proposal_merge_margins.csv"
TIE_ROWS_OUTPUT = "root_tie_rank_null_proposal_tie_rows.csv"
MIXED_ROWS_OUTPUT = "root_tie_rank_null_proposal_mixed_rows.csv"
COMBINED_FEASIBILITY_ROWS_OUTPUT = (
    "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)
COMBINED_FEASIBILITY_STRATA_OUTPUT = (
    "root_tie_rank_null_proposal_combined_feasibility_strata.csv"
)
COMBINED_FEASIBILITY_SUMMARY_OUTPUT = (
    "root_tie_rank_null_proposal_combined_feasibility_summary.csv"
)
TARGET_SUPPORT_OUTPUT = "root_tie_rank_null_proposal_target_support.csv"
TARGET_FRONTIER_OUTPUT = "root_tie_rank_null_proposal_target_frontier.csv"
PROPOSAL_SUMMARY_OUTPUT = "root_tie_rank_null_proposal_summary.csv"
FAILURES_OUTPUT = "root_tie_rank_null_proposal_failures.csv"
MANIFEST_OUTPUT = "manifest.json"

METADATA_COLUMNS = (
    "case_id",
    "base_case_id",
    "data_role",
    "calibration_role",
    "replicate",
    "proposal_family",
    "proposal_calibration_status",
    "null_seed",
    "null_feature_probability",
    "proposal_beta_concentration",
    "proposal_two_block_delta",
    "proposal_spike_feature_fraction",
    "proposal_spike_delta",
    "proposal_spike_feature_count",
    "generated_feature_probability_mean",
    "generated_feature_probability_min",
    "generated_feature_probability_max",
    "generated_matrix_density",
    "target_null_log_probability",
    "proposal_log_probability",
    "importance_log_weight",
    "importance_law_status",
    "root_active_feature_count",
    "root_full_eigenvalue_count",
    "root_full_component_eigenvalues_json",
    "root_projected_eigenvalues_json",
    "root_mp_upper_bound",
    "root_raw_mp_signal_count",
    "root_mp_threshold_rows",
)

FAILURE_COLUMNS = (
    "schema_version",
    "study_role",
    "base_case_id",
    "proposal_family",
    "replicate",
    "simulation_case_id",
    "seed",
    "failure_type",
    "failure_message",
)

TARGET_FRONTIER_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_base_case_id",
    "root_conditioning_stratum_key",
    "proposal_family",
    "target_root_sibling_selected_ratio",
    "generated_row_count",
    "generated_calibration_support_count",
    "generated_exceedance_count",
    "generated_calibration_exceedance_count",
    "max_generated_root_sibling_selected_ratio",
    "median_generated_root_sibling_selected_ratio",
    "frontier_hit_status",
)

PROPOSAL_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "proposal_role",
    "generated_root_count",
    "generated_stratum_count",
    "generated_calibration_support_count",
    "observed_target_case_count",
    "target_case_hit_count",
    "target_case_calibration_support_hit_count",
    "target_case_exceedance_count",
    "target_case_calibration_exceedance_count",
    "median_root_sibling_selected_ratio",
    "max_root_sibling_selected_ratio",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankNullProposalFrontierConfig:
    """Configuration for the root tie-rank proposal frontier."""

    output_dir: Path
    observed_mixed_region_rows_path: Path = DEFAULT_OBSERVED_MIXED_ROWS
    topology_frontier_rows_path: Path | None = None
    suite: str = "full"
    case_names: tuple[str, ...] | None = None
    proposal_families: tuple[str, ...] = DEFAULT_PROPOSAL_FAMILIES
    replicates_per_case: int = DEFAULT_REPLICATES_PER_CASE
    seed_offset: int = DEFAULT_SEED_OFFSET
    target_alpha: float = DEFAULT_TARGET_ALPHA
    relative_se_target: float = DEFAULT_RELATIVE_SE_TARGET
    beta_concentration: float = DEFAULT_BETA_CONCENTRATION
    two_block_delta: float = DEFAULT_TWO_BLOCK_DELTA
    spike_feature_fraction: float = DEFAULT_SPIKE_FEATURE_FRACTION
    spike_delta: float = DEFAULT_SPIKE_DELTA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-mixed-region-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_MIXED_ROWS,
    )
    parser.add_argument(
        "--topology-frontier-rows-path",
        type=Path,
        default=None,
        help=(
            "Optional selected-neighborhood topology-frontier rows to join "
            "when rebuilding generated proposal mixed-law rows."
        ),
    )
    parser.add_argument("--suite", default="full")
    parser.add_argument(
        "--case-names",
        default=None,
        help=(
            "Optional comma-separated base case names. Defaults to case_id values "
            "from the observed mixed-law rows."
        ),
    )
    parser.add_argument(
        "--proposal-families",
        default=",".join(DEFAULT_PROPOSAL_FAMILIES),
        help=f"Comma-separated proposal families from {DEFAULT_PROPOSAL_FAMILIES!r}.",
    )
    parser.add_argument("--replicates-per-case", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument("--target-alpha", type=float, default=DEFAULT_TARGET_ALPHA)
    parser.add_argument(
        "--relative-se-target",
        type=float,
        default=DEFAULT_RELATIVE_SE_TARGET,
    )
    parser.add_argument(
        "--beta-concentration",
        type=float,
        default=DEFAULT_BETA_CONCENTRATION,
        help="Column probability concentration for column_beta_bernoulli.",
    )
    parser.add_argument(
        "--two-block-delta",
        type=float,
        default=DEFAULT_TWO_BLOCK_DELTA,
        help="Probability tilt for two_block_tilt_proposal.",
    )
    parser.add_argument(
        "--spike-feature-fraction",
        type=float,
        default=DEFAULT_SPIKE_FEATURE_FRACTION,
        help="Feature fraction tilted in sparse_block_spike_proposal.",
    )
    parser.add_argument(
        "--spike-delta",
        type=float,
        default=DEFAULT_SPIKE_DELTA,
        help="Probability tilt on active features for sparse_block_spike_proposal.",
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankNullProposalFrontierConfig):
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


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing}.")
    return [by_name[name].copy() for name in case_names]


def _validate_proposal_families(families: Sequence[str]) -> tuple[str, ...]:
    valid = set(SUPPORTED_PROPOSAL_FAMILIES)
    parsed = tuple(str(family) for family in families)
    unknown = sorted(set(parsed) - valid)
    if unknown:
        raise ValueError(f"Unknown proposal families: {unknown!r}.")
    if not parsed:
        raise ValueError("At least one proposal family is required.")
    return parsed


def _proposal_role(family: str) -> str:
    if family == IID_MARGINAL_BERNOULLI:
        return "calibration_candidate_support"
    if family in IMPORTANCE_EXTERNAL_NULL_FAMILIES:
        return "importance_weighted_external_null_support"
    return "diagnostic_proposal_not_calibration"


def _data_role_for_family(family: str) -> str:
    if family == IID_MARGINAL_BERNOULLI:
        return "selected_null"
    if family in IMPORTANCE_EXTERNAL_NULL_FAMILIES:
        return "external_selected_null"
    return "diagnostic_proposal"


def _calibration_role_for_family(family: str) -> str:
    if family == IID_MARGINAL_BERNOULLI:
        return "selected_null_candidate_support"
    if family in IMPORTANCE_EXTERNAL_NULL_FAMILIES:
        return "external_null_support"
    return "diagnostic_proposal_not_null_support"


def _safe_family_id(family: str) -> str:
    return family.replace("-", "_").replace("/", "_")


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


def _bernoulli_log_probability(
    matrix: np.ndarray,
    probabilities: np.ndarray | float,
) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    values = np.asarray(matrix, dtype=float)
    return float(np.sum(values * np.log(clipped) + (1.0 - values) * np.log1p(-clipped)))


def _column_beta_probabilities(
    *,
    mean_probability: float,
    n_features: int,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if concentration <= 0.0:
        raise ValueError("beta_concentration must be positive.")
    mean = float(np.clip(mean_probability, 1e-3, 1.0 - 1e-3))
    alpha = max(mean * float(concentration), 1e-6)
    beta = max((1.0 - mean) * float(concentration), 1e-6)
    return np.clip(rng.beta(alpha, beta, size=int(n_features)), 1e-3, 1.0 - 1e-3)


def generate_binary_proposal_matrix(
    *,
    base_case: dict[str, object],
    proposal_family: str,
    seed: int,
    beta_concentration: float = DEFAULT_BETA_CONCENTRATION,
    two_block_delta: float = DEFAULT_TWO_BLOCK_DELTA,
    spike_feature_fraction: float = DEFAULT_SPIKE_FEATURE_FRACTION,
    spike_delta: float = DEFAULT_SPIKE_DELTA,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate one binary matrix and metadata for a proposal family."""
    family = str(proposal_family)
    _validate_proposal_families((family,))
    n_samples = int(base_case["n_samples"])
    n_features = int(base_case["n_features"])
    base_probability = binary_null_probability_for_case(base_case)
    rng = np.random.default_rng(seed)
    active_count = 0
    if family == IID_MARGINAL_BERNOULLI:
        feature_probabilities = np.full(n_features, base_probability, dtype=float)
        probability_matrix = np.broadcast_to(
            feature_probabilities[None, :],
            (n_samples, n_features),
        )
        matrix = rng.binomial(1, probability_matrix)
    elif family == COLUMN_BETA_BERNOULLI:
        feature_probabilities = _column_beta_probabilities(
            mean_probability=base_probability,
            n_features=n_features,
            concentration=float(beta_concentration),
            rng=rng,
        )
        probability_matrix = np.broadcast_to(
            feature_probabilities[None, :],
            (n_samples, n_features),
        )
        matrix = rng.binomial(1, probability_matrix)
    elif family in {TWO_BLOCK_TILT_PROPOSAL, IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL}:
        if float(two_block_delta) < 0.0:
            raise ValueError("two_block_delta must be nonnegative.")
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
        block_labels = np.zeros(n_samples, dtype=int)
        block_labels[n_samples // 2 :] = 1
        rng.shuffle(block_labels)
        probability_matrix = np.where(
            block_labels[:, None] == 0,
            left_probabilities[None, :],
            right_probabilities[None, :],
        )
        feature_probabilities = np.mean(probability_matrix, axis=0)
        matrix = rng.binomial(1, probability_matrix)
    elif family == SPARSE_BLOCK_SPIKE_PROPOSAL:
        if not (0.0 < float(spike_feature_fraction) <= 1.0):
            raise ValueError("spike_feature_fraction must be in (0, 1].")
        if float(spike_delta) < 0.0:
            raise ValueError("spike_delta must be nonnegative.")
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
        feature_probabilities = np.mean(probability_matrix, axis=0)
        matrix = rng.binomial(1, probability_matrix)
    else:
        if float(two_block_delta) < 0.0:
            raise ValueError("two_block_delta must be nonnegative.")
        if not (0.0 < float(spike_feature_fraction) <= 1.0):
            raise ValueError("spike_feature_fraction must be in (0, 1].")
        if float(spike_delta) < 0.0:
            raise ValueError("spike_delta must be nonnegative.")
        active_count = max(
            1,
            int(math.ceil(float(spike_feature_fraction) * float(n_features))),
        )
        active_features = rng.choice(n_features, size=active_count, replace=False)
        dense_signs = rng.choice(np.array([-1.0, 1.0]), size=n_features)
        sparse_signs = np.zeros(n_features, dtype=float)
        sparse_signs[active_features] = 1.0
        combined_shift = (
            float(two_block_delta) * dense_signs
            + float(spike_delta) * sparse_signs
        )
        left_probabilities = np.clip(
            base_probability + combined_shift,
            1e-3,
            1.0 - 1e-3,
        )
        right_probabilities = np.clip(
            base_probability - combined_shift,
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
        feature_probabilities = np.mean(probability_matrix, axis=0)
        matrix = rng.binomial(1, probability_matrix)
    matrix = _ensure_nonempty_feature_columns(matrix.astype(int), rng=rng)
    target_probability_matrix = np.full_like(
        np.asarray(matrix, dtype=float),
        base_probability,
        dtype=float,
    )
    target_null_log_probability = _bernoulli_log_probability(
        matrix,
        target_probability_matrix,
    )
    proposal_log_probability = _bernoulli_log_probability(matrix, probability_matrix)
    importance_log_weight = target_null_log_probability - proposal_log_probability
    metadata = {
        "null_feature_probability": float(base_probability),
        "proposal_beta_concentration": float(beta_concentration)
        if family == COLUMN_BETA_BERNOULLI
        else math.nan,
        "proposal_two_block_delta": float(two_block_delta)
        if family
        in {
            TWO_BLOCK_TILT_PROPOSAL,
            COUPLED_EDGE_SPECTRAL_PROPOSAL,
            IMPORTANCE_TWO_BLOCK_EXTERNAL_NULL,
            IMPORTANCE_COUPLED_EXTERNAL_NULL,
        }
        else math.nan,
        "proposal_spike_feature_fraction": float(spike_feature_fraction)
        if family
        in {
            SPARSE_BLOCK_SPIKE_PROPOSAL,
            COUPLED_EDGE_SPECTRAL_PROPOSAL,
            IMPORTANCE_COUPLED_EXTERNAL_NULL,
        }
        else math.nan,
        "proposal_spike_delta": float(spike_delta)
        if family
        in {
            SPARSE_BLOCK_SPIKE_PROPOSAL,
            COUPLED_EDGE_SPECTRAL_PROPOSAL,
            IMPORTANCE_COUPLED_EXTERNAL_NULL,
        }
        else math.nan,
        "proposal_spike_feature_count": int(active_count)
        if family
        in {
            SPARSE_BLOCK_SPIKE_PROPOSAL,
            COUPLED_EDGE_SPECTRAL_PROPOSAL,
            IMPORTANCE_COUPLED_EXTERNAL_NULL,
        }
        else 0,
        "generated_feature_probability_mean": float(np.mean(feature_probabilities)),
        "generated_feature_probability_min": float(np.min(feature_probabilities)),
        "generated_feature_probability_max": float(np.max(feature_probabilities)),
        "generated_matrix_density": float(np.mean(matrix)),
        "target_null_log_probability": target_null_log_probability,
        "proposal_log_probability": proposal_log_probability,
        "importance_log_weight": importance_log_weight
        if family in IMPORTANCE_EXTERNAL_NULL_FAMILIES
        else math.nan,
        "importance_law_status": (
            "target_iid_bernoulli_over_tilted_proposal"
            if family in IMPORTANCE_EXTERNAL_NULL_FAMILIES
            else "not_importance_weighted"
        ),
    }
    return matrix, metadata


def _write_binary_proposal_case(
    *,
    base_case: dict[str, object],
    simulation_case_id: str,
    proposal_family: str,
    seed: int,
    matrix_dir: Path,
    beta_concentration: float,
    two_block_delta: float,
    spike_feature_fraction: float,
    spike_delta: float,
) -> tuple[dict[str, object], dict[str, object]]:
    matrix, proposal_metadata = generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=proposal_family,
        seed=seed,
        beta_concentration=beta_concentration,
        two_block_delta=two_block_delta,
        spike_feature_fraction=spike_feature_fraction,
        spike_delta=spike_delta,
    )
    sample_names = [f"L{i + 1}" for i in range(matrix.shape[0])]
    feature_names = [f"F{j}" for j in range(matrix.shape[1])]
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{simulation_case_id}.csv"
    pd.DataFrame(matrix, index=sample_names, columns=feature_names).to_csv(matrix_path)
    case_metadata = {
        "name": simulation_case_id,
        "generator": "preloaded",
        "file_path": str(matrix_path),
        "sep": ",",
        "n_clusters": 1,
        "category": "root_tie_rank_null_proposal_frontier",
        "baseline_case_name": str(base_case["name"]),
        "null_generation": proposal_family,
        "proposal_family": proposal_family,
        "proposal_calibration_status": _proposal_role(proposal_family),
        "seed": int(seed),
    }
    case_metadata.update(proposal_metadata)
    return case_metadata, proposal_metadata


def collect_root_tie_rank_proposal_rows(
    *,
    base_cases: Sequence[dict[str, object]],
    output_dir: Path,
    proposal_families: Sequence[str] = DEFAULT_PROPOSAL_FAMILIES,
    replicates_per_case: int = DEFAULT_REPLICATES_PER_CASE,
    seed_offset: int = DEFAULT_SEED_OFFSET,
    beta_concentration: float = DEFAULT_BETA_CONCENTRATION,
    two_block_delta: float = DEFAULT_TWO_BLOCK_DELTA,
    spike_feature_fraction: float = DEFAULT_SPIKE_FEATURE_FRACTION,
    spike_delta: float = DEFAULT_SPIKE_DELTA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate proposal roots and return root rows, merge margins, and failures."""
    families = _validate_proposal_families(proposal_families)
    if int(replicates_per_case) <= 0:
        raise ValueError("replicates_per_case must be positive.")
    root_records: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    matrix_dir = Path(output_dir) / "generated_proposal_matrices"
    for family_index, family in enumerate(families):
        for base_index, base_case in enumerate(base_cases):
            base_case_id = str(base_case["name"])
            for replicate in range(int(replicates_per_case)):
                seed = (
                    int(seed_offset)
                    + family_index * 10_000_000
                    + base_index * 100_000
                    + replicate
                )
                simulation_case_id = (
                    f"{base_case_id}__{_safe_family_id(family)}_r{replicate:04d}"
                )
                try:
                    proposal_case, proposal_metadata = _write_binary_proposal_case(
                        base_case=base_case,
                        simulation_case_id=simulation_case_id,
                        proposal_family=family,
                        seed=seed,
                        matrix_dir=matrix_dir,
                        beta_concentration=float(beta_concentration),
                        two_block_delta=float(two_block_delta),
                        spike_feature_fraction=float(spike_feature_fraction),
                        spike_delta=float(spike_delta),
                    )
                    row, margins = collect_observed_root_selected_region_row(
                        proposal_case
                    )
                    row.update(
                        {
                            "base_case_id": base_case_id,
                            "data_role": _data_role_for_family(family),
                            "calibration_role": _calibration_role_for_family(family),
                            "replicate": int(replicate),
                            "proposal_family": family,
                            "proposal_calibration_status": _proposal_role(family),
                            "null_seed": int(seed),
                        }
                    )
                    row.update(proposal_metadata)
                    root_records.append(row)
                    case_margins = margins.copy()
                    case_margins.insert(0, "case_id", row["case_id"])
                    case_margins.insert(1, "schema_version", ROOT_MARGIN_SCHEMA_VERSION)
                    case_margins.insert(2, "study_role", STUDY_ROLE)
                    case_margins.insert(3, "base_case_id", base_case_id)
                    case_margins.insert(4, "proposal_family", family)
                    margin_tables.append(case_margins)
                except Exception as exc:  # pragma: no cover - integration safeguard
                    failures.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "study_role": STUDY_ROLE,
                            "base_case_id": base_case_id,
                            "proposal_family": family,
                            "replicate": int(replicate),
                            "simulation_case_id": simulation_case_id,
                            "seed": int(seed),
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    )
    root_rows = pd.DataFrame.from_records(root_records)
    merge_margins = (
        pd.concat(margin_tables, ignore_index=True)
        if margin_tables
        else pd.DataFrame()
    )
    failure_rows = pd.DataFrame.from_records(failures, columns=FAILURE_COLUMNS)
    return root_rows, merge_margins, failure_rows


def _metadata_by_case(rows: pd.DataFrame) -> pd.DataFrame:
    present = [column for column in METADATA_COLUMNS if column in rows.columns]
    if not present:
        return pd.DataFrame()
    return rows[present].drop_duplicates("case_id").copy()


def _attach_metadata(rows: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or metadata.empty:
        return rows
    metadata_columns = [
        column
        for column in metadata.columns
        if column == "case_id" or column not in rows.columns
    ]
    if metadata_columns == ["case_id"]:
        return rows.copy()
    return rows.merge(metadata[metadata_columns], on="case_id", how="left")


def build_proposal_mixed_rows(
    *,
    root_rows: pd.DataFrame,
    merge_margins: pd.DataFrame,
    topology_frontier_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tie-burden and mixed-law rows for generated proposal roots."""
    if root_rows.empty or merge_margins.empty:
        return pd.DataFrame(), pd.DataFrame()
    tie_rows = build_root_selected_tie_cell_burden_rows(
        root_summary=root_rows,
        merge_margins=merge_margins,
    )
    mixed_rows = build_root_selected_mixed_region_law_rows(
        root_summary=root_rows,
        tie_cell_burden_rows=tie_rows,
        topology_frontier_rows=topology_frontier_rows,
    )
    metadata = _metadata_by_case(root_rows)
    return _attach_metadata(tie_rows, metadata), _attach_metadata(mixed_rows, metadata)


def _observed_target_rows(observed_mixed_rows: pd.DataFrame) -> pd.DataFrame:
    observed = observed_mixed_rows.copy()
    observed["data_role"] = "observed_target"
    observed["calibration_role"] = "observed_target_not_null_support"
    observed["proposal_family"] = "observed_target"
    observed["proposal_calibration_status"] = "observed_target_not_null_support"
    if "base_case_id" not in observed.columns:
        observed["base_case_id"] = observed["case_id"].astype(str)
    if "replicate" not in observed.columns:
        observed["replicate"] = -1
    return observed


def _annotated_feasibility_rows(
    *,
    feasibility_rows: pd.DataFrame,
    combined_mixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in METADATA_COLUMNS
        if column in combined_mixed_rows.columns and column not in feasibility_rows.columns
    ]
    if not metadata_columns:
        return feasibility_rows.copy()
    metadata = combined_mixed_rows[["case_id", *metadata_columns]].drop_duplicates(
        "case_id"
    )
    return feasibility_rows.merge(metadata, on="case_id", how="left")


def _is_calibration_support(role: object) -> bool:
    return str(role) in CALIBRATION_SUPPORT_ROLES


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def build_target_frontier_rows(
    *,
    observed_case_ids: Sequence[str],
    annotated_feasibility_rows: pd.DataFrame,
    proposal_families: Sequence[str],
) -> pd.DataFrame:
    """Report whether proposal families hit each observed target stratum."""
    if annotated_feasibility_rows.empty:
        return pd.DataFrame(columns=TARGET_FRONTIER_COLUMNS)
    _require_columns(
        annotated_feasibility_rows,
        {
            "case_id",
            "calibration_role",
            "root_conditioning_stratum_key",
            "root_sibling_selected_ratio",
        },
        "annotated feasibility rows",
    )
    observed_set = {str(case_id) for case_id in observed_case_ids}
    families = _validate_proposal_families(proposal_families)
    targets = annotated_feasibility_rows[
        annotated_feasibility_rows["case_id"].astype(str).isin(observed_set)
    ].copy()
    generated = annotated_feasibility_rows[
        ~annotated_feasibility_rows["case_id"].astype(str).isin(observed_set)
    ].copy()
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        stratum = str(target["root_conditioning_stratum_key"])
        target_ratio = float(target["root_sibling_selected_ratio"])
        target_base_case_id = (
            str(target["base_case_id"])
            if "base_case_id" in target and pd.notna(target["base_case_id"])
            else str(target["case_id"])
        )
        for family in families:
            same = generated[
                generated["root_conditioning_stratum_key"].astype(str).eq(stratum)
                & generated["proposal_family"].astype(str).eq(family)
            ]
            ratios = _finite_numeric(same.get("root_sibling_selected_ratio", pd.Series()))
            calibration_mask = same["calibration_role"].map(_is_calibration_support)
            calibration_rows = same[calibration_mask]
            calibration_ratios = _finite_numeric(
                calibration_rows.get("root_sibling_selected_ratio", pd.Series())
            )
            generated_count = int(same.shape[0])
            calibration_count = int(calibration_rows.shape[0])
            exceedance_count = int((ratios >= target_ratio).sum())
            calibration_exceedance_count = int(
                (calibration_ratios >= target_ratio).sum()
            )
            if calibration_count > 0:
                status = "target_stratum_hit_by_calibration_support"
            elif generated_count > 0:
                status = "target_stratum_hit_by_diagnostic_proposal_only"
            else:
                status = "no_target_stratum_hit"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "target_case_id": str(target["case_id"]),
                    "target_base_case_id": target_base_case_id,
                    "root_conditioning_stratum_key": stratum,
                    "proposal_family": family,
                    "target_root_sibling_selected_ratio": target_ratio,
                    "generated_row_count": generated_count,
                    "generated_calibration_support_count": calibration_count,
                    "generated_exceedance_count": exceedance_count,
                    "generated_calibration_exceedance_count": (
                        calibration_exceedance_count
                    ),
                    "max_generated_root_sibling_selected_ratio": float(ratios.max())
                    if not ratios.empty
                    else math.nan,
                    "median_generated_root_sibling_selected_ratio": float(
                        ratios.median()
                    )
                    if not ratios.empty
                    else math.nan,
                    "frontier_hit_status": status,
                }
            )
    return pd.DataFrame.from_records(records, columns=TARGET_FRONTIER_COLUMNS)


def summarize_proposal_frontier(
    *,
    annotated_feasibility_rows: pd.DataFrame,
    target_frontier_rows: pd.DataFrame,
    observed_case_ids: Sequence[str],
    proposal_families: Sequence[str],
) -> pd.DataFrame:
    """Summarize generated stratum coverage and target hits by family."""
    observed_set = {str(case_id) for case_id in observed_case_ids}
    families = _validate_proposal_families(proposal_families)
    generated = (
        annotated_feasibility_rows[
            ~annotated_feasibility_rows["case_id"].astype(str).isin(observed_set)
        ].copy()
        if not annotated_feasibility_rows.empty
        else pd.DataFrame()
    )
    records: list[dict[str, object]] = []
    for family in families:
        family_rows = (
            generated[generated["proposal_family"].astype(str).eq(family)]
            if not generated.empty and "proposal_family" in generated
            else pd.DataFrame()
        )
        family_frontier = (
            target_frontier_rows[
                target_frontier_rows["proposal_family"].astype(str).eq(family)
            ]
            if not target_frontier_rows.empty
            else pd.DataFrame()
        )
        ratios = _finite_numeric(
            family_rows.get("root_sibling_selected_ratio", pd.Series())
        )
        calibration_count = (
            int(family_rows["calibration_role"].map(_is_calibration_support).sum())
            if not family_rows.empty and "calibration_role" in family_rows
            else 0
        )
        target_hits = (
            int((family_frontier["generated_row_count"] > 0).sum())
            if not family_frontier.empty
            else 0
        )
        calibration_hits = (
            int((family_frontier["generated_calibration_support_count"] > 0).sum())
            if not family_frontier.empty
            else 0
        )
        exceedance_hits = (
            int((family_frontier["generated_exceedance_count"] > 0).sum())
            if not family_frontier.empty
            else 0
        )
        calibration_exceedance_hits = (
            int((family_frontier["generated_calibration_exceedance_count"] > 0).sum())
            if not family_frontier.empty
            else 0
        )
        if calibration_hits > 0:
            status = "calibration_candidate_hits_observed_target_strata"
        elif target_hits > 0:
            status = "diagnostic_proposal_hits_observed_target_strata_not_calibration"
        elif not family_rows.empty:
            status = "generated_no_observed_target_stratum_hits"
        else:
            status = "no_generated_rows"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": family,
                "proposal_role": _proposal_role(family),
                "generated_root_count": int(family_rows.shape[0]),
                "generated_stratum_count": int(
                    family_rows["root_conditioning_stratum_key"].nunique()
                )
                if "root_conditioning_stratum_key" in family_rows
                else 0,
                "generated_calibration_support_count": calibration_count,
                "observed_target_case_count": int(len(observed_set)),
                "target_case_hit_count": target_hits,
                "target_case_calibration_support_hit_count": calibration_hits,
                "target_case_exceedance_count": exceedance_hits,
                "target_case_calibration_exceedance_count": (
                    calibration_exceedance_hits
                ),
                "median_root_sibling_selected_ratio": float(ratios.median())
                if not ratios.empty
                else math.nan,
                "max_root_sibling_selected_ratio": float(ratios.max())
                if not ratios.empty
                else math.nan,
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=PROPOSAL_SUMMARY_COLUMNS)


def evaluate_root_tie_rank_null_proposal_frontier(
    config: RootTieRankNullProposalFrontierConfig,
) -> dict[str, pd.DataFrame]:
    """Run proposal frontier evaluation and return output tables."""
    families = _validate_proposal_families(config.proposal_families)
    observed_mixed = pd.read_csv(config.observed_mixed_region_rows_path)
    topology_frontier_rows = (
        pd.read_csv(config.topology_frontier_rows_path, low_memory=False)
        if config.topology_frontier_rows_path is not None
        and Path(config.topology_frontier_rows_path).exists()
        else None
    )
    _require_columns(observed_mixed, {"case_id"}, "observed mixed-law rows")
    observed_case_ids = tuple(observed_mixed["case_id"].astype(str).tolist())
    case_names = tuple(config.case_names) if config.case_names else observed_case_ids
    base_cases = _select_cases(suite=config.suite, case_names=case_names)
    root_rows, merge_margins, failures = collect_root_tie_rank_proposal_rows(
        base_cases=base_cases,
        output_dir=config.output_dir,
        proposal_families=families,
        replicates_per_case=int(config.replicates_per_case),
        seed_offset=int(config.seed_offset),
        beta_concentration=float(config.beta_concentration),
        two_block_delta=float(config.two_block_delta),
        spike_feature_fraction=float(config.spike_feature_fraction),
        spike_delta=float(config.spike_delta),
    )
    tie_rows, proposal_mixed_rows = build_proposal_mixed_rows(
        root_rows=root_rows,
        merge_margins=merge_margins,
        topology_frontier_rows=topology_frontier_rows,
    )
    observed_targets = _observed_target_rows(observed_mixed)
    combined_mixed = pd.concat(
        [observed_targets, proposal_mixed_rows],
        ignore_index=True,
        sort=False,
    )
    feasibility_rows = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=combined_mixed,
        target_alpha=float(config.target_alpha),
        relative_se_target=float(config.relative_se_target),
    )
    annotated_feasibility = _annotated_feasibility_rows(
        feasibility_rows=feasibility_rows,
        combined_mixed_rows=combined_mixed,
    )
    feasibility_strata = summarize_root_tie_rank_calibration_strata(feasibility_rows)
    feasibility_summary = summarize_root_tie_rank_calibration_feasibility(
        feasibility_rows,
        feasibility_strata,
    )
    target_support = build_selected_null_target_support_rows(
        observed_case_ids=observed_case_ids,
        combined_feasibility_rows=feasibility_rows,
    )
    target_support["schema_version"] = SCHEMA_VERSION
    target_support["study_role"] = STUDY_ROLE
    target_frontier = build_target_frontier_rows(
        observed_case_ids=observed_case_ids,
        annotated_feasibility_rows=annotated_feasibility,
        proposal_families=families,
    )
    proposal_summary = summarize_proposal_frontier(
        annotated_feasibility_rows=annotated_feasibility,
        target_frontier_rows=target_frontier,
        observed_case_ids=observed_case_ids,
        proposal_families=families,
    )
    proposal_mixed_with_strata = annotated_feasibility[
        ~annotated_feasibility["case_id"].astype(str).isin(
            {str(case_id) for case_id in observed_case_ids}
        )
    ][["case_id", "root_conditioning_stratum_key"]].merge(
        proposal_mixed_rows,
        on="case_id",
        how="right",
    )
    return {
        "root_rows": root_rows,
        "merge_margins": merge_margins,
        "tie_rows": tie_rows,
        "mixed_rows": proposal_mixed_with_strata,
        "combined_feasibility_rows": annotated_feasibility,
        "combined_feasibility_strata": feasibility_strata,
        "combined_feasibility_summary": feasibility_summary,
        "target_support": target_support,
        "target_frontier": target_frontier,
        "proposal_summary": proposal_summary,
        "failures": failures,
    }


def run_root_tie_rank_null_proposal_frontier(
    config: RootTieRankNullProposalFrontierConfig,
) -> dict[str, Path]:
    """Run the proposal frontier and write outputs."""
    start = perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_tie_rank_null_proposal_frontier(config)
    paths = {
        "root_rows": output_dir / ROOT_ROWS_OUTPUT,
        "merge_margins": output_dir / MERGE_MARGINS_OUTPUT,
        "tie_rows": output_dir / TIE_ROWS_OUTPUT,
        "mixed_rows": output_dir / MIXED_ROWS_OUTPUT,
        "combined_feasibility_rows": output_dir / COMBINED_FEASIBILITY_ROWS_OUTPUT,
        "combined_feasibility_strata": output_dir / COMBINED_FEASIBILITY_STRATA_OUTPUT,
        "combined_feasibility_summary": output_dir / COMBINED_FEASIBILITY_SUMMARY_OUTPUT,
        "target_support": output_dir / TARGET_SUPPORT_OUTPUT,
        "target_frontier": output_dir / TARGET_FRONTIER_OUTPUT,
        "proposal_summary": output_dir / PROPOSAL_SUMMARY_OUTPUT,
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
    outputs = run_root_tie_rank_null_proposal_frontier(
        RootTieRankNullProposalFrontierConfig(
            output_dir=args.output_dir,
            observed_mixed_region_rows_path=args.observed_mixed_region_rows_path,
            topology_frontier_rows_path=args.topology_frontier_rows_path,
            suite=str(args.suite),
            case_names=_parse_csv_list(args.case_names),
            proposal_families=_parse_csv_list(args.proposal_families)
            or DEFAULT_PROPOSAL_FAMILIES,
            replicates_per_case=int(args.replicates_per_case),
            seed_offset=int(args.seed_offset),
            target_alpha=float(args.target_alpha),
            relative_se_target=float(args.relative_se_target),
            beta_concentration=float(args.beta_concentration),
            two_block_delta=float(args.two_block_delta),
            spike_feature_fraction=float(args.spike_feature_fraction),
            spike_delta=float(args.spike_delta),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
