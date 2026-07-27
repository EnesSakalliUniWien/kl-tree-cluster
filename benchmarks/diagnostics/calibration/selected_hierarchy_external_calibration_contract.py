"""External selected-hierarchy calibration admissibility diagnostic.

This diagnostic defines a strict contract for when selected-hierarchy null
simulations would be admissible as an external calibration object. It also
checks whether a scalar mean inflation model is descriptively adequate, or
whether the selected-ratio tail law must be modeled.

The script is diagnostic-only. It does not add a production calibration path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.selected_hierarchy_null_audit import (
    DEFAULT_CASE_NAMES,
    _parse_csv_list,
    _selected_cases,
)
from benchmarks.diagnostics.calibration.selected_hierarchy_stratification_diagnostic import (
    _diagnose_case,
)
from benchmarks.shared.util.time import format_timestamp_utc

CONDITIONING_SCOPE = "same_data_selected_hierarchy_edge_path_open"
STRATUM_ROLE = "external_selected_hierarchy_contract_diagnostic_only"
GROUP_COLUMNS = (
    "case_id",
    "feature_family",
    "n_samples",
    "feature_dimension",
    "projection_dimension",
    "parent_size_bin",
)


@dataclass(frozen=True)
class ExternalCalibrationContract:
    """Thresholds for production-admissible external selected-hierarchy evidence."""

    sibling_alpha: float = float(DEFAULT_SIBLING_ALPHA)
    tail_resolution_fraction_of_alpha: float = 0.2
    min_matching_simulations: int | None = None
    min_matched_records: int | None = None
    max_relative_c_simulation_se: float = 0.05
    scalar_rejection_absolute_tolerance: float = 0.005
    scalar_tail_quantile_excess_limit: float = 1.25
    min_scalar_uniformity_ks_p_value: float = 0.01

    def resolved_min_matching_simulations(self) -> int:
        """Return the required independent simulation count for tail resolution."""
        if self.min_matching_simulations is not None:
            return int(self.min_matching_simulations)
        return int(
            ceil(
                1.0
                / (self.sibling_alpha * self.tail_resolution_fraction_of_alpha)
            )
            - 1
        )

    def resolved_min_matched_records(self) -> int:
        """Return the required selected-record count for tail-law diagnostics."""
        if self.min_matched_records is not None:
            return int(self.min_matched_records)
        return self.resolved_min_matching_simulations()


def _require_contract(contract: ExternalCalibrationContract) -> None:
    if not 0.0 < contract.sibling_alpha < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    if not 0.0 < contract.tail_resolution_fraction_of_alpha < 1.0:
        raise ValueError("tail_resolution_fraction_of_alpha must lie in (0, 1).")
    if contract.resolved_min_matching_simulations() <= 0:
        raise ValueError("min_matching_simulations must be positive.")
    if contract.resolved_min_matched_records() <= 0:
        raise ValueError("min_matched_records must be positive.")
    if contract.max_relative_c_simulation_se <= 0.0:
        raise ValueError("max_relative_c_simulation_se must be positive.")
    if contract.scalar_rejection_absolute_tolerance < 0.0:
        raise ValueError("scalar_rejection_absolute_tolerance cannot be negative.")
    if contract.scalar_tail_quantile_excess_limit <= 1.0:
        raise ValueError("scalar_tail_quantile_excess_limit must exceed one.")
    if not 0.0 < contract.min_scalar_uniformity_ks_p_value < 1.0:
        raise ValueError("min_scalar_uniformity_ks_p_value must lie in (0, 1).")


def _simulation_relative_se(records: pd.DataFrame, ratios: np.ndarray) -> float:
    simulation_means = records.groupby("replicate_index")[
        "selected_hierarchy_ratio"
    ].mean()
    if simulation_means.shape[0] <= 1:
        return np.nan
    c_hat = float(np.mean(ratios))
    if c_hat == 0.0:
        return np.nan
    se = float(
        np.std(simulation_means.to_numpy(dtype=float), ddof=1)
        / np.sqrt(simulation_means.shape[0])
    )
    return float(se / c_hat)


def _admissibility_failure_reasons(
    *,
    n_matching_simulations: int,
    n_records: int,
    relative_c_se: float,
    contract: ExternalCalibrationContract,
) -> list[str]:
    reasons: list[str] = []
    if n_matching_simulations < contract.resolved_min_matching_simulations():
        reasons.append(
            "matching_simulations_below_tail_resolution_contract"
        )
    if n_records < contract.resolved_min_matched_records():
        reasons.append("matched_records_below_tail_resolution_contract")
    if (
        not np.isfinite(relative_c_se)
        or relative_c_se > contract.max_relative_c_simulation_se
    ):
        reasons.append("relative_c_simulation_se_above_contract")
    return reasons


def _scalar_c_summary(
    records: pd.DataFrame,
    *,
    ratios: np.ndarray,
    c_hat: float,
    contract: ExternalCalibrationContract,
) -> dict[str, object]:
    statistic = records["statistic"].to_numpy(dtype=float)
    reference_scale = records["reference_scale"].to_numpy(dtype=float)
    degrees_of_freedom = records["degrees_of_freedom"].to_numpy(dtype=float)
    if np.any(reference_scale <= 0.0):
        raise ValueError("Scalar-c diagnostic requires positive reference_scale.")
    if np.any(degrees_of_freedom <= 0.0):
        raise ValueError("Scalar-c diagnostic requires positive degrees_of_freedom.")

    scalar_p_values = chi2.sf(
        statistic / (reference_scale * c_hat),
        df=degrees_of_freedom,
    )
    scalar_rejection_rate = float(
        np.mean(scalar_p_values <= contract.sibling_alpha)
    )
    scalar_threshold_ratios = (
        c_hat
        * chi2.isf(contract.sibling_alpha, df=degrees_of_freedom)
        / degrees_of_freedom
    )
    selected_ratio_tail_quantile = float(
        np.quantile(ratios, 1.0 - contract.sibling_alpha)
    )
    scalar_tail_threshold_ratio = float(np.mean(scalar_threshold_ratios))
    if len(scalar_p_values) > 1:
        ks_result = kstest(scalar_p_values, "uniform")
        ks_statistic = float(ks_result.statistic)
        ks_p_value = float(ks_result.pvalue)
    else:
        ks_statistic = np.nan
        ks_p_value = np.nan

    tail_quantile_excess = float(
        selected_ratio_tail_quantile / scalar_tail_threshold_ratio
    )
    scalar_rejection_limit = (
        contract.sibling_alpha + contract.scalar_rejection_absolute_tolerance
    )
    scalar_c_descriptive_status = (
        "descriptive_scalar_c_plausible"
        if (
            scalar_rejection_rate <= scalar_rejection_limit
            and np.isfinite(ks_p_value)
            and ks_p_value >= contract.min_scalar_uniformity_ks_p_value
            and tail_quantile_excess <= contract.scalar_tail_quantile_excess_limit
        )
        else "descriptive_full_tail_law_indicated"
    )
    return {
        "scalar_c_rejection_rate_at_alpha": scalar_rejection_rate,
        "scalar_c_rejection_limit": float(scalar_rejection_limit),
        "scalar_c_uniformity_ks_statistic": ks_statistic,
        "scalar_c_uniformity_ks_p_value": ks_p_value,
        "selected_ratio_tail_quantile_at_alpha": selected_ratio_tail_quantile,
        "scalar_c_tail_threshold_ratio_at_alpha": scalar_tail_threshold_ratio,
        "selected_ratio_tail_quantile_over_scalar_threshold": tail_quantile_excess,
        "scalar_c_descriptive_status": scalar_c_descriptive_status,
    }


def _evaluate_stratum(
    group_values: tuple[object, ...],
    records: pd.DataFrame,
    *,
    contract: ExternalCalibrationContract,
) -> dict[str, object]:
    ratios = records["selected_hierarchy_ratio"].to_numpy(dtype=float)
    c_hat = float(np.mean(ratios))
    n_matching_simulations = int(records["replicate_index"].nunique())
    n_records = int(records.shape[0])
    relative_c_se = _simulation_relative_se(records, ratios)
    failure_reasons = _admissibility_failure_reasons(
        n_matching_simulations=n_matching_simulations,
        n_records=n_records,
        relative_c_se=relative_c_se,
        contract=contract,
    )
    production_admissible = not failure_reasons
    scalar_summary = _scalar_c_summary(
        records,
        ratios=ratios,
        c_hat=c_hat,
        contract=contract,
    )
    if not production_admissible:
        estimator_decision = "undefined_not_production_admissible"
    elif scalar_summary["scalar_c_descriptive_status"] == "descriptive_scalar_c_plausible":
        estimator_decision = "scalar_c_external_model_candidate"
    else:
        estimator_decision = "full_selected_ratio_tail_law_required"

    row = {
        column: value for column, value in zip(GROUP_COLUMNS, group_values)
    }
    row.update(
        {
            "conditioning_scope": CONDITIONING_SCOPE,
            "stratum_role": STRATUM_ROLE,
            "n_records": n_records,
            "n_matching_simulations": n_matching_simulations,
            "matching_simulation_tail_resolution": float(
                1.0 / (n_matching_simulations + 1)
            ),
            "record_tail_resolution": float(1.0 / (n_records + 1)),
            "required_min_matching_simulations": (
                contract.resolved_min_matching_simulations()
            ),
            "required_min_matched_records": contract.resolved_min_matched_records(),
            "selected_hierarchy_c_hat": c_hat,
            "selected_hierarchy_c_hat_relative_simulation_se": relative_c_se,
            "required_max_relative_c_simulation_se": (
                contract.max_relative_c_simulation_se
            ),
            "selected_hierarchy_ratio_median": float(np.quantile(ratios, 0.5)),
            "selected_hierarchy_ratio_q95": float(np.quantile(ratios, 0.95)),
            "selected_hierarchy_ratio_q99": float(np.quantile(ratios, 0.99)),
            "production_external_calibration_admissible": production_admissible,
            "admissibility_failure_reasons": ";".join(failure_reasons),
            "estimator_family_decision": estimator_decision,
        }
    )
    row.update(scalar_summary)
    return row


def evaluate_external_calibration_contract(
    records: pd.DataFrame,
    *,
    contract: ExternalCalibrationContract,
) -> pd.DataFrame:
    """Evaluate admissibility and scalar-vs-tail behavior for selected records."""
    _require_contract(contract)
    missing_columns = set(GROUP_COLUMNS) | {
        "replicate_index",
        "statistic",
        "reference_scale",
        "degrees_of_freedom",
        "selected_hierarchy_ratio",
    }
    missing_columns -= set(records.columns)
    if missing_columns:
        raise ValueError(
            "Selected-hierarchy contract records are missing columns: "
            f"{sorted(missing_columns)}."
        )
    if records.empty:
        raise ValueError("Selected-hierarchy contract diagnostic received no records.")

    rows: list[dict[str, object]] = []
    for group_values, group in records.groupby(list(GROUP_COLUMNS), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        rows.append(
            _evaluate_stratum(
                group_values,
                group,
                contract=contract,
            )
        )
    return pd.DataFrame.from_records(rows).sort_values(list(GROUP_COLUMNS)).reset_index(drop=True)


def run_external_calibration_contract_diagnostic(
    *,
    case_names: list[str],
    output_dir: Path,
    n_replicates: int,
    seed: int,
    contract: ExternalCalibrationContract,
) -> dict[str, pd.DataFrame]:
    """Run selected-hierarchy support and scalar-vs-tail diagnostics."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    _require_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()

    record_tables: list[pd.DataFrame] = []
    case_summaries: list[dict[str, object]] = []
    selected_cases = _selected_cases(case_names)
    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index}/{len(selected_cases)}] {case['name']}", flush=True)
        records, case_summary = _diagnose_case(
            case,
            n_replicates=int(n_replicates),
            seed=int(seed) + index * 1_000_000,
        )
        case_summary["status"] = "ok"
        case_summary["skip_reason"] = ""
        record_tables.append(records)
        case_summaries.append(case_summary)

    selected_records = pd.concat(record_tables, ignore_index=True)
    case_summary = pd.DataFrame.from_records(case_summaries)
    contract_summary = evaluate_external_calibration_contract(
        selected_records,
        contract=contract,
    )

    outputs = {
        "case_summary": case_summary,
        "external_calibration_contract": contract_summary,
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "case_names": case_names,
        "conditioning_scope": CONDITIONING_SCOPE,
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(contract.sibling_alpha),
        "contract": asdict(contract),
        "resolved_min_matching_simulations": contract.resolved_min_matching_simulations(),
        "resolved_min_matched_records": contract.resolved_min_matched_records(),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "note": (
            "Diagnostic-only external selected-hierarchy calibration contract. "
            "Rows that fail admissibility define no production external "
            "calibration estimate. Scalar-c decisions are descriptive unless "
            "the row is production-admissible."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected-hierarchy external calibration admissibility and "
            "compare scalar-c calibration against selected-ratio tail behavior."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--tail-resolution-fraction-of-alpha",
        type=float,
        default=0.2,
    )
    parser.add_argument("--min-matching-simulations", type=int, default=None)
    parser.add_argument("--min-matched-records", type=int, default=None)
    parser.add_argument("--max-relative-c-simulation-se", type=float, default=0.05)
    parser.add_argument("--scalar-rejection-absolute-tolerance", type=float, default=0.005)
    parser.add_argument("--scalar-tail-quantile-excess-limit", type=float, default=1.25)
    parser.add_argument("--min-scalar-uniformity-ks-p-value", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"selected_hierarchy_external_calibration_contract_{format_timestamp_utc()}"
        )
    outputs = run_external_calibration_contract_diagnostic(
        case_names=_parse_csv_list(str(args.case_names)),
        output_dir=output_dir,
        n_replicates=int(args.n_replicates),
        seed=int(args.seed),
        contract=ExternalCalibrationContract(
            tail_resolution_fraction_of_alpha=float(
                args.tail_resolution_fraction_of_alpha
            ),
            min_matching_simulations=args.min_matching_simulations,
            min_matched_records=args.min_matched_records,
            max_relative_c_simulation_se=float(args.max_relative_c_simulation_se),
            scalar_rejection_absolute_tolerance=float(
                args.scalar_rejection_absolute_tolerance
            ),
            scalar_tail_quantile_excess_limit=float(
                args.scalar_tail_quantile_excess_limit
            ),
            min_scalar_uniformity_ks_p_value=float(
                args.min_scalar_uniformity_ks_p_value
            ),
        ),
    )
    print(outputs["case_summary"].to_string(index=False))
    print(outputs["external_calibration_contract"].to_string(index=False))
    print(f"Wrote selected-hierarchy external calibration contract outputs to {output_dir}")


if __name__ == "__main__":
    main()
