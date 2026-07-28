"""Promotion gate for spectral transport traversal evidence.

This diagnostic turns overlap benchmark evidence into an explicit promotion
decision. A spectral traversal rule is promotable only if it preserves signal
quality in standard dispatch and reduces selected-null false splitting in the
selected-family null/signal panel. The gate is intentionally conservative:
missing or neutral null-side evidence keeps the rule diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "spectral_transport_promotion_gate/v1"
STUDY_ROLE = "diagnostic_spectral_transport_promotion_gate_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.spectral_transport.spectral_transport_promotion_gate"
)

BASELINE_PROFILE = "fixed_coordinate_global_passthrough_refined_v1"
CANDIDATE_PROFILE = "fixed_coordinate_spectral_transport_passthrough_v1"
BASELINE_METHOD = "tbs_global_passthrough_refined_diagnostic"
CANDIDATE_METHOD = "tbs_spectral_transport_passthrough"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_DISPATCH_PAIRWISE = (
    DEFAULT_RESULT_ROOT
    / "spectral_transport_overlap_dispatch_panel_promoted"
    / "spectral_transport_overlap_dispatch_pairwise.csv"
)
DEFAULT_SELECTED_FAMILY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "selected_family_traversal_spectral_transport_promoted_replicates"
    / "selected_family_traversal_rows.csv"
)

COMPONENT_OUTPUT = "spectral_transport_promotion_components.csv"
SUMMARY_OUTPUT = "spectral_transport_promotion_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class SpectralTransportPromotionGateConfig:
    """Inputs and thresholds for the spectral traversal promotion gate."""

    output_dir: Path
    dispatch_pairwise_path: Path = DEFAULT_DISPATCH_PAIRWISE
    selected_family_rows_path: Path = DEFAULT_SELECTED_FAMILY_ROWS
    baseline_method: str = BASELINE_METHOD
    candidate_method: str = CANDIDATE_METHOD
    baseline_profile: str = BASELINE_PROFILE
    candidate_profile: str = CANDIDATE_PROFILE
    max_allowed_signal_ari_drop: float = 0.0
    min_null_false_split_reduction: int = 1
    min_standard_dispatch_cases: int = 3
    min_selected_family_null_cases: int = 3
    min_selected_family_signal_cases: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dispatch-pairwise-path",
        type=Path,
        default=DEFAULT_DISPATCH_PAIRWISE,
    )
    parser.add_argument(
        "--selected-family-rows-path",
        type=Path,
        default=DEFAULT_SELECTED_FAMILY_ROWS,
    )
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--candidate-method", default=CANDIDATE_METHOD)
    parser.add_argument("--baseline-profile", default=BASELINE_PROFILE)
    parser.add_argument("--candidate-profile", default=CANDIDATE_PROFILE)
    parser.add_argument("--max-allowed-signal-ari-drop", type=float, default=0.0)
    parser.add_argument("--min-null-false-split-reduction", type=int, default=1)
    parser.add_argument("--min-standard-dispatch-cases", type=int, default=3)
    parser.add_argument("--min-selected-family-null-cases", type=int, default=3)
    parser.add_argument("--min-selected-family-signal-cases", type=int, default=3)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _finite_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.min()) if not numeric.empty else float("nan")


def _finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if not numeric.empty else float("nan")


def evaluate_standard_dispatch_component(
    pairwise: pd.DataFrame,
    *,
    baseline_method: str = BASELINE_METHOD,
    candidate_method: str = CANDIDATE_METHOD,
    max_allowed_signal_ari_drop: float = 0.0,
    min_cases: int = 3,
) -> dict[str, object]:
    """Evaluate standard-dispatch signal-retention evidence."""
    required = {
        "baseline_method",
        "candidate_method",
        "baseline_status",
        "candidate_status",
        "delta_ari_candidate_minus_baseline",
        "partition_ari_between_methods",
    }
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"dispatch pairwise rows missing required columns: {sorted(missing)!r}.")

    table = pairwise[
        pairwise["baseline_method"].astype(str).eq(str(baseline_method))
        & pairwise["candidate_method"].astype(str).eq(str(candidate_method))
    ].copy()
    n_cases = int(table["case_id"].nunique()) if "case_id" in table.columns else int(len(table))
    both_ok = table["baseline_status"].astype(str).eq("ok") & table["candidate_status"].astype(
        str
    ).eq("ok")
    min_delta_ari = _finite_min(table["delta_ari_candidate_minus_baseline"])
    mean_delta_ari = _finite_mean(table["delta_ari_candidate_minus_baseline"])
    mean_partition_ari = _finite_mean(table["partition_ari_between_methods"])
    threshold = -float(max_allowed_signal_ari_drop)
    passes = bool(
        n_cases >= int(min_cases)
        and bool(both_ok.all())
        and np.isfinite(min_delta_ari)
        and min_delta_ari >= threshold
    )
    return {
        "component_id": "standard_dispatch_signal_retention",
        "component_type": "standard_dispatch_pairwise",
        "required_for_promotion": True,
        "component_status": "passes" if passes else "fails",
        "n_cases": n_cases,
        "n_ok_pairs": int(both_ok.sum()),
        "min_required_cases": int(min_cases),
        "min_delta_ari_candidate_minus_baseline": min_delta_ari,
        "mean_delta_ari_candidate_minus_baseline": mean_delta_ari,
        "mean_partition_ari_between_methods": mean_partition_ari,
        "threshold": threshold,
        "failure_reason": "" if passes else "standard_dispatch_signal_regression_or_coverage",
    }


def _paired_selected_family_rows(
    rows: pd.DataFrame,
    *,
    baseline_profile: str,
    candidate_profile: str,
    data_role: str,
) -> pd.DataFrame:
    required = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "ari",
        "false_split",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"selected-family rows missing required columns: {sorted(missing)!r}.")
    subset = rows[rows["data_role"].astype(str).eq(str(data_role))].copy()
    baseline = subset[subset["method_id"].astype(str).eq(str(baseline_profile))].copy()
    candidate = subset[subset["method_id"].astype(str).eq(str(candidate_profile))].copy()
    key = ["case_id", "data_role", "replicate"]
    return baseline.merge(
        candidate,
        on=key,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )


def evaluate_selected_family_signal_component(
    rows: pd.DataFrame,
    *,
    baseline_profile: str = BASELINE_PROFILE,
    candidate_profile: str = CANDIDATE_PROFILE,
    max_allowed_signal_ari_drop: float = 0.0,
    min_cases: int = 3,
) -> dict[str, object]:
    """Evaluate selected-family signal retention."""
    paired = _paired_selected_family_rows(
        rows,
        baseline_profile=baseline_profile,
        candidate_profile=candidate_profile,
        data_role="signal",
    )
    n_cases = int(paired["case_id"].nunique()) if not paired.empty else 0
    delta_ari = (
        pd.to_numeric(paired["ari_candidate"], errors="coerce")
        - pd.to_numeric(paired["ari_baseline"], errors="coerce")
        if not paired.empty
        else pd.Series(dtype=float)
    )
    min_delta_ari = _finite_min(delta_ari)
    mean_delta_ari = _finite_mean(delta_ari)
    threshold = -float(max_allowed_signal_ari_drop)
    passes = bool(
        n_cases >= int(min_cases) and np.isfinite(min_delta_ari) and min_delta_ari >= threshold
    )
    return {
        "component_id": "selected_family_signal_retention",
        "component_type": "selected_family_signal",
        "required_for_promotion": True,
        "component_status": "passes" if passes else "fails",
        "n_cases": n_cases,
        "min_required_cases": int(min_cases),
        "min_delta_ari_candidate_minus_baseline": min_delta_ari,
        "mean_delta_ari_candidate_minus_baseline": mean_delta_ari,
        "threshold": threshold,
        "failure_reason": "" if passes else "selected_family_signal_regression_or_coverage",
    }


def evaluate_selected_family_null_component(
    rows: pd.DataFrame,
    *,
    baseline_profile: str = BASELINE_PROFILE,
    candidate_profile: str = CANDIDATE_PROFILE,
    min_false_split_reduction: int = 1,
    min_cases: int = 3,
) -> dict[str, object]:
    """Evaluate selected-null false-split reduction."""
    paired = _paired_selected_family_rows(
        rows,
        baseline_profile=baseline_profile,
        candidate_profile=candidate_profile,
        data_role="selected_null",
    )
    n_cases = int(paired["case_id"].nunique()) if not paired.empty else 0
    baseline_false = (
        paired["false_split_baseline"].map(_to_bool) if not paired.empty else pd.Series(dtype=bool)
    )
    candidate_false = (
        paired["false_split_candidate"].map(_to_bool) if not paired.empty else pd.Series(dtype=bool)
    )
    baseline_count = int(baseline_false.sum()) if not paired.empty else 0
    candidate_count = int(candidate_false.sum()) if not paired.empty else 0
    reduction = int(baseline_count - candidate_count)
    passes = bool(
        n_cases >= int(min_cases)
        and reduction >= int(min_false_split_reduction)
        and candidate_count < baseline_count
    )
    return {
        "component_id": "selected_family_null_false_split_reduction",
        "component_type": "selected_family_null",
        "required_for_promotion": True,
        "component_status": "passes" if passes else "fails",
        "n_cases": n_cases,
        "min_required_cases": int(min_cases),
        "baseline_false_split_count": baseline_count,
        "candidate_false_split_count": candidate_count,
        "false_split_reduction": reduction,
        "min_false_split_reduction": int(min_false_split_reduction),
        "failure_reason": "" if passes else "selected_null_false_split_not_reduced",
    }


def summarize_promotion_components(components: pd.DataFrame) -> pd.DataFrame:
    """Summarize component rows into a single promotion decision."""
    if components.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "promotion_contract_id": "spectral_transport_passthrough_promotion_v1",
                    "promotion_decision": "diagnostic_only_not_promoted",
                    "blocking_component_ids": "no_components",
                    "n_components": 0,
                    "n_required_components": 0,
                    "n_required_passed": 0,
                    "n_required_failed": 0,
                }
            ]
        )
    table = components.copy()
    required = table["required_for_promotion"].map(_to_bool)
    required_table = table[required]
    passed = required_table["component_status"].astype(str).eq("passes")
    blocking = required_table.loc[~passed, "component_id"].astype(str).tolist()
    decision = (
        "promotion_admissible"
        if required_table.shape[0] > 0 and not blocking
        else "diagnostic_only_not_promoted"
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "promotion_contract_id": "spectral_transport_passthrough_promotion_v1",
                "promotion_decision": decision,
                "blocking_component_ids": ";".join(blocking),
                "n_components": int(table.shape[0]),
                "n_required_components": int(required_table.shape[0]),
                "n_required_passed": int(passed.sum()),
                "n_required_failed": int((~passed).sum()),
            }
        ]
    )


def evaluate_spectral_transport_promotion_gate(
    *,
    dispatch_pairwise: pd.DataFrame,
    selected_family_rows: pd.DataFrame,
    config: SpectralTransportPromotionGateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all promotion components."""
    component_rows = [
        evaluate_standard_dispatch_component(
            dispatch_pairwise,
            baseline_method=config.baseline_method,
            candidate_method=config.candidate_method,
            max_allowed_signal_ari_drop=config.max_allowed_signal_ari_drop,
            min_cases=config.min_standard_dispatch_cases,
        ),
        evaluate_selected_family_signal_component(
            selected_family_rows,
            baseline_profile=config.baseline_profile,
            candidate_profile=config.candidate_profile,
            max_allowed_signal_ari_drop=config.max_allowed_signal_ari_drop,
            min_cases=config.min_selected_family_signal_cases,
        ),
        evaluate_selected_family_null_component(
            selected_family_rows,
            baseline_profile=config.baseline_profile,
            candidate_profile=config.candidate_profile,
            min_false_split_reduction=config.min_null_false_split_reduction,
            min_cases=config.min_selected_family_null_cases,
        ),
    ]
    components = pd.DataFrame.from_records(component_rows)
    components.insert(0, "schema_version", SCHEMA_VERSION)
    components.insert(1, "study_role", STUDY_ROLE)
    summary = summarize_promotion_components(components)
    return components, summary


def run_spectral_transport_promotion_gate(
    config: SpectralTransportPromotionGateConfig,
) -> dict[str, Path]:
    """Run the gate from existing diagnostic CSVs and write outputs."""
    dispatch_pairwise = pd.read_csv(config.dispatch_pairwise_path)
    selected_family_rows = pd.read_csv(config.selected_family_rows_path)
    components, summary = evaluate_spectral_transport_promotion_gate(
        dispatch_pairwise=dispatch_pairwise,
        selected_family_rows=selected_family_rows,
        config=config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    components_path = config.output_dir / COMPONENT_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT
    components.to_csv(components_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "dispatch_pairwise_path": config.dispatch_pairwise_path,
        "selected_family_rows_path": config.selected_family_rows_path,
        "baseline_method": config.baseline_method,
        "candidate_method": config.candidate_method,
        "baseline_profile": config.baseline_profile,
        "candidate_profile": config.candidate_profile,
        "max_allowed_signal_ari_drop": config.max_allowed_signal_ari_drop,
        "min_null_false_split_reduction": config.min_null_false_split_reduction,
        "outputs": {
            "components": components_path,
            "summary": summary_path,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return {
        "components": components_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    run_spectral_transport_promotion_gate(
        SpectralTransportPromotionGateConfig(
            output_dir=args.output_dir,
            dispatch_pairwise_path=args.dispatch_pairwise_path,
            selected_family_rows_path=args.selected_family_rows_path,
            baseline_method=str(args.baseline_method),
            candidate_method=str(args.candidate_method),
            baseline_profile=str(args.baseline_profile),
            candidate_profile=str(args.candidate_profile),
            max_allowed_signal_ari_drop=float(args.max_allowed_signal_ari_drop),
            min_null_false_split_reduction=int(args.min_null_false_split_reduction),
            min_standard_dispatch_cases=int(args.min_standard_dispatch_cases),
            min_selected_family_null_cases=int(args.min_selected_family_null_cases),
            min_selected_family_signal_cases=int(args.min_selected_family_signal_cases),
        )
    )


if __name__ == "__main__":
    main()
