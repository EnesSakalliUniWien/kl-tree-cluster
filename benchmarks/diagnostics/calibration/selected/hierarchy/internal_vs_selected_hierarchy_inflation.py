"""Compare internal empirical inflation with selected-hierarchy tail behavior.

This script is diagnostic-only. It measures whether the active internal
empirical-null inflation estimate agrees with a regenerated same-data
selected-hierarchy null for the same observed sibling target. It does not
install the selected-hierarchy estimate as a production calibration fallback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import chi2
from tree_break_selection.core_utils.tree_utils import compute_node_depths
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
    predict_empirical_inflation_factor,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from benchmarks.diagnostics.calibration.selected.hierarchy.selected_hierarchy_null_audit import (
    SelectedSiblingContext,
    SelectedSiblingRecord,
    SiblingRecordSample,
    _choose_observed_target,
    _run_edge_and_sibling_records,
    _selected_hierarchy_summary,
    _simulate_null_data,
)
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_internal_vs_selected_hierarchy_inflation_not_calibration"

DEFAULT_CASE_NAMES = (
    "gauss_null_large",
    "dim_diffuse_6c_136f",
    "binary_low_noise_4c",
    "cat_highcard_20cat_4c",
)


@dataclass(frozen=True)
class ObservedSiblingTarget:
    """Observed target plus the full sibling records from the same tree."""

    context: SelectedSiblingContext
    record: SiblingPairRecord
    all_records: tuple[SiblingPairRecord, ...]
    observed_sample: SiblingRecordSample
    tree_distance_metric: str
    tree_distance_source: str
    feature_family: str


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _selected_cases(case_names: list[str]) -> list[dict[str, object]]:
    case_by_name = {str(case["name"]): case for case in get_default_test_cases()}
    missing = [case_name for case_name in case_names if case_name not in case_by_name]
    if missing:
        raise ValueError(f"Unknown benchmark case name(s): {missing!r}.")
    return [case_by_name[case_name].copy() for case_name in case_names]


def required_inflation_to_block(record: SiblingPairRecord, *, alpha: float) -> float:
    """Return the scalar inflation that moves one record exactly to alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if record.degrees_of_freedom <= 0.0 or record.stat <= 0.0:
        return float("nan")
    critical_value = float(chi2.isf(alpha, df=float(record.degrees_of_freedom)))
    return float(record.stat / (record.reference_scale * critical_value))


def inflation_adjusted_p_value(
    record: SiblingPairRecord,
    *,
    inflation_factor: float,
) -> float:
    """Evaluate the projected-Wald tail after a scalar inflation factor."""
    if record.degrees_of_freedom <= 0.0:
        return 1.0
    if not np.isfinite(inflation_factor) or inflation_factor <= 0.0:
        raise ValueError(f"inflation_factor must be positive; got {inflation_factor!r}.")
    adjusted_statistic = float(record.stat / (record.reference_scale * inflation_factor))
    return float(chi2.sf(adjusted_statistic, df=float(record.degrees_of_freedom)))


def _feature_family(feature_space: FeatureSpace | None) -> str:
    return "bernoulli" if feature_space is None else feature_space.family_label


def _selected_records_from_full_tree(
    *,
    records: tuple[SiblingPairRecord, ...],
    node_depths: dict[object, int],
) -> tuple[SelectedSiblingRecord, ...]:
    return tuple(
        SelectedSiblingRecord(
            record=record,
            parent_depth=int(node_depths[record.parent]),
        )
        for record in records
        if (
            not record.is_null_like
            and not record.is_edge_blocked
            and record.degrees_of_freedom > 0.0
        )
    )


def _observed_target(
    case: dict[str, object],
    *,
    target_mode: str,
) -> tuple[ObservedSiblingTarget, pd.DataFrame, dict[str, object]]:
    context = build_tbs_tree_context(case, populate_node_distributions=True)
    edge_df, spectral_context = annotate_child_parent_divergence(
        context.tree,
        context.tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        context.tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, _non_binary_nodes = collect_sibling_pair_records(
        context.tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=context.feature_space,
    )
    node_depths = compute_node_depths(context.tree)
    selected_records = _selected_records_from_full_tree(
        records=tuple(records),
        node_depths=node_depths,
    )
    target_context = _choose_observed_target(
        str(context.metadata["name"]),
        selected_records,
        target_mode=target_mode,
    )
    matching_records = [
        selected_record.record
        for selected_record in selected_records
        if selected_record.record.parent == target_context.parent
    ]
    if len(matching_records) != 1:
        raise ValueError(
            "Observed target selection requires exactly one sibling record for "
            f"parent={target_context.parent!r}; found {len(matching_records)}."
        )
    tested_edges = edge_df["Child_Parent_Divergence_Tested"].astype(bool)
    significant_edges = edge_df["Child_Parent_Divergence_Significant"].astype(bool)
    observed_sample = SiblingRecordSample(
        selected_records=selected_records,
        candidate_records=int(len(records)),
        tested_edges=int(tested_edges.sum()),
        significant_edges=int(significant_edges.sum()),
    )
    target = ObservedSiblingTarget(
        context=target_context,
        record=matching_records[0],
        all_records=tuple(records),
        observed_sample=observed_sample,
        tree_distance_metric=context.tree_distance_metric,
        tree_distance_source=context.tree_distance_source,
        feature_family=_feature_family(context.feature_space),
    )
    return target, context.data, context.metadata


def _internal_empirical_inflation_summary(
    *,
    target_record: SiblingPairRecord,
    records: tuple[SiblingPairRecord, ...],
    alpha: float,
) -> dict[str, object]:
    required_c = required_inflation_to_block(target_record, alpha=alpha)
    support_records = tuple(
        record
        for record in records
        if record.degrees_of_freedom > 0.0
        and record.sibling_null_weight > 0.0
        and (record.is_null_like or record.is_edge_blocked)
    )
    positive_weight_records = tuple(
        record
        for record in records
        if record.degrees_of_freedom > 0.0 and record.sibling_null_weight > 0.0
    )
    try:
        model = fit_empirical_null_inflation_model(list(records))
        c_hat = predict_empirical_inflation_factor(model, target_record)
        p_value = inflation_adjusted_p_value(
            target_record,
            inflation_factor=c_hat,
        )
    except ValueError as exc:
        return {
            "internal_support_status": "unsupported_internal_empirical_null",
            "internal_support_error": str(exc),
            "internal_c_hat": np.nan,
            "internal_p_value": np.nan,
            "internal_blocks_at_alpha": np.nan,
            "internal_c_over_required": np.nan,
            "internal_required_c_to_block": required_c,
            "internal_model_calibration_records": int(len(support_records)),
            "internal_positive_weight_records": int(len(positive_weight_records)),
            "internal_strict_null_records": int(
                sum(record.is_null_like for record in support_records)
            ),
            "internal_stopped_or_null_records": int(len(support_records)),
            "internal_effective_sample_size": np.nan,
            "internal_baseline_c_hat": np.nan,
        }
    return {
        "internal_support_status": "supported_internal_empirical_null",
        "internal_support_error": "",
        "internal_c_hat": float(c_hat),
        "internal_p_value": p_value,
        "internal_blocks_at_alpha": bool(p_value >= alpha),
        "internal_c_over_required": (
            float(c_hat / required_c) if np.isfinite(required_c) and required_c > 0.0 else np.nan
        ),
        "internal_required_c_to_block": required_c,
        "internal_model_calibration_records": int(model.n_calibration),
        "internal_positive_weight_records": int(len(positive_weight_records)),
        "internal_strict_null_records": int(model.n_strict_null_calibration),
        "internal_stopped_or_null_records": int(model.n_stopped_or_null_calibration),
        "internal_effective_sample_size": float(model.effective_sample_size),
        "internal_baseline_c_hat": float(model.baseline_empirical_inflation_factor),
    }


def _comparison_fields(
    *,
    internal: dict[str, object],
    selected_hierarchy: dict[str, object],
) -> dict[str, object]:
    internal_c = float(internal["internal_c_hat"])
    selected_c = float(selected_hierarchy["selected_hierarchy_c_hat"])
    required_c = float(internal["internal_required_c_to_block"])
    return {
        "internal_to_selected_c_ratio": (
            float(internal_c / selected_c)
            if np.isfinite(internal_c) and np.isfinite(selected_c) and selected_c > 0.0
            else np.nan
        ),
        "selected_hierarchy_c_over_required": (
            float(selected_c / required_c)
            if np.isfinite(selected_c) and np.isfinite(required_c) and required_c > 0.0
            else np.nan
        ),
        "internal_minus_selected_c": (
            float(internal_c - selected_c)
            if np.isfinite(internal_c) and np.isfinite(selected_c)
            else np.nan
        ),
        "calibration_comparison_status": _comparison_status(
            internal_status=str(internal["internal_support_status"]),
            selected_status=str(selected_hierarchy["selected_hierarchy_support_status"]),
        ),
    }


def _comparison_status(*, internal_status: str, selected_status: str) -> str:
    internal_supported = internal_status == "supported_internal_empirical_null"
    selected_supported = selected_status == "matched_selected_hierarchy_records"
    if internal_supported and selected_supported:
        return "internal_and_selected_hierarchy_supported"
    if internal_supported:
        return "internal_supported_selected_hierarchy_unsupported"
    if selected_supported:
        return "internal_unsupported_selected_hierarchy_supported"
    return "internal_and_selected_hierarchy_unsupported"


def _selected_hierarchy_samples(
    *,
    data: pd.DataFrame,
    metadata: dict[str, object],
    feature_space: FeatureSpace | None,
    n_replicates: int,
    seed: int,
) -> list[SiblingRecordSample]:
    if feature_space is not None and feature_space.family_label == "continuous":
        raise ValueError(
            "Continuous selected-hierarchy null comparison requires a validated "
            "continuous covariance generator before use."
        )
    rng = np.random.default_rng(int(seed))
    samples: list[SiblingRecordSample] = []
    for _replicate_index in range(int(n_replicates)):
        simulated = _simulate_null_data(data, feature_space, rng=rng)
        sample, _tree_metric = _run_edge_and_sibling_records(
            simulated,
            metadata,
            feature_space,
        )
        samples.append(sample)
    return samples


def _diagnose_case(
    case: dict[str, object],
    *,
    n_replicates: int,
    seed: int,
    target_mode: str,
    context_match: str,
) -> dict[str, object]:
    target, data, metadata = _observed_target(case, target_mode=target_mode)
    feature_space = metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Prepared feature_space metadata must be a FeatureSpace.")
    samples = _selected_hierarchy_samples(
        data=data,
        metadata=metadata,
        feature_space=feature_space,
        n_replicates=n_replicates,
        seed=seed,
    )
    internal = _internal_empirical_inflation_summary(
        target_record=target.record,
        records=target.all_records,
        alpha=float(DEFAULT_SIBLING_ALPHA),
    )
    selected_hierarchy = _selected_hierarchy_summary(
        target=target.context,
        samples=samples,
        context_match=context_match,
        n_replicates=int(n_replicates),
    )
    row = {
        "study_role": STUDY_ROLE,
        "case_id": str(metadata["name"]),
        "case_category": str(metadata["category"]),
        "feature_family": target.feature_family,
        "tree_distance_metric": target.tree_distance_metric,
        "tree_distance_source": target.tree_distance_source,
        "target_mode": target_mode,
        "context_match": context_match,
        "observed_parent": target.context.parent,
        "observed_statistic": target.context.statistic,
        "observed_reference_expectation": target.context.reference_expectation,
        "observed_stat_over_reference": (
            target.context.statistic / target.context.reference_expectation
        ),
        "observed_raw_p_value": target.context.raw_p_value,
        "observed_degrees_of_freedom": target.context.degrees_of_freedom,
        "observed_projection_dimension": target.context.projection_dimension,
        "observed_parent_sample_size": target.context.parent_sample_size,
        "observed_parent_depth": target.context.parent_depth,
        "observed_selected_records": len(target.observed_sample.selected_records),
        "observed_candidate_records": target.observed_sample.candidate_records,
        "observed_edge_rejection_rate": (
            target.observed_sample.significant_edges / target.observed_sample.tested_edges
            if target.observed_sample.tested_edges
            else 0.0
        ),
    }
    row.update(internal)
    row.update(selected_hierarchy)
    row.update(
        _comparison_fields(
            internal=internal,
            selected_hierarchy=selected_hierarchy,
        )
    )
    return row


def run_internal_vs_selected_hierarchy_inflation_study(
    *,
    case_names: list[str],
    output_dir: Path,
    n_replicates: int,
    seed: int,
    target_mode: str,
    context_match: str,
) -> pd.DataFrame:
    """Write one diagnostic table comparing internal and selected-hierarchy c."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started_at = perf_counter()
    cases = _selected_cases(case_names)
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['name']}", flush=True)
        try:
            row = _diagnose_case(
                case,
                n_replicates=int(n_replicates),
                seed=int(seed) + index * 1_000_000,
                target_mode=target_mode,
                context_match=context_match,
            )
            row["status"] = "ok"
            row["skip_reason"] = ""
        except ValueError as exc:
            row = {
                "study_role": STUDY_ROLE,
                "case_id": str(case["name"]),
                "case_category": str(case["category"]),
                "target_mode": target_mode,
                "context_match": context_match,
                "status": "skip",
                "skip_reason": str(exc),
            }
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "internal_vs_selected_hierarchy_inflation.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "study_role": STUDY_ROLE,
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "case_names": case_names,
        "target_mode": target_mode,
        "context_match": context_match,
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "summary_csv": str(summary_path),
        "note": (
            "Diagnostic-only comparison between active internal empirical-null "
            "inflation and a regenerated same-data selected-hierarchy sibling "
            "null. Unsupported support remains undefined; selected-hierarchy "
            "estimates are not used as production fallback calibration."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare internal empirical sibling inflation against regenerated "
            "selected-hierarchy selected-ratio behavior for the same observed target."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--n-replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument(
        "--target-mode",
        choices=(
            "root",
            "strongest",
            "median_parent_size",
            "non_root_strongest",
            "non_root_median_parent_size",
        ),
        default="root",
    )
    parser.add_argument(
        "--context-match",
        choices=(
            "any",
            "projection",
            "projection_and_parent_size",
            "projection_parent_size_depth",
        ),
        default="projection",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"internal_vs_selected_hierarchy_inflation_{format_timestamp_utc()}"
        )
    summary = run_internal_vs_selected_hierarchy_inflation_study(
        case_names=_parse_csv_list(str(args.case_names)),
        output_dir=output_dir,
        n_replicates=int(args.n_replicates),
        seed=int(args.seed),
        target_mode=str(args.target_mode),
        context_match=str(args.context_match),
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir / 'internal_vs_selected_hierarchy_inflation.csv'}")


if __name__ == "__main__":
    main()
