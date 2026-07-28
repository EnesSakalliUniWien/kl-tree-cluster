"""Selected-hierarchy null audit for same-data Tree-Break Selection inference.

This diagnostic rebuilds the hierarchy inside each null replicate, then reruns
the edge gate and raw sibling-record collection. It estimates the selected
same-data null for sibling statistics after the tree and edge path have already
selected focal sibling contexts.

The script is diagnostic-only. It does not install an external calibration
fallback in production code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from tree_break_selection.core_utils.tree_utils import compute_node_depths
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence,
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
from tree_break_selection.tree.construction import tree_from_linkage
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

DEFAULT_CASE_NAMES = (
    "gauss_null_large",
    "gauss_clear_medium",
    "binary_low_noise_4c",
    "cat_clear_3cat_4c",
)


@dataclass(frozen=True)
class SelectedSiblingContext:
    """Observed sibling target used for selected-hierarchy null comparison."""

    case_id: str
    parent: object
    feature_family: str
    target_mode: str
    statistic: float
    reference_expectation: float
    raw_p_value: float
    degrees_of_freedom: float
    projection_dimension: int
    parent_sample_size: int
    parent_depth: int


@dataclass(frozen=True)
class SelectedSiblingRecord:
    """Sibling statistic selected by the same-data hierarchy and edge path."""

    record: SiblingPairRecord
    parent_depth: int


@dataclass(frozen=True)
class SiblingRecordSample:
    """Raw selected sibling statistics collected from one hierarchy run."""

    selected_records: tuple[SelectedSiblingRecord, ...]
    candidate_records: int
    tested_edges: int
    significant_edges: int


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _selected_cases(case_names: list[str]) -> list[dict[str, object]]:
    case_by_name = {str(case["name"]): case for case in get_default_test_cases()}
    missing = [case_name for case_name in case_names if case_name not in case_by_name]
    if missing:
        raise ValueError(f"Unknown benchmark case name(s): {missing!r}.")
    return [case_by_name[case_name].copy() for case_name in case_names]


def _build_tree(data: pd.DataFrame, metadata: dict[str, object]) -> tuple[PosetTree, str]:
    if bool(metadata["requires_precomputed_tbs_distance"]):
        raise ValueError(
            "Selected-hierarchy null audit does not support cases requiring "
            "precomputed TBS tree distances. The null generator must own the "
            "tree-distance contract before those cases are audited."
        )
    params = METHOD_SPECS["tbs"].param_grid[0]
    metric = str(params["tree_distance_metric"])
    linkage_method = str(params["tree_linkage_method"])
    tree = tree_from_linkage(
        linkage(pdist(data.values, metric=metric), method=linkage_method),
        leaf_names=data.index.tolist(),
    )
    return tree, metric


def _run_edge_and_sibling_records(
    data: pd.DataFrame,
    metadata: dict[str, object],
    feature_space: FeatureSpace | None,
) -> tuple[SiblingRecordSample, str]:
    tree, tree_metric = _build_tree(data, metadata)
    tree.populate_node_divergences(data, feature_space=feature_space)
    edge_df, spectral_context = annotate_child_parent_divergence(
        tree,
        tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=data,
        feature_space=feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, _non_binary_nodes = collect_sibling_pair_records(
        tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    node_depths = compute_node_depths(tree)
    selected_records = tuple(
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
    tested_edges = edge_df["Child_Parent_Divergence_Tested"].astype(bool)
    significant_edges = edge_df["Child_Parent_Divergence_Significant"].astype(bool)
    return (
        SiblingRecordSample(
            selected_records=selected_records,
            candidate_records=int(len(records)),
            tested_edges=int(tested_edges.sum()),
            significant_edges=int(significant_edges.sum()),
        ),
        tree_metric,
    )


def _choose_observed_target(
    case_id: str,
    records: tuple[SelectedSiblingRecord, ...],
    *,
    target_mode: str,
) -> SelectedSiblingContext:
    focal_records = list(records)
    if not focal_records:
        raise ValueError(f"Case {case_id!r} has no sibling records selected by the edge path.")

    if target_mode == "root":
        max_parent_sample_size = max(record.record.n_parent for record in focal_records)
        candidates = [
            record for record in focal_records if record.record.n_parent == max_parent_sample_size
        ]
        if not candidates:
            raise ValueError(f"Case {case_id!r} has no focal root-like sibling record.")
        target = candidates[0]
    elif target_mode == "strongest":
        target = min(focal_records, key=lambda item: item.record.p_value)
    elif target_mode == "median_parent_size":
        ordered = sorted(focal_records, key=lambda item: item.record.n_parent)
        target = ordered[len(ordered) // 2]
    elif target_mode == "non_root_strongest":
        max_parent_sample_size = max(record.record.n_parent for record in focal_records)
        non_root_records = [
            record for record in focal_records if record.record.n_parent < max_parent_sample_size
        ]
        if not non_root_records:
            raise ValueError(
                f"Case {case_id!r} has no non-root sibling record selected by the edge path."
            )
        target = min(non_root_records, key=lambda item: item.record.p_value)
    elif target_mode == "non_root_median_parent_size":
        max_parent_sample_size = max(record.record.n_parent for record in focal_records)
        non_root_records = [
            record for record in focal_records if record.record.n_parent < max_parent_sample_size
        ]
        if not non_root_records:
            raise ValueError(
                f"Case {case_id!r} has no non-root sibling record selected by the edge path."
            )
        ordered = sorted(non_root_records, key=lambda item: item.record.n_parent)
        target = ordered[len(ordered) // 2]
    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}.")

    target_record = target.record
    return SelectedSiblingContext(
        case_id=case_id,
        parent=target_record.parent,
        feature_family=target_record.feature_family,
        target_mode=target_mode,
        statistic=float(target_record.stat),
        reference_expectation=float(
            target_record.reference_scale * target_record.degrees_of_freedom
        ),
        raw_p_value=float(target_record.p_value),
        degrees_of_freedom=float(target_record.degrees_of_freedom),
        projection_dimension=int(target_record.sibling_projection_dimension),
        parent_sample_size=int(target_record.n_parent),
        parent_depth=int(target.parent_depth),
    )


def _bernoulli_null(
    data: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    probabilities = data.to_numpy(dtype=float).mean(axis=0)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    simulated = (rng.random(size=data.shape) < probabilities[None, :]).astype(int)
    return pd.DataFrame(simulated, index=data.index, columns=data.columns)


def _categorical_null(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if feature_space.family_label != "categorical":
        raise ValueError("Categorical null simulation requires a categorical feature_space.")
    simulated = np.zeros(data.shape, dtype=int)
    values = data.to_numpy(dtype=float)
    for block in feature_space.blocks:
        if block.family != "categorical":
            raise ValueError("Mixed feature spaces are not supported by this diagnostic.")
        columns = list(block.column_indices)
        probabilities = values[:, columns].mean(axis=0)
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0:
            raise ValueError(f"Categorical block {block.name!r} has zero mass.")
        probabilities = probabilities / probability_sum
        draws = rng.choice(len(columns), size=data.shape[0], p=probabilities)
        simulated[np.arange(data.shape[0]), np.asarray(columns)[draws]] = 1
    return pd.DataFrame(simulated, index=data.index, columns=data.columns)


def _simulate_null_data(
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if feature_space is None:
        return _bernoulli_null(data, rng=rng)
    if feature_space.family_label == "bernoulli":
        return _bernoulli_null(data, rng=rng)
    if feature_space.family_label == "categorical":
        return _categorical_null(data, feature_space, rng=rng)
    raise ValueError(
        "Selected-hierarchy null audit currently supports Bernoulli and "
        "categorical feature families only. Continuous selected-hierarchy null "
        "simulation needs a validated covariance generator before use."
    )


def _record_matches_context(
    record_sample: SelectedSiblingRecord,
    target: SelectedSiblingContext,
    *,
    context_match: str,
) -> bool:
    record = record_sample.record
    if record.feature_family != target.feature_family:
        return False
    if context_match == "any":
        return True
    if context_match in {
        "projection",
        "projection_and_parent_size",
        "projection_parent_size_depth",
    }:
        if int(record.sibling_projection_dimension) != int(target.projection_dimension):
            return False
    if context_match in {
        "projection_and_parent_size",
        "projection_parent_size_depth",
    }:
        lower = max(1, int(np.floor(0.75 * target.parent_sample_size)))
        upper = int(np.ceil(1.25 * target.parent_sample_size))
        if not lower <= int(record.n_parent) <= upper:
            return False
    if context_match == "projection_parent_size_depth":
        return int(record_sample.parent_depth) == int(target.parent_depth)
    if context_match not in {
        "any",
        "projection",
        "projection_and_parent_size",
        "projection_parent_size_depth",
    }:
        raise ValueError(f"Unknown context_match={context_match!r}.")
    return True


def _selected_hierarchy_summary(
    *,
    target: SelectedSiblingContext,
    samples: list[SiblingRecordSample],
    context_match: str,
    n_replicates: int,
) -> dict[str, object]:
    matched_records: list[SelectedSiblingRecord] = []
    matched_records_by_simulation: list[list[SelectedSiblingRecord]] = []
    n_simulations_with_match = 0
    for sample in samples:
        matches = [
            record
            for record in sample.selected_records
            if _record_matches_context(record, target, context_match=context_match)
        ]
        if matches:
            n_simulations_with_match += 1
            matched_records_by_simulation.append(matches)
            matched_records.extend(matches)

    candidate_records = sum(sample.candidate_records for sample in samples)
    selected_records = sum(len(sample.selected_records) for sample in samples)
    tested_edges = sum(sample.tested_edges for sample in samples)
    significant_edges = sum(sample.significant_edges for sample in samples)

    if not matched_records:
        return {
            "n_replicates": int(n_replicates),
            "n_candidate_records": int(candidate_records),
            "n_selected_records": int(selected_records),
            "n_matched_records": 0,
            "n_simulations_with_match": 0,
            "simulation_acceptance_rate": 0.0,
            "selected_record_rate": (
                float(selected_records / candidate_records) if candidate_records else 0.0
            ),
            "edge_rejection_rate": (
                float(significant_edges / tested_edges) if tested_edges else 0.0
            ),
            "selected_hierarchy_support_status": (
                "unsupported_no_matched_selected_hierarchy_records"
            ),
            "selected_hierarchy_c_hat": np.nan,
            "selected_hierarchy_c_hat_record_se": np.nan,
            "selected_hierarchy_c_hat_simulation_se": np.nan,
            "selected_hierarchy_c_hat_relative_simulation_se": np.nan,
            "selected_hierarchy_mean_scaled_chi2_p_value": np.nan,
            "selected_hierarchy_blocks_at_alpha": np.nan,
            "selected_hierarchy_mean_statistic": np.nan,
            "selected_hierarchy_median_statistic": np.nan,
            "selected_hierarchy_q95_statistic": np.nan,
            "selected_hierarchy_q99_statistic": np.nan,
            "selected_hierarchy_empirical_tail_p_value": np.nan,
            "selected_hierarchy_empirical_tail_record_se": np.nan,
            "selected_hierarchy_empirical_tail_simulation_se": np.nan,
            "selected_hierarchy_record_tail_resolution": np.nan,
            "selected_hierarchy_matching_simulation_tail_resolution": np.nan,
        }

    def _statistics_and_ratios(
        records: list[SelectedSiblingRecord],
    ) -> tuple[np.ndarray, np.ndarray]:
        statistics = np.asarray(
            [record_sample.record.stat for record_sample in records],
            dtype=float,
        )
        reference_expectations = np.asarray(
            [
                float(
                    record_sample.record.reference_scale * record_sample.record.degrees_of_freedom
                )
                for record_sample in records
            ],
            dtype=float,
        )
        return statistics, statistics / reference_expectations

    statistics, ratios = _statistics_and_ratios(matched_records)
    c_hat = float(np.mean(ratios))
    c_hat_record_se = (
        float(np.std(ratios, ddof=1) / np.sqrt(len(ratios))) if len(ratios) > 1 else np.nan
    )
    simulation_ratio_means = np.asarray(
        [
            float(np.mean(_statistics_and_ratios(records)[1]))
            for records in matched_records_by_simulation
        ],
        dtype=float,
    )
    c_hat_simulation_se = (
        float(np.std(simulation_ratio_means, ddof=1) / np.sqrt(len(simulation_ratio_means)))
        if len(simulation_ratio_means) > 1
        else np.nan
    )
    c_hat_relative_simulation_se = (
        float(c_hat_simulation_se / c_hat)
        if np.isfinite(c_hat_simulation_se) and c_hat != 0.0
        else np.nan
    )
    target_reference_scale = float(target.reference_expectation / target.degrees_of_freedom)
    mean_scaled_statistic = float(target.statistic / (target_reference_scale * c_hat))
    mean_scaled_chi2_p_value = float(
        chi2.sf(mean_scaled_statistic, df=float(target.degrees_of_freedom))
    )
    empirical_tail = float(
        (1 + np.count_nonzero(statistics >= target.statistic)) / (len(statistics) + 1)
    )
    empirical_tail_record_se = float(
        np.sqrt(empirical_tail * (1.0 - empirical_tail) / (len(statistics) + 1))
    )
    simulation_tail_means = np.asarray(
        [
            float(
                np.count_nonzero(_statistics_and_ratios(records)[0] >= target.statistic)
                / len(records)
            )
            for records in matched_records_by_simulation
        ],
        dtype=float,
    )
    empirical_tail_simulation_se = (
        float(np.std(simulation_tail_means, ddof=1) / np.sqrt(len(simulation_tail_means)))
        if len(simulation_tail_means) > 1
        else np.nan
    )
    return {
        "n_replicates": int(n_replicates),
        "n_candidate_records": int(candidate_records),
        "n_selected_records": int(selected_records),
        "n_matched_records": int(len(matched_records)),
        "n_simulations_with_match": int(n_simulations_with_match),
        "simulation_acceptance_rate": float(n_simulations_with_match / n_replicates),
        "selected_record_rate": (
            float(selected_records / candidate_records) if candidate_records else 0.0
        ),
        "edge_rejection_rate": (float(significant_edges / tested_edges) if tested_edges else 0.0),
        "selected_hierarchy_support_status": "matched_selected_hierarchy_records",
        "selected_hierarchy_c_hat": c_hat,
        "selected_hierarchy_c_hat_record_se": c_hat_record_se,
        "selected_hierarchy_c_hat_simulation_se": c_hat_simulation_se,
        "selected_hierarchy_c_hat_relative_simulation_se": (c_hat_relative_simulation_se),
        "selected_hierarchy_mean_scaled_chi2_p_value": mean_scaled_chi2_p_value,
        "selected_hierarchy_blocks_at_alpha": bool(
            mean_scaled_chi2_p_value >= DEFAULT_SIBLING_ALPHA
        ),
        "selected_hierarchy_mean_statistic": float(np.mean(statistics)),
        "selected_hierarchy_median_statistic": float(np.quantile(statistics, 0.5)),
        "selected_hierarchy_q95_statistic": float(np.quantile(statistics, 0.95)),
        "selected_hierarchy_q99_statistic": float(np.quantile(statistics, 0.99)),
        "selected_hierarchy_empirical_tail_p_value": empirical_tail,
        "selected_hierarchy_empirical_tail_record_se": empirical_tail_record_se,
        "selected_hierarchy_empirical_tail_simulation_se": empirical_tail_simulation_se,
        "selected_hierarchy_record_tail_resolution": float(1.0 / (len(statistics) + 1)),
        "selected_hierarchy_matching_simulation_tail_resolution": float(
            1.0 / (len(simulation_tail_means) + 1)
        ),
    }


def _diagnose_case(
    case: dict[str, object],
    *,
    n_replicates: int,
    seed: int,
    target_mode: str,
    context_match: str,
) -> dict[str, object]:
    inputs = prepare_case_inputs(case, ["tbs"])
    feature_space = inputs.metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        raise ValueError("Prepared feature_space metadata must be a FeatureSpace.")
    if feature_space is not None and feature_space.family_label == "continuous":
        raise ValueError(
            "Continuous selected-hierarchy null simulation requires a validated "
            "continuous null covariance generator and an audit-owned continuous "
            "tree-distance contract before use."
        )

    observed_sample, tree_metric = _run_edge_and_sibling_records(
        inputs.data,
        inputs.metadata,
        feature_space,
    )
    target = _choose_observed_target(
        str(inputs.metadata["name"]),
        observed_sample.selected_records,
        target_mode=target_mode,
    )

    rng = np.random.default_rng(int(seed))
    samples: list[SiblingRecordSample] = []
    for _replicate in range(int(n_replicates)):
        simulated = _simulate_null_data(inputs.data, feature_space, rng=rng)
        sample, _tree_metric = _run_edge_and_sibling_records(
            simulated,
            inputs.metadata,
            feature_space,
        )
        samples.append(sample)

    row = {
        "case_id": str(inputs.metadata["name"]),
        "case_category": str(inputs.metadata["category"]),
        "feature_family": "bernoulli" if feature_space is None else feature_space.family_label,
        "tree_distance_metric": tree_metric,
        "target_mode": target_mode,
        "context_match": context_match,
        "observed_parent": target.parent,
        "observed_statistic": target.statistic,
        "observed_reference_expectation": target.reference_expectation,
        "observed_stat_over_reference": target.statistic / target.reference_expectation,
        "observed_raw_p_value": target.raw_p_value,
        "observed_degrees_of_freedom": target.degrees_of_freedom,
        "observed_projection_dimension": target.projection_dimension,
        "observed_parent_sample_size": target.parent_sample_size,
        "observed_parent_depth": target.parent_depth,
        "observed_selected_records": len(observed_sample.selected_records),
        "observed_candidate_records": observed_sample.candidate_records,
        "observed_edge_rejection_rate": (
            observed_sample.significant_edges / observed_sample.tested_edges
            if observed_sample.tested_edges
            else 0.0
        ),
    }
    row.update(
        _selected_hierarchy_summary(
            target=target,
            samples=samples,
            context_match=context_match,
            n_replicates=int(n_replicates),
        )
    )
    return row


def run_selected_hierarchy_null_audit(
    *,
    case_names: list[str],
    output_dir: Path,
    n_replicates: int,
    seed: int,
    target_mode: str,
    context_match: str,
) -> pd.DataFrame:
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started_at = perf_counter()
    for index, case in enumerate(_selected_cases(case_names), start=1):
        print(f"[{index}/{len(case_names)}] {case['name']}", flush=True)
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
        except Exception as exc:
            row = {
                "case_id": str(case["name"]),
                "case_category": str(case["category"]),
                "target_mode": target_mode,
                "context_match": context_match,
                "status": "skip",
                "skip_reason": str(exc),
            }
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "selected_hierarchy_null_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
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
            "Diagnostic-only selected same-data hierarchy null. Production code "
            "still fails closed when internal empirical-null support is absent. "
            "Selected records require an open child-parent edge path. Missing "
            "matched selected-hierarchy records are reported as unsupported "
            "descriptive status, not converted into a calibration fallback."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate selected-hierarchy sibling-null behavior by regenerating "
            "null data, rebuilding the hierarchy, and collecting selected focal "
            "sibling statistics."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument("--n-replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260601)
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
            / f"selected_hierarchy_null_audit_{format_timestamp_utc()}"
        )
    summary = run_selected_hierarchy_null_audit(
        case_names=_parse_csv_list(str(args.case_names)),
        output_dir=output_dir,
        n_replicates=int(args.n_replicates),
        seed=int(args.seed),
        target_mode=str(args.target_mode),
        context_match=str(args.context_match),
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir / 'selected_hierarchy_null_audit_summary.csv'}")


if __name__ == "__main__":
    main()
