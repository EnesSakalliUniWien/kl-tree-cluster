"""Compare pooled versus within-child discrete sibling covariance models.

This diagnostic keeps the selected tree, branch-time multiplier, and fixed
coordinate/global p-value summaries fixed.  It only changes the covariance model
used to whiten binary/categorical sibling contrasts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.generators.case_data_contracts import one_hot_encode_categorical
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    compute_sibling_branch_length_sum,
    extract_branch_length_observation,
)
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    fixed_coordinate_bh_p_value,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)
from tree_break_selection.tree.poset_tree import PosetTree

DEFAULT_CASES = (
    "binary_null_small",
    "binary_null_medium",
    "binary_2clusters",
    "binary_moderate_4c",
    "overlap_heavy_4c_small_feat",
    "overlap_unbal_4c_small",
    "cat_null_3cat_120x60",
    "cat_clear_3cat_4c",
    "cat_mod_3cat_4c",
    "cat_highcard_20cat_4c",
    "cat_overlap_3cat_4c",
    "cat_simplex_face_rare_20cat",
    "cat_dirichlet_overdispersed_20cat",
)


@dataclass(frozen=True)
class PreparedDiagnosticCase:
    case: dict[str, object]
    data: pd.DataFrame
    labels: np.ndarray
    feature_space: FeatureSpace
    distance_condensed: np.ndarray
    source_family: str
    feature_representation: str


def _regularized_binary_probabilities(
    probabilities: np.ndarray,
    sample_size: float,
    *,
    pseudo_count: float,
) -> np.ndarray:
    return (sample_size * probabilities + pseudo_count) / (
        sample_size + 2.0 * pseudo_count
    )


def _regularized_simplex_probabilities(
    probabilities: np.ndarray,
    sample_size: float,
    *,
    pseudo_count: float,
) -> np.ndarray:
    category_count = probabilities.shape[0]
    return (sample_size * probabilities + pseudo_count) / (
        sample_size + category_count * pseudo_count
    )


def _time_multiplier(
    branch_length_sum: float | None,
    mean_branch_length: float | None,
) -> float:
    if branch_length_sum is None or branch_length_sum <= 0.0:
        return 1.0
    if mean_branch_length is None or mean_branch_length <= 0.0:
        return 1.0
    return 1.0 + float(branch_length_sum) / (2.0 * float(mean_branch_length))


def _current_discrete_z(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
) -> np.ndarray:
    tree_time_normalizer = (
        None if mean_branch_length is None else 2.0 * float(mean_branch_length)
    )
    return compute_whitened_wald_contrast(
        first,
        second,
        first_sample_size,
        second_sample_size,
        comparison="sibling",
        feature_space=feature_space,
        tree_time=branch_length_sum,
        tree_time_normalizer=tree_time_normalizer,
    )


def _binary_within_z(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
    pseudo_count: float,
    pooled_guard_weight: float,
    ridge: float = 1e-12,
) -> np.ndarray:
    column_indices = np.asarray(
        [block.column_indices[0] for block in feature_space.blocks],
        dtype=np.int64,
    )
    first_values = first[column_indices]
    second_values = second[column_indices]
    first_reg = _regularized_binary_probabilities(
        first_values,
        first_sample_size,
        pseudo_count=pseudo_count,
    )
    second_reg = _regularized_binary_probabilities(
        second_values,
        second_sample_size,
        pseudo_count=pseudo_count,
    )
    within_variance = (
        first_reg * (1.0 - first_reg) / float(first_sample_size)
        + second_reg * (1.0 - second_reg) / float(second_sample_size)
    )

    pooled = (
        float(first_sample_size) * first_values
        + float(second_sample_size) * second_values
    ) / (float(first_sample_size) + float(second_sample_size))
    pooled_variance = pooled * (1.0 - pooled) * (
        1.0 / float(first_sample_size) + 1.0 / float(second_sample_size)
    )
    variance = within_variance + float(pooled_guard_weight) * pooled_variance
    variance = variance * _time_multiplier(branch_length_sum, mean_branch_length)
    variance = variance + float(ridge)
    return (first_values - second_values) / np.sqrt(variance)


def _simplex_covariance_reduced(probabilities: np.ndarray) -> np.ndarray:
    reduced = probabilities[:-1]
    covariance = -np.outer(reduced, reduced)
    diagonal = np.arange(reduced.shape[0])
    covariance[diagonal, diagonal] += reduced
    return covariance


def _categorical_within_z(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
    pseudo_count: float,
    pooled_guard_weight: float,
    ridge: float = 1e-12,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    multiplier = _time_multiplier(branch_length_sum, mean_branch_length)
    for block in feature_space.blocks:
        indices = np.asarray(block.column_indices, dtype=np.int64)
        first_prob = first[indices]
        second_prob = second[indices]
        first_reg = _regularized_simplex_probabilities(
            first_prob,
            first_sample_size,
            pseudo_count=pseudo_count,
        )
        second_reg = _regularized_simplex_probabilities(
            second_prob,
            second_sample_size,
            pseudo_count=pseudo_count,
        )
        pooled = (
            float(first_sample_size) * first_prob
            + float(second_sample_size) * second_prob
        ) / (float(first_sample_size) + float(second_sample_size))
        covariance = (
            _simplex_covariance_reduced(first_reg) / float(first_sample_size)
            + _simplex_covariance_reduced(second_reg) / float(second_sample_size)
            + float(pooled_guard_weight)
            * _simplex_covariance_reduced(pooled)
            * (1.0 / float(first_sample_size) + 1.0 / float(second_sample_size))
        )
        covariance = covariance * multiplier
        diagonal = np.arange(covariance.shape[0])
        covariance[diagonal, diagonal] += float(ridge)
        contrast = first_prob[:-1] - second_prob[:-1]
        blocks.append(np.linalg.solve(np.linalg.cholesky(covariance), contrast))
    return np.concatenate(blocks)


def _within_child_z(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
    pseudo_count: float,
    pooled_guard_weight: float,
) -> np.ndarray:
    if feature_space.family_label == "bernoulli":
        return _binary_within_z(
            first,
            second,
            first_sample_size,
            second_sample_size,
            feature_space,
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
            pseudo_count=pseudo_count,
            pooled_guard_weight=pooled_guard_weight,
        )
    if feature_space.family_label == "categorical":
        return _categorical_within_z(
            first,
            second,
            first_sample_size,
            second_sample_size,
            feature_space,
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
            pseudo_count=pseudo_count,
            pooled_guard_weight=pooled_guard_weight,
        )
    raise ValueError(
        f"Discrete diagnostic supports Bernoulli/categorical spaces; got {feature_space.family_label!r}."
    )


def _pvalue_summary(z: np.ndarray) -> dict[str, float]:
    vector = np.asarray(z, dtype=np.float64)
    coordinate_p = chi2.sf(vector * vector, df=1.0)
    return {
        "fixed_coordinate_bh": fixed_coordinate_bh_p_value(vector),
        "global_chi_square": float(chi2.sf(float(np.dot(vector, vector)), df=vector.size)),
        "min_coordinate_p": float(np.min(coordinate_p)) if coordinate_p.size else 1.0,
        "wald_norm2": float(np.dot(vector, vector)),
        "max_abs_z": float(np.max(np.abs(vector))) if vector.size else 0.0,
    }


def _label_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts, strict=True)}


def _context_label(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
) -> str:
    parent_labels = np.concatenate([left_labels, right_labels])
    if np.unique(parent_labels).size == 1:
        return "strict_null"
    left_unique = set(np.unique(left_labels).tolist())
    right_unique = set(np.unique(right_labels).tolist())
    if len(left_unique) == 1 and len(right_unique) == 1 and left_unique.isdisjoint(right_unique):
        return "pure_signal"
    return "mixed_or_overlap"


def _log10_p(value: float) -> float:
    return float(-np.log10(max(float(value), np.finfo(float).tiny)))


def _build_tree(inputs: PreparedDiagnosticCase) -> PosetTree:
    linkage_matrix = linkage(inputs.distance_condensed, method="average")
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=inputs.data.index.tolist())
    tree.populate_node_divergences(inputs.data, feature_space=inputs.feature_space)
    return tree


def _diagnose_case(
    inputs: PreparedDiagnosticCase,
    *,
    root_only: bool,
) -> list[dict[str, object]]:
    tree = _build_tree(inputs)
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    label_series = pd.Series(inputs.labels, index=inputs.data.index)
    mean_branch_length = compute_mean_branch_length(tree)
    root = tree.root()
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        if root_only and parent != root:
            continue
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            continue
        left, right = children
        left_branch = extract_branch_length_observation(tree, parent, left)
        right_branch = extract_branch_length_observation(tree, parent, right)
        branch_sum = compute_sibling_branch_length_sum(left_branch, right_branch)
        first = np.asarray(tree.nodes[left]["distribution"], dtype=np.float64)
        second = np.asarray(tree.nodes[right]["distribution"], dtype=np.float64)
        first_n = float(tree.nodes[left]["leaf_count"])
        second_n = float(tree.nodes[right]["leaf_count"])

        current = _pvalue_summary(
            _current_discrete_z(
                first,
                second,
                first_n,
                second_n,
                inputs.feature_space,
                branch_length_sum=branch_sum,
                mean_branch_length=mean_branch_length,
            )
        )
        within_jeffreys = _pvalue_summary(
            _within_child_z(
                first,
                second,
                first_n,
                second_n,
                inputs.feature_space,
                branch_length_sum=branch_sum,
                mean_branch_length=mean_branch_length,
                pseudo_count=0.5,
                pooled_guard_weight=0.0,
            )
        )
        guarded_25 = _pvalue_summary(
            _within_child_z(
                first,
                second,
                first_n,
                second_n,
                inputs.feature_space,
                branch_length_sum=branch_sum,
                mean_branch_length=mean_branch_length,
                pseudo_count=0.5,
                pooled_guard_weight=0.25,
            )
        )

        left_leaf_labels = sorted(descendant_sets[left])
        right_leaf_labels = sorted(descendant_sets[right])
        left_truth = label_series.loc[left_leaf_labels].to_numpy()
        right_truth = label_series.loc[right_leaf_labels].to_numpy()
        row = {
            "case_id": inputs.case["name"],
            "suite_category": inputs.case.get("category", ""),
            "generator": inputs.case.get("generator", ""),
            "source_family": inputs.source_family,
            "feature_representation": inputs.feature_representation,
            "feature_family": inputs.feature_space.family_label,
            "samples": int(inputs.data.shape[0]),
            "raw_features": int(inputs.data.shape[1]),
            "contrast_dimension": int(inputs.feature_space.contrast_dimension),
            "parent": parent,
            "left": left,
            "right": right,
            "is_root": bool(parent == root),
            "left_n": int(first_n),
            "right_n": int(second_n),
            "parent_n": int(first_n + second_n),
            "left_branch_length": left_branch,
            "right_branch_length": right_branch,
            "branch_length_sum": branch_sum,
            "mean_branch_length": mean_branch_length,
            "time_multiplier": _time_multiplier(branch_sum, mean_branch_length),
            "truth_context": _context_label(left_truth, right_truth),
            "left_truth_counts": _label_counts(left_truth),
            "right_truth_counts": _label_counts(right_truth),
        }
        for prefix, summary in (
            ("current_pooled", current),
            ("within_jeffreys", within_jeffreys),
            ("guarded_within_25pct_pooled", guarded_25),
        ):
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_neglog10_bh"] = _log10_p(summary["fixed_coordinate_bh"])
            row[f"{prefix}_neglog10_global"] = _log10_p(summary["global_chi_square"])
        rows.append(row)
    return rows


def _benchmark_cases_by_name() -> dict[str, dict[str, object]]:
    cases = {}
    for suite in ("binary", "categorical"):
        for case in get_test_cases_by_suite(suite):
            cases[str(case["name"])] = case.copy()
    return cases


def _prepare_benchmark_case(case: dict[str, object]) -> PreparedDiagnosticCase:
    selected_methods = ["tbs"]
    inputs = prepare_case_inputs(case, selected_methods)
    feature_space = inputs.metadata.get("feature_space")
    if feature_space is None:
        feature_space = bernoulli_feature_space_from_columns(tuple(inputs.data.columns))
    if not isinstance(feature_space, FeatureSpace):
        raise TypeError("metadata feature_space must be a FeatureSpace.")
    distance_condensed = inputs.distance_condensed
    if distance_condensed is None:
        distance_condensed = pdist(inputs.data.to_numpy(dtype=float), metric="hamming")
    return PreparedDiagnosticCase(
        case=case,
        data=inputs.data,
        labels=inputs.labels,
        feature_space=feature_space,
        distance_condensed=distance_condensed,
        source_family=str(inputs.metadata["source_family"]),
        feature_representation=str(inputs.metadata["feature_representation"]),
    )


def _prepare_categorical_null_case() -> PreparedDiagnosticCase:
    rng = np.random.default_rng(20260623)
    n_samples = 120
    n_features = 60
    n_categories = 3
    matrix = rng.integers(0, n_categories, size=(n_samples, n_features))
    sample_names = [f"cat_null_{i}" for i in range(n_samples)]
    data, _raw_dimension, feature_space = one_hot_encode_categorical(
        matrix,
        n_categories,
        sample_names,
    )
    from scipy.spatial.distance import pdist

    return PreparedDiagnosticCase(
        case={
            "name": "cat_null_3cat_120x60",
            "category": "diagnostic_categorical_null",
            "generator": "diagnostic_categorical_null",
            "seed": 20260623,
        },
        data=data,
        labels=np.zeros(n_samples, dtype=int),
        feature_space=feature_space,
        distance_condensed=pdist(data.to_numpy(dtype=float), metric="hamming"),
        source_family="categorical_multinomial_null",
        feature_representation="categorical_one_hot",
    )


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    group_columns = ["case_id", "feature_family", "truth_context"]
    for keys, group in rows.groupby(group_columns, dropna=False):
        current_open = group["current_pooled_fixed_coordinate_bh"] <= 0.01
        within_open = group["within_jeffreys_fixed_coordinate_bh"] <= 0.01
        guarded_open = group["guarded_within_25pct_pooled_fixed_coordinate_bh"] <= 0.01
        grouped.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "nodes": int(len(group)),
                "current_open_at_001": int(current_open.sum()),
                "within_jeffreys_open_at_001": int(within_open.sum()),
                "guarded_25pct_open_at_001": int(guarded_open.sum()),
                "min_current_bh": float(group["current_pooled_fixed_coordinate_bh"].min()),
                "min_within_jeffreys_bh": float(
                    group["within_jeffreys_fixed_coordinate_bh"].min()
                ),
                "min_guarded_25pct_bh": float(
                    group["guarded_within_25pct_pooled_fixed_coordinate_bh"].min()
                ),
                "median_current_neglog10_bh": float(
                    group["current_pooled_neglog10_bh"].median()
                ),
                "median_within_neglog10_bh": float(
                    group["within_jeffreys_neglog10_bh"].median()
                ),
                "median_guarded_25pct_neglog10_bh": float(
                    group["guarded_within_25pct_pooled_neglog10_bh"].median()
                ),
            }
        )
    return pd.DataFrame(grouped).sort_values(group_columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw/assets/benchmark-results/discrete_covariance_guard_check_20260623"),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case ids to run. Include cat_null_3cat_120x60 for the synthetic categorical null.",
    )
    parser.add_argument(
        "--case-set",
        choices=("default", "all-suite"),
        default="default",
        help="Use the explicit --cases list or every binary/categorical suite case.",
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="Only diagnose the selected tree root sibling contrast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cases = _benchmark_cases_by_name()
    requested_cases = (
        sorted(benchmark_cases)
        if args.case_set == "all-suite"
        else [str(case_id) for case_id in args.cases]
    )
    prepared_cases = []
    for case_id in requested_cases:
        if case_id == "cat_null_3cat_120x60":
            prepared_cases.append(_prepare_categorical_null_case())
            continue
        if case_id not in benchmark_cases:
            raise KeyError(f"Unknown case id {case_id!r}.")
        prepared_cases.append(_prepare_benchmark_case(benchmark_cases[case_id]))

    all_rows: list[dict[str, object]] = []
    for prepared in prepared_cases:
        print(f"diagnosing {prepared.case['name']} ...", flush=True)
        all_rows.extend(_diagnose_case(prepared, root_only=bool(args.root_only)))

    rows = pd.DataFrame(all_rows)
    summary = _summarize(rows)
    rows.to_csv(args.output_dir / "node_covariance_comparison.csv", index=False)
    summary.to_csv(args.output_dir / "case_context_summary.csv", index=False)
    manifest = {
        "generated_at": generated_at,
        "command": " ".join(sys.argv),
        "output_dir": str(args.output_dir),
        "case_set": str(args.case_set),
        "cases": requested_cases,
        "root_only": bool(args.root_only),
        "node_rows": int(len(rows)),
        "summary_rows": int(len(summary)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} node rows to {args.output_dir}")
    print(summary.to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
