#!/usr/bin/env python3
"""Compare missing-sibling-space policies on the 17-case KL regression suite.

Variants:

- A: current behavior (child-derived sibling k, else JL fallback)
- B: inject parent Gate 2 MP k for missing leaf-leaf sibling pairs, but keep
     the projection basis neutral by not supplying parent PCA projections
- C: same as B, but injected k is derived from effective rank of the parent's
     stored Gate 2 eigenvalues
- E: keep the current sibling path, but replace the capped auto floor used by
     JL fallback with an uncapped global effective-rank floor
- F: keep the current sibling path, but replace the capped auto floor used by
     JL fallback with the participation ratio of the global excess spectrum
- G*: keep the current sibling path, and set the fallback floor to
      max(current stability floor, ceil(c * r_signal)) for a small grid of c

This is an experiment harness only. It does not modify production code.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection.dimension import (
    resolve_minimum_projection_dimension,
    set_resolved_minimum_projection_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection.floor import (
    estimate_projection_dimension_floor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    effective_rank,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree


VariantName = Literal[
    "A_current_jl",
    "B_parent_gate2_random",
    "C_effective_rank_random",
    "E_global_erank_random_uncapped",
    "F_signal_excess_pr_random",
    "G_signal_excess_plus_floor_c1",
    "G_signal_excess_plus_floor_c1_5",
    "G_signal_excess_plus_floor_c2",
    "G_signal_excess_plus_floor_c3",
    "G_signal_excess_plus_floor_c4",
    "G_signal_excess_plus_floor_c6",
    "G_signal_excess_plus_floor_c8",
]

SIGNAL_EXCESS_PLUS_FLOOR_OVERSAMPLING: dict[str, float] = {
    "G_signal_excess_plus_floor_c1": 1.0,
    "G_signal_excess_plus_floor_c1_5": 1.5,
    "G_signal_excess_plus_floor_c2": 2.0,
    "G_signal_excess_plus_floor_c3": 3.0,
    "G_signal_excess_plus_floor_c4": 4.0,
    "G_signal_excess_plus_floor_c6": 6.0,
    "G_signal_excess_plus_floor_c8": 8.0,
}

# Match the live KL benchmark runner, which uses one shared significance level
# for both Gate 2 and Gate 3.
BENCHMARK_SIGNIFICANCE_LEVEL = float(config.SIBLING_ALPHA)

DEFAULT_BASELINE_RUN = (
    PROJECT_ROOT / "benchmarks" / "results" / "run_20260328_103732Z"
)
DEFAULT_CANDIDATE_RUN = (
    PROJECT_ROOT / "benchmarks" / "results" / "run_20260328_113818Z"
)


@dataclass(frozen=True)
class PreparedCase:
    case_num: int
    case_name: str
    case_category: str
    true_clusters: int
    data_df: pd.DataFrame
    y_true: np.ndarray
    tree: PosetTree
    base_annotations: pd.DataFrame
    edge_annotated_df: pd.DataFrame
    sibling_dims_current: dict[str, int] | None
    sibling_pca_projections_current: dict[str, np.ndarray] | None
    sibling_pca_eigenvalues_current: dict[str, np.ndarray] | None


class _PreAnnotatedTreeDecomposition(TreeDecomposition):
    """Use an already-annotated dataframe without rerunning the gate pipeline."""

    def _prepare_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        return annotations_df


def _full_leaf_spectrum(
    leaf_feature_matrix: pd.DataFrame,
) -> tuple[np.ndarray, int, int, float]:
    """Return the global spectrum on the same scale as the auto floor backend."""
    leaf_feature_values = leaf_feature_matrix.values.astype(np.float64)
    n_samples, n_features = leaf_feature_values.shape

    if n_samples < 2 or n_features < 2:
        return np.array([], dtype=np.float64), n_samples, n_features, 1.0

    feature_variances = np.var(leaf_feature_values, axis=0)
    nonconstant_feature_mask = feature_variances > 0
    n_nonconstant_features = int(np.sum(nonconstant_feature_mask))
    if n_nonconstant_features < 2:
        return np.array([], dtype=np.float64), n_samples, n_nonconstant_features, 1.0

    nonconstant_feature_matrix = leaf_feature_values[:, nonconstant_feature_mask]
    if n_samples < n_nonconstant_features:
        feature_means = nonconstant_feature_matrix.mean(axis=0)
        feature_standard_deviations = nonconstant_feature_matrix.std(axis=0, ddof=0)
        feature_standard_deviations[feature_standard_deviations == 0] = 1.0
        standardized_feature_matrix = (
            nonconstant_feature_matrix - feature_means
        ) / feature_standard_deviations
        gram_matrix = standardized_feature_matrix @ standardized_feature_matrix.T
        gram_matrix /= n_nonconstant_features
        spectrum_eigenvalues = np.sort(np.linalg.eigvalsh(gram_matrix))[::-1]
        rho = float(n_samples) / float(n_nonconstant_features)
    else:
        correlation_matrix = np.corrcoef(nonconstant_feature_matrix.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)
        spectrum_eigenvalues = np.sort(np.linalg.eigvalsh(correlation_matrix))[::-1]
        rho = float(n_nonconstant_features) / float(n_samples)

    rho = min(max(rho, np.finfo(np.float64).tiny), 1.0)
    return (
        np.maximum(np.asarray(spectrum_eigenvalues, dtype=np.float64), 0.0),
        n_samples,
        n_nonconstant_features,
        float(rho),
    )


def _mp_density(x: float, rho: float) -> float:
    """Marchenko-Pastur density for 0 < rho <= 1 and unit noise variance."""
    lambda_minus = (1.0 - np.sqrt(rho)) ** 2
    lambda_plus = (1.0 + np.sqrt(rho)) ** 2
    if x <= lambda_minus or x >= lambda_plus:
        return 0.0
    return np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (
        2.0 * np.pi * rho * x
    )


@lru_cache(maxsize=None)
def _mp_positive_median(rho: float) -> float:
    """Median of the positive MP spectrum for 0 < rho <= 1."""
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"Expected aspect ratio in (0, 1], got rho={rho!r}.")

    lambda_minus = (1.0 - np.sqrt(rho)) ** 2
    lambda_plus = (1.0 + np.sqrt(rho)) ** 2
    lo, hi = lambda_minus, lambda_plus

    for _ in range(64):
        mid = (lo + hi) / 2.0
        cdf_at_mid, _ = quad(_mp_density, lambda_minus, mid, args=(rho,))
        if cdf_at_mid < 0.5:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def _estimate_signal_excess_pr_floor(leaf_feature_matrix: pd.DataFrame) -> int:
    """Estimate fallback k from the participation ratio of the excess spectrum."""
    participation_ratio = _estimate_signal_excess_participation_ratio(leaf_feature_matrix)
    return max(2, int(np.ceil(participation_ratio)))


def _estimate_signal_excess_participation_ratio(leaf_feature_matrix: pd.DataFrame) -> float:
    """Estimate signal-only effective rank from the excess spectrum."""
    spectrum_eigenvalues, _, _, rho = _full_leaf_spectrum(leaf_feature_matrix)
    positive_eigenvalues = spectrum_eigenvalues[spectrum_eigenvalues > 0.0]
    if positive_eigenvalues.size == 0:
        return 0.0

    observed_positive_median = float(np.median(positive_eigenvalues))
    if observed_positive_median <= 0.0:
        return 0.0

    lambda_plus_unit = (1.0 + np.sqrt(rho)) ** 2
    mp_positive_median = _mp_positive_median(rho)
    estimated_noise_edge = observed_positive_median * (
        lambda_plus_unit / mp_positive_median
    )

    excess_spectrum = np.maximum(spectrum_eigenvalues - estimated_noise_edge, 0.0)
    excess_sum = float(np.sum(excess_spectrum))
    if excess_sum <= 0.0:
        return 0.0

    excess_sq_sum = float(np.sum(excess_spectrum**2))
    if excess_sq_sum <= 0.0:
        return 0.0

    return float((excess_sum**2) / excess_sq_sum)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare missing-sibling-space policies on the 17 changed KL cases."
        )
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=DEFAULT_BASELINE_RUN,
        help="Baseline benchmark run directory used to derive the 17-case suite.",
    )
    parser.add_argument(
        "--candidate-run",
        type=Path,
        default=DEFAULT_CANDIDATE_RUN,
        help="Candidate benchmark run directory used to derive the 17-case suite.",
    )
    parser.add_argument(
        "--case-nums",
        type=str,
        default="",
        help="Optional comma-separated test_case ids. Default: the 17 changed KL cases.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=(
            "A_current_jl,B_parent_gate2_random,C_effective_rank_random,"
            "E_global_erank_random_uncapped,F_signal_excess_pr_random"
        ),
        help="Comma-separated variant names to run.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the per-case variant table.",
    )
    return parser.parse_args()


def _load_changed_kl_case_nums(baseline_run: Path, candidate_run: Path) -> list[int]:
    baseline_df = pd.read_csv(baseline_run / "full_benchmark_comparison.csv")
    candidate_df = pd.read_csv(candidate_run / "full_benchmark_comparison.csv")
    key = ["test_case", "case_id", "method", "Params"]
    merged = baseline_df.merge(candidate_df, on=key, suffixes=("_old", "_new"))
    changed = merged[
        (
            (merged["ari_old"].round(12) != merged["ari_new"].round(12))
            | (merged["found_clusters_old"] != merged["found_clusters_new"])
            | (merged["Status_old"] != merged["Status_new"])
        )
        & (merged["method"] == "kl")
    ][["test_case"]].drop_duplicates()
    return sorted(int(value) for value in changed["test_case"].tolist())


def _resolve_case_nums(args: argparse.Namespace) -> list[int]:
    if args.case_nums.strip():
        return sorted(
            {
                int(part.strip())
                for part in args.case_nums.split(",")
                if part.strip()
            }
        )
    return _load_changed_kl_case_nums(args.baseline_run, args.candidate_run)


def _resolve_variants(args: argparse.Namespace) -> list[VariantName]:
    allowed = {
        "A_current_jl",
        "B_parent_gate2_random",
        "C_effective_rank_random",
        "E_global_erank_random_uncapped",
        "F_signal_excess_pr_random",
    } | set(SIGNAL_EXCESS_PLUS_FLOOR_OVERSAMPLING)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in allowed]
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return variants  # type: ignore[return-value]


def _prepare_case(case_num: int) -> PreparedCase:
    cases = get_default_test_cases()
    case = cases[case_num - 1]
    data_df, y_true, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])
    linkage_matrix = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)
    base_annotations = tree.annotations_df.copy()
    edge_annotated_df = annotate_child_parent_divergence(
        tree,
        base_annotations.copy(),
        significance_level_alpha=BENCHMARK_SIGNIFICANCE_LEVEL,
        leaf_data=data_df,
    )
    sibling_dims_current = derive_sibling_projection_dimensions_from_child_edge_comparisons(tree, edge_annotated_df)
    sibling_pca_projections_current, sibling_pca_eigenvalues_current = (
        collect_parent_principal_component_inputs_for_sibling_tests(edge_annotated_df, sibling_dims_current)
    )
    return PreparedCase(
        case_num=case_num,
        case_name=str(case["name"]),
        case_category=str(case.get("category", "")),
        true_clusters=int(case.get("n_clusters", len(np.unique(np.asarray(y_true))))),
        data_df=data_df,
        y_true=np.asarray(y_true),
        tree=tree,
        base_annotations=base_annotations,
        edge_annotated_df=edge_annotated_df,
        sibling_dims_current=sibling_dims_current,
        sibling_pca_projections_current=sibling_pca_projections_current,
        sibling_pca_eigenvalues_current=sibling_pca_eigenvalues_current,
    )


def _eligible_leaf_leaf_missing_parents(
    prepared: PreparedCase,
) -> list[str]:
    current_dims = prepared.sibling_dims_current or {}
    eligible: list[str] = []
    for parent in prepared.tree.nodes:
        children = list(prepared.tree.successors(parent))
        if len(children) != 2:
            continue
        if parent in current_dims:
            continue
        if not all(bool(prepared.tree.nodes[child].get("is_leaf", False)) for child in children):
            continue
        eligible.append(parent)
    return eligible


def _build_variant_inputs(
    prepared: PreparedCase,
    variant: VariantName,
) -> tuple[
    dict[str, int] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, int],
]:
    current_dims = dict(prepared.sibling_dims_current or {})
    current_projections = prepared.sibling_pca_projections_current
    current_eigenvalues = prepared.sibling_pca_eigenvalues_current
    injected_parent_k: dict[str, int] = {}

    if variant == "A_current_jl":
        return (
            prepared.sibling_dims_current,
            current_projections,
            current_eigenvalues,
            injected_parent_k,
        )

    if variant == "E_global_erank_random_uncapped":
        return (
            prepared.sibling_dims_current,
            current_projections,
            current_eigenvalues,
            injected_parent_k,
        )

    if variant == "F_signal_excess_pr_random":
        return (
            prepared.sibling_dims_current,
            current_projections,
            current_eigenvalues,
            injected_parent_k,
        )

    if variant in SIGNAL_EXCESS_PLUS_FLOOR_OVERSAMPLING:
        return (
            prepared.sibling_dims_current,
            current_projections,
            current_eigenvalues,
            injected_parent_k,
        )

    edge_spectral_dims = prepared.edge_annotated_df.attrs.get("_spectral_dims", {}) or {}
    edge_pca_eigenvalues = prepared.edge_annotated_df.attrs.get("_pca_eigenvalues", {}) or {}

    for parent in _eligible_leaf_leaf_missing_parents(prepared):
        if variant == "B_parent_gate2_random":
            parent_k = int(edge_spectral_dims.get(parent, 0))
        elif variant == "C_effective_rank_random":
            parent_eigs = edge_pca_eigenvalues.get(parent)
            if parent_eigs is None:
                parent_k = 0
            else:
                parent_k = max(2, int(round(effective_rank(np.asarray(parent_eigs, dtype=float)))))
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        if parent_k > 0:
            current_dims[parent] = parent_k
            injected_parent_k[parent] = parent_k

    return (
        current_dims if current_dims else None,
        current_projections,
        current_eigenvalues,
        injected_parent_k,
    )


def _run_variant(
    prepared: PreparedCase,
    variant: VariantName,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    # Match TreeDecomposition.__init__ so the neutral JL fallback uses the same
    # per-case resolved minimum projection dimension as the live KL runner,
    # unless the variant explicitly overrides the global floor.
    if variant == "E_global_erank_random_uncapped":
        uncapped_floor = estimate_projection_dimension_floor(
            prepared.data_df,
            maximum_dimension_cap=int(prepared.data_df.shape[1]),
        )
        set_resolved_minimum_projection_dimension(int(uncapped_floor))
    elif variant == "F_signal_excess_pr_random":
        signal_excess_floor = _estimate_signal_excess_pr_floor(prepared.data_df)
        set_resolved_minimum_projection_dimension(int(signal_excess_floor))
    elif variant in SIGNAL_EXCESS_PLUS_FLOOR_OVERSAMPLING:
        stability_floor = estimate_projection_dimension_floor(prepared.data_df)
        signal_excess_rank = _estimate_signal_excess_participation_ratio(prepared.data_df)
        oversampling = SIGNAL_EXCESS_PLUS_FLOOR_OVERSAMPLING[variant]
        oversampled_signal_floor = int(np.ceil(oversampling * signal_excess_rank))
        resolved_floor = max(int(stability_floor), int(oversampled_signal_floor))
        set_resolved_minimum_projection_dimension(int(resolved_floor))
    else:
        resolve_minimum_projection_dimension(
            config.PROJECTION_MINIMUM_DIMENSION,
            leaf_data=prepared.data_df,
        )
    sibling_dims, sibling_pca_projections, sibling_pca_eigenvalues, injected_parent_k = (
        _build_variant_inputs(prepared, variant)
    )
    annotated_df = annotate_sibling_divergence(
        tree=prepared.tree,
        annotations_df=prepared.edge_annotated_df.copy(),
        significance_level_alpha=BENCHMARK_SIGNIFICANCE_LEVEL,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca_projections,
        pca_eigenvalues=sibling_pca_eigenvalues,
    )

    decomposer = _PreAnnotatedTreeDecomposition(
        tree=prepared.tree,
        annotations_df=annotated_df.copy(),
        alpha_local=BENCHMARK_SIGNIFICANCE_LEVEL,
        sibling_alpha=BENCHMARK_SIGNIFICANCE_LEVEL,
        leaf_data=prepared.data_df,
    )
    decomposition = decomposer.decompose_tree()
    labels = np.asarray(_labels_from_decomposition(decomposition, prepared.data_df.index.tolist()))
    ari = float(adjusted_rand_score(prepared.y_true, labels))
    metrics: dict[str, object] = {
        "found_clusters": int(decomposition["num_clusters"]),
        "ari": ari,
    }
    return annotated_df, metrics, injected_parent_k


def _projection_counts(audit: dict[str, object]) -> tuple[int, int, int]:
    source_counts = audit.get("projection_dimension_source_counts", {}) or {}
    total_pairs = int(audit.get("total_pairs", 0))
    return (
        int(source_counts.get("spectral", 0)),
        int(source_counts.get("johnson_lindenstrauss_fallback", 0)),
        total_pairs,
    )


def _build_case_variant_rows(case_nums: list[int], variants: list[VariantName]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, case_num in enumerate(case_nums, start=1):
        prepared = _prepare_case(case_num)
        print(
            f"[{index:>2d}/{len(case_nums):>2d}] case {prepared.case_num}: "
            f"{prepared.case_name}"
        )

        baseline_annotations, baseline_metrics, baseline_injected = _run_variant(
            prepared,
            "A_current_jl",
        )
        baseline_row = baseline_annotations
        baseline_bh = baseline_row["Sibling_BH_Different"].astype(bool)
        baseline_audit = baseline_annotations.attrs.get("sibling_divergence_audit", {}) or {}
        baseline_n_spectral, baseline_n_jl, baseline_n_total = _projection_counts(baseline_audit)

        variant_outputs: dict[VariantName, tuple[pd.DataFrame, dict[str, object], dict[str, int]]] = {
            "A_current_jl": (baseline_annotations, baseline_metrics, baseline_injected)
        }
        for variant in variants:
            if variant == "A_current_jl":
                continue
            variant_outputs[variant] = _run_variant(prepared, variant)

        for variant in variants:
            annotated_df, metrics, injected_parent_k = variant_outputs[variant]
            audit = annotated_df.attrs.get("sibling_divergence_audit", {}) or {}
            n_spectral_pairs, n_jl_pairs, n_total_pairs = _projection_counts(audit)
            bh_current = annotated_df["Sibling_BH_Different"].astype(bool)
            shared_nodes = baseline_bh.index.intersection(bh_current.index)
            bh_flip_mask = baseline_bh.loc[shared_nodes] != bh_current.loc[shared_nodes]
            bh_flipped_parents = [str(node) for node in shared_nodes[bh_flip_mask]]
            injected_flip_count = sum(parent in injected_parent_k for parent in bh_flipped_parents)

            found_clusters = int(metrics["found_clusters"])
            true_clusters = int(prepared.true_clusters)
            rows.append(
                {
                    "case_num": prepared.case_num,
                    "case_name": prepared.case_name,
                    "case_category": prepared.case_category,
                    "variant": variant,
                    "true_clusters": true_clusters,
                    "found_clusters": found_clusters,
                    "cluster_delta_vs_A": found_clusters - int(baseline_metrics["found_clusters"]),
                    "ari": float(metrics["ari"]),
                    "ari_delta_vs_A": float(metrics["ari"]) - float(baseline_metrics["ari"]),
                    "exact_k_match": found_clusters == true_clusters,
                    "k_is_one": found_clusters == 1,
                    "over_split": found_clusters > true_clusters,
                    "under_split": found_clusters < true_clusters,
                    "cluster_abs_error": abs(found_clusters - true_clusters),
                    "n_spectral_pairs": n_spectral_pairs,
                    "n_jl_pairs": n_jl_pairs,
                    "n_total_pairs": n_total_pairs,
                    "eligible_leaf_leaf_jl_pairs": len(_eligible_leaf_leaf_missing_parents(prepared)),
                    "injected_parent_random_pairs": len(injected_parent_k),
                    "bh_flip_count_vs_A": int(bh_flip_mask.sum()),
                    "bh_flip_injected_vs_A": int(injected_flip_count),
                    "bh_flipped_parents_vs_A": ";".join(sorted(bh_flipped_parents)),
                    "baseline_n_spectral_pairs": baseline_n_spectral,
                    "baseline_n_jl_pairs": baseline_n_jl,
                    "baseline_n_total_pairs": baseline_n_total,
                }
            )
    return pd.DataFrame(rows)


def _print_summary(results_df: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("Variant Summary")
    print("=" * 100)
    baseline_by_case = (
        results_df[results_df["variant"] == "A_current_jl"]
        .set_index("case_num")[["cluster_abs_error", "exact_k_match", "ari"]]
        .rename(
            columns={
                "cluster_abs_error": "baseline_cluster_abs_error",
                "exact_k_match": "baseline_exact_k_match",
                "ari": "baseline_ari",
            }
        )
    )
    summary_rows: list[dict[str, object]] = []
    for variant, frame in results_df.groupby("variant", sort=False):
        merged = frame.join(baseline_by_case, on="case_num", how="left")
        explosion = frame["found_clusters"] / frame["true_clusters"].clip(lower=1)
        max_explosion_index = explosion.idxmax()
        summary_rows.append(
            {
                "variant": variant,
                "cases": int(len(frame)),
                "exact_k": int(frame["exact_k_match"].sum()),
                "k1": int(frame["k_is_one"].sum()),
                "over_split": int(frame["over_split"].sum()),
                "under_split": int(frame["under_split"].sum()),
                "mean_ari": float(frame["ari"].mean()),
                "exact_k_improved_vs_A": int(
                    ((~merged["baseline_exact_k_match"]) & merged["exact_k_match"]).sum()
                ),
                "exact_k_regressed_vs_A": int(
                    (merged["baseline_exact_k_match"] & (~merged["exact_k_match"])).sum()
                ),
                "cluster_error_improved_vs_A": int(
                    (merged["cluster_abs_error"] < merged["baseline_cluster_abs_error"]).sum()
                ),
                "cluster_error_regressed_vs_A": int(
                    (merged["cluster_abs_error"] > merged["baseline_cluster_abs_error"]).sum()
                ),
                "ari_improved_vs_A": int((frame["ari_delta_vs_A"] > 0).sum()),
                "ari_regressed_vs_A": int((frame["ari_delta_vs_A"] < 0).sum()),
                "total_bh_flips_vs_A": int(frame["bh_flip_count_vs_A"].sum()),
                "max_explosion_factor": float(explosion.loc[max_explosion_index]),
                "max_explosion_case": str(frame.loc[max_explosion_index, "case_name"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print()
    print("=" * 100)
    print("Top Changed Cases By |cluster_delta_vs_A|")
    print("=" * 100)
    top_changed = results_df[results_df["variant"] != "A_current_jl"].copy()
    top_changed = top_changed.reindex(
        top_changed["cluster_delta_vs_A"].abs().sort_values(ascending=False).index
    )
    columns = [
        "variant",
        "case_num",
        "case_name",
        "true_clusters",
        "found_clusters",
        "cluster_delta_vs_A",
        "ari",
        "ari_delta_vs_A",
        "n_spectral_pairs",
        "n_jl_pairs",
        "injected_parent_random_pairs",
        "bh_flip_count_vs_A",
    ]
    print(top_changed.head(20)[columns].to_string(index=False))


def main() -> None:
    args = _parse_args()
    case_nums = _resolve_case_nums(args)
    variants = _resolve_variants(args)
    results_df = _build_case_variant_rows(case_nums, variants)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nWrote CSV to {args.output_csv}")
    _print_summary(results_df)


if __name__ == "__main__":
    main()
