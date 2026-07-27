"""Null-law decomposition diagnostics for sibling projected-Wald tests.

This module separates three fixed-topology sibling null laws:

* current adaptive projection learned from the tested sample;
* independent projection learned from the tree-construction sample;
* random fixed orthonormal projection independent of the tested contrast.

It is diagnostic-only and does not install a production calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)

from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.diagnostics.calibration.statistic_distribution_shape_panel import (
    infer_satterthwaite_reference_from_eigenvalues,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    _annotate_edges,
    _build_tree_from_data,
    _case_contract,
    _projection_inputs,
    _select_cases,
    parse_alpha_grid,
    parse_names,
    regenerate_null_case,
)

STUDY_ROLE = "diagnostic_null_law_decomposition_not_calibration"
SCHEMA_VERSION = "null_law_decomposition_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.null_law_decomposition_panel"
PROJECTION_SOURCES = (
    "adaptive_same_sample",
    "independent_tree_sample",
    "random_fixed_orthonormal",
)
DEFAULT_EDGE_ALPHA_GRID = (0.001,)


@dataclass(frozen=True)
class NullLawDecompositionConfig:
    """Runtime contract for null-law decomposition diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "null_law_decomposition_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "null_law_decomposition_summary.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def projection_operator_eigenvalues(projection: np.ndarray) -> np.ndarray:
    """Return eigenvalues of the quadratic operator induced by ``||Pz||^2``."""
    matrix = np.asarray(projection, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"projection must be two-dimensional; got {matrix.shape}.")
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=float)
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return np.sort(eigenvalues)[::-1]


def projection_orthonormality_error(projection: np.ndarray) -> float:
    """Return Frobenius error of row-orthonormality for a projection matrix."""
    matrix = np.asarray(projection, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"projection must be two-dimensional; got {matrix.shape}.")
    if matrix.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(matrix @ matrix.T - np.eye(matrix.shape[0]), ord="fro"))


def _quadratic_reference_fields(projection: np.ndarray) -> dict[str, float]:
    weights = projection_operator_eigenvalues(projection)
    positive = weights[weights > 1e-12]
    if positive.size == 0:
        return {
            "operator_weight_min": float("nan"),
            "operator_weight_max": float("nan"),
            "operator_satterthwaite_scale": float("nan"),
            "operator_satterthwaite_df": float("nan"),
            "operator_df_ratio_to_projection": float("nan"),
        }
    scale, inferred_df = infer_satterthwaite_reference_from_eigenvalues(positive)
    k = float(np.asarray(projection).shape[0])
    return {
        "operator_weight_min": float(np.min(positive)),
        "operator_weight_max": float(np.max(positive)),
        "operator_satterthwaite_scale": float(scale),
        "operator_satterthwaite_df": float(inferred_df),
        "operator_df_ratio_to_projection": float(inferred_df / k) if k > 0 else math.nan,
    }


def _df_bin(value: float) -> str:
    if value <= 1.0:
        return "df_0_1"
    if value <= 2.0:
        return "df_1_2"
    if value <= 4.0:
        return "df_2_4"
    if value <= 8.0:
        return "df_4_8"
    return "df_ge8"


def _random_orthonormal_projection(
    *,
    rng: np.random.Generator,
    width: int,
    projection_dimension: int,
) -> np.ndarray:
    if projection_dimension <= 0:
        return np.zeros((0, int(width)), dtype=float)
    if projection_dimension > width:
        raise ValueError(
            f"projection_dimension={projection_dimension} exceeds width={width}."
        )
    basis, _ = np.linalg.qr(rng.normal(size=(int(width), int(projection_dimension))))
    return basis[:, : int(projection_dimension)].T


def _projection_context(
    *,
    tree_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_space,
    projection_source: str,
    edge_alpha: float,
):
    tree = _build_tree_from_data(tree_data)
    if projection_source == "adaptive_same_sample":
        _annotations, spectral_context = _annotate_edges(
            tree,
            test_data,
            edge_alpha=edge_alpha,
            feature_space=feature_space,
        )
    elif projection_source in {"independent_tree_sample", "random_fixed_orthonormal"}:
        _annotations, spectral_context = _annotate_edges(
            tree,
            tree_data,
            edge_alpha=edge_alpha,
            feature_space=feature_space,
        )
        tree.populate_node_divergences(test_data, feature_space=feature_space)
    else:
        raise ValueError(f"Unknown projection_source: {projection_source!r}.")
    return tree, _projection_inputs(tree, spectral_context)


def _rows_for_projection_source(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    replicate: int,
    data_seed: int,
    tree_seed: int,
    edge_alpha: float,
    sibling_alpha: float,
    tree_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_space,
    projection_source: str,
    run_id: str,
) -> list[dict[str, object]]:
    tree, (projection_dimensions, parent_projections, parent_eigenvalues) = (
        _projection_context(
            tree_data=tree_data,
            test_data=test_data,
            feature_space=feature_space,
            projection_source=projection_source,
            edge_alpha=edge_alpha,
        )
    )
    rng = np.random.default_rng(
        int(data_seed)
        + {
            "adaptive_same_sample": 11_003,
            "independent_tree_sample": 23_009,
            "random_fixed_orthonormal": 37_019,
        }[projection_source]
    )
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None or parent not in projection_dimensions:
            continue
        k = int(projection_dimensions[parent])
        if k <= 0:
            continue
        left, right = children
        left_distribution = np.asarray(tree.nodes[left]["distribution"], dtype=float)
        right_distribution = np.asarray(tree.nodes[right]["distribution"], dtype=float)
        left_n = float(tree.nodes[left]["leaf_count"])
        right_n = float(tree.nodes[right]["leaf_count"])
        contrast = build_contrast_covariance(
            left_distribution,
            right_distribution,
            left_n,
            right_n,
            comparison="sibling",
            feature_space=feature_space,
        )
        z = contrast.whitened_vector()
        if projection_source == "random_fixed_orthonormal":
            projection = _random_orthonormal_projection(
                rng=rng,
                width=int(z.shape[0]),
                projection_dimension=k,
            )
            parent_eigenvalue_min = math.nan
            parent_eigenvalue_max = math.nan
        else:
            projection = np.asarray(parent_projections[parent], dtype=float)[:k]
            eigenvalues = np.asarray(parent_eigenvalues[parent], dtype=float)[:k]
            parent_eigenvalue_min = float(np.min(eigenvalues)) if eigenvalues.size else math.nan
            parent_eigenvalue_max = float(np.max(eigenvalues)) if eigenvalues.size else math.nan
        if projection.shape[1] != z.shape[0]:
            raise ValueError(
                f"projection width {projection.shape[1]} must match z width {z.shape[0]}."
            )
        statistic = float(np.sum((projection @ z) ** 2))
        p_value = float(chi2.sf(statistic, df=float(k)))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "run_id": run_id,
                "case_id": case_id,
                "mode": "fixed_topology",
                "projection_source": projection_source,
                "replicate": int(replicate),
                "parent_id": str(parent),
                "left_child_id": str(left),
                "right_child_id": str(right),
                "source_family": source_family,
                "feature_representation": feature_representation,
                "edge_alpha": float(edge_alpha),
                "sibling_alpha": float(sibling_alpha),
                "data_seed": int(data_seed),
                "tree_seed": int(tree_seed),
                "sibling_df": float(k),
                "df_bin": _df_bin(float(k)),
                "test_statistic": statistic,
                "p_value": p_value,
                "tail_reject_at_0.05": bool(p_value <= 0.05),
                "z_norm": float(np.linalg.norm(z)),
                "projection_orthonormality_error": projection_orthonormality_error(
                    projection
                ),
                "parent_eigenvalue_min": parent_eigenvalue_min,
                "parent_eigenvalue_max": parent_eigenvalue_max,
                **_quadratic_reference_fields(projection),
            }
        )
    return rows


def _null_law_status(
    group: pd.DataFrame,
    *,
    alpha: float,
    tolerance: float,
    min_rows: int,
) -> str:
    if group.shape[0] < int(min_rows):
        return "null_law_insufficient_rows"
    tail_rate = float(group["tail_reject_at_0.05"].mean())
    projection_source = str(group["projection_source"].iloc[0])
    if abs(tail_rate - float(alpha)) <= float(tolerance):
        return "null_law_fixed_projection_candidate"
    if projection_source == "adaptive_same_sample" and tail_rate > float(alpha) + float(
        tolerance
    ):
        return "null_law_adaptive_projection_tail_inflated"
    return "null_law_fixed_projection_tail_misaligned"


def summarize_null_law_decomposition_rows(
    rows: pd.DataFrame,
    *,
    alpha: float = 0.05,
    tolerance: float = 0.05,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize empirical tail behavior by projection source and df bin."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    group_columns = ("case_id", "mode", "projection_source", "source_family", "df_bin")
    for key, group in rows.groupby(list(group_columns), sort=True):
        p_values = pd.to_numeric(group["p_value"], errors="coerce")
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_rows": int(group.shape[0]),
                "tail_rate_at_0.05": float(group["tail_reject_at_0.05"].mean()),
                "p_value_q50": float(p_values.quantile(0.50)),
                "p_value_q05": float(p_values.quantile(0.05)),
                "test_statistic_q50": float(group["test_statistic"].quantile(0.50)),
                "projection_orthonormality_error_max": float(
                    group["projection_orthonormality_error"].max()
                ),
                "operator_weight_min_q50": float(group["operator_weight_min"].quantile(0.50)),
                "operator_weight_max_q50": float(group["operator_weight_max"].quantile(0.50)),
                "operator_satterthwaite_scale_q50": float(
                    group["operator_satterthwaite_scale"].quantile(0.50)
                ),
                "operator_satterthwaite_df_q50": float(
                    group["operator_satterthwaite_df"].quantile(0.50)
                ),
                "null_law_status": _null_law_status(
                    group,
                    alpha=alpha,
                    tolerance=tolerance,
                    min_rows=min_rows,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def build_null_law_production_components(summary: pd.DataFrame) -> pd.DataFrame:
    """Build production-admissibility components from null-law summaries."""
    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        context = (
            f"case_id={row['case_id']}|mode={row['mode']}|"
            f"projection_source={row['projection_source']}|"
            f"source_family={row['source_family']}|df_bin={row['df_bin']}"
        )
        records.append(
            {
                "contract_id": "sibling_projected_wald_null_law",
                "component_id": f"null_law:{context}",
                "component_type": "null_law_decomposition_panel",
                "component_status": str(row["null_law_status"]),
                "required_for_production": True,
                "notes": "Diagnostic status for sibling projected-Wald null law.",
            }
        )
    return pd.DataFrame.from_records(records)


def run_null_law_decomposition_panel(
    config: NullLawDecompositionConfig,
) -> dict[str, Path]:
    """Run null-law decomposition diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "null_law_decomposition__"
        f"{config.suite}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    rows: list[dict[str, object]] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for replicate in range(int(config.replicates)):
            for edge_alpha in config.edge_alphas:
                data_seed = (
                    int(config.base_seed)
                    + replicate * 1009
                    + int(float(edge_alpha) * 1_000_000)
                )
                tree_seed = int(config.base_seed) + replicate * 917 + 17
                tree_data, tree_metadata = regenerate_null_case(
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    seed=tree_seed,
                )
                test_data, _test_metadata = regenerate_null_case(
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    seed=data_seed,
                )
                feature_space = tree_metadata["feature_space"]
                for projection_source in PROJECTION_SOURCES:
                    rows.extend(
                        _rows_for_projection_source(
                            case_id=case_id,
                            source_family=source_family,
                            feature_representation=feature_representation,
                            replicate=replicate,
                            data_seed=data_seed,
                            tree_seed=tree_seed,
                            edge_alpha=float(edge_alpha),
                            sibling_alpha=float(config.sibling_alpha),
                            tree_data=tree_data,
                            test_data=test_data,
                            feature_space=feature_space,
                            projection_source=projection_source,
                            run_id=run_id,
                        )
                    )
    row_table = pd.DataFrame.from_records(rows)
    summary = summarize_null_law_decomposition_rows(row_table)
    components = build_null_law_production_components(summary)
    production_rows = evaluate_production_admissibility_components(components)
    production_summary = summarize_production_admissibility_contracts(production_rows)

    row_table.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    production_rows.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": config.suite,
        "case_names": list(config.case_names),
        "edge_alphas": list(config.edge_alphas),
        "sibling_alpha": float(config.sibling_alpha),
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "projection_sources": list(PROJECTION_SOURCES),
        "n_rows": int(row_table.shape[0]),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "production_components": str(config.production_components_path),
            "production_summary": str(config.production_summary_path),
        },
        "interpretation": (
            "Diagnostic decomposition of the sibling projected-Wald null law. "
            "Adaptive same-sample projection is expected to fail the fixed-subspace "
            "chi-square reference even when independent fixed projections pass."
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "production_components": config.production_components_path,
        "production_summary": config.production_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument(
        "--edge-alphas",
        default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260613)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_null_law_decomposition_panel(
        NullLawDecompositionConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
            sibling_alpha=float(args.sibling_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
        )
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "NullLawDecompositionConfig",
    "build_null_law_production_components",
    "projection_operator_eigenvalues",
    "projection_orthonormality_error",
    "run_null_law_decomposition_panel",
    "summarize_null_law_decomposition_rows",
]
