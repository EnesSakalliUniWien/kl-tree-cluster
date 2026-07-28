"""Post-run analysis for enriched selected-edge sibling artifacts.

This diagnostic consumes ``selected_edge_geometry_siblings.csv`` files emitted
by ``benchmarks.validation.statistics.selected_edge_type1_geometry`` and writes
distribution-shape plus selected edge+sibling equation summaries. It is
diagnostic-only and does not install a production calibration rule.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.edge.selected_edge_sibling_null_equation import (
    evaluate_selected_edge_sibling_null_equation,
    summarize_selected_edge_sibling_null_equation,
)
from benchmarks.diagnostics.calibration.statistics.statistic_distribution_shape_panel import (
    normalize_statistic_distribution_records,
    summarize_statistic_distribution_shape,
)
from benchmarks.diagnostics.calibration.traversal.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_selected_edge_sibling_postrun_analysis_not_calibration"
SCHEMA_VERSION = "selected_edge_sibling_postrun_analysis/v1"
REQUIRED_SIBLING_COLUMNS = {
    "case_id",
    "mode",
    "replicate",
    "parent_id",
    "source_family",
    "sibling_raw_stat",
    "sibling_df",
    "left_edge_raw_p",
    "right_edge_raw_p",
    "n_left",
    "n_right",
    "sibling_projection_dimension",
    "edge_path_open",
}
LAPLACIAN_STATUS_COLUMNS = (
    ("sibling_contrast_laplacian_status", "sibling_contrast_covariance_laplacian"),
    ("parent_spectral_laplacian_status", "parent_spectral_covariance_laplacian"),
)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _finite_rows(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=table.index)
    for column in columns:
        mask &= pd.to_numeric(table[column], errors="coerce").notna()
    return table[mask].copy()


def build_distribution_shape_records(siblings: pd.DataFrame) -> pd.DataFrame:
    """Build statistic-distribution records from enriched sibling rows."""
    missing = REQUIRED_SIBLING_COLUMNS - set(siblings.columns)
    if missing:
        raise ValueError(f"Sibling artifact is missing columns: {sorted(missing)!r}.")
    source = _finite_rows(siblings, ("sibling_raw_stat", "sibling_df"))
    records = pd.DataFrame(
        {
            "record_id": (
                source["case_id"].astype(str)
                + ":"
                + source["mode"].astype(str)
                + ":"
                + source["replicate"].astype(str)
                + ":"
                + source["parent_id"].astype(str)
            ),
            "test_family": source["source_family"].astype(str),
            "statistic_context_role": (
                source["mode"].astype(str)
                + "__"
                + source.get(
                    "calibration_support_status",
                    pd.Series("unknown_support", index=source.index),
                ).astype(str)
            ),
            "test_statistic": source["sibling_raw_stat"],
            "degrees_of_freedom": source["sibling_df"],
            "case_id": source["case_id"],
            "replicate_id": source["replicate"],
            "parent_id": source["parent_id"],
            "edge_path_open": source["edge_path_open"],
            "sibling_projection_dimension": source["sibling_projection_dimension"],
            "parent_sample_size": source.get("n_parent", np.nan),
        }
    )
    if "covariance_inferred_df" in source.columns:
        records["alternate_degrees_of_freedom"] = source["covariance_inferred_df"]
    if "covariance_inferred_reference_scale" in source.columns:
        records["alternate_reference_scale"] = source["covariance_inferred_reference_scale"]
    return records


def build_selected_edge_sibling_equation_records(siblings: pd.DataFrame) -> pd.DataFrame:
    """Build selected edge+sibling equation records from sibling rows."""
    missing = REQUIRED_SIBLING_COLUMNS - set(siblings.columns)
    if missing:
        raise ValueError(f"Sibling artifact is missing columns: {sorted(missing)!r}.")
    source = _finite_rows(
        siblings,
        (
            "sibling_raw_stat",
            "left_edge_raw_p",
            "right_edge_raw_p",
            "n_left",
            "n_right",
            "sibling_projection_dimension",
        ),
    )
    return pd.DataFrame(
        {
            "record_id": (
                source["case_id"].astype(str)
                + ":"
                + source["mode"].astype(str)
                + ":"
                + source["replicate"].astype(str)
                + ":"
                + source["parent_id"].astype(str)
            ),
            "sibling_test_statistic": source["sibling_raw_stat"],
            "left_edge_p_value": source["left_edge_raw_p"],
            "right_edge_p_value": source["right_edge_raw_p"],
            "left_child_sample_size": source["n_left"],
            "right_child_sample_size": source["n_right"],
            "sibling_projection_dimension": source["sibling_projection_dimension"],
            "feature_family": source["source_family"].astype(str),
            "edge_path_open": _bool_series(source["edge_path_open"]),
            "is_null_context": True,
            "is_signal_context": False,
            "case_id": source["case_id"],
        }
    )


def _component_context_id(row: pd.Series, columns: Sequence[str]) -> str:
    pieces = []
    for column in columns:
        if column in row.index:
            pieces.append(f"{column}={row[column]}")
    return "|".join(pieces) if pieces else "all"


def build_postrun_production_admissibility_components(
    *,
    siblings: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    equation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build production-admissibility components from post-run evidence.

    The resulting components are intentionally conservative: distribution and
    selected edge+sibling equation summaries are required for production, while
    covariance Laplacian statuses are included as required diagnostics. The
    downstream production-admissibility contract decides whether that evidence
    is production-ready, diagnostic-only, or fail-closed.
    """
    records: list[dict[str, object]] = []
    contract_id = "selected_edge_sibling_postrun"
    for _, row in distribution_summary.iterrows():
        context_id = _component_context_id(
            row,
            ("test_family", "statistic_context_role", "df_bin"),
        )
        records.append(
            {
                "contract_id": contract_id,
                "component_id": f"distribution_shape:{context_id}",
                "component_type": "statistic_distribution_shape_panel",
                "component_status": str(row["distribution_shape_status"]),
                "required_for_production": True,
                "notes": (
                    "Current projected-Wald distribution shape for selected-edge sibling rows."
                ),
            }
        )
    for _, row in equation_summary.iterrows():
        status = str(row["selected_edge_sibling_status"])
        records.append(
            {
                "contract_id": contract_id,
                "component_id": f"selected_edge_sibling_equation:{status}",
                "component_type": "selected_edge_sibling_null_equation",
                "component_status": status,
                "required_for_production": True,
                "notes": ("Conditional empirical selected edge+sibling equation support."),
            }
        )
    for column, component_type in LAPLACIAN_STATUS_COLUMNS:
        if column not in siblings.columns:
            continue
        for status in sorted(siblings[column].dropna().astype(str).unique()):
            records.append(
                {
                    "contract_id": contract_id,
                    "component_id": f"{component_type}:{status}",
                    "component_type": component_type,
                    "component_status": status,
                    "required_for_production": True,
                    "notes": (
                        "Covariance Laplacian geometry summary; diagnostic-only "
                        "unless separately promoted."
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def run_selected_edge_sibling_postrun_analysis(
    *,
    siblings_path: Path,
    output_dir: Path,
    min_null_records: int = 30,
) -> dict[str, Path]:
    """Run post-run sibling distribution and equation analyses."""
    siblings = pd.read_csv(siblings_path)
    distribution_records = build_distribution_shape_records(siblings)
    distribution_rows = normalize_statistic_distribution_records(distribution_records)
    distribution_summary = summarize_statistic_distribution_shape(
        distribution_rows,
        group_columns=("test_family", "statistic_context_role", "df_bin"),
        min_rows=min_null_records,
    )

    equation_records = build_selected_edge_sibling_equation_records(siblings)
    equation_rows, equation_contexts = evaluate_selected_edge_sibling_null_equation(
        equation_records,
        min_null_records=min_null_records,
    )
    equation_summary = summarize_selected_edge_sibling_null_equation(equation_rows)
    production_components = build_postrun_production_admissibility_components(
        siblings=siblings,
        distribution_summary=distribution_summary,
        equation_summary=equation_summary,
    )
    production_component_rows = evaluate_production_admissibility_components(
        production_components,
    )
    production_summary = summarize_production_admissibility_contracts(
        production_component_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_rows_path = output_dir / "sibling_distribution_shape_rows.csv"
    distribution_summary_path = output_dir / "sibling_distribution_shape_summary.csv"
    equation_rows_path = output_dir / "selected_edge_sibling_equation_rows.csv"
    equation_contexts_path = output_dir / "selected_edge_sibling_equation_contexts.csv"
    equation_summary_path = output_dir / "selected_edge_sibling_equation_summary.csv"
    production_components_path = output_dir / "production_admissibility_components.csv"
    production_summary_path = output_dir / "production_admissibility_summary.csv"
    manifest_path = output_dir / "manifest.json"

    distribution_rows.to_csv(distribution_rows_path, index=False)
    distribution_summary.to_csv(distribution_summary_path, index=False)
    equation_rows.to_csv(equation_rows_path, index=False)
    equation_contexts.to_csv(equation_contexts_path, index=False)
    equation_summary.to_csv(equation_summary_path, index=False)
    production_component_rows.to_csv(production_components_path, index=False)
    production_summary.to_csv(production_summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "siblings_path": str(siblings_path),
        "min_null_records": int(min_null_records),
        "outputs": {
            "distribution_rows": str(distribution_rows_path),
            "distribution_summary": str(distribution_summary_path),
            "equation_rows": str(equation_rows_path),
            "equation_contexts": str(equation_contexts_path),
            "equation_summary": str(equation_summary_path),
            "production_components": str(production_components_path),
            "production_summary": str(production_summary_path),
        },
        "interpretation": (
            "Diagnostic post-run analysis over enriched selected-edge sibling "
            "rows. The production summary is conservative: required diagnostic "
            "or failed components prevent production promotion."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "distribution_rows": distribution_rows_path,
        "distribution_summary": distribution_summary_path,
        "equation_rows": equation_rows_path,
        "equation_contexts": equation_contexts_path,
        "equation_summary": equation_summary_path,
        "production_components": production_components_path,
        "production_summary": production_summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siblings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-null-records", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_edge_sibling_postrun_analysis(
        siblings_path=args.siblings,
        output_dir=args.output_dir,
        min_null_records=args.min_null_records,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "build_postrun_production_admissibility_components",
    "build_distribution_shape_records",
    "build_selected_edge_sibling_equation_records",
    "run_selected_edge_sibling_postrun_analysis",
]
