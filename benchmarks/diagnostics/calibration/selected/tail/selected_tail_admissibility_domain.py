"""Summarize admissible and non-admissible selected-tail contexts.

This diagnostic combines selected-ratio tail-law CSVs from selected-hierarchy
studies. It does not create a calibration path; it reports where the existing
external selected-tail support contract is or is not satisfied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

REQUIRED_COLUMNS = {
    "source_family",
    "feature_family",
    "parent_size_bin",
    "sibling_projection_dimension",
    "edge_action_bin",
    "n_records",
    "n_matching_simulations",
    "required_min_matching_simulations",
    "required_min_matched_records",
    "max_exceedance_standard_error",
    "production_tail_law_admissible",
    "tail_law_admissibility_failure_reasons",
    "heldout_exceedance_rate",
    "heldout_exceedance_standard_error",
}


@dataclass(frozen=True)
class SelectedTailRun:
    """Named selected-ratio tail-law table."""

    run_id: str
    path: Path


def parse_run_specs(raw_specs: list[str]) -> tuple[SelectedTailRun, ...]:
    """Parse CLI specs of the form run_id=/path/to/selected_ratio_tail_law.csv."""
    runs: list[SelectedTailRun] = []
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Selected-tail run specs must use run_id=csv_path; got {raw_spec!r}.")
        run_id, raw_path = raw_spec.split("=", maxsplit=1)
        run_id = run_id.strip()
        raw_path = raw_path.strip()
        if not run_id:
            raise ValueError(f"Selected-tail run spec has empty run_id: {raw_spec!r}.")
        if not raw_path:
            raise ValueError(f"Selected-tail run spec has empty path: {raw_spec!r}.")
        runs.append(SelectedTailRun(run_id=run_id, path=Path(raw_path)))
    if not runs:
        raise ValueError("At least one selected-tail run spec is required.")
    return tuple(runs)


def _classify_context(row: pd.Series) -> str:
    if bool(row["production_tail_law_admissible"]):
        return "production_admissible_context"
    failures = {
        failure
        for failure in str(row["tail_law_admissibility_failure_reasons"]).split(";")
        if failure and failure != "nan"
    }
    if failures == {"heldout_exceedance_se_above_contract"}:
        return "support_met_tail_precision_failed"
    if failures <= {
        "matching_simulations_below_tail_resolution_contract",
        "matched_records_below_tail_resolution_contract",
    }:
        return "support_failed_tail_precision_met_or_unchecked"
    if "matching_simulations_below_tail_resolution_contract" in failures:
        return "simulation_support_failed"
    if "matched_records_below_tail_resolution_contract" in failures:
        return "record_support_failed"
    if "heldout_exceedance_se_above_contract" in failures:
        return "tail_precision_failed"
    return "non_admissible_context"


def _validate_selected_tail_table(table: pd.DataFrame, *, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            f"Selected-tail table {path} is missing required columns: {sorted(missing)!r}."
        )


def build_admissibility_domain_table(
    runs: tuple[SelectedTailRun, ...],
) -> pd.DataFrame:
    """Build one row per selected-tail context and run."""
    rows: list[pd.DataFrame] = []
    for run in runs:
        table = pd.read_csv(run.path)
        _validate_selected_tail_table(table, path=run.path)
        table = table.copy()
        table.insert(0, "run_id", run.run_id)
        table.insert(1, "selected_tail_law_csv", str(run.path))
        table["simulation_support_margin"] = (
            table["n_matching_simulations"] - table["required_min_matching_simulations"]
        )
        table["record_support_margin"] = table["n_records"] - table["required_min_matched_records"]
        table["tail_precision_margin"] = (
            table["max_exceedance_standard_error"] - table["heldout_exceedance_standard_error"]
        )
        table["admissibility_class"] = table.apply(_classify_context, axis=1)
        rows.append(table)
    combined = pd.concat(rows, ignore_index=True)
    ordered_columns = [
        "run_id",
        "selected_tail_law_csv",
        "source_family",
        "feature_family",
        "parent_size_bin",
        "edge_action_bin",
        "sibling_projection_dimension",
        "n_matching_simulations",
        "n_records",
        "required_min_matching_simulations",
        "required_min_matched_records",
        "max_exceedance_standard_error",
        "simulation_support_margin",
        "record_support_margin",
        "heldout_exceedance_rate",
        "heldout_exceedance_standard_error",
        "tail_precision_margin",
        "production_tail_law_admissible",
        "admissibility_class",
        "tail_law_admissibility_failure_reasons",
    ]
    return combined[ordered_columns]


def summarize_admissibility_domain(domain: pd.DataFrame) -> pd.DataFrame:
    """Summarize admissibility counts by run, source family, and feature family."""
    grouped = domain.groupby(
        ["run_id", "source_family", "feature_family"],
        dropna=False,
    )
    summary = grouped.agg(
        n_contexts=("production_tail_law_admissible", "size"),
        n_admissible_contexts=("production_tail_law_admissible", "sum"),
        max_matching_simulations=("n_matching_simulations", "max"),
        max_records=("n_records", "max"),
        min_heldout_exceedance_standard_error=(
            "heldout_exceedance_standard_error",
            "min",
        ),
    ).reset_index()
    summary["has_admissible_context"] = summary["n_admissible_contexts"] > 0
    return summary


def nearest_boundary_contexts(domain: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Return contexts nearest the support boundary, admissible rows first."""
    scored = domain.copy()
    scored["min_support_margin"] = np.minimum(
        scored["simulation_support_margin"],
        scored["record_support_margin"],
    )
    scored["absolute_simulation_margin"] = scored["simulation_support_margin"].abs()
    scored = scored.sort_values(
        [
            "production_tail_law_admissible",
            "absolute_simulation_margin",
            "n_matching_simulations",
            "n_records",
        ],
        ascending=[False, True, False, False],
    )
    return scored.head(int(top_n))


def run_selected_tail_admissibility_domain(
    *,
    runs: tuple[SelectedTailRun, ...],
    output_dir: Path,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    """Write combined selected-tail admissibility-domain tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    domain = build_admissibility_domain_table(runs)
    summary = summarize_admissibility_domain(domain)
    boundary = nearest_boundary_contexts(domain, top_n=top_n)
    outputs = {
        "context_admissibility_domain": domain,
        "admissibility_summary_by_run_family": summary,
        "nearest_boundary_contexts": boundary,
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "diagnostic": "selected_tail_admissibility_domain",
        "role": "diagnostic_selected_tail_domain_not_calibration",
        "run_specs": [
            {"run_id": run.run_id, "selected_tail_law_csv": str(run.path)} for run in runs
        ],
        "top_n": int(top_n),
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "note": (
            "Diagnostic-only admissibility-domain summary. Admissible rows "
            "identify contexts where the existing selected-tail support "
            "contract is satisfied; non-admissible rows remain undefined for "
            "production external calibration."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine selected-ratio tail-law CSVs into admissible and "
            "non-admissible selected-tail context summaries."
        )
    )
    parser.add_argument(
        "--run-spec",
        action="append",
        required=True,
        help="Run spec in the form run_id=/path/to/selected_ratio_tail_law.csv.",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"selected_tail_admissibility_domain_{format_timestamp_utc()}"
        )
    outputs = run_selected_tail_admissibility_domain(
        runs=parse_run_specs(list(args.run_spec)),
        output_dir=output_dir,
        top_n=int(args.top_n),
    )
    print(outputs["admissibility_summary_by_run_family"].to_string(index=False))
    print(outputs["nearest_boundary_contexts"].to_string(index=False))
    print(f"Wrote selected-tail admissibility-domain outputs to {output_dir}")


if __name__ == "__main__":
    main()
