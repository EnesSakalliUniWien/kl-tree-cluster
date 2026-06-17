"""Compare current leaf-only MP spectra with legacy internal-barycenter spectra.

This panel runs the current KL spectral context and the commit-era diagnostic
variant that appends descendant internal node distributions to each local
spectral matrix. The legacy variant is intentionally opt-in: internal
barycenters are deterministic tree-filtered leaf averages, not independent
samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

SCHEMA_VERSION = "legacy_internal_spectral_comparison_panel/v1"
STUDY_ROLE = "diagnostic_legacy_internal_spectral_comparison_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "legacy_internal_spectral_comparison_panel"
)

CURRENT_VARIANT = "current_leaf_only_spectral"
LEGACY_VARIANT = "legacy_internal_barycenter_spectral"
CURRENT_METHOD = "kl"
LEGACY_METHOD = "kl_legacy_internal_spectral_diagnostic"

DEFAULT_CASE_NAMES = (
    "overlap_part_4c_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
)

ROWS_OUTPUT = "legacy_internal_spectral_comparison_rows.csv"
PAIRWISE_OUTPUT = "legacy_internal_spectral_comparison_pairwise.csv"
SUMMARY_OUTPUT = "legacy_internal_spectral_comparison_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class LegacyInternalSpectralComparisonConfig:
    """Configuration for the current-vs-legacy internal spectral comparison."""

    output_dir: Path
    suite: str = "binary"
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    data_roles: tuple[str, ...] = ("null", "signal")
    replicates: int = 1
    base_seed: int = 20260613
    edge_alpha: float = 0.001
    sibling_alpha: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", type=parse_names, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--data-roles", type=parse_names, default=("null", "signal"))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _data_role_label(raw_role: str) -> str:
    return "selected_null" if raw_role == "null" else "signal"


def _method_params(variant_id: str) -> tuple[str, dict[str, object]]:
    params: dict[str, object] = {
        "tree_distance_metric": "hamming",
        "tree_linkage_method": "average",
    }
    if variant_id == CURRENT_VARIANT:
        return CURRENT_METHOD, params
    if variant_id == LEGACY_VARIANT:
        params["spectral_include_internal_barycenters"] = True
        return LEGACY_METHOD, params
    raise ValueError(f"Unknown spectral comparison variant: {variant_id!r}.")


def _spectral_context_summary(result_extra: dict[str, object] | None) -> dict[str, object]:
    if not result_extra or "gate_bundle" not in result_extra:
        return {
            "spectral_raw_mp_signal_count_sum": np.nan,
            "spectral_test_dimension_sum": np.nan,
            "spectral_effective_independent_rows_sum": np.nan,
            "spectral_mp_threshold_rows_sum": np.nan,
            "spectral_context_sec": np.nan,
        }
    gate_bundle = result_extra["gate_bundle"]
    spectral_context = gate_bundle.edge_gate_result.spectral_context  # type: ignore[attr-defined]

    def _sum_values(values: object) -> float:
        if not isinstance(values, dict):
            return np.nan
        numeric = [float(value) for value in values.values()]
        return float(np.sum(numeric)) if numeric else 0.0

    timings = getattr(spectral_context, "stage_timings", {})
    return {
        "spectral_raw_mp_signal_count_sum": _sum_values(
            getattr(spectral_context, "raw_mp_signal_counts_by_node", {})
        ),
        "spectral_test_dimension_sum": _sum_values(
            getattr(spectral_context, "test_projection_dimensions_by_node", {})
        ),
        "spectral_effective_independent_rows_sum": _sum_values(
            getattr(spectral_context, "effective_independent_rows_by_node", {})
        ),
        "spectral_mp_threshold_rows_sum": _sum_values(
            getattr(spectral_context, "mp_threshold_rows_by_node", {})
        ),
        "spectral_context_sec": float(timings.get("spectral_context_sec", np.nan)),
    }


def _run_one_variant(
    *,
    data: pd.DataFrame,
    feature_space: object,
    truth_labels: np.ndarray,
    true_clusters: int,
    case_id: str,
    data_role: str,
    replicate: int,
    data_seed: int,
    variant_id: str,
    edge_alpha: float,
    sibling_alpha: float,
) -> tuple[dict[str, object], np.ndarray | None]:
    method_id, params = _method_params(variant_id)
    result = run_clustering_result(
        data,
        method_id=method_id,
        params=params,
        seed=data_seed,
        significance_level=float(sibling_alpha),
        edge_alpha=float(edge_alpha),
        feature_space=feature_space,  # type: ignore[arg-type]
    )
    labels = np.asarray(result.labels) if result.labels is not None else None
    ari = (
        float(adjusted_rand_score(np.asarray(truth_labels, dtype=int), labels))
        if result.status == "ok" and labels is not None and labels.size == len(truth_labels)
        else np.nan
    )
    found_clusters = int(result.found_clusters)
    spectral_summary = _spectral_context_summary(result.extra)
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": data_role,
        "variant_id": variant_id,
        "method_id": method_id,
        "spectral_include_internal_barycenters": bool(
            params.get("spectral_include_internal_barycenters", False)
        ),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "true_clusters": int(true_clusters),
        "found_clusters": found_clusters,
        "ari": ari,
        "exact_cluster_count": bool(
            result.status == "ok" and found_clusters == int(true_clusters)
        ),
        "false_split": bool(
            result.status == "ok" and found_clusters > int(true_clusters)
        ),
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        **spectral_summary,
    }
    return row, labels


def run_comparison_rows(
    config: LegacyInternalSpectralComparisonConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run current and legacy spectral variants and return rows plus pairwise table."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    roles = validate_data_roles(config.data_roles)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)

    rows: list[dict[str, object]] = []
    labels_by_key: dict[tuple[str, str, int, str], np.ndarray | None] = {}
    for case in cases:
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for raw_role in roles:
            data_role = _data_role_label(raw_role)
            for replicate in range(int(config.replicates)):
                data_seed = int(config.base_seed + replicate * 1009)
                data, feature_space, truth_labels, true_clusters = (
                    _generate_data_with_truth(
                        case=case,
                        case_id=case_id,
                        source_family=source_family,
                        feature_representation=feature_representation,
                        n_samples=n_samples,
                        n_features=n_features,
                        n_categories=n_categories,
                        data_role=raw_role,
                        seed=data_seed,
                    )
                )
                for variant_id in (CURRENT_VARIANT, LEGACY_VARIANT):
                    row, labels = _run_one_variant(
                        data=data,
                        feature_space=feature_space,
                        truth_labels=truth_labels,
                        true_clusters=true_clusters,
                        case_id=case_id,
                        data_role=data_role,
                        replicate=replicate,
                        data_seed=data_seed,
                        variant_id=variant_id,
                        edge_alpha=config.edge_alpha,
                        sibling_alpha=config.sibling_alpha,
                    )
                    rows.append(row)
                    labels_by_key[(case_id, data_role, replicate, variant_id)] = labels

    rows_df = pd.DataFrame.from_records(rows)
    pairwise = build_pairwise_rows(rows_df, labels_by_key)
    return rows_df, pairwise


def build_pairwise_rows(
    rows: pd.DataFrame,
    labels_by_key: dict[tuple[str, str, int, str], np.ndarray | None],
) -> pd.DataFrame:
    """Compare legacy rows against current rows by case, role, and replicate."""
    if rows.empty:
        return pd.DataFrame()
    current = rows[rows["variant_id"].astype(str).eq(CURRENT_VARIANT)].copy()
    legacy = rows[rows["variant_id"].astype(str).eq(LEGACY_VARIANT)].copy()
    key = ["case_id", "data_role", "replicate"]
    merged = current.merge(
        legacy,
        on=key,
        how="inner",
        suffixes=("_current", "_legacy"),
        validate="one_to_one",
    )
    pairwise_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        case_id = str(row["case_id"])
        data_role = str(row["data_role"])
        replicate = int(row["replicate"])
        labels_current = labels_by_key[(case_id, data_role, replicate, CURRENT_VARIANT)]
        labels_legacy = labels_by_key[(case_id, data_role, replicate, LEGACY_VARIANT)]
        partition_ari = np.nan
        if (
            labels_current is not None
            and labels_legacy is not None
            and len(labels_current) == len(labels_legacy)
        ):
            partition_ari = float(adjusted_rand_score(labels_current, labels_legacy))
        pairwise_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": data_role,
                "replicate": replicate,
                "current_status": row["status_current"],
                "legacy_status": row["status_legacy"],
                "current_found_clusters": int(row["found_clusters_current"]),
                "legacy_found_clusters": int(row["found_clusters_legacy"]),
                "delta_clusters_legacy_minus_current": int(
                    row["found_clusters_legacy"]
                )
                - int(row["found_clusters_current"]),
                "current_ari": float(row["ari_current"]),
                "legacy_ari": float(row["ari_legacy"]),
                "delta_ari_legacy_minus_current": float(row["ari_legacy"])
                - float(row["ari_current"]),
                "current_false_split": bool(row["false_split_current"]),
                "legacy_false_split": bool(row["false_split_legacy"]),
                "partition_ari_between_variants": partition_ari,
                "delta_raw_mp_signal_count_sum": float(
                    row["spectral_raw_mp_signal_count_sum_legacy"]
                )
                - float(row["spectral_raw_mp_signal_count_sum_current"]),
                "delta_mp_threshold_rows_sum": float(
                    row["spectral_mp_threshold_rows_sum_legacy"]
                )
                - float(row["spectral_mp_threshold_rows_sum_current"]),
                "delta_test_dimension_sum": float(
                    row["spectral_test_dimension_sum_legacy"]
                )
                - float(row["spectral_test_dimension_sum_current"]),
            }
        )
    return pd.DataFrame.from_records(pairwise_rows)


def summarize_comparison(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise current-vs-legacy spectral behavior."""
    if pairwise.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "data_role": "all",
                    "row_count": 0,
                    "comparison_status": "no_pairwise_rows",
                }
            ]
        )
    summaries: list[dict[str, object]] = []
    for data_role, group in pairwise.groupby("data_role", dropna=False):
        delta_ari = pd.to_numeric(
            group["delta_ari_legacy_minus_current"],
            errors="coerce",
        )
        delta_clusters = pd.to_numeric(
            group["delta_clusters_legacy_minus_current"],
            errors="coerce",
        )
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": str(data_role),
                "row_count": int(len(group)),
                "legacy_signal_regression_count": int((delta_ari < -1e-12).sum()),
                "legacy_signal_improvement_count": int((delta_ari > 1e-12).sum()),
                "mean_delta_ari_legacy_minus_current": float(delta_ari.mean()),
                "min_delta_ari_legacy_minus_current": float(delta_ari.min()),
                "mean_delta_clusters_legacy_minus_current": float(delta_clusters.mean()),
                "current_false_split_count": int(group["current_false_split"].sum()),
                "legacy_false_split_count": int(group["legacy_false_split"].sum()),
                "mean_partition_ari_between_variants": float(
                    pd.to_numeric(
                        group["partition_ari_between_variants"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_delta_raw_mp_signal_count_sum": float(
                    pd.to_numeric(
                        group["delta_raw_mp_signal_count_sum"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_delta_mp_threshold_rows_sum": float(
                    pd.to_numeric(
                        group["delta_mp_threshold_rows_sum"],
                        errors="coerce",
                    ).mean()
                ),
                "comparison_status": "diagnostic_only_not_calibration",
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_legacy_internal_spectral_comparison_panel(
    config: LegacyInternalSpectralComparisonConfig,
) -> dict[str, Path]:
    """Run the comparison panel and write CSV outputs plus manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, pairwise = run_comparison_rows(config)
    summary = summarize_comparison(pairwise)

    rows_path = config.output_dir / ROWS_OUTPUT
    pairwise_path = config.output_dir / PAIRWISE_OUTPUT
    summary_path = config.output_dir / SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT

    rows.to_csv(rows_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "suite": config.suite,
            "case_names": list(config.case_names),
            "data_roles": list(config.data_roles),
            "replicates": int(config.replicates),
            "base_seed": int(config.base_seed),
            "edge_alpha": float(config.edge_alpha),
            "sibling_alpha": float(config.sibling_alpha),
        },
        "variants": {
            CURRENT_VARIANT: {
                "method_id": CURRENT_METHOD,
                "spectral_include_internal_barycenters": False,
            },
            LEGACY_VARIANT: {
                "method_id": LEGACY_METHOD,
                "spectral_include_internal_barycenters": True,
            },
        },
        "outputs": {
            "rows": rows_path,
            "pairwise": pairwise_path,
            "summary": summary_path,
            "manifest": manifest_path,
        },
        "n_rows": int(len(rows)),
        "n_pairwise_rows": int(len(pairwise)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return {
        "rows": rows_path,
        "pairwise": pairwise_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    run_legacy_internal_spectral_comparison_panel(
        LegacyInternalSpectralComparisonConfig(
            output_dir=args.output_dir,
            suite=args.suite,
            case_names=tuple(args.case_names),
            data_roles=tuple(args.data_roles),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            edge_alpha=float(args.edge_alpha),
            sibling_alpha=float(args.sibling_alpha),
        )
    )


if __name__ == "__main__":
    main()
