"""Shared runner mechanics for overlap calibration diagnostic panels."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from benchmarks.diagnostics.calibration.selected.family.selected_family_traversal_panel import (
    _build_node_decisions,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_panel import (
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.statistics.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)

OverlapRowBuilder = Callable[
    ...,
    pd.DataFrame,
]
OverlapSummaryBuilder = Callable[[pd.DataFrame], pd.DataFrame]
OverlapNodeRowBuilder = Callable[..., dict[str, object] | None]


def run_binary_overlap_panel(
    *,
    config: Any,
    row_columns: tuple[str, ...],
    row_builder: OverlapRowBuilder,
    summarize_rows: OverlapSummaryBuilder,
    schema_version: str,
    study_role: str,
    generated_by: str,
    unsupported_family_label: str,
    manifest_extra: Mapping[str, object] | None = None,
    prepare_rows: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Run a binary-template overlap diagnostic panel and write its outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive.")
    validate_data_roles(config.data_roles)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    row_frames: list[pd.DataFrame] = []
    skipped_cases: list[dict[str, object]] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        try:
            (
                case_id,
                source_family,
                feature_representation,
                n_samples,
                n_features,
                n_categories,
            ) = _case_contract(case)
        except ValueError as exc:
            case_id = str(case.get("case_id", case.get("name", "unknown_case")))
            if not bool(config.skip_unsupported_cases):
                raise
            skipped_cases.append({"case_id": case_id, "reason": str(exc)})
            continue
        if source_family != "binary_template":
            if not bool(config.skip_unsupported_cases):
                raise ValueError(f"Unsupported {unsupported_family_label} family: {source_family}")
            skipped_cases.append(
                {
                    "case_id": case_id,
                    "reason": f"unsupported source_family={source_family}",
                }
            )
            continue
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate
            for data_role in config.data_roles:
                row_frames.append(
                    row_builder(
                        case=case,
                        case_id=case_id,
                        source_family=source_family,
                        feature_representation=feature_representation,
                        n_samples=n_samples,
                        n_features=n_features,
                        n_categories=n_categories,
                        data_role=data_role,
                        replicate=replicate,
                        data_seed=data_seed,
                        config=config,
                    )
                )

    rows = (
        pd.concat([frame for frame in row_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in row_frames)
        else pd.DataFrame(columns=row_columns)
    )
    if prepare_rows is not None:
        rows = prepare_rows(rows)
    summary = summarize_rows(rows)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)

    manifest = {
        "schema_version": schema_version,
        "study_role": study_role,
        "generated_by": generated_by,
        "generated_at_utc": format_timestamp_utc(),
        "suite": config.suite,
        "case_names": list(config.case_names),
        "data_roles": list(config.data_roles),
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "profile_id": str(config.profile_id),
        "sibling_alpha": float(config.sibling_alpha),
        "edge_alpha": float(config.edge_alpha),
        "top_k": int(config.top_k),
        "rows": int(rows.shape[0]),
        "skipped_cases": skipped_cases,
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        **dict(manifest_extra or {}),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def build_binary_overlap_case_node_rows(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    replicate: int,
    data_seed: int,
    config: Any,
    row_columns: tuple[str, ...],
    relevant_node_rows: Callable[[pd.DataFrame], pd.DataFrame],
    node_row_builder: OverlapNodeRowBuilder,
) -> pd.DataFrame:
    """Build diagnostic rows for one binary-template case/role/replicate."""
    data, feature_space, truth_labels, _true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role=data_role,
        seed=data_seed,
    )
    distance = pdist(data.to_numpy(dtype=float), metric="hamming")
    result = run_tbs_on_distance(
        data,
        distance,
        sibling_significance_level=float(config.sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(config.edge_alpha),
        feature_space=feature_space,
        sibling_gate_profile=str(config.profile_id),
        trace_level="full",
    )
    node_decisions = _build_node_decisions(
        case_id=case_id,
        data_role=data_role,
        method_id=str(config.profile_id),
        replicate=replicate,
        data_seed=data_seed,
        result=result,
    )
    truth_by_label = {
        str(label): int(value)
        for label, value in zip(data.index.astype(str), np.asarray(truth_labels, dtype=int))
    }
    records: list[dict[str, object]] = []
    for _, node_row in relevant_node_rows(node_decisions).iterrows():
        row = node_row_builder(
            case_id=case_id,
            data_role=data_role,
            replicate=replicate,
            data_seed=data_seed,
            profile_id=str(config.profile_id),
            node_row=node_row,
            result=result,
            data=data,
            truth_by_label=truth_by_label,
            config=config,
        )
        if row is not None:
            records.append(row)
    return pd.DataFrame.from_records(records, columns=row_columns)


__all__ = ["build_binary_overlap_case_node_rows", "run_binary_overlap_panel"]
