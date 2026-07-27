"""Data-independent sibling gate diagnostics.

This panel tests same-data fixes for the selected sibling null-law failure
without cross-fitting. The candidate repair removes PCA rows and projection
dimension learned from the tested sample, then evaluates predeclared fixed
global, coordinate-wise, or feature-block Wald gates with a selected-topology
penalty.
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
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    fixed_coordinate_bh_p_value,
    fixed_subspace_sibling_p_value,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)

from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.validation.selected_edge_type1_geometry import (
    _build_tree_from_data,
    _case_contract,
    _select_cases,
    parse_names,
    regenerate_null_case,
)

STUDY_ROLE = "diagnostic_data_independent_sibling_gate_not_calibration"
SCHEMA_VERSION = "data_independent_sibling_gate_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel"
DEFAULT_DATA_ROLES = ("null", "signal")
DEFAULT_CANDIDATE_METHODS = (
    "global_chi_square",
    "coordinate_bonferroni",
    "coordinate_bh",
    "block_bonferroni",
    "block_bh",
)
DEFAULT_SELECTED_TOPOLOGY_PENALTIES = (1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class DataIndependentSiblingGateConfig:
    """Runtime contract for data-independent sibling gate diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    data_roles: tuple[str, ...]
    candidate_methods: tuple[str, ...]
    sibling_alpha: float
    selected_topology_penalties: tuple[float, ...]
    replicates: int
    base_seed: int
    min_rows: int = 100
    min_signal_rejection_rate: float = 0.15

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_summary.csv"

    @property
    def transfer_summary_path(self) -> Path:
        return self.output_dir / "data_independent_sibling_gate_penalty_transfer_summary.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def validate_data_roles(data_roles: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(role) for role in data_roles)
    invalid = sorted(set(values) - set(DEFAULT_DATA_ROLES))
    if invalid:
        raise ValueError(
            f"Unknown data role(s): {invalid!r}; allowed={sorted(DEFAULT_DATA_ROLES)!r}."
        )
    if not values:
        raise ValueError("At least one data role is required.")
    return values


def validate_candidate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(method) for method in methods)
    invalid = sorted(set(values) - set(DEFAULT_CANDIDATE_METHODS))
    if invalid:
        raise ValueError(
            f"Unknown candidate method(s): {invalid!r}; "
            f"allowed={sorted(DEFAULT_CANDIDATE_METHODS)!r}."
        )
    if not values:
        raise ValueError("At least one candidate method is required.")
    return values


def validate_selected_topology_penalties(penalties: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(penalty) for penalty in penalties)
    if not values:
        raise ValueError("At least one selected topology penalty is required.")
    invalid = [penalty for penalty in values if penalty <= 0.0 or not math.isfinite(penalty)]
    if invalid:
        raise ValueError(
            "selected_topology_penalties must be finite and positive; "
            f"got {invalid!r}."
        )
    return tuple(sorted(set(values)))


def _selected_alpha(*, sibling_alpha: float, selected_topology_penalty: float) -> float:
    penalty = float(selected_topology_penalty)
    if penalty <= 0.0 or not math.isfinite(penalty):
        raise ValueError("selected_topology_penalty must be finite and positive.")
    alpha = float(sibling_alpha) / penalty
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Effective selected alpha must lie in (0, 1); got {alpha!r}.")
    return alpha


def data_independent_coordinate_gate_p_value(
    z: np.ndarray,
    *,
    candidate_method: str,
) -> float:
    """Return a data-independent coordinate-wise sibling p-value."""
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if vector.size == 0:
        return 1.0
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    coordinate_p = chi2.sf(vector * vector, df=1.0)
    if candidate_method == "coordinate_bonferroni":
        return float(min(1.0, vector.size * float(np.min(coordinate_p))))
    if candidate_method == "coordinate_bh":
        return fixed_coordinate_bh_p_value(vector)
    raise ValueError(f"Unknown candidate_method: {candidate_method!r}.")


def data_independent_global_gate_p_value(
    z: np.ndarray,
    feature_space: FeatureSpace,
    *,
    candidate_method: str,
) -> float:
    """Return the full fixed-subspace chi-square sibling p-value."""
    if candidate_method == "global_chi_square":
        return fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_global_chi_square",
        )
    raise ValueError(f"Unknown candidate_method: {candidate_method!r}.")


def _feature_block_p_values(z: np.ndarray, feature_space: FeatureSpace) -> np.ndarray:
    vector = np.asarray(z, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError("z must contain only finite values.")
    p_values: list[float] = []
    offset = 0
    for block in feature_space.blocks:
        width = int(block.contrast_dimension)
        if width <= 0:
            raise ValueError(
                f"Feature block {block.name!r} has invalid contrast_dimension={width}."
            )
        block_z = vector[offset : offset + width]
        if block_z.shape[0] != width:
            raise ValueError(
                "Feature blocks do not cover the whitened contrast vector: "
                f"block={block.name!r}, offset={offset}, width={width}, "
                f"z_dimension={vector.shape[0]}."
            )
        block_stat = float(np.dot(block_z, block_z))
        p_values.append(float(chi2.sf(block_stat, df=width)))
        offset += width
    if offset != vector.shape[0]:
        raise ValueError(
            "Feature blocks do not cover the whitened contrast vector: "
            f"covered={offset}, z_dimension={vector.shape[0]}."
        )
    return np.asarray(p_values, dtype=float)


def data_independent_feature_block_gate_p_value(
    z: np.ndarray,
    feature_space: FeatureSpace,
    *,
    candidate_method: str,
) -> float:
    """Return a feature-block fixed-subspace sibling p-value."""
    block_p = _feature_block_p_values(z, feature_space)
    if block_p.size == 0:
        return 1.0
    if candidate_method == "block_bonferroni":
        return float(min(1.0, block_p.size * float(np.min(block_p))))
    if candidate_method == "block_bh":
        return fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_block_bh",
        )
    raise ValueError(f"Unknown candidate_method: {candidate_method!r}.")


def data_independent_gate_p_value(
    z: np.ndarray,
    feature_space: FeatureSpace,
    *,
    candidate_method: str,
) -> float:
    """Return a p-value for one predeclared data-independent gate method."""
    if candidate_method == "global_chi_square":
        return data_independent_global_gate_p_value(
            z,
            feature_space,
            candidate_method=candidate_method,
        )
    if candidate_method.startswith("coordinate_"):
        return data_independent_coordinate_gate_p_value(
            z,
            candidate_method=candidate_method,
        )
    if candidate_method.startswith("block_"):
        return data_independent_feature_block_gate_p_value(
            z,
            feature_space,
            candidate_method=candidate_method,
        )
    raise ValueError(f"Unknown candidate_method: {candidate_method!r}.")


def _feature_space_from_signal_data(data: pd.DataFrame) -> FeatureSpace:
    return bernoulli_feature_space_from_columns(tuple(data.columns))


def _generate_data(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    seed: int,
) -> tuple[pd.DataFrame, FeatureSpace]:
    if data_role == "null":
        data, metadata = regenerate_null_case(
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            seed=seed,
        )
        return data, metadata["feature_space"]  # type: ignore[return-value]
    if data_role == "signal":
        signal_case = dict(case)
        signal_case["seed"] = int(seed)
        data, _labels, _original, metadata = generate_case_data(signal_case)
        feature_space = metadata.get("feature_space")
        if isinstance(feature_space, FeatureSpace):
            return data, feature_space
        if metadata.get("source_family") == "binary_template":
            return data, _feature_space_from_signal_data(data)
        raise ValueError(
            "Signal diagnostics require a FeatureSpace for non-binary cases."
        )
    raise ValueError(f"Unknown data_role: {data_role!r}.")


def _rows_for_replicate(
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
    candidate_methods: tuple[str, ...],
    sibling_alpha: float,
    selected_topology_penalties: tuple[float, ...],
    run_id: str,
) -> list[dict[str, object]]:
    data, feature_space = _generate_data(
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
    tree = _build_tree_from_data(data)
    tree.populate_node_divergences(data, feature_space=feature_space)
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
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
        for candidate_method in candidate_methods:
            p_value = data_independent_gate_p_value(
                z,
                feature_space,
                candidate_method=candidate_method,
            )
            for selected_topology_penalty in selected_topology_penalties:
                effective_alpha = _selected_alpha(
                    sibling_alpha=sibling_alpha,
                    selected_topology_penalty=selected_topology_penalty,
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "run_id": run_id,
                        "case_id": case_id,
                        "data_role": data_role,
                        "topology_mode": "selected_topology",
                        "candidate_method": candidate_method,
                        "replicate": int(replicate),
                        "parent_id": str(parent),
                        "left_child_id": str(left),
                        "right_child_id": str(right),
                        "source_family": source_family,
                        "feature_representation": feature_representation,
                        "sibling_alpha": float(sibling_alpha),
                        "selected_topology_penalty": float(selected_topology_penalty),
                        "effective_selected_alpha": float(effective_alpha),
                        "data_seed": int(data_seed),
                        "parent_sample_size": float(left_n + right_n),
                        "left_sample_size": left_n,
                        "right_sample_size": right_n,
                    "contrast_dimension": float(z.shape[0]),
                    "data_independent_gate_p_value": p_value,
                    "coordinate_gate_p_value": p_value,
                        "rejected_at_sibling_alpha": bool(
                            p_value <= float(sibling_alpha)
                        ),
                        "rejected_at_effective_alpha": bool(p_value <= effective_alpha),
                        "z_norm": float(np.linalg.norm(z)),
                        "max_abs_z": float(np.max(np.abs(z))) if z.size else 0.0,
                    }
                )
    return rows


def _candidate_status(
    group: pd.DataFrame,
    *,
    min_rows: int,
    min_signal_rejection_rate: float,
) -> str:
    if group.shape[0] < int(min_rows):
        return "data_independent_gate_insufficient_rows"
    data_role = str(group["data_role"].iloc[0])
    rejection_rate = float(group["rejected_at_effective_alpha"].mean())
    if data_role == "null":
        sibling_alpha = float(group["sibling_alpha"].iloc[0])
        if rejection_rate <= sibling_alpha:
            return "data_independent_gate_null_candidate"
        return "data_independent_gate_null_inflated"
    if data_role == "signal":
        if rejection_rate >= float(min_signal_rejection_rate):
            return "data_independent_gate_signal_retained"
        return "data_independent_gate_signal_weak"
    raise ValueError(f"Unknown data role in summary: {data_role!r}.")


def summarize_data_independent_sibling_gate_rows(
    rows: pd.DataFrame,
    *,
    min_rows: int = 100,
    min_signal_rejection_rate: float = 0.15,
) -> pd.DataFrame:
    """Summarize data-independent gate behavior by role and method."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    group_columns = (
        "case_id",
        "data_role",
        "topology_mode",
        "candidate_method",
        "selected_topology_penalty",
        "source_family",
    )
    for key, group in rows.groupby(list(group_columns), sort=True):
        p_values = pd.to_numeric(
            group["data_independent_gate_p_value"],
            errors="coerce",
        )
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_rows": int(group.shape[0]),
                "rejection_rate_at_sibling_alpha": float(
                    group["rejected_at_sibling_alpha"].mean()
                ),
                "rejection_rate_at_effective_alpha": float(
                    group["rejected_at_effective_alpha"].mean()
                ),
                "effective_selected_alpha": float(
                    group["effective_selected_alpha"].iloc[0]
                ),
                "p_value_q50": float(p_values.quantile(0.50)),
                "p_value_q05": float(p_values.quantile(0.05)),
                "parent_sample_size_q90": float(
                    group["parent_sample_size"].quantile(0.90)
                ),
                "large_parent_rejection_rate_at_effective_alpha": float(
                    group.loc[
                        group["parent_sample_size"]
                        >= group["parent_sample_size"].quantile(0.90),
                        "rejected_at_effective_alpha",
                    ].mean()
                ),
                "data_independent_gate_status": _candidate_status(
                    group,
                    min_rows=min_rows,
                    min_signal_rejection_rate=min_signal_rejection_rate,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def _transfer_status(group: pd.DataFrame) -> str:
    null_rows = group[group["data_role"].eq("null")]
    signal_rows = group[group["data_role"].eq("signal")]
    if null_rows.empty or signal_rows.empty:
        return "data_independent_gate_penalty_insufficient_coverage"
    null_ok = null_rows["data_independent_gate_status"].eq(
        "data_independent_gate_null_candidate"
    )
    signal_ok = signal_rows["data_independent_gate_status"].eq(
        "data_independent_gate_signal_retained"
    )
    if not bool(null_ok.all()):
        return "data_independent_gate_penalty_null_inflated"
    if not bool(signal_ok.all()):
        return "data_independent_gate_penalty_signal_weak"
    return "data_independent_gate_penalty_transfer_candidate"


def summarize_data_independent_gate_penalty_transfer(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether a penalty transfers across included cases."""
    if summary.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    group_columns = (
        "candidate_method",
        "selected_topology_penalty",
        "topology_mode",
        "source_family",
    )
    for key, group in summary.groupby(list(group_columns), sort=True):
        row = dict(zip(group_columns, key))
        null_rows = group[group["data_role"].eq("null")]
        signal_rows = group[group["data_role"].eq("signal")]
        row.update(
            {
                "n_case_role_rows": int(group.shape[0]),
                "n_null_case_rows": int(null_rows.shape[0]),
                "n_signal_case_rows": int(signal_rows.shape[0]),
                "max_null_rejection_rate_at_effective_alpha": (
                    float(null_rows["rejection_rate_at_effective_alpha"].max())
                    if not null_rows.empty
                    else math.nan
                ),
                "min_signal_rejection_rate_at_effective_alpha": (
                    float(signal_rows["rejection_rate_at_effective_alpha"].min())
                    if not signal_rows.empty
                    else math.nan
                ),
                "n_null_candidates": int(
                    null_rows["data_independent_gate_status"]
                    .eq("data_independent_gate_null_candidate")
                    .sum()
                ),
                "n_signal_retained": int(
                    signal_rows["data_independent_gate_status"]
                    .eq("data_independent_gate_signal_retained")
                    .sum()
                ),
                "data_independent_gate_transfer_status": _transfer_status(group),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _empty_components() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contract_id",
            "component_id",
            "component_type",
            "component_status",
            "required_for_production",
            "notes",
        ]
    )


def build_data_independent_gate_components(
    summary: pd.DataFrame,
    transfer_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build production-admissibility components from gate summaries."""
    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        contract_context = (
            f"topology_mode={row['topology_mode']}|"
            f"candidate_method={row['candidate_method']}|"
            f"selected_topology_penalty={row['selected_topology_penalty']}|"
            f"source_family={row['source_family']}"
        )
        context = (
            f"case_id={row['case_id']}|data_role={row['data_role']}|"
            f"{contract_context}"
        )
        records.append(
            {
                "contract_id": (
                    "data_independent_sibling_gate_candidate:"
                    f"{contract_context}"
                ),
                "component_id": f"data_independent_gate:{context}",
                "component_type": "data_independent_sibling_gate_panel",
                "component_status": str(row["data_independent_gate_status"]),
                "required_for_production": True,
                "notes": "Diagnostic candidate for same-data sibling gate repair.",
            }
        )
    if transfer_summary is not None:
        for _, row in transfer_summary.iterrows():
            context = (
                f"topology_mode={row['topology_mode']}|"
                f"candidate_method={row['candidate_method']}|"
                f"selected_topology_penalty={row['selected_topology_penalty']}|"
                f"source_family={row['source_family']}"
            )
            records.append(
                {
                    "contract_id": (
                        "data_independent_sibling_gate_candidate:"
                        f"{context}"
                    ),
                    "component_id": f"data_independent_gate_transfer:{context}",
                    "component_type": "data_independent_sibling_gate_transfer",
                    "component_status": str(
                        row["data_independent_gate_transfer_status"]
                    ),
                    "required_for_production": True,
                    "notes": "Diagnostic transfer check for selected-topology penalty.",
                }
            )
    if not records:
        return _empty_components()
    return pd.DataFrame.from_records(records)


def run_data_independent_sibling_gate_panel(
    config: DataIndependentSiblingGateConfig,
) -> dict[str, Path]:
    """Run data-independent sibling gate diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "data_independent_sibling_gate__"
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
        if source_family not in {"binary_template", "categorical_multinomial"}:
            raise ValueError(
                "Data-independent sibling gate V1 supports binary and direct "
                f"categorical cases only; got {source_family!r}."
            )
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate * 1009
            for data_role in config.data_roles:
                rows.extend(
                    _rows_for_replicate(
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
                        candidate_methods=config.candidate_methods,
                        sibling_alpha=float(config.sibling_alpha),
                        selected_topology_penalties=config.selected_topology_penalties,
                        run_id=run_id,
                    )
                )
    row_table = pd.DataFrame.from_records(rows)
    summary = summarize_data_independent_sibling_gate_rows(
        row_table,
        min_rows=int(config.min_rows),
        min_signal_rejection_rate=float(config.min_signal_rejection_rate),
    )
    transfer_summary = summarize_data_independent_gate_penalty_transfer(summary)
    components = build_data_independent_gate_components(summary, transfer_summary)
    production_rows = evaluate_production_admissibility_components(components)
    production_summary = summarize_production_admissibility_contracts(production_rows)

    row_table.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    transfer_summary.to_csv(config.transfer_summary_path, index=False)
    production_rows.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": config.suite,
        "case_names": list(config.case_names),
        "data_roles": list(config.data_roles),
        "candidate_methods": list(config.candidate_methods),
        "sibling_alpha": float(config.sibling_alpha),
        "selected_topology_penalties": list(config.selected_topology_penalties),
        "effective_selected_alphas": [
            _selected_alpha(
                sibling_alpha=float(config.sibling_alpha),
                selected_topology_penalty=float(penalty),
            )
            for penalty in config.selected_topology_penalties
        ],
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "n_rows": int(row_table.shape[0]),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "transfer_summary": str(config.transfer_summary_path),
            "production_components": str(config.production_components_path),
            "production_summary": str(config.production_summary_path),
        },
        "interpretation": (
            "Diagnostic same-data repair candidate. Removes adaptive parent PCA "
            "projection/dimension from sibling gate and applies a predeclared "
            "selected-topology penalty to coordinate-wise Wald p-values."
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "transfer_summary": config.transfer_summary_path,
        "production_components": config.production_components_path,
        "production_summary": config.production_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument("--data-roles", default=",".join(DEFAULT_DATA_ROLES))
    parser.add_argument(
        "--candidate-methods",
        default=",".join(DEFAULT_CANDIDATE_METHODS),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument(
        "--selected-topology-penalties",
        default=",".join(str(value) for value in DEFAULT_SELECTED_TOPOLOGY_PENALTIES),
    )
    parser.add_argument(
        "--selected-topology-penalty",
        type=float,
        help="Deprecated single-penalty alias for --selected-topology-penalties.",
    )
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-signal-rejection-rate", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    selected_topology_penalties = (
        (float(args.selected_topology_penalty),)
        if args.selected_topology_penalty is not None
        else tuple(float(value) for value in parse_names(args.selected_topology_penalties))
    )
    outputs = run_data_independent_sibling_gate_panel(
        DataIndependentSiblingGateConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            data_roles=validate_data_roles(parse_names(args.data_roles)),
            candidate_methods=validate_candidate_methods(parse_names(args.candidate_methods)),
            sibling_alpha=float(args.sibling_alpha),
            selected_topology_penalties=validate_selected_topology_penalties(
                selected_topology_penalties
            ),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            min_rows=int(args.min_rows),
            min_signal_rejection_rate=float(args.min_signal_rejection_rate),
        )
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DataIndependentSiblingGateConfig",
    "build_data_independent_gate_components",
    "data_independent_coordinate_gate_p_value",
    "data_independent_feature_block_gate_p_value",
    "data_independent_gate_p_value",
    "data_independent_global_gate_p_value",
    "run_data_independent_sibling_gate_panel",
    "summarize_data_independent_gate_penalty_transfer",
    "summarize_data_independent_sibling_gate_rows",
    "validate_candidate_methods",
    "validate_data_roles",
    "validate_selected_topology_penalties",
]
