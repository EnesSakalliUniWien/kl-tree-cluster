"""Validate fixed sibling-gate profiles through the shared KL runner.

This diagnostic is intentionally narrow: it verifies that named production-facing
profiles route through fixed-subspace sibling statistics and records traversal
point/confidence evidence. It does not promote a profile to production.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    SIBLING_GATE_PROFILES,
    resolve_sibling_gate_profile,
    resolve_sibling_gate_profile_config,
)
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    DEFAULT_DATA_ROLES,
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
    _mean_lower_bound,
    _required_replicates_for_wilson_upper_bound,
    _required_zero_false_split_replicates,
    _wilson_upper_bound,
)
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

STUDY_ROLE = "diagnostic_fixed_sibling_gate_profile_validation_not_calibration"
SCHEMA_VERSION = "fixed_sibling_gate_profile_validation/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation"
)
DEFAULT_PROFILES = ("fixed_coordinate_guarded_v1", "fixed_global_guarded_v1")
COMMON_EVIDENCE_FIELDS = (
    "validation_design",
    "source_artifact_paths",
    "code_commit",
    "run_command",
    "random_seed_policy",
    "n_replicates",
    "baseline_setting",
    "candidate_setting",
    "primary_endpoint",
    "effect_estimate",
    "confidence_interval",
    "decision_rule",
    "limitations",
)
PROFILE_CONSTANT_FIELDS = {
    "sibling_gate_profile": (
        "profile_grid",
        "profile_metadata",
        "adaptive_projection_avoidance_check",
        "root_selective_permutation_guard",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "fixed_sibling_gate_alpha_penalty": (
        "penalty_grid",
        "effective_sibling_alpha",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "root_stability_guard_threshold": (
        "threshold_grid",
        "root_stability_distribution",
        "null_root_block_rate",
        "signal_root_block_rate",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "root_stability_subsample_replicates": (
        "subsample_replicate_grid",
        "guard_decision_stability",
        "runtime_cost_summary",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "root_stability_feature_fraction": (
        "feature_fraction_grid",
        "root_stability_distribution",
        "null_root_block_rate",
        "signal_root_block_rate",
    ),
    "root_selective_permutation_guard_replicates": (
        "permutation_replicate_grid",
        "root_selective_p_value_distribution",
        "null_root_block_rate",
        "signal_root_block_rate",
        "runtime_cost_summary",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "root_selective_permutation_guard_alpha": (
        "alpha_grid",
        "root_selective_p_value_distribution",
        "null_root_block_rate",
        "signal_root_block_rate",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
    "root_selective_permutation_guard_scope": (
        "scope_grid",
        "root_selective_p_value_distribution",
        "null_false_split_confidence",
        "signal_ari_confidence",
    ),
}


@dataclass(frozen=True)
class FixedSiblingGateProfileValidationConfig:
    """Runtime contract for fixed sibling-gate profile validation."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    profiles: tuple[str, ...]
    data_roles: tuple[str, ...]
    sibling_alpha: float
    edge_alpha: float
    replicates: int
    base_seed: int
    root_selective_permutation_guard_replicates: int = 0
    root_selective_permutation_guard_seed: int = 0
    root_selective_permutation_guard_alpha: float | None = None
    max_null_split_rate: float = 0.05
    min_signal_mean_ari: float = 0.75
    resume_from_checkpoints: bool = False

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "fixed_sibling_gate_profile_validation_rows.csv"

    @property
    def checkpoint_rows_dir(self) -> Path:
        return self.output_dir / "checkpoint_rows"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "fixed_sibling_gate_profile_validation_summary.csv"

    @property
    def transfer_summary_path(self) -> Path:
        return self.output_dir / "fixed_sibling_gate_profile_validation_transfer_summary.csv"

    @property
    def evidence_fields_path(self) -> Path:
        return self.output_dir / "method_constant_evidence_fields.json"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    """Return validated profile ids."""
    values = tuple(str(profile) for profile in profiles)
    if not values:
        raise ValueError("At least one sibling-gate profile is required.")
    invalid = sorted(set(values) - set(SIBLING_GATE_PROFILES))
    if invalid:
        raise ValueError(
            f"Unknown sibling-gate profile(s): {invalid!r}; "
            f"allowed={tuple(sorted(SIBLING_GATE_PROFILES))!r}."
        )
    return values


def _resolved_profile_metadata(
    config: FixedSiblingGateProfileValidationConfig,
) -> dict[str, dict[str, object]]:
    """Resolve each requested profile exactly as the runner will execute it."""
    metadata: dict[str, dict[str, object]] = {}
    for requested_profile_id in config.profiles:
        profile = resolve_sibling_gate_profile(requested_profile_id)
        if profile is None:
            raise ValueError(f"Unknown profile: {requested_profile_id!r}.")
        (
            profile_id,
            method,
            penalty,
            threshold,
            stability_replicates,
            feature_fraction,
            stability_seed,
            root_selective_replicates,
            root_selective_seed,
            root_selective_alpha,
            root_selective_scope,
        ) = resolve_sibling_gate_profile_config(
            sibling_gate_profile=requested_profile_id,
            root_selective_permutation_guard_replicates=(
                config.root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(
                config.root_selective_permutation_guard_seed
            ),
            root_selective_permutation_guard_alpha=(
                config.root_selective_permutation_guard_alpha
            ),
        )
        profile_record = asdict(profile)
        profile_record["resolved_runtime_config"] = {
            "sibling_gate_profile_id": profile_id,
            "sibling_gate_method": method,
            "sibling_gate_alpha_penalty": float(penalty),
            "root_stability_guard_threshold": threshold,
            "root_stability_subsample_replicates": int(stability_replicates),
            "root_stability_feature_fraction": float(feature_fraction),
            "root_stability_seed": int(stability_seed),
            "root_selective_permutation_guard_replicates": int(
                root_selective_replicates
            ),
            "root_selective_permutation_guard_seed": int(root_selective_seed),
            "root_selective_permutation_guard_alpha": (
                None
                if root_selective_alpha is None
                else float(root_selective_alpha)
            ),
            "root_selective_permutation_guard_scope": str(root_selective_scope),
        }
        metadata[requested_profile_id] = profile_record
    return metadata


def _sorted_unique_values(values: Sequence[object]) -> list[object]:
    return sorted(set(values), key=lambda value: (value is None, str(value)))


def _status(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float,
    min_signal_mean_ari: float,
) -> str:
    if group.empty:
        return "fixed_profile_insufficient_coverage"
    if not bool(group["adaptive_projection_avoided"].astype(bool).all()):
        return "fixed_profile_adaptive_projection_detected"
    role = str(group["data_role"].iloc[0])
    if role == "null":
        if float(group["false_split"].mean()) <= float(max_null_split_rate):
            return "fixed_profile_null_candidate"
        return "fixed_profile_null_inflated"
    if role == "signal":
        if float(group["ari"].mean()) >= float(min_signal_mean_ari):
            return "fixed_profile_signal_retained"
        return "fixed_profile_signal_weak"
    raise ValueError(f"Unknown data role: {role!r}.")


def _transfer_status(group: pd.DataFrame) -> str:
    if not bool(group["adaptive_projection_avoided_rate"].eq(1.0).all()):
        return "fixed_profile_adaptive_projection_detected"
    null_rows = group[group["data_role"].eq("null")]
    signal_rows = group[group["data_role"].eq("signal")]
    if null_rows.empty or signal_rows.empty:
        return "fixed_profile_insufficient_coverage"
    if not bool(null_rows["profile_validation_status"].eq("fixed_profile_null_candidate").all()):
        return "fixed_profile_null_inflated"
    if not bool(
        signal_rows["profile_validation_status"].eq("fixed_profile_signal_retained").all()
    ):
        return "fixed_profile_signal_weak"
    return "fixed_profile_transfer_candidate"


def _confidence_status(
    group: pd.DataFrame,
    *,
    max_null_split_rate: float,
    min_signal_mean_ari: float,
) -> str:
    if not bool(group["adaptive_projection_avoided_rate"].eq(1.0).all()):
        return "fixed_profile_adaptive_projection_detected"
    null_rows = group[group["data_role"].eq("null")]
    signal_rows = group[group["data_role"].eq("signal")]
    if null_rows.empty or signal_rows.empty:
        return "fixed_profile_insufficient_coverage"
    max_null_upper = pd.to_numeric(
        null_rows["false_split_rate_upper_confidence"],
        errors="coerce",
    ).max()
    min_signal_lower = pd.to_numeric(
        signal_rows["mean_ari_lower_confidence"],
        errors="coerce",
    ).min()
    if pd.isna(max_null_upper) or pd.isna(min_signal_lower):
        return "fixed_profile_insufficient_coverage"
    if float(max_null_upper) > float(max_null_split_rate):
        return "fixed_profile_confidence_null_uncertain"
    if float(min_signal_lower) < float(min_signal_mean_ari):
        return "fixed_profile_confidence_signal_uncertain"
    return "fixed_profile_confidence_candidate"


def _annotation_methods(annotations: pd.DataFrame) -> tuple[str, ...]:
    if "Sibling_Test_Method" not in annotations:
        return ()
    values = annotations["Sibling_Test_Method"].dropna().astype(str)
    return tuple(sorted({value for value in values if value and value != "nan"}))


def _safe_path_token(value: object) -> str:
    token = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in token)


def _checkpoint_row_path(
    *,
    checkpoint_dir: Path,
    case_id: str,
    data_role: str,
    profile_id: str,
    replicate: int,
) -> Path:
    return checkpoint_dir / (
        f"case={_safe_path_token(case_id)}"
        f"__role={_safe_path_token(data_role)}"
        f"__profile={_safe_path_token(profile_id)}"
        f"__replicate={int(replicate):04d}.csv"
    )


def _write_checkpoint_row(row: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([row]).to_csv(path, index=False)
    return path


def _read_checkpoint_row(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.shape[0] != 1:
        raise ValueError(f"Checkpoint row file must contain exactly one row: {path}")
    return frame.iloc[0].to_dict()


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
    profile_id: str,
    sibling_alpha: float,
    edge_alpha: float,
    root_selective_permutation_guard_replicates: int,
    root_selective_permutation_guard_seed: int,
    root_selective_permutation_guard_alpha: float | None,
) -> dict[str, object]:
    profile = resolve_sibling_gate_profile(profile_id)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_id!r}.")
    data, feature_space, truth_labels, true_clusters = _generate_data_with_truth(
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
    result = _run_kl_method(
        data,
        distance,
        sibling_significance_level=float(sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(edge_alpha),
        feature_space=feature_space,
        sibling_gate_profile=profile_id,
        root_selective_permutation_guard_replicates=int(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=int(
            root_selective_permutation_guard_seed
        ),
        root_selective_permutation_guard_alpha=(
            None
            if root_selective_permutation_guard_alpha is None
            else float(root_selective_permutation_guard_alpha)
        ),
    )
    annotations = result.extra["annotations"]
    tree = result.extra["tree"]
    root = tree.root() if hasattr(tree, "root") else tree.graph.get("root")
    gate_bundle = result.extra["gate_bundle"]
    config = gate_bundle.metadata.config
    methods = _annotation_methods(annotations)
    fixed_method_rows = (
        int(annotations["Sibling_Test_Method"].eq(profile.sibling_gate_method).sum())
        if "Sibling_Test_Method" in annotations
        else 0
    )
    adaptive_projection_detected = "projected_wald_inflation" in methods
    adaptive_projection_avoided = (
        not adaptive_projection_detected
        and str(config.sibling_gate_profile_id) == profile_id
        and str(config.sibling_gate_method) == profile.sibling_gate_method
        and fixed_method_rows > 0
    )
    root_guard_block_count = (
        int(annotations["Root_Stability_Guard_Blocked"].fillna(False).sum())
        if "Root_Stability_Guard_Blocked" in annotations
        else 0
    )
    root_sibling_p_value = (
        float(annotations.at[root, "Sibling_Divergence_P_Value"])
        if root in annotations.index
        and "Sibling_Divergence_P_Value" in annotations
        and pd.notna(annotations.at[root, "Sibling_Divergence_P_Value"])
        else np.nan
    )
    root_sibling_open = (
        bool(annotations.at[root, "Sibling_BH_Different"])
        if root in annotations.index and "Sibling_BH_Different" in annotations
        else False
    )
    root_stability_guard_blocked = (
        bool(annotations.at[root, "Root_Stability_Guard_Blocked"])
        if root in annotations.index and "Root_Stability_Guard_Blocked" in annotations
        else False
    )
    root_stability_mean_ari = (
        float(annotations.at[root, "Root_Stability_Subsample_Mean_ARI"])
        if root in annotations.index
        and "Root_Stability_Subsample_Mean_ARI" in annotations
        and pd.notna(annotations.at[root, "Root_Stability_Subsample_Mean_ARI"])
        else np.nan
    )
    root_stability_median_ari = (
        float(annotations.at[root, "Root_Stability_Subsample_Median_ARI"])
        if root in annotations.index
        and "Root_Stability_Subsample_Median_ARI" in annotations
        and pd.notna(annotations.at[root, "Root_Stability_Subsample_Median_ARI"])
        else np.nan
    )
    root_stability_q10_ari = (
        float(annotations.at[root, "Root_Stability_Subsample_Q10_ARI"])
        if root in annotations.index
        and "Root_Stability_Subsample_Q10_ARI" in annotations
        and pd.notna(annotations.at[root, "Root_Stability_Subsample_Q10_ARI"])
        else np.nan
    )
    root_selective_p_value = (
        float(annotations.at[root, "Root_Selective_Permutation_P_Value"])
        if root in annotations.index
        and "Root_Selective_Permutation_P_Value" in annotations
        and pd.notna(annotations.at[root, "Root_Selective_Permutation_P_Value"])
        else np.nan
    )
    root_selective_guard_blocked = (
        bool(annotations.at[root, "Root_Selective_Permutation_Guard_Blocked"])
        if root in annotations.index
        and "Root_Selective_Permutation_Guard_Blocked" in annotations
        else False
    )
    root_selective_guard_would_block = (
        bool(annotations.at[root, "Root_Selective_Permutation_Guard_Would_Block"])
        if root in annotations.index
        and "Root_Selective_Permutation_Guard_Would_Block" in annotations
        else False
    )
    selective_p_values = (
        pd.to_numeric(
            annotations["Selective_Permutation_P_Value"],
            errors="coerce",
        )
        if "Selective_Permutation_P_Value" in annotations
        else pd.Series(dtype=float)
    )
    selective_guard_block_count = (
        int(annotations["Selective_Permutation_Guard_Blocked"].fillna(False).sum())
        if "Selective_Permutation_Guard_Blocked" in annotations
        else 0
    )
    selective_guard_would_block_count = (
        int(
            annotations["Selective_Permutation_Guard_Would_Block"]
            .fillna(False)
            .sum()
        )
        if "Selective_Permutation_Guard_Would_Block" in annotations
        else 0
    )
    predicted = np.asarray(result.labels)
    ari = float(adjusted_rand_score(np.asarray(truth_labels, dtype=int), predicted))
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": data_role,
        "source_family": source_family,
        "feature_representation": feature_representation,
        "profile_id": profile_id,
        "profile_status": profile.status,
        "expected_sibling_gate_method": profile.sibling_gate_method,
        "observed_sibling_gate_method": str(config.sibling_gate_method),
        "observed_sibling_gate_profile_id": str(config.sibling_gate_profile_id),
        "observed_sibling_gate_alpha_penalty": float(
            config.sibling_gate_alpha_penalty
        ),
        "observed_root_stability_guard_threshold": (
            np.nan
            if config.root_stability_guard_threshold is None
            else float(config.root_stability_guard_threshold)
        ),
        "observed_root_stability_subsample_replicates": int(
            config.root_stability_subsample_replicates
        ),
        "observed_root_stability_feature_fraction": float(
            config.root_stability_feature_fraction
        ),
        "observed_root_stability_seed": int(config.root_stability_seed),
        "observed_root_stability_tree_distance_metric": str(
            config.root_stability_tree_distance_metric
        ),
        "observed_root_stability_tree_linkage_method": str(
            config.root_stability_tree_linkage_method
        ),
        "observed_root_selective_permutation_guard_replicates": int(
            config.root_selective_permutation_guard_replicates
        ),
        "observed_root_selective_permutation_guard_seed": int(
            config.root_selective_permutation_guard_seed
        ),
        "observed_root_selective_permutation_guard_alpha": (
            np.nan
            if config.root_selective_permutation_guard_alpha is None
            else float(config.root_selective_permutation_guard_alpha)
        ),
        "observed_root_selective_permutation_guard_scope": str(
            config.root_selective_permutation_guard_scope
        ),
        "observed_root_selective_permutation_guard_tree_distance_metric": str(
            config.root_selective_permutation_guard_tree_distance_metric
        ),
        "observed_root_selective_permutation_guard_tree_linkage_method": str(
            config.root_selective_permutation_guard_tree_linkage_method
        ),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "effective_sibling_alpha": (
            float(sibling_alpha) / float(config.sibling_gate_alpha_penalty)
        ),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "true_clusters": int(true_clusters),
        "found_clusters": int(result.found_clusters),
        "ari": ari,
        "exact_cluster_count": bool(int(result.found_clusters) == int(true_clusters)),
        "false_split": bool(data_role == "null" and int(result.found_clusters) > 1),
        "adaptive_projection_detected": bool(adaptive_projection_detected),
        "adaptive_projection_avoided": bool(adaptive_projection_avoided),
        "observed_annotation_methods": ";".join(methods),
        "fixed_method_annotation_rows": int(fixed_method_rows),
        "sibling_open_count": int(
            annotations["Sibling_BH_Different"].fillna(False).sum()
        )
        if "Sibling_BH_Different" in annotations
        else 0,
        "root_sibling_p_value": root_sibling_p_value,
        "root_sibling_open": root_sibling_open,
        "root_stability_guard_block_count": int(root_guard_block_count),
        "root_stability_guard_blocked": root_stability_guard_blocked,
        "root_stability_subsample_mean_ari": root_stability_mean_ari,
        "root_stability_subsample_median_ari": root_stability_median_ari,
        "root_stability_subsample_q10_ari": root_stability_q10_ari,
        "root_selective_permutation_p_value": root_selective_p_value,
        "root_selective_permutation_guard_blocked": root_selective_guard_blocked,
        "root_selective_permutation_guard_would_block": (
            root_selective_guard_would_block
        ),
        "selective_permutation_guard_tested_count": int(
            selective_p_values.notna().sum()
        ),
        "selective_permutation_guard_block_count": int(selective_guard_block_count),
        "selective_permutation_guard_would_block_count": int(
            selective_guard_would_block_count
        ),
        "mean_selective_permutation_p_value": (
            float(selective_p_values.mean())
            if int(selective_p_values.notna().sum()) > 0
            else np.nan
        ),
        "min_selective_permutation_p_value": (
            float(selective_p_values.min())
            if int(selective_p_values.notna().sum()) > 0
            else np.nan
        ),
        "tree_linkage_method": "average",
        "tree_distance_metric": "hamming",
    }


def summarize_profile_validation_rows(
    rows: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> pd.DataFrame:
    """Summarize profile validation rows by case, role, and profile."""
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    group_columns = ("case_id", "data_role", "source_family", "profile_id")
    for key, group in rows.groupby(list(group_columns), sort=True):
        false_split_count = int(group["false_split"].astype(bool).sum())
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_replicates": int(group.shape[0]),
                "adaptive_projection_avoided_rate": float(
                    group["adaptive_projection_avoided"].astype(bool).mean()
                ),
                "adaptive_projection_detected_rate": float(
                    group["adaptive_projection_detected"].astype(bool).mean()
                ),
                "mean_ari": float(group["ari"].mean()),
                "mean_ari_lower_confidence": _mean_lower_bound(group["ari"]),
                "false_split_rate": float(group["false_split"].mean()),
                "false_split_rate_upper_confidence": _wilson_upper_bound(
                    false_split_count,
                    int(group.shape[0]),
                ),
                "mean_found_clusters": float(group["found_clusters"].mean()),
                "exact_cluster_count_rate": float(group["exact_cluster_count"].mean()),
                "mean_sibling_open_count": float(group["sibling_open_count"].mean()),
                "root_sibling_open_rate": (
                    float(group["root_sibling_open"].astype(bool).mean())
                    if "root_sibling_open" in group
                    else np.nan
                ),
                "mean_root_sibling_p_value": (
                    float(
                        pd.to_numeric(
                            group["root_sibling_p_value"],
                            errors="coerce",
                        ).mean()
                    )
                    if "root_sibling_p_value" in group
                    else np.nan
                ),
                "root_stability_guard_block_rate": float(
                    (group["root_stability_guard_block_count"] > 0).mean()
                ),
                "mean_root_stability_subsample_mean_ari": (
                    float(
                        pd.to_numeric(
                            group["root_stability_subsample_mean_ari"],
                            errors="coerce",
                        ).mean()
                    )
                    if "root_stability_subsample_mean_ari" in group
                    else np.nan
                ),
                "mean_root_stability_subsample_q10_ari": (
                    float(
                        pd.to_numeric(
                            group["root_stability_subsample_q10_ari"],
                            errors="coerce",
                        ).mean()
                    )
                    if "root_stability_subsample_q10_ari" in group
                    else np.nan
                ),
                "root_selective_permutation_guard_block_rate": (
                    float(
                        group["root_selective_permutation_guard_blocked"]
                        .astype(bool)
                        .mean()
                    )
                    if "root_selective_permutation_guard_blocked" in group
                    else np.nan
                ),
                "root_selective_permutation_guard_would_block_rate": (
                    float(
                        group["root_selective_permutation_guard_would_block"]
                        .astype(bool)
                        .mean()
                    )
                    if "root_selective_permutation_guard_would_block" in group
                    else np.nan
                ),
                "mean_root_selective_permutation_p_value": (
                    float(
                        pd.to_numeric(
                            group["root_selective_permutation_p_value"],
                            errors="coerce",
                        ).mean()
                    )
                    if "root_selective_permutation_p_value" in group
                    else np.nan
                ),
                "mean_selective_permutation_guard_tested_count": (
                    float(group["selective_permutation_guard_tested_count"].mean())
                    if "selective_permutation_guard_tested_count" in group
                    else np.nan
                ),
                "mean_selective_permutation_guard_block_count": (
                    float(group["selective_permutation_guard_block_count"].mean())
                    if "selective_permutation_guard_block_count" in group
                    else np.nan
                ),
                "mean_selective_permutation_guard_would_block_count": (
                    float(
                        group["selective_permutation_guard_would_block_count"].mean()
                    )
                    if "selective_permutation_guard_would_block_count" in group
                    else np.nan
                ),
                "mean_selective_permutation_p_value": (
                    float(
                        pd.to_numeric(
                            group["mean_selective_permutation_p_value"],
                            errors="coerce",
                        ).mean()
                    )
                    if "mean_selective_permutation_p_value" in group
                    else np.nan
                ),
                "min_selective_permutation_p_value": (
                    float(
                        pd.to_numeric(
                            group["min_selective_permutation_p_value"],
                            errors="coerce",
                        ).min()
                    )
                    if "min_selective_permutation_p_value" in group
                    else np.nan
                ),
                "observed_sibling_gate_method": ";".join(
                    sorted(set(group["observed_sibling_gate_method"].astype(str)))
                ),
                "profile_validation_status": _status(
                    group,
                    max_null_split_rate=max_null_split_rate,
                    min_signal_mean_ari=min_signal_mean_ari,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_profile_validation_transfer(
    summary: pd.DataFrame,
    *,
    max_null_split_rate: float = 0.05,
    min_signal_mean_ari: float = 0.75,
) -> pd.DataFrame:
    """Summarize transfer across included cases for each profile and family."""
    if summary.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    group_columns = ("source_family", "profile_id")
    for key, group in summary.groupby(list(group_columns), sort=True):
        null_rows = group[group["data_role"].eq("null")]
        signal_rows = group[group["data_role"].eq("signal")]
        required_zero_false = _required_zero_false_split_replicates(
            max_null_split_rate=float(max_null_split_rate)
        )
        min_null_replicates = (
            int(pd.to_numeric(null_rows["n_replicates"], errors="coerce").min())
            if not null_rows.empty and "n_replicates" in null_rows
            else 0
        )
        observed_false_counts: list[int] = []
        required_for_observed_counts: list[int] = []
        additional_for_observed_counts: list[int] = []
        if not null_rows.empty:
            for _, null_row in null_rows.iterrows():
                n_replicates = int(null_row["n_replicates"])
                false_count = int(
                    round(float(null_row["false_split_rate"]) * n_replicates)
                )
                required_for_count = _required_replicates_for_wilson_upper_bound(
                    false_count,
                    max_null_split_rate=float(max_null_split_rate),
                )
                observed_false_counts.append(false_count)
                required_for_observed_counts.append(required_for_count)
                additional_for_observed_counts.append(
                    max(0, required_for_count - n_replicates)
                )
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_case_role_rows": int(group.shape[0]),
                "n_null_case_rows": int(null_rows.shape[0]),
                "n_signal_case_rows": int(signal_rows.shape[0]),
                "min_replicates_per_case_role": int(group["n_replicates"].min()),
                "min_null_replicates_per_case": int(min_null_replicates),
                "required_zero_false_split_null_replicates_per_case": int(
                    required_zero_false
                ),
                "additional_zero_false_split_null_replicates_per_case": int(
                    max(0, required_zero_false - min_null_replicates)
                ),
                "max_observed_null_false_split_count": (
                    int(max(observed_false_counts))
                    if observed_false_counts
                    else 0
                ),
                "max_required_null_replicates_given_observed_false_splits": (
                    int(max(required_for_observed_counts))
                    if required_for_observed_counts
                    else 0
                ),
                "max_additional_zero_false_null_replicates_given_observed_false_splits": (
                    int(max(additional_for_observed_counts))
                    if additional_for_observed_counts
                    else 0
                ),
                "adaptive_projection_avoided": bool(
                    group["adaptive_projection_avoided_rate"].eq(1.0).all()
                ),
                "max_null_false_split_rate": (
                    float(null_rows["false_split_rate"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "max_null_false_split_rate_upper_confidence": (
                    float(null_rows["false_split_rate_upper_confidence"].max())
                    if not null_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari": (
                    float(signal_rows["mean_ari"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "min_signal_mean_ari_lower_confidence": (
                    float(signal_rows["mean_ari_lower_confidence"].min())
                    if not signal_rows.empty
                    else np.nan
                ),
                "profile_transfer_status": _transfer_status(group),
                "profile_confidence_status": _confidence_status(
                    group,
                    max_null_split_rate=max_null_split_rate,
                    min_signal_mean_ari=min_signal_mean_ari,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_profile_production_components(
    transfer_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build conservative production-admissibility rows for profile evidence."""
    if transfer_summary.empty:
        components = pd.DataFrame(
            [
                {
                    "contract_id": "fixed_sibling_gate_profile:empty",
                    "component_id": "fixed_sibling_gate_profile:empty",
                    "component_type": "fixed_sibling_gate_profile_validation",
                    "component_status": "fixed_profile_insufficient_coverage",
                    "required_for_production": True,
                    "notes": "No fixed profile validation rows were available.",
                }
            ]
        )
        rows = evaluate_production_admissibility_components(components)
        return rows, summarize_production_admissibility_contracts(rows)
    records: list[dict[str, object]] = []
    for _, row in transfer_summary.iterrows():
        context = f"{row['source_family']}:{row['profile_id']}"
        adaptive_status = (
            "fixed_profile_adaptive_projection_avoided"
            if bool(row["adaptive_projection_avoided"])
            else "fixed_profile_adaptive_projection_detected"
        )
        records.extend(
            [
                {
                    "contract_id": f"fixed_sibling_gate_profile:{context}",
                    "component_id": f"fixed_profile_adaptive_projection:{context}",
                    "component_type": "fixed_profile_adaptive_projection_check",
                    "component_status": adaptive_status,
                    "required_for_production": True,
                    "notes": (
                        "Profile must avoid projected_wald_inflation sibling rows "
                        "before production promotion can be considered."
                    ),
                },
                {
                    "contract_id": f"fixed_sibling_gate_profile:{context}",
                    "component_id": f"fixed_profile_transfer:{context}",
                    "component_type": "fixed_profile_traversal_transfer",
                    "component_status": str(row["profile_transfer_status"]),
                    "required_for_production": True,
                    "notes": "Point traversal evidence for the fixed sibling profile.",
                },
                {
                    "contract_id": f"fixed_sibling_gate_profile:{context}",
                    "component_id": f"fixed_profile_confidence:{context}",
                    "component_type": "fixed_profile_traversal_confidence",
                    "component_status": str(row["profile_confidence_status"]),
                    "required_for_production": True,
                    "notes": (
                        "Confidence-bound evidence. Smoke runs should remain "
                        "non-production until null and signal bounds pass."
                    ),
                },
            ]
        )
    components = pd.DataFrame.from_records(records)
    rows = evaluate_production_admissibility_components(components)
    return rows, summarize_production_admissibility_contracts(rows)


def build_method_constant_evidence_fields(
    *,
    config: FixedSiblingGateProfileValidationConfig,
    outputs: dict[str, Path],
    summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> dict[str, object]:
    """Build manifest-style evidence fields for fixed profile constants."""
    profile_metadata = _resolved_profile_metadata(config)
    resolved_configs = {
        profile_id: profile["resolved_runtime_config"]
        for profile_id, profile in profile_metadata.items()
    }
    root_selective_guard_by_profile = {
        profile_id: {
            "replicates": int(
                resolved["root_selective_permutation_guard_replicates"]
            ),
            "seed": int(resolved["root_selective_permutation_guard_seed"]),
            "alpha": resolved["root_selective_permutation_guard_alpha"],
            "scope": resolved["root_selective_permutation_guard_scope"],
            "status": (
                "enabled"
                if int(resolved["root_selective_permutation_guard_replicates"]) > 0
                else "disabled"
            ),
        }
        for profile_id, resolved in resolved_configs.items()
    }
    root_selective_guard_enabled = any(
        guard["status"] == "enabled"
        for guard in root_selective_guard_by_profile.values()
    )
    paths = {key: str(path) for key, path in outputs.items()}
    all_avoided = (
        bool(summary["adaptive_projection_avoided_rate"].eq(1.0).all())
        if not summary.empty
        else False
    )
    base = {
        "validation_design": (
            "Shared KL runner smoke over selected null/signal benchmark cases; "
            "diagnostic-only fixed sibling-gate profile validation."
        ),
        "source_artifact_paths": paths,
        "code_commit": "working_tree",
        "run_command": (
            "python -m benchmarks.diagnostics.calibration."
            "fixed_sibling_gate_profile_validation"
        ),
        "random_seed_policy": (
            f"base_seed={int(config.base_seed)}; replicate seed = base + 1009*r; "
            "root_selective_permutation_guard_seed="
            f"{int(config.root_selective_permutation_guard_seed)}"
        ),
        "n_replicates": int(config.replicates),
        "baseline_setting": "projected_wald_inflation default, not rerun here",
        "candidate_setting": ",".join(config.profiles),
        "primary_endpoint": (
            "adaptive_projection_avoided, null false split rate, signal mean ARI"
        ),
        "effect_estimate": {
            "summary_rows": int(summary.shape[0]),
            "transfer_rows": int(transfer_summary.shape[0]),
        },
        "confidence_interval": {
            "null": "Wilson upper bound for false split rate",
            "signal": "one-sided t lower bound for mean ARI",
        },
        "decision_rule": (
            "Diagnostic candidate only: all rows must avoid adaptive projection; "
            f"null upper confidence <= {config.max_null_split_rate:g}; "
            f"signal lower confidence >= {config.min_signal_mean_ari:g}."
        ),
        "limitations": (
            "This artifact validates routing and smoke-level traversal evidence. "
            "It is not a broad production calibration and does not compare "
            "TooManyCells as a direct benchmark method."
        ),
        "root_selective_permutation_guard": {
            "status": "enabled" if root_selective_guard_enabled else "disabled",
            "by_profile": root_selective_guard_by_profile,
        },
        "profile_grid": tuple(config.profiles),
        "profile_metadata": profile_metadata,
        "adaptive_projection_avoidance_check": {
            "all_summary_rows_avoided_adaptive_projection": all_avoided,
        },
        "null_false_split_confidence": transfer_summary[
            [
                "source_family",
                "profile_id",
                "max_null_false_split_rate",
                "max_null_false_split_rate_upper_confidence",
                "required_zero_false_split_null_replicates_per_case",
                "additional_zero_false_split_null_replicates_per_case",
                "max_observed_null_false_split_count",
                "max_required_null_replicates_given_observed_false_splits",
                "max_additional_zero_false_null_replicates_given_observed_false_splits",
            ]
        ].to_dict(orient="records")
        if not transfer_summary.empty
        else [],
        "signal_ari_confidence": transfer_summary[
            [
                "source_family",
                "profile_id",
                "min_signal_mean_ari",
                "min_signal_mean_ari_lower_confidence",
            ]
        ].to_dict(orient="records")
        if not transfer_summary.empty
        else [],
        "penalty_grid": sorted(
            {
                float(resolved["sibling_gate_alpha_penalty"])
                for resolved in resolved_configs.values()
            }
        ),
        "effective_sibling_alpha": {
            profile_id: (
                float(config.sibling_alpha)
                / float(resolved_configs[profile_id]["sibling_gate_alpha_penalty"])
            )
            for profile_id in profile_metadata
        },
        "threshold_grid": sorted(
            {
                resolved["root_stability_guard_threshold"]
                for resolved in resolved_configs.values()
                if resolved["root_stability_guard_threshold"] is not None
            }
        ),
        "root_stability_distribution": "see rows and summary outputs",
        "null_root_block_rate": "see summary root_stability_guard_block_rate",
        "signal_root_block_rate": "see summary root_stability_guard_block_rate",
        "root_selective_p_value_distribution": (
            "see rows root_selective_permutation_p_value and summary "
            "mean_root_selective_permutation_p_value"
        ),
        "subsample_replicate_grid": sorted(
            {
                int(resolved["root_stability_subsample_replicates"])
                for resolved in resolved_configs.values()
            }
        ),
        "permutation_replicate_grid": sorted(
            {
                int(resolved["root_selective_permutation_guard_replicates"])
                for resolved in resolved_configs.values()
            }
        ),
        "alpha_grid": _sorted_unique_values(
            tuple(
                resolved["root_selective_permutation_guard_alpha"]
                for resolved in resolved_configs.values()
            )
        ),
        "scope_grid": _sorted_unique_values(
            tuple(
                resolved["root_selective_permutation_guard_scope"]
                for resolved in resolved_configs.values()
            )
        ),
        "guard_decision_stability": "see rows and summary outputs",
        "runtime_cost_summary": "see manifest and runner timings in MethodRunResult extras",
        "feature_fraction_grid": sorted(
            {
                float(resolved["root_stability_feature_fraction"])
                for resolved in resolved_configs.values()
            }
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "created_at_utc": format_timestamp_utc(),
        "constants": {
            constant_id: {
                field: base[field]
                for field in COMMON_EVIDENCE_FIELDS + extra_fields
            }
            for constant_id, extra_fields in PROFILE_CONSTANT_FIELDS.items()
        },
    }


def run_fixed_sibling_gate_profile_validation(
    config: FixedSiblingGateProfileValidationConfig,
) -> dict[str, Path]:
    """Run fixed sibling-gate profile validation and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if not math.isfinite(float(config.sibling_alpha)) or not 0.0 < float(
        config.sibling_alpha
    ) < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    if not math.isfinite(float(config.edge_alpha)) or not 0.0 < float(
        config.edge_alpha
    ) < 1.0:
        raise ValueError("edge_alpha must lie in (0, 1).")
    if int(config.root_selective_permutation_guard_replicates) < 0:
        raise ValueError(
            "root_selective_permutation_guard_replicates must be nonnegative."
        )
    if (
        config.root_selective_permutation_guard_alpha is not None
        and not 0.0 < float(config.root_selective_permutation_guard_alpha) < 1.0
    ):
        raise ValueError(
            "root_selective_permutation_guard_alpha must lie in (0, 1)."
        )
    validate_profiles(config.profiles)
    validate_data_roles(config.data_roles)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    checkpoint_paths: list[Path] = []
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
                "Fixed profile validation supports binary and direct categorical "
                f"cases only; got {source_family!r}."
            )
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate * 1009
            for data_role in config.data_roles:
                for profile_id in config.profiles:
                    checkpoint_path = _checkpoint_row_path(
                        checkpoint_dir=config.checkpoint_rows_dir,
                        case_id=case_id,
                        data_role=data_role,
                        profile_id=profile_id,
                        replicate=replicate,
                    )
                    if bool(config.resume_from_checkpoints) and checkpoint_path.exists():
                        row = _read_checkpoint_row(checkpoint_path)
                    else:
                        row = _rows_for_replicate(
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
                            profile_id=profile_id,
                            sibling_alpha=float(config.sibling_alpha),
                            edge_alpha=float(config.edge_alpha),
                            root_selective_permutation_guard_replicates=int(
                                config.root_selective_permutation_guard_replicates
                            ),
                            root_selective_permutation_guard_seed=int(
                                config.root_selective_permutation_guard_seed
                            ),
                            root_selective_permutation_guard_alpha=(
                                None
                                if config.root_selective_permutation_guard_alpha
                                is None
                                else float(
                                    config.root_selective_permutation_guard_alpha
                                )
                            ),
                        )
                        _write_checkpoint_row(row, checkpoint_path)
                    rows.append(row)
                    checkpoint_paths.append(checkpoint_path)

    rows_df = pd.DataFrame.from_records(rows)
    summary = summarize_profile_validation_rows(
        rows_df,
        max_null_split_rate=float(config.max_null_split_rate),
        min_signal_mean_ari=float(config.min_signal_mean_ari),
    )
    transfer_summary = summarize_profile_validation_transfer(
        summary,
        max_null_split_rate=float(config.max_null_split_rate),
        min_signal_mean_ari=float(config.min_signal_mean_ari),
    )
    production_components, production_summary = build_profile_production_components(
        transfer_summary
    )
    outputs = {
        "rows": config.rows_path,
        "checkpoint_rows_dir": config.checkpoint_rows_dir,
        "summary": config.summary_path,
        "transfer_summary": config.transfer_summary_path,
        "production_components": config.production_components_path,
        "production_summary": config.production_summary_path,
    }
    rows_df.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    transfer_summary.to_csv(config.transfer_summary_path, index=False)
    production_components.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    evidence_fields = build_method_constant_evidence_fields(
        config=config,
        outputs=outputs,
        summary=summary,
        transfer_summary=transfer_summary,
    )
    config.evidence_fields_path.write_text(
        json.dumps(evidence_fields, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "config": {
            "suite": config.suite,
            "case_names": list(config.case_names),
            "profiles": list(config.profiles),
            "data_roles": list(config.data_roles),
            "sibling_alpha": float(config.sibling_alpha),
            "edge_alpha": float(config.edge_alpha),
            "replicates": int(config.replicates),
            "base_seed": int(config.base_seed),
            "resume_from_checkpoints": bool(config.resume_from_checkpoints),
            "root_selective_permutation_guard_replicates": int(
                config.root_selective_permutation_guard_replicates
            ),
            "root_selective_permutation_guard_seed": int(
                config.root_selective_permutation_guard_seed
            ),
            "root_selective_permutation_guard_alpha": (
                None
                if config.root_selective_permutation_guard_alpha is None
                else float(config.root_selective_permutation_guard_alpha)
            ),
            "max_null_split_rate": float(config.max_null_split_rate),
            "min_signal_mean_ari": float(config.min_signal_mean_ari),
        },
        "outputs": {
            **{key: str(path) for key, path in outputs.items()},
            "checkpoint_rows": [str(path) for path in checkpoint_paths],
            "method_constant_evidence_fields": str(config.evidence_fields_path),
        },
        "interpretation": (
            "Diagnostic-only validation that named fixed sibling-gate profiles "
            "avoid adaptive sibling projection in the shared KL runner and record "
            "traversal point/confidence evidence."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **outputs,
        "method_constant_evidence_fields": config.evidence_fields_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", type=parse_names, default=())
    parser.add_argument("--profiles", type=parse_names, default=DEFAULT_PROFILES)
    parser.add_argument("--data-roles", type=parse_names, default=DEFAULT_DATA_ROLES)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--root-selective-permutation-guard-replicates", type=int, default=0)
    parser.add_argument("--root-selective-permutation-guard-seed", type=int, default=0)
    parser.add_argument("--root-selective-permutation-guard-alpha", type=float)
    parser.add_argument("--max-null-split-rate", type=float, default=0.05)
    parser.add_argument("--min-signal-mean-ari", type=float, default=0.75)
    parser.add_argument("--resume-from-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = FixedSiblingGateProfileValidationConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=tuple(args.case_names),
        profiles=validate_profiles(args.profiles),
        data_roles=validate_data_roles(args.data_roles),
        sibling_alpha=float(args.sibling_alpha),
        edge_alpha=float(args.edge_alpha),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
        root_selective_permutation_guard_replicates=int(
            args.root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=int(
            args.root_selective_permutation_guard_seed
        ),
        root_selective_permutation_guard_alpha=(
            None
            if args.root_selective_permutation_guard_alpha is None
            else float(args.root_selective_permutation_guard_alpha)
        ),
        max_null_split_rate=float(args.max_null_split_rate),
        min_signal_mean_ari=float(args.min_signal_mean_ari),
        resume_from_checkpoints=bool(args.resume_from_checkpoints),
    )
    outputs = run_fixed_sibling_gate_profile_validation(config)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
