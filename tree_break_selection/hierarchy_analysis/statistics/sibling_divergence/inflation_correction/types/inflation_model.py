"""Empirical-null inflation model result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

CalibrationDecisionStatus = Literal[
    "internal_admissible",
    "undefined_no_internal_support",
    "undefined_no_family_support",
    "undefined_sparse_context",
]


@dataclass(frozen=True)
class CalibrationDecision:
    """Focal sibling calibration decision and support evidence."""

    status: CalibrationDecisionStatus
    c_hat: float | None
    p_value: float | None
    estimator: str
    support: dict[str, float | int | str | bool]
    exact_context: dict[str, object]
    descriptive_strata: dict[str, object]


@dataclass(frozen=True)
class CalibrationSupportThresholds:
    """Validation thresholds for the internal calibration support contract."""

    min_supported_records: int = 30
    min_family_supported_records: int = 15
    min_stopped_or_null_records: int = 10
    min_family_effective_sample_size: float = 10.0
    min_local_effective_sample_size: float = 8.0
    max_weight_share: float = 0.25
    max_leave_one_record_delta_log_c: float = float(np.log(1.25))


DEFAULT_INTERNAL_SUPPORT_THRESHOLDS = CalibrationSupportThresholds()


@dataclass
class EmpiricalNullInflationModel:
    """Result of fitting the context-weighted empirical-null inflation."""

    method: str
    n_calibration: int
    baseline_empirical_inflation_factor: float
    effective_sample_size: float
    n_positive_weight_records: int = 0
    n_selected_nonnull_positive_weight_records: int = 0
    n_strict_null_calibration: int = 0
    n_edge_blocked_calibration: int = 0
    n_stopped_or_null_calibration: int = 0
    context_center: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    context_bandwidth: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_contexts: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    sample_feature_families: tuple[str, ...] = ()
    sample_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_parent_ids: tuple[object, ...] = ()
    sample_is_strict_null: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    sample_is_edge_blocked: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    sample_statistics: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_reference_scales: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_degrees_of_freedom: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


__all__ = [
    "CalibrationDecision",
    "CalibrationDecisionStatus",
    "CalibrationSupportThresholds",
    "DEFAULT_INTERNAL_SUPPORT_THRESHOLDS",
    "EmpiricalNullInflationModel",
]
