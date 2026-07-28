"""Production contract for promoted external selected-tail calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .types.inflation_model import CalibrationDecision

ExternalContextValue = str | int | float | bool


def _normalize_context_value(value: object) -> ExternalContextValue:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("External selected-tail context values must be finite.")
        return int(value) if value.is_integer() else float(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        "External selected-tail context values must be str, int, float, or bool; "
        f"got {type(value).__name__}."
    )


def _normalize_context(
    context: Mapping[str, object],
) -> tuple[tuple[str, ExternalContextValue], ...]:
    if not context:
        raise ValueError("External selected-tail context cannot be empty.")
    return tuple(
        sorted((str(key), _normalize_context_value(value)) for key, value in context.items())
    )


def _context_dict(
    context: Mapping[str, object],
) -> dict[str, ExternalContextValue]:
    return {str(key): _normalize_context_value(value) for key, value in context.items()}


def _external_adjusted_p_value(
    record: SiblingPairRecord,
    *,
    c_hat: float,
) -> float:
    if record.degrees_of_freedom == 0:
        return 1.0
    adjusted_statistic = float(record.stat / (record.reference_scale * c_hat))
    return float(chi2.sf(adjusted_statistic, df=float(record.degrees_of_freedom)))


@dataclass(frozen=True)
class ExternalSelectedTailCalibrationRule:
    """One pre-promoted external selected-tail calibration context."""

    exact_context: Mapping[str, object]
    c_hat: float
    support: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    descriptive_strata: Mapping[str, object] = field(default_factory=dict)
    estimator: str = "external_selected_tail_scalar"

    def __post_init__(self) -> None:
        _normalize_context(self.exact_context)
        if not np.isfinite(self.c_hat) or self.c_hat < 1.0:
            raise ValueError("External selected-tail c_hat must be finite and >= 1.0.")

    @property
    def normalized_context(self) -> tuple[tuple[str, ExternalContextValue], ...]:
        return _normalize_context(self.exact_context)


@dataclass(frozen=True)
class ExternalSelectedTailCalibrationModel:
    """Exact-match lookup for externally promoted selected-tail contexts."""

    rules: tuple[ExternalSelectedTailCalibrationRule, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[tuple[str, ExternalContextValue], ...]] = set()
        for rule in self.rules:
            normalized = rule.normalized_context
            if normalized in seen:
                raise ValueError("External selected-tail calibration contexts must be unique.")
            seen.add(normalized)

    def decision_for(
        self,
        record: SiblingPairRecord,
        *,
        external_context: Mapping[str, object] | None,
        internal_decision: CalibrationDecision | None = None,
    ) -> CalibrationDecision:
        """Return an external calibration decision for a focal sibling record."""
        context = {
            "feature_family": record.feature_family,
            "sibling_projection_dimension": record.sibling_projection_dimension,
        }
        if external_context is not None:
            context.update(dict(external_context))
        normalized_context = _context_dict(context)
        rule = self._find_rule(normalized_context)
        internal_status = "not_evaluated" if internal_decision is None else internal_decision.status
        if rule is None:
            return CalibrationDecision(
                status="undefined_external_not_admissible",
                c_hat=None,
                p_value=None,
                estimator="external_selected_tail_scalar",
                support={
                    "external_selected_tail_status": "not_matched",
                    "n_external_selected_tail_rules": int(len(self.rules)),
                },
                exact_context=normalized_context,
                descriptive_strata={
                    "reason": "no_matching_external_selected_tail_context",
                    "internal_status": internal_status,
                },
            )

        support = dict(rule.support)
        support.update(
            {
                "external_selected_tail_status": "matched",
                "n_external_selected_tail_rules": int(len(self.rules)),
            }
        )
        descriptive_strata = dict(rule.descriptive_strata)
        descriptive_strata["internal_status"] = internal_status
        return CalibrationDecision(
            status="external_admissible_scalar",
            c_hat=float(rule.c_hat),
            p_value=_external_adjusted_p_value(record, c_hat=float(rule.c_hat)),
            estimator=str(rule.estimator),
            support=support,
            exact_context=normalized_context,
            descriptive_strata=descriptive_strata,
        )

    def _find_rule(
        self,
        context: Mapping[str, ExternalContextValue],
    ) -> ExternalSelectedTailCalibrationRule | None:
        for rule in self.rules:
            rule_context = dict(rule.normalized_context)
            if all(context.get(key) == value for key, value in rule_context.items()):
                return rule
        return None


__all__ = [
    "ExternalSelectedTailCalibrationModel",
    "ExternalSelectedTailCalibrationRule",
]
