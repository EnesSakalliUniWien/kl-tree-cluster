"""Typed reasons for scientifically unsupported benchmark outcomes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class UnsupportedReasonCode(str, Enum):
    """Registered machine-readable unsupported outcome reasons."""

    EMPIRICAL_NULL_NO_INTERNAL_SUPPORT = "empirical_null_no_internal_support"


@dataclass(frozen=True)
class UnsupportedEvidence:
    """Stable numeric evidence attached to an unsupported outcome."""

    focal_record_count: int | None = None
    admissible_support_count: int | None = None
    invalid_record_count: int | None = None
    upstream_tested_count: int | None = None
    upstream_rejected_count: int | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Unsupported evidence {field.name} must be a non-negative integer "
                    f"or None; got {value!r}."
                )


@dataclass(frozen=True)
class UnsupportedReason:
    """Machine-readable and human-readable unsupported outcome description."""

    code: UnsupportedReasonCode | str
    stage: str
    message: str
    evidence: UnsupportedEvidence

    def __post_init__(self) -> None:
        try:
            code = UnsupportedReasonCode(self.code)
        except ValueError as exc:
            raise ValueError(f"Unknown unsupported reason code: {self.code!r}.") from exc
        object.__setattr__(self, "code", code)

        if code is UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT:
            if self.stage != "sibling_calibration":
                raise ValueError(
                    "empirical_null_no_internal_support requires "
                    "stage='sibling_calibration'."
                )
            evidence_values = [
                getattr(self.evidence, field.name) for field in fields(self.evidence)
            ]
            if any(value is None for value in evidence_values):
                raise ValueError(
                    "empirical_null_no_internal_support requires all evidence counts."
                )
            if self.evidence.admissible_support_count != 0:
                raise ValueError(
                    "empirical_null_no_internal_support requires "
                    "admissible_support_count=0."
                )

        normalized_message = self.message.strip()
        if not normalized_message:
            raise ValueError("Unsupported reason message must be non-empty.")
        object.__setattr__(self, "message", normalized_message)


__all__ = [
    "UnsupportedEvidence",
    "UnsupportedReason",
    "UnsupportedReasonCode",
]
