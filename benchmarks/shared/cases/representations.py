"""Helpers for explicit representation-forwarding benchmark cases."""

from __future__ import annotations


def representation_variant(
    case: dict,
    *,
    generator: str,
    name_suffix: str,
    representation_role: str,
) -> dict:
    """Return a case that forwards one stochastic source to another representation."""
    variant = case.copy()
    variant["name"] = f"{case['name']}{name_suffix}"
    variant["generator"] = generator
    variant["baseline_case_name"] = case["name"]
    variant["representation_role"] = representation_role
    return variant


def representation_variants_by_name(
    cases: list[dict],
    *,
    selected_names: tuple[str, ...],
    generator: str,
    name_suffix: str,
    representation_role: str,
) -> list[dict]:
    """Forward selected source cases to a second representation."""
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Cannot build representation variants for missing cases: {missing}")
    return [
        representation_variant(
            by_name[name],
            generator=generator,
            name_suffix=name_suffix,
            representation_role=representation_role,
        )
        for name in selected_names
    ]


__all__ = ["representation_variant", "representation_variants_by_name"]
