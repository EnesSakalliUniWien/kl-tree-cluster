"""Barycentric action helpers for diagnostic equation traces."""

from __future__ import annotations

import math

import numpy as np


def _as_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {array.shape}.")
    return array


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


def barycentric_contrast_residuals(
    *,
    theta_parent: np.ndarray,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    left_size: int,
    right_size: int,
) -> dict[str, float | str]:
    """Return residuals for the exact binary-parent barycentric identities."""
    left = _as_vector(theta_left)
    right = _as_vector(theta_right)
    parent = _as_vector(theta_parent)
    if left.shape != right.shape or left.shape != parent.shape:
        raise ValueError("Parent, left, and right vectors must have the same shape.")

    total = int(left_size) + int(right_size)
    if total <= 0 or left_size <= 0 or right_size <= 0:
        return {"status": "invalid_sizes"}

    beta = float(left_size / total)
    delta = left - right
    expected_parent = beta * left + (1.0 - beta) * right
    return {
        "status": "ok",
        "beta": beta,
        "parent_barycenter_residual_norm": _norm(parent - expected_parent),
        "left_contrast_residual_norm": _norm((left - parent) - (1.0 - beta) * delta),
        "right_contrast_residual_norm": _norm((right - parent) + beta * delta),
        "sibling_delta_norm": _norm(delta),
    }


def z_identity_residuals(
    *,
    z_edge_left: np.ndarray | None,
    z_edge_right: np.ndarray | None,
    z_sibling: np.ndarray | None,
) -> dict[str, float | str]:
    """Return residuals for the default no-branch-scaling edge/sibling identity."""
    if z_edge_left is None or z_edge_right is None or z_sibling is None:
        return {"status": "missing_edge_context"}

    left = _as_vector(z_edge_left)
    right = _as_vector(z_edge_right)
    sibling = _as_vector(z_sibling)
    if left.shape != sibling.shape or right.shape != sibling.shape:
        return {"status": "incompatible_projection"}

    return {
        "status": "ok",
        "left_z_identity_residual_norm": _norm(left - sibling),
        "right_z_identity_residual_norm": _norm(right + sibling),
    }


def action_budget(
    *,
    coords: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> dict[str, float | str]:
    """Return the parallel-axis split action and its share of parent inertia."""
    values = np.asarray(coords, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D coordinates, got shape {values.shape}.")

    left_ids = np.asarray(left_indices, dtype=int)
    right_ids = np.asarray(right_indices, dtype=int)
    if left_ids.size == 0 or right_ids.size == 0:
        return {"status": "invalid_sizes"}

    parent_ids = np.concatenate([left_ids, right_ids])
    left_centroid = values[left_ids].mean(axis=0)
    right_centroid = values[right_ids].mean(axis=0)
    parent_centroid = values[parent_ids].mean(axis=0)
    parent_centered = values[parent_ids] - parent_centroid

    n_left = int(left_ids.size)
    n_right = int(right_ids.size)
    n_parent = n_left + n_right
    beta = n_left / n_parent
    split_action = (
        n_parent * beta * (1.0 - beta) * float(np.sum((left_centroid - right_centroid) ** 2))
    )
    parent_inertia = float(np.sum(parent_centered**2))
    return {
        "status": "ok",
        "beta": float(beta),
        "split_action": split_action,
        "parent_inertia": parent_inertia,
        "within_child_inertia": parent_inertia - split_action,
        "action_fraction": (
            float(split_action / parent_inertia) if parent_inertia > 0.0 else math.nan
        ),
    }
