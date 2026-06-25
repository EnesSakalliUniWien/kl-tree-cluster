"""Mass-weighted distributional movement diagnostics for tree edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE = "none"
DISTRIBUTIONAL_ACTION_SPLIT_FILTER_OPEN_SPLIT_QUANTILE = "open_split_quantile"

_DISTRIBUTIONAL_ACTION_SPLIT_FILTER_POLICIES = {
    DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    DISTRIBUTIONAL_ACTION_SPLIT_FILTER_OPEN_SPLIT_QUANTILE,
}


@dataclass(frozen=True)
class EdgeDistributionalActionSummary:
    parent_mass: float
    child_mass: float
    child_parent_mass_fraction: float
    squared_displacement: float
    action: float


@dataclass(frozen=True)
class BinarySplitDistributionalActionSummary:
    left_mass: float
    right_mass: float
    parent_mass: float
    parent_mean: np.ndarray
    left_parent_mass_fraction: float
    right_parent_mass_fraction: float
    squared_child_displacement: float
    left_edge_action: float
    right_edge_action: float
    action: float


@dataclass(frozen=True)
class SplitDistributionalActionSummary:
    child_masses: tuple[float, ...]
    parent_mass: float
    parent_mean: np.ndarray
    child_parent_mass_fractions: tuple[float, ...]
    squared_child_parent_displacements: tuple[float, ...]
    child_edge_actions: tuple[float, ...]
    action: float


def _as_1d_float_vector(value: ArrayLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _as_positive_count(value: int | float, *, name: str) -> float:
    count = float(value)
    if not np.isfinite(count) or count <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return count


def validate_distributional_action_split_filter_policy(policy: str) -> str:
    resolved = str(policy)
    if resolved not in _DISTRIBUTIONAL_ACTION_SPLIT_FILTER_POLICIES:
        raise ValueError(
            "distributional_action_split_filter_policy must be one of "
            f"{sorted(_DISTRIBUTIONAL_ACTION_SPLIT_FILTER_POLICIES)!r}; got {policy!r}."
        )
    return resolved


def validate_distributional_action_split_filter_quantile(quantile: float) -> float:
    value = float(quantile)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            "distributional_action_split_filter_quantile must be a finite value in [0, 1]."
        )
    return value


def _standardization_scale(leaf_data: pd.DataFrame) -> np.ndarray:
    values = leaf_data.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("leaf_data must have at least one feature column.")
    if not np.all(np.isfinite(values)):
        raise ValueError("leaf_data must contain only finite values.")
    scale = np.std(values, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    return scale


def edge_distributional_action(
    parent_mean: ArrayLike,
    child_mean: ArrayLike,
    child_leaf_count: int | float,
) -> float:
    """Return child mass times squared child-parent mean displacement."""
    parent = _as_1d_float_vector(parent_mean, name="parent_mean")
    child = _as_1d_float_vector(child_mean, name="child_mean")
    if parent.shape != child.shape:
        raise ValueError("parent_mean and child_mean must have the same shape.")
    child_count = _as_positive_count(child_leaf_count, name="child_leaf_count")
    delta = child - parent
    return float(child_count * np.dot(delta, delta))


def edge_distributional_action_summary(
    parent_mean: ArrayLike,
    child_mean: ArrayLike,
    parent_leaf_count: int | float,
    child_leaf_count: int | float,
) -> EdgeDistributionalActionSummary:
    """Return edge action together with parent and child distribution masses."""
    parent = _as_1d_float_vector(parent_mean, name="parent_mean")
    child = _as_1d_float_vector(child_mean, name="child_mean")
    if parent.shape != child.shape:
        raise ValueError("parent_mean and child_mean must have the same shape.")
    parent_count = _as_positive_count(parent_leaf_count, name="parent_leaf_count")
    child_count = _as_positive_count(child_leaf_count, name="child_leaf_count")
    delta = child - parent
    squared_displacement = float(np.dot(delta, delta))
    return EdgeDistributionalActionSummary(
        parent_mass=parent_count,
        child_mass=child_count,
        child_parent_mass_fraction=float(child_count / parent_count),
        squared_displacement=squared_displacement,
        action=float(child_count * squared_displacement),
    )


def annotate_distributional_action_split_filter(
    tree,
    annotated_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
    *,
    policy: str = DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    quantile: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Annotate edge/split action columns without changing clustering gates.

    Edge action is the child contribution to the parent split. The split action is
    ``sum_i m_i * mean(((child_i_mean - parent_mean) / leaf_feature_std) ** 2)``.
    It is a diagnostic quantity, not a calibrated replacement for the projected-Wald
    edge or sibling gates used by the clustering traversal.
    """
    resolved_policy = validate_distributional_action_split_filter_policy(policy)
    resolved_quantile = validate_distributional_action_split_filter_quantile(quantile)
    if resolved_policy != DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE:
        raise ValueError(
            "distributional-action split filters are diagnostic-only until they are "
            "calibrated against the projected-Wald edge/sibling gates; use policy='none'."
        )
    if "Child_Parent_Divergence_Significant" not in annotated_df.columns:
        raise ValueError(
            "annotated_df must include Child_Parent_Divergence_Significant before "
            "distributional-action split filtering."
        )

    out = annotated_df.copy()
    scale = _standardization_scale(leaf_data)
    feature_count = float(scale.size)
    columns = {
        "Distributional_Action_Parent_Mass": np.nan,
        "Distributional_Action_Child_Mass": np.nan,
        "Distributional_Action_Child_Parent_Mass_Fraction": np.nan,
        "Distributional_Action_Standardized_Delta_Sq_Mean": np.nan,
        "Distributional_Action": np.nan,
        "Distributional_Action_Parent_Fraction_Action": np.nan,
        "Distributional_Split_Action_Parent": "",
        "Distributional_Split_Action_Parent_Mass": np.nan,
        "Distributional_Split_Action_Child_Count": np.nan,
        "Distributional_Split_Action": np.nan,
        "Distributional_Split_Action_Child_Edge_Share": np.nan,
        "Distributional_Split_Action_Filter_Policy": resolved_policy,
        "Distributional_Split_Action_Filter_Quantile": resolved_quantile,
        "Distributional_Split_Action_Filter_Threshold": np.nan,
        "Distributional_Split_Action_Filter_Passes": True,
        "Distributional_Split_Action_Filtered": False,
    }
    for column, default in columns.items():
        out[column] = default

    split_action_by_parent: dict[object, float] = {}
    children_by_parent: dict[object, list[object]] = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if not children:
            continue
        child_means = [
            np.asarray(tree.nodes[child]["distribution"], dtype=float) / scale
            for child in children
        ]
        child_masses = [tree.nodes[child]["leaf_count"] for child in children]
        split_summary = split_distributional_action_summary(child_means, child_masses)
        split_action = split_summary.action / feature_count
        split_action_by_parent[parent] = split_action
        children_by_parent[parent] = children
        for child_index, child in enumerate(children):
            if child not in out.index:
                continue
            child_edge_action = split_summary.child_edge_actions[child_index] / feature_count
            squared_delta_sq_mean = (
                split_summary.squared_child_parent_displacements[child_index]
                / feature_count
            )
            out.loc[child, "Distributional_Action_Parent_Mass"] = (
                split_summary.parent_mass
            )
            out.loc[child, "Distributional_Action_Child_Mass"] = (
                split_summary.child_masses[child_index]
            )
            out.loc[child, "Distributional_Action_Child_Parent_Mass_Fraction"] = (
                split_summary.child_parent_mass_fractions[child_index]
            )
            out.loc[child, "Distributional_Action_Standardized_Delta_Sq_Mean"] = (
                squared_delta_sq_mean
            )
            out.loc[child, "Distributional_Action"] = child_edge_action
            out.loc[child, "Distributional_Action_Parent_Fraction_Action"] = (
                split_summary.child_parent_mass_fractions[child_index]
                * squared_delta_sq_mean
            )
            out.loc[child, "Distributional_Split_Action_Parent"] = parent
            out.loc[child, "Distributional_Split_Action_Parent_Mass"] = (
                split_summary.parent_mass
            )
            out.loc[child, "Distributional_Split_Action_Child_Count"] = len(children)
            out.loc[child, "Distributional_Split_Action"] = split_action
            if split_action > 0.0:
                out.loc[child, "Distributional_Split_Action_Child_Edge_Share"] = (
                    child_edge_action / split_action
                )

    edge_open_before = out["Child_Parent_Divergence_Significant"].fillna(False).astype(bool)
    split_open_before = {}
    for parent, children in children_by_parent.items():
        if len(children) != 2:
            continue
        child_rows = [child for child in children if child in out.index]
        split_open_before[parent] = bool(edge_open_before.reindex(child_rows).fillna(False).any())

    candidate_parents = [
        parent
        for parent, is_open in split_open_before.items()
        if is_open and np.isfinite(split_action_by_parent[parent])
    ]
    threshold = np.nan
    filtered_edges = pd.Series(False, index=out.index)
    out["Distributional_Split_Action_Filter_Threshold"] = threshold
    out["Distributional_Split_Action_Filtered"] = filtered_edges.astype(bool)
    edge_open_after = out["Child_Parent_Divergence_Significant"].fillna(False).astype(bool)
    open_parent_after_count = 0
    for parent in candidate_parents:
        children = [child for child in children_by_parent[parent] if child in out.index]
        if edge_open_after.reindex(children).fillna(False).any():
            open_parent_after_count += 1

    metadata = {
        "distributional_action_split_filter_policy": resolved_policy,
        "distributional_action_split_filter_quantile": resolved_quantile,
        "distributional_action_split_filter_threshold": threshold,
        "distributional_action_split_filter_candidate_parent_count": len(candidate_parents),
        "distributional_action_split_filter_open_parent_count_before": len(candidate_parents),
        "distributional_action_split_filter_open_parent_count_after": open_parent_after_count,
        "distributional_action_split_filter_filtered_parent_count": 0,
        "distributional_action_split_filter_filtered_edge_count": int(filtered_edges.sum()),
        "distributional_action_split_filter_edge_count": int(
            np.isfinite(pd.to_numeric(out["Distributional_Action"], errors="coerce")).sum()
        ),
        "distributional_action_split_filter_definition": (
            "sum_i child_leaf_count_i * "
            "mean(((child_i_mean - parent_mean) / leaf_feature_std) ** 2)"
        ),
    }
    return out, metadata


def binary_split_distributional_action(
    left_mean: ArrayLike,
    right_mean: ArrayLike,
    left_leaf_count: int | float,
    right_leaf_count: int | float,
) -> float:
    """Return between-child distributional action for a binary split."""
    return binary_split_distributional_action_summary(
        left_mean,
        right_mean,
        left_leaf_count,
        right_leaf_count,
    ).action


def split_distributional_action_summary(
    child_means: list[ArrayLike] | tuple[ArrayLike, ...],
    child_masses: list[int | float] | tuple[int | float, ...],
) -> SplitDistributionalActionSummary:
    """Return the barycentric split action for one internal node.

    This is the between-child term in the parallel-axis decomposition:
    ``sum_i m_i ||mu_i - mu_parent||^2``.
    """
    if len(child_means) != len(child_masses):
        raise ValueError("child_means and child_masses must have the same length.")
    if len(child_means) == 0:
        raise ValueError("At least one child distribution is required.")
    means = tuple(
        _as_1d_float_vector(mean, name=f"child_means[{index}]")
        for index, mean in enumerate(child_means)
    )
    first_shape = means[0].shape
    if any(mean.shape != first_shape for mean in means):
        raise ValueError("All child means must have the same shape.")
    masses = tuple(
        _as_positive_count(mass, name=f"child_masses[{index}]")
        for index, mass in enumerate(child_masses)
    )
    parent_mass = float(sum(masses))
    parent_mean = sum(mass * mean for mass, mean in zip(masses, means, strict=True))
    parent_mean = np.asarray(parent_mean, dtype=float) / parent_mass
    squared_displacements = tuple(
        float(np.dot(mean - parent_mean, mean - parent_mean)) for mean in means
    )
    edge_actions = tuple(
        float(mass * squared_displacement)
        for mass, squared_displacement in zip(masses, squared_displacements, strict=True)
    )
    return SplitDistributionalActionSummary(
        child_masses=masses,
        parent_mass=parent_mass,
        parent_mean=parent_mean,
        child_parent_mass_fractions=tuple(mass / parent_mass for mass in masses),
        squared_child_parent_displacements=squared_displacements,
        child_edge_actions=edge_actions,
        action=float(sum(edge_actions)),
    )


def binary_split_distributional_action_summary(
    left_mean: ArrayLike,
    right_mean: ArrayLike,
    left_leaf_count: int | float,
    right_leaf_count: int | float,
) -> BinarySplitDistributionalActionSummary:
    """Return split action together with child and parent distribution masses."""
    left = _as_1d_float_vector(left_mean, name="left_mean")
    right = _as_1d_float_vector(right_mean, name="right_mean")
    if left.shape != right.shape:
        raise ValueError("left_mean and right_mean must have the same shape.")
    left_count = _as_positive_count(left_leaf_count, name="left_leaf_count")
    right_count = _as_positive_count(right_leaf_count, name="right_leaf_count")
    total_count = left_count + right_count
    delta = left - right
    squared_displacement = float(np.dot(delta, delta))
    split_summary = split_distributional_action_summary(
        (left, right),
        (left_count, right_count),
    )
    return BinarySplitDistributionalActionSummary(
        left_mass=left_count,
        right_mass=right_count,
        parent_mass=total_count,
        parent_mean=split_summary.parent_mean,
        left_parent_mass_fraction=float(left_count / total_count),
        right_parent_mass_fraction=float(right_count / total_count),
        squared_child_displacement=squared_displacement,
        left_edge_action=split_summary.child_edge_actions[0],
        right_edge_action=split_summary.child_edge_actions[1],
        action=float((left_count * right_count / total_count) * squared_displacement),
    )
