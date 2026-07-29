"""DataFrame enrichment for mass-weighted tree distributional-action diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.distributional_action import (
    split_distributional_action_summary,
)


def _standardization_scale(leaf_data: pd.DataFrame) -> np.ndarray:
    values = leaf_data.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("leaf_data must have at least one feature column.")
    if not np.all(np.isfinite(values)):
        raise ValueError("leaf_data must contain only finite values.")
    scale = np.std(values, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    return scale


def annotate_distributional_action_diagnostics(
    tree,
    annotated_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return annotations enriched with non-gating edge and split action diagnostics."""
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
    }
    for column, default in columns.items():
        out[column] = default

    action_rows: dict[object, dict[str, object]] = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if not children:
            continue
        child_means = [
            np.asarray(tree.nodes[child]["distribution"], dtype=float) / scale for child in children
        ]
        child_masses = [tree.nodes[child]["leaf_count"] for child in children]
        split_summary = split_distributional_action_summary(child_means, child_masses)
        split_action = split_summary.action / feature_count
        for child_index, child in enumerate(children):
            if child not in out.index:
                continue
            child_edge_action = split_summary.child_edge_actions[child_index] / feature_count
            squared_delta_sq_mean = (
                split_summary.squared_child_parent_displacements[child_index] / feature_count
            )
            action_rows[child] = {
                "Distributional_Action_Parent_Mass": split_summary.parent_mass,
                "Distributional_Action_Child_Mass": split_summary.child_masses[child_index],
                "Distributional_Action_Child_Parent_Mass_Fraction": (
                    split_summary.child_parent_mass_fractions[child_index]
                ),
                "Distributional_Action_Standardized_Delta_Sq_Mean": squared_delta_sq_mean,
                "Distributional_Action": child_edge_action,
                "Distributional_Action_Parent_Fraction_Action": (
                    split_summary.child_parent_mass_fractions[child_index] * squared_delta_sq_mean
                ),
                "Distributional_Split_Action_Parent": parent,
                "Distributional_Split_Action_Parent_Mass": split_summary.parent_mass,
                "Distributional_Split_Action_Child_Count": len(children),
                "Distributional_Split_Action": split_action,
                "Distributional_Split_Action_Child_Edge_Share": (
                    child_edge_action / split_action if split_action > 0.0 else np.nan
                ),
            }

    if action_rows:
        action_frame = pd.DataFrame.from_dict(action_rows, orient="index")
        out.loc[action_frame.index, action_frame.columns] = action_frame
    return out


__all__ = ["annotate_distributional_action_diagnostics"]
