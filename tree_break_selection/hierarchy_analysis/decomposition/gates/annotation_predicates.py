"""Shared predicates over gate annotation tables."""

from __future__ import annotations

import pandas as pd


def annotation_bool(
    annotations_df: pd.DataFrame,
    node: object,
    column: str,
) -> bool:
    """Return a boolean annotation value, treating missing values as false."""
    if node not in annotations_df.index or column not in annotations_df.columns:
        return False
    value = annotations_df.loc[node, column]
    return bool(pd.notna(value) and bool(value))


def node_split_prerequisites(
    tree,
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    """Return whether a binary node has child-parent evidence to consider splitting."""
    children = list(tree.successors(node))
    if len(children) != 2:
        return False
    return any(
        annotation_bool(
            annotations_df,
            child,
            "Child_Parent_Divergence_Significant",
        )
        for child in children
    )


def node_sibling_gate_open(
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    """Return whether the sibling gate opened and was not skipped at a node."""
    return annotation_bool(
        annotations_df,
        node,
        "Sibling_BH_Different",
    ) and not annotation_bool(
        annotations_df,
        node,
        "Sibling_Divergence_Skipped",
    )


__all__ = [
    "annotation_bool",
    "node_sibling_gate_open",
    "node_split_prerequisites",
]
