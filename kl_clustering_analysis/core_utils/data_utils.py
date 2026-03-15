from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

_TRUE_BOOL_TOKENS = {"1", "true"}
_FALSE_BOOL_TOKENS = {"0", "false"}
_BOOL_NULL_POLICIES = {"raise", "false", "true"}


def _normalize_strict_bool(
    value: object,
    *,
    null_policy: Literal["raise", "false", "true"] = "raise",
) -> bool | None:
    """Normalize strict boolean-like values to bool, otherwise ``None``.

    Accepted values are:
    - bool / np.bool_
    - integers 0 or 1
    - finite floats 0.0 or 1.0
    - case-insensitive strings in {_TRUE_BOOL_TOKENS, _FALSE_BOOL_TOKENS}

    Parameters
    ----------
    value
        Value to normalize.
    null_policy
        How to handle nulls (None/NaN):
        - ``"raise"``: return None (caller should raise)
        - ``"false"``: coerce null to False
        - ``"true"``: coerce null to True
    """
    if null_policy not in _BOOL_NULL_POLICIES:
        raise ValueError(
            f"Invalid null_policy {null_policy!r}. " "Expected one of {'raise', 'false', 'true'}."
        )

    if pd.isna(value):
        if null_policy == "false":
            return False
        if null_policy == "true":
            return True
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
        return None

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return None
        if float(value) in (0.0, 1.0):
            return bool(int(value))
        return None

    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_BOOL_TOKENS:
            return True
        if token in _FALSE_BOOL_TOKENS:
            return False

    return None


def normalize_bool_series(
    series: pd.Series,
    *,
    column_name: str,
    null_policy: Literal["raise", "false", "true"] = "raise",
) -> pd.Series:
    """Normalize a pandas Series to strict boolean values.

    Returns a boolean-dtype series with the same index as the input.
    """
    if null_policy not in _BOOL_NULL_POLICIES:
        raise ValueError(
            f"Invalid null_policy {null_policy!r}. " "Expected one of {'raise', 'false', 'true'}."
        )

    normalized: dict[object, bool] = {}
    invalid_entries: list[tuple[object, object]] = []
    null_entries: list[object] = []

    for row_id, raw_value in series.items():
        normalized_value = _normalize_strict_bool(raw_value, null_policy=null_policy)
        if normalized_value is None:
            if pd.isna(raw_value):
                null_entries.append(row_id)
            else:
                invalid_entries.append((row_id, raw_value))
            continue
        normalized[row_id] = normalized_value

    if null_entries and null_policy == "raise":
        preview = ", ".join(map(repr, null_entries[:5]))
        raise ValueError(
            f"Column {column_name!r} contains missing values for nodes: {preview}. "
            "Set null_policy to 'false' or 'true' to coerce missing values."
        )

    if invalid_entries:
        preview = ", ".join(
            f"{node_id!r}={raw_value!r}" for node_id, raw_value in invalid_entries[:5]
        )
        raise ValueError(
            f"Column {column_name!r} contains non-boolean values for nodes: {preview}. "
            "Accepted values: bool, 0/1, 0.0/1.0, or string tokens {'true','false','1','0'}."
        )

    return pd.Series(normalized, index=series.index, dtype=bool, name=series.name)


def extract_leaf_counts(annotations_df: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    """Extract leaf counts for specified nodes.

    Parameters
    ----------
    annotations_df
        DataFrame with node statistics including 'leaf_count' column
    node_ids
        List of node identifiers

    Returns
    -------
    np.ndarray
        Leaf counts aligned to node_ids

    Raises
    ------
    KeyError
        If 'leaf_count' column is missing
    ValueError
        If any nodes have missing leaf counts
    """
    if "leaf_count" not in annotations_df.columns:
        raise KeyError("Missing required column 'leaf_count' in annotations dataframe.")

    leaf_counts = annotations_df["leaf_count"].reindex(node_ids).to_numpy()
    if np.isnan(leaf_counts).any():
        missing = [node_ids[i] for i, v in enumerate(leaf_counts) if np.isnan(v)]
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(f"Missing leaf_count values for nodes: {preview}.")

    return leaf_counts


def extract_node_distribution(tree: nx.DiGraph, node_id: str) -> np.ndarray:
    """Extract distribution for a single node, converted to float64.

    Parameters
    ----------
    tree
        Directed acyclic graph with 'distribution' attribute on nodes
    node_id
        Node identifier to extract distribution for

    Returns
    -------
    np.ndarray
        Distribution array as float64

    Raises
    ------
    ValueError
        If distribution is not available for the node
    """
    node_data = tree.nodes.get(node_id, {})
    distribution = node_data.get("distribution")

    if distribution is None:
        raise ValueError(
            f"Missing 'distribution' attribute for node {node_id!r}. "
            "Ensure the tree is properly annotated before running sibling tests."
        )

    return np.asarray(distribution, dtype=np.float64)


def extract_node_sample_size(tree: nx.DiGraph, node_id: str) -> int:
    """Extract sample size (leaf count) for a node.

    Checks for the canonical 'leaf_count' attribute first, then falls back
    to alternative names and finally counts descendants if needed.

    Parameters
    ----------
    tree
        Directed acyclic graph with node attributes
    node_id
        Node identifier to get sample size for

    Returns
    -------
    int
        Number of leaves under this node (or 1 if leaf)
    """
    node_data = tree.nodes.get(node_id, {})

    # Check canonical attribute first
    if "leaf_count" in node_data:
        return int(node_data["leaf_count"])

    # Fallback to alternative names
    if "sample_size" in node_data:
        return int(node_data["sample_size"])

    if "n_leaves" in node_data:
        return int(node_data["n_leaves"])

    # Check if node is a leaf
    if node_data.get("is_leaf", False) or tree.out_degree(node_id) == 0:
        return 1

    # Count leaf descendants as last resort
    descendants = list(nx.descendants(tree, node_id))
    count = sum(
        1
        for desc in descendants
        if tree.nodes.get(desc, {}).get("is_leaf", False) or tree.out_degree(desc) == 0
    )

    return max(1, count)


def assign_divergence_results(
    annotations_df: pd.DataFrame,
    child_ids: list[str],
    p_values: np.ndarray,
    p_values_corrected: np.ndarray,
    reject_null: np.ndarray,
    degrees_of_freedom: np.ndarray,
    invalid_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assign child-parent divergence test results to the annotations dataframe.

    Initializes result columns with default values, then assigns the computed
    test results to the appropriate child node rows.

    Parameters
    ----------
    annotations_df
        DataFrame to update (modified in place)
    child_ids
        List of child node identifiers
    p_values
        Raw chi-square p-values for each edge
    p_values_corrected
        FDR-corrected p-values
    reject_null
        Boolean array indicating significant edges
    degrees_of_freedom
        Effective degrees of freedom for each edge
    invalid_mask
        Optional boolean mask (aligned to ``child_ids``) indicating tests
        that were invalid and routed through the conservative p-value path.

    Returns
    -------
    pd.DataFrame
        The updated annotations dataframe with divergence columns
    """
    # Initialize columns with default values
    annotations_df["Child_Parent_Divergence_P_Value"] = np.nan
    annotations_df["Child_Parent_Divergence_P_Value_BH"] = np.nan
    annotations_df["Child_Parent_Divergence_Significant"] = False
    annotations_df["Child_Parent_Divergence_df"] = np.nan
    annotations_df["Child_Parent_Divergence_Invalid"] = False

    # Assign results to child nodes
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values_corrected
    annotations_df.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null
    annotations_df.loc[child_ids, "Child_Parent_Divergence_df"] = degrees_of_freedom

    if invalid_mask is not None:
        invalid_array = np.asarray(invalid_mask, dtype=bool)

        if invalid_array.shape[0] != len(child_ids):

            raise ValueError(
                "invalid_mask must be aligned to child_ids. "
                f"Got len(invalid_mask)={invalid_array.shape[0]}, len(child_ids)={len(child_ids)}."
            )

        annotations_df.loc[child_ids, "Child_Parent_Divergence_Invalid"] = invalid_array

    annotations_df["Child_Parent_Divergence_Significant"] = normalize_bool_series(
        annotations_df["Child_Parent_Divergence_Significant"],
        column_name="Child_Parent_Divergence_Significant",
        null_policy="raise",
    )

    return annotations_df


def initialize_sibling_divergence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize sibling divergence output columns in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to initialize columns in.

    Returns
    -------
    pd.DataFrame
        The dataframe with initialized columns.
    """
    df["Sibling_Divergence_Skipped"] = False
    df["Sibling_Test_Statistic"] = np.nan
    df["Sibling_Degrees_of_Freedom"] = np.nan
    df["Sibling_Divergence_P_Value"] = np.nan
    df["Sibling_Divergence_P_Value_Corrected"] = np.nan
    df["Sibling_Divergence_Invalid"] = False
    df["Sibling_BH_Different"] = False  # Reject H₀: siblings are different
    df["Sibling_BH_Same"] = False  # Fail to reject: siblings are similar
    return df


def extract_bool_column_dict(
    df: object,
    column_name: str,
    *,
    null_policy: Literal["raise", "false", "true"] = "raise",
) -> dict[str, bool]:
    """Extract a boolean column from DataFrame as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame or similar
        DataFrame containing the column.
    column_name : str
        Name of the column to extract.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping index to boolean values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for {column_name!r} extraction.")
    if df.empty:
        raise ValueError(f"Empty DataFrame; missing required column {column_name!r}.")
    if column_name not in df.columns:
        raise KeyError(f"Missing required column {column_name!r} in dataframe.")

    series = normalize_bool_series(
        df[column_name], column_name=column_name, null_policy=null_policy
    )
    return {str(node_id): bool(value) for node_id, value in series.items()}
