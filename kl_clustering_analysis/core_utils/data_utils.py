from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


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
    tested_mask: np.ndarray | None = None,
    ancestor_blocked_mask: np.ndarray | None = None,
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
    tested_mask
        Optional boolean mask aligned to ``child_ids`` indicating whether the
        edge was actually tested by the multiple-testing procedure.
    ancestor_blocked_mask
        Optional boolean mask aligned to ``child_ids`` indicating TreeBH
        descendants that were not tested because an ancestor family failed.

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
    annotations_df["Child_Parent_Divergence_Tested"] = False
    annotations_df["Child_Parent_Divergence_Ancestor_Blocked"] = False

    # Assign results to child nodes
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
    annotations_df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values_corrected
    annotations_df.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null
    annotations_df.loc[child_ids, "Child_Parent_Divergence_df"] = degrees_of_freedom
    if tested_mask is None:
        annotations_df.loc[child_ids, "Child_Parent_Divergence_Tested"] = True
    else:
        tested_array = np.asarray(tested_mask, dtype=bool)
        if tested_array.shape[0] != len(child_ids):
            raise ValueError(
                "tested_mask must be aligned to child_ids. "
                f"Got len(tested_mask)={tested_array.shape[0]}, len(child_ids)={len(child_ids)}."
            )
        annotations_df.loc[child_ids, "Child_Parent_Divergence_Tested"] = tested_array

    if ancestor_blocked_mask is not None:
        blocked_array = np.asarray(ancestor_blocked_mask, dtype=bool)
        if blocked_array.shape[0] != len(child_ids):
            raise ValueError(
                "ancestor_blocked_mask must be aligned to child_ids. "
                "Got "
                f"len(ancestor_blocked_mask)={blocked_array.shape[0]}, len(child_ids)={len(child_ids)}."
            )
        annotations_df.loc[child_ids, "Child_Parent_Divergence_Ancestor_Blocked"] = blocked_array

    if invalid_mask is not None:
        invalid_array = np.asarray(invalid_mask, dtype=bool)

        if invalid_array.shape[0] != len(child_ids):

            raise ValueError(
                "invalid_mask must be aligned to child_ids. "
                f"Got len(invalid_mask)={invalid_array.shape[0]}, len(child_ids)={len(child_ids)}."
            )

        annotations_df.loc[child_ids, "Child_Parent_Divergence_Invalid"] = invalid_array

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

    series = df[column_name]
    if series.isna().any():
        raise ValueError(
            f"Column {column_name!r} contains missing values. "
            "Ensure all nodes are annotated before extraction."
        )
    return {str(node_id): bool(value) for node_id, value in series.items()}
