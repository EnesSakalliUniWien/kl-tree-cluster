from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def extract_leaf_counts(
    nodes_dataframe: pd.DataFrame, node_ids: list[str]
) -> np.ndarray:
    """Extract leaf counts for specified nodes.

    Parameters
    ----------
    nodes_dataframe
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
    if "leaf_count" not in nodes_dataframe.columns:
        raise KeyError("Missing required column 'leaf_count' in nodes dataframe.")

    leaf_counts = nodes_dataframe["leaf_count"].reindex(node_ids).to_numpy()
    if np.isnan(leaf_counts).any():
        missing = [node_ids[i] for i, v in enumerate(leaf_counts) if np.isnan(v)]
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(f"Missing leaf_count values for nodes: {preview}.")

    return leaf_counts


def extract_parent_distributions(
    tree: nx.DiGraph, parent_ids: list[str]
) -> list[np.ndarray | None]:
    """Extract parent node distributions for variance-weighted degrees of freedom.

    Parameters
    ----------
    tree
        Directed acyclic graph with 'distribution' attribute on nodes
    parent_ids
        List of parent node identifiers

    Returns
    -------
    list[np.ndarray | None]
        List of distribution arrays (or None if not available) aligned to parent_ids
    """
    return [extract_node_distribution(tree, pid) for pid in parent_ids]


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


def raise_leaf_only_tree_error(_: pd.DataFrame) -> pd.DataFrame:
    """Raise error when edge-level statistics cannot be computed for leaf-only tree."""
    raise ValueError(
        "Child-parent divergence annotations cannot be computed for a leaf-only tree."
    )


def assign_divergence_results(
    nodes_dataframe: pd.DataFrame,
    child_ids: list[str],
    p_values: np.ndarray,
    p_values_corrected: np.ndarray,
    reject_null: np.ndarray,
    degrees_of_freedom: np.ndarray,
    global_weights: np.ndarray,
) -> pd.DataFrame:
    """Assign child-parent divergence test results to the nodes dataframe.

    Initializes result columns with default values, then assigns the computed
    test results to the appropriate child node rows.

    Parameters
    ----------
    nodes_dataframe
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
    global_weights
        Global weighting factors applied to each edge

    Returns
    -------
    pd.DataFrame
        The updated nodes dataframe with divergence columns
    """
    # Initialize columns with default values
    nodes_dataframe["Child_Parent_Divergence_P_Value"] = np.nan
    nodes_dataframe["Child_Parent_Divergence_P_Value_BH"] = np.nan
    nodes_dataframe["Child_Parent_Divergence_Significant"] = False
    nodes_dataframe["Child_Parent_Divergence_df"] = np.nan
    nodes_dataframe["Child_Parent_Divergence_Global_Weight"] = 1.0

    # Assign results to child nodes
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = (
        p_values_corrected
    )
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_Significant"] = reject_null
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_df"] = degrees_of_freedom
    nodes_dataframe.loc[child_ids, "Child_Parent_Divergence_Global_Weight"] = (
        global_weights
    )

    # Validate and convert significance column
    if nodes_dataframe["Child_Parent_Divergence_Significant"].isna().any():
        raise ValueError(
            "Child_Parent_Divergence_Significant contains missing values after "
            "annotation; aborting."
        )
    with pd.option_context("future.no_silent_downcasting", True):
        nodes_dataframe["Child_Parent_Divergence_Significant"] = nodes_dataframe[
            "Child_Parent_Divergence_Significant"
        ].astype(bool)

    return nodes_dataframe


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
    df["Sibling_BH_Different"] = False  # Reject H₀: siblings are different
    df["Sibling_BH_Same"] = False  # Fail to reject: siblings are similar
    return df


def extract_bool_column_dict(
    df: object, column_name: str, default: bool = False
) -> dict[str, bool]:
    """Extract a boolean column from DataFrame as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame or similar
        DataFrame containing the column.
    column_name : str
        Name of the column to extract.
    default : bool
        Default value for missing/NaN entries.

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
        missing = series[series.isna()].index.tolist()
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(
            f"Column {column_name!r} contains missing values for nodes: {preview}."
        )
    return series.astype(bool).to_dict()
