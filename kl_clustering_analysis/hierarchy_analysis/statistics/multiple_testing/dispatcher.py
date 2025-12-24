"""Dispatcher for multiple testing correction methods.

This module provides a unified interface for selecting and applying
different correction methods based on a string identifier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import networkx as nx

from .flat_correction import flat_bh_correction
from .level_wise_correction import level_wise_bh_correction
from .tree_bh_correction import tree_bh_correction


def apply_multiple_testing_correction(
    p_values: np.ndarray,
    child_ids: List[str],
    child_depths: np.ndarray,
    alpha: float,
    method: str = "flat",
    tree: Optional[nx.DiGraph] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply BH correction with optional hierarchical structure awareness.

    This is the main entry point for multiple testing correction. It dispatches
    to the appropriate correction method based on the `method` parameter.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values for each edge
    child_ids : List[str]
        List of child node identifiers
    child_depths : np.ndarray
        Array of depths for each child node
    alpha : float
        Base significance level
    method : str
        Correction method:
        - "flat" (default): Standard BH across all edges
        - "level_wise": BH applied separately at each tree level
        - "tree_bh": Family-wise BH with ancestor-adjusted thresholds
                      (Bogomolov et al. 2021)
    tree : nx.DiGraph, optional
        Required for method="tree_bh". The tree structure.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input

    Raises
    ------
    ValueError
        If method="tree_bh" but tree is not provided
        If method is not one of the supported values

    Examples
    --------
    >>> import numpy as np
    >>> p_values = np.array([0.01, 0.02, 0.03, 0.04])
    >>> child_ids = ["A", "B", "C", "D"]
    >>> depths = np.array([1, 1, 2, 2])
    >>> rejected, adjusted = apply_multiple_testing_correction(
    ...     p_values, child_ids, depths, alpha=0.05, method="flat"
    ... )
    """
    if method == "tree_bh":
        if tree is None:
            raise ValueError("tree parameter required for method='tree_bh'")
        result = tree_bh_correction(tree, p_values, child_ids, alpha=alpha)
        return result.reject, result.adjusted_p
    elif method == "level_wise":
        return level_wise_bh_correction(p_values, child_depths, alpha)
    elif method == "flat":
        return flat_bh_correction(p_values, alpha)
    else:
        raise ValueError(
            f"Unknown correction method: {method!r}. "
            f"Supported methods: 'flat', 'level_wise', 'tree_bh'"
        )


__all__ = ["apply_multiple_testing_correction"]
