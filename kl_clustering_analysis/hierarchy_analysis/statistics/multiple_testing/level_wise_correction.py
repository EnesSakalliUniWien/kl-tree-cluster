"""Level-wise BH correction for hierarchical structures.

This module applies BH correction separately at each tree level,
providing level-specific FDR control.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .base import benjamini_hochberg_correction


def level_wise_bh_correction(
    p_values: np.ndarray,
    node_depths: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply BH correction separately at each tree level.

    This is a simpler hierarchical variant that applies standard BH
    at each level independently, providing level-specific FDR control.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values for each edge/test
    node_depths : np.ndarray
        Array of depths for each node (root = 0)
    alpha : float
        Significance level for BH correction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (reject_null, adjusted_p_values) arrays aligned to input

    Notes
    -----
    This method groups tests by their depth in the tree and applies
    BH correction independently within each level. This is less
    conservative than flat BH but simpler than full TreeBH.

    Examples
    --------
    >>> import numpy as np
    >>> p_values = np.array([0.01, 0.02, 0.03, 0.04])
    >>> depths = np.array([1, 1, 2, 2])  # Two at level 1, two at level 2
    >>> rejected, adjusted = level_wise_bh_correction(p_values, depths, alpha=0.05)
    """
    n = len(p_values)
    reject_null = np.zeros(n, dtype=bool)
    adjusted_p = np.ones(n, dtype=float)

    if n == 0:
        return reject_null, adjusted_p

    # Group tests by depth level
    levels = sorted(set(node_depths))
    level_indices: Dict[int, List[int]] = defaultdict(list)
    for i, depth in enumerate(node_depths):
        level_indices[depth].append(i)

    for level in levels:
        indices = level_indices[level]
        if not indices:
            continue

        # Extract p-values for this level
        level_p_values = p_values[indices]

        # Apply standard BH at this level
        if len(level_p_values) > 0:
            level_reject, level_adjusted, _ = benjamini_hochberg_correction(
                level_p_values, alpha=alpha
            )

            # Store results back to original positions
            for j, idx in enumerate(indices):
                reject_null[idx] = level_reject[j]
                adjusted_p[idx] = level_adjusted[j]

    return reject_null, adjusted_p


__all__ = ["level_wise_bh_correction"]
