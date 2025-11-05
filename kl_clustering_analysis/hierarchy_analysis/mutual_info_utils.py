"""
Mutual information utilities for binary patterns and threshold estimation.

This module provides functions for calculating mutual information between binary
feature patterns and estimating statistical thresholds using permutation testing.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.metrics import mutual_info_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tree.poset_tree import PosetTree


def _binary_pattern(p: np.ndarray, thresh: float) -> np.ndarray:
    """
    Apply a fixed threshold to a probability vector (>= thresh -> 1, else 0).
    """
    return (np.asarray(p, dtype=float) >= float(thresh)).astype(np.int8)


def _binary_entropy(b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy (nats) of a binary array.
    """
    b = np.asarray(b, dtype=np.int8)
    p1 = b.mean()
    p1 = np.clip(p1, eps, 1 - eps)
    return float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))


def _mutual_info_binary_normalized(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    MI(b1,b2) / max(H(b1), H(b2)) in nats; returns in [0,1].
    If both entropies ~0 (constant), returns 0.0 (we treat constants as uninformative).
    """
    b1 = np.asarray(b1, dtype=np.int8)
    b2 = np.asarray(b2, dtype=np.int8)
    # joint histogram for mutual_info_score
    # sklearn's mutual_info_score expects discrete labels; b1,b2 already discrete
    mi = float(mutual_info_score(b1, b2))  # in nats
    h1 = _binary_entropy(b1)
    h2 = _binary_entropy(b2)
    denom = max(h1, h2)
    if denom <= 1e-12:
        return 0.0
    return mi / denom


def estimate_global_mi_threshold(
    G: PosetTree,
    results_df: pd.DataFrame,
    binarization_threshold: float = 0.5,
    alpha: float = 0.05,
    n_permutations: int = 256,
    random_state: Optional[int] = None,
) -> float:
    """
    Estimate a *global* MI cutoff from a permutation null built over all sibling pairs
    where the parent and both children pass the significance gate.

    We compute MI between child binary patterns; then we build a null by shuffling
    one child's bits (features) relative to the other and re-computing MI.
    The threshold is the (1 - alpha) quantile of the pooled permuted MI values.

    Returns:
        mi_threshold in [0,1]
    """
    rng = check_random_state(random_state)
    root = G.root()

    permuted_mis: List[float] = []

    for parent in G.nodes:
        if parent == root:
            pass  # still evaluate at root too (it may not be marked significant)
        # need exactly two children to compare siblings
        children = list(G.successors(parent))
        if len(children) != 2:
            continue

        # significance gate: parent and internal children must be significant (if present)
        def _sig(n: str) -> bool:
            if results_df.index.isin([n]).any():
                return bool(results_df.loc[n, "Are_Features_Dependent"])
            return False

        p_ok = _sig(parent) or (parent == root)  # allow root to try even if not marked
        c1, c2 = children
        c1_ok = G.nodes[c1].get("is_leaf", False) or _sig(c1)
        c2_ok = G.nodes[c2].get("is_leaf", False) or _sig(c2)
        if not (p_ok and c1_ok and c2_ok):
            continue

        p1 = _binary_pattern(G.nodes[c1]["distribution"], binarization_threshold)
        p2 = _binary_pattern(G.nodes[c2]["distribution"], binarization_threshold)

        # permutation null: shuffle p2 against p1
        for _ in range(n_permutations):
            p2_perm = p2.copy()
            rng.shuffle(p2_perm)
            mi_perm = _mutual_info_binary_normalized(p1, p2_perm)
            permuted_mis.append(mi_perm)

    if len(permuted_mis) == 0:
        # fallback small threshold if not enough data
        return 0.1

    threshold = float(np.quantile(np.asarray(permuted_mis, dtype=float), 1.0 - alpha))
    # ensure strictly between (0,1]
    return max(1e-6, min(1.0, threshold))
