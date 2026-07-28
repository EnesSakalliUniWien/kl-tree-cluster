"""Metamorphic checks for benchmark traceability."""

from __future__ import annotations

import numpy as np


def same_partition_up_to_labels(left: np.ndarray, right: np.ndarray) -> bool:
    """Return True when two label vectors induce the same sample partition."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    for i in range(left.size):
        for j in range(i + 1, left.size):
            if (left[i] == left[j]) != (right[i] == right[j]):
                return False
    return True


def permute_sample_order(
    matrix: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation transform for sample-order invariance tests."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(matrix).shape[0])
    return np.asarray(matrix)[order], np.asarray(labels)[order]


def permute_feature_order(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Permutation transform for feature-order invariance tests."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(matrix).shape[1])
    return np.asarray(matrix)[:, order]
