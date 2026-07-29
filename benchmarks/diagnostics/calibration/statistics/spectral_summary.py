"""Spectral summaries used only by calibration and validation reports."""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Return the exponentiated Shannon entropy of a nonnegative spectrum."""
    nonnegative = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(nonnegative))
    if total <= 0.0:
        return 1.0
    weights = nonnegative / total
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        return 1.0
    entropy = -float(np.sum(positive_weights * np.log(positive_weights)))
    return float(np.exp(entropy))


__all__ = ["effective_rank"]
