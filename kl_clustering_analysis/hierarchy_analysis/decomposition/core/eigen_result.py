"""Eigendecomposition result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EigenResult:
    """Container for eigendecomposition outputs."""

    eigenvalues: np.ndarray
    active_mask: np.ndarray
    d_active: int
    use_dual: bool
    eigenvectors_active: Optional[np.ndarray] = None
    gram_vecs: Optional[np.ndarray] = None
    X_std: Optional[np.ndarray] = None
