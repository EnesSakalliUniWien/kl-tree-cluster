"""Conditional Mutual Information (CMI) calculations for binary data.

Provides efficient computation of conditional mutual information using
stratification by the conditioning variable.
"""

from __future__ import annotations

import numpy as np

from .mi import _mi_binary_vec_accel


def _cmi_binary_vec(x_1d: np.ndarray, y_2d: np.ndarray, z_1d: np.ndarray) -> np.ndarray:
    """
    Vectorized Conditional MI I(X;Y|Z) for binary arrays.

    Uses the decomposition:
    I(X;Y|Z) = sum_z P(Z=z) * I(X;Y | Z=z)

    This reuses the mutual-information helper on each stratum defined by Z.

    Parameters
    ----------
    x_1d : np.ndarray
        Shape (F,), binary values
    y_2d : np.ndarray
        Shape (P, F), binary rows - multiple Y vectors
    z_1d : np.ndarray
        Shape (F,), binary conditioning variable

    Returns
    -------
    np.ndarray
        Shape (P,), CMI values for each Y vector
    """
    z = np.ascontiguousarray(z_1d, dtype=np.uint8)
    P = y_2d.shape[0]
    if z.size == 0:
        return np.zeros(P, dtype=float)

    # Split by Z=0 and Z=1 strata
    m0 = z == 0
    m1 = ~m0
    n = float(z.size)
    out = np.zeros(P, dtype=float)

    # Compute MI within Z=0 stratum
    n0 = int(m0.sum())
    if n0 > 0:
        x0 = np.ascontiguousarray(x_1d[m0], dtype=np.uint8)
        Y0 = np.ascontiguousarray(y_2d[:, m0], dtype=np.uint8)
        out += (n0 / n) * _mi_binary_vec_accel(x0, Y0)

    # Compute MI within Z=1 stratum
    n1 = int(m1.sum())
    if n1 > 0:
        x1 = np.ascontiguousarray(x_1d[m1], dtype=np.uint8)
        Y1 = np.ascontiguousarray(y_2d[:, m1], dtype=np.uint8)
        out += (n1 / n) * _mi_binary_vec_accel(x1, Y1)

    return out


__all__ = ["_cmi_binary_vec"]
