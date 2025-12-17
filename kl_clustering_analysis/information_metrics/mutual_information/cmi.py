"""Conditional Mutual Information (CMI) calculations for binary data.

Provides efficient computation of conditional mutual information using
stratification by the conditioning variable.
"""

from __future__ import annotations

import numpy as np

from .mi import _mi_binary_vec


def _cmi_binary_vec(
    x_vector: np.ndarray, y_matrix: np.ndarray, z_condition: np.ndarray
) -> np.ndarray:
    """
    Vectorized Conditional MI I(X;Y|Z) for discrete arrays.

    Uses the decomposition:
    I(X;Y|Z) = sum_z P(Z=z) * I(X;Y | Z=z)

    This reuses the mutual-information helper on each stratum defined by Z.
    Supports arbitrary discrete values for Z (not just binary).

    Parameters
    ----------
    x_vector : np.ndarray
        Shape (F,), discrete values
    y_matrix : np.ndarray
        Shape (P, F), discrete rows - multiple Y vectors
    z_condition : np.ndarray
        Shape (F,), discrete conditioning variable

    Returns
    -------
    np.ndarray
        Shape (P,), CMI values for each Y vector
    """
    # Ensure Z is integer-like for unique
    z_discrete = np.ascontiguousarray(z_condition)
    num_y_vectors = y_matrix.shape[0]

    if z_discrete.size == 0:
        return np.zeros(num_y_vectors, dtype=float)

    total_samples = float(z_discrete.size)
    cmi_values = np.zeros(num_y_vectors, dtype=float)

    # Iterate over all unique values of Z (strata)
    unique_z = np.unique(z_discrete)

    for z_val in unique_z:
        stratum_mask = z_discrete == z_val
        stratum_count = int(stratum_mask.sum())

        if stratum_count > 0:
            # Extract stratum data
            x_stratum = x_vector[stratum_mask]
            y_stratum = y_matrix[:, stratum_mask]

            # Calculate MI for this stratum
            mi_stratum = _mi_binary_vec(x_stratum, y_stratum)

            # Add weighted contribution
            cmi_values += (stratum_count / total_samples) * mi_stratum

    return cmi_values


__all__ = ["_cmi_binary_vec"]
