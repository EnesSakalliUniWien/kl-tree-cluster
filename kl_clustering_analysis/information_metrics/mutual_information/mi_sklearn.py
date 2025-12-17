"""Scikit-learn based Mutual Information (MI) calculations for binary data."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def _mi_binary_vec_sklearn(x_vector: np.ndarray, y_vectors: np.ndarray) -> np.ndarray:
    """
    Calculate Mutual Information using scikit-learn's mutual_info_classif.

    Parameters
    ----------
    x_vector : np.ndarray
        Binary vector X of shape (n_samples,).
    y_vectors : np.ndarray
        Matrix of binary vectors Y of shape (n_vectors, n_samples).

    Returns
    -------
    np.ndarray
        Array of shape (n_vectors,) containing MI values in nats.
    """
    n_samples = x_vector.shape[0]
    n_vectors = y_vectors.shape[0]

    if n_samples == 0:
        return np.zeros(n_vectors, dtype=float)

    if n_vectors == 0:
        return np.array([], dtype=float)

    # mutual_info_classif expects X as (n_samples, n_features)
    # We treat y_vectors as the features matrix X
    X = y_vectors.T
    y = x_vector

    # discrete_features=True ensures we use the count-based estimator
    # instead of the KNN-based estimator used for continuous data.
    return mutual_info_classif(X, y, discrete_features=True)
