"""NumPy-based Mutual Information (MI) calculations for binary data."""

from __future__ import annotations

import numpy as np


def _safe_mi_contrib(p_xy: np.ndarray, p_x: np.ndarray, p_y: np.ndarray) -> np.ndarray:
    """
    Calculate the element-wise contribution to Mutual Information.

    Computes the term:
    $$ p(x,y) \cdot \log \left( \frac{p(x,y)}{p(x) \cdot p(y)} \right) $$

    Handles cases where probabilities are zero by returning 0.0, consistent with
    the limit $\lim_{p \to 0} p \log p = 0$.

    Parameters
    ----------
    p_xy : np.ndarray
        Joint probabilities $p(x,y)$.
    p_x : np.ndarray
        Marginal probabilities $p(x)$.
    p_y : np.ndarray
        Marginal probabilities $p(y)$.

    Returns
    -------
    np.ndarray
        Element-wise MI contributions. Zeros where input probabilities are zero.
    """
    p_xy = np.asarray(p_xy, dtype=float)
    p_x = np.asarray(p_x, dtype=float)
    p_y = np.asarray(p_y, dtype=float)

    contribution = np.zeros_like(p_xy, dtype=float)
    denominator = p_x * p_y

    # Only compute where joint probability and denominator are positive
    valid_mask = (p_xy > 0.0) & (denominator > 0.0)

    if np.any(valid_mask):
        contribution[valid_mask] = p_xy[valid_mask] * np.log(
            p_xy[valid_mask] / denominator[valid_mask]
        )

    return contribution


def _mi_binary_vec_numpy(x_vector: np.ndarray, y_vectors: np.ndarray) -> np.ndarray:
    """
    Vectorized Mutual Information I(X;Y) for a binary vector X vs many Y vectors (NumPy).

    Computes Mutual Information in nats:
    $$ I(X;Y) = \sum_{x \in \{0,1\}} \sum_{y \in \{0,1\}} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right) $$

    Parameters
    ----------
    x_vector : np.ndarray
        Binary vector X of shape (n_samples,). Values must be in {0, 1}.
    y_vectors : np.ndarray
        Matrix of binary vectors Y of shape (n_vectors, n_samples).
        Each row represents a different variable Y.

    Returns
    -------
    np.ndarray
        Array of shape (n_vectors,) containing the MI between X and each row of Y.
    """
    # Ensure inputs are uint8 for efficient processing
    x_u8 = np.ascontiguousarray(x_vector, dtype=np.uint8)
    y_u8 = np.ascontiguousarray(y_vectors, dtype=np.uint8)

    # Upcast to int32 to avoid overflow during dot-products if n_samples > 255
    x_int = x_u8.astype(np.int32, copy=False)
    y_int = y_u8.astype(np.int32, copy=False)

    n_vectors, n_samples = y_int.shape
    if n_samples == 0:
        return np.zeros(n_vectors, dtype=float)

    # --- Count Joint and Marginal Events ---
    # n_x_ones: Count where X=1
    n_x_ones = int(x_int.sum())

    # n_y_ones: Count where Y=1 (for each vector in Y) -> Shape (n_vectors,)
    n_y_ones = y_int.sum(axis=1)

    # n11: Count where X=1 AND Y=1 (dot product) -> Shape (n_vectors,)
    n11 = y_int @ x_int

    # Derive other counts from inclusion-exclusion principles:
    # n10: X=1, Y=0 -> n(X=1) - n(X=1, Y=1)
    n10 = n_x_ones - n11

    # n01: X=0, Y=1 -> n(Y=1) - n(X=1, Y=1)
    n01 = n_y_ones - n11

    # n00: X=0, Y=0 -> Total - (n11 + n10 + n01)
    n00 = n_samples - (n11 + n10 + n01)

    # --- Calculate Probabilities ---
    n_samples_float = float(n_samples)

    # Marginals for X
    p_x1 = n_x_ones / n_samples_float
    p_x0 = 1.0 - p_x1

    # Marginals for Y (vectors)
    p_y1 = n_y_ones / n_samples_float
    p_y0 = 1.0 - p_y1

    # Joint probabilities
    p_xy00 = n00 / n_samples_float
    p_xy01 = n01 / n_samples_float
    p_xy10 = n10 / n_samples_float
    p_xy11 = n11 / n_samples_float

    # Broadcast p_x scalars to match shape of p_y (n_vectors,)
    p_x0_vec = np.full(n_vectors, p_x0, dtype=float)
    p_x1_vec = np.full(n_vectors, p_x1, dtype=float)

    # --- Sum Contributions ---
    mi = (
        _safe_mi_contrib(p_xy00, p_x0_vec, p_y0)
        + _safe_mi_contrib(p_xy01, p_x0_vec, p_y1)
        + _safe_mi_contrib(p_xy10, p_x1_vec, p_y0)
        + _safe_mi_contrib(p_xy11, p_x1_vec, p_y1)
    )
    return mi
