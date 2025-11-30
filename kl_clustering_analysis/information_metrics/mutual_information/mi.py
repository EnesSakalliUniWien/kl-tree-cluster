"""Mutual Information (MI) calculations for binary data.

Provides efficient implementations of mutual information for binary/discrete
distributions, with optional Numba acceleration for large-scale computations.
"""

from __future__ import annotations

import numpy as np

# ----------------------------
# Optional Numba acceleration
# ----------------------------
try:
    from numba import njit, prange  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def _safe_mi_contrib(pxy: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Elementwise contribution pxy * log(pxy / (px * py)), masked at zeros.

    Shapes must be broadcastable and result 1-D along the batch axis.
    This encodes the summand in the binary mutual information identity:
    MI(X;Y) = sum_{x,y} p(x,y) log(p(x,y) / (p(x)p(y)))

    Parameters
    ----------
    pxy : np.ndarray
        Joint probabilities p(x,y)
    px : np.ndarray
        Marginal probabilities p(x)
    py : np.ndarray
        Marginal probabilities p(y)

    Returns
    -------
    np.ndarray
        Element-wise MI contributions, with 0 for invalid entries
    """
    pxy = np.asarray(pxy, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    out = np.zeros_like(pxy, dtype=float)
    denom = px * py
    mask = (pxy > 0.0) & (denom > 0.0)
    if np.any(mask):
        out[mask] = pxy[mask] * np.log(pxy[mask] / denom[mask])
    return out


def _mi_binary_vec_numpy(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
    """
    Vectorized Mutual Information I(X;Y) for binary x vs many ys (NumPy).

    Computes MI(X;Y) = sum_{x,y in {0,1}} p(x,y) log(p(x,y) / (p(x)p(y))) in nats.

    Parameters
    ----------
    x_1d : np.ndarray
        Shape (F,), values in {0,1}
    y_2d : np.ndarray
        Shape (P, F), rows are different Y vectors

    Returns
    -------
    np.ndarray
        Shape (P,), MI per row
    """
    x_u8 = np.ascontiguousarray(x_1d, dtype=np.uint8)
    Y_u8 = np.ascontiguousarray(y_2d, dtype=np.uint8)

    # Upcast to avoid overflow during dot-products when F > 255
    x_int = x_u8.astype(np.int32, copy=False)
    Y_int = Y_u8.astype(np.int32, copy=False)

    P, F = Y_int.shape
    if F == 0:
        return np.zeros(P, dtype=float)

    # Count joint events: n_ab = |{f : X_f=a, Y_f=b}|
    sx = int(x_int.sum())
    sy = Y_int.sum(axis=1)  # (P,)
    n11 = Y_int @ x_int  # (P,)
    n10 = sx - n11  # (P,)
    n01 = sy - n11  # (P,)
    n00 = F - (n11 + n10 + n01)  # (P,)

    Ff = float(F)
    px1 = sx / Ff
    px0 = 1.0 - px1
    py1 = sy / Ff
    py0 = 1.0 - py1

    # Joint probabilities
    pxy00 = n00 / Ff
    pxy01 = n01 / Ff
    pxy10 = n10 / Ff
    pxy11 = n11 / Ff

    # Broadcast px scalars to vectors where needed
    px0v = np.full(P, px0, dtype=float)
    px1v = np.full(P, px1, dtype=float)

    mi = (
        _safe_mi_contrib(pxy00, px0v, py0)
        + _safe_mi_contrib(pxy01, px0v, py1)
        + _safe_mi_contrib(pxy10, px1v, py0)
        + _safe_mi_contrib(pxy11, px1v, py1)
    )
    return mi


if _NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _mi_binary_vec_numba(
        x_1d: np.ndarray, y_2d: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """
        Numba-accelerated vectorized MI calculation for binary data.

        Parameters
        ----------
        x_1d : np.ndarray
            Shape (F,), values in {0,1}
        y_2d : np.ndarray
            Shape (P, F), rows are different Y vectors

        Returns
        -------
        np.ndarray
            Shape (P,), MI per row
        """
        P = y_2d.shape[0]
        F = y_2d.shape[1]
        out = np.zeros(P, dtype=np.float64)
        if F == 0:
            return out

        # counts for x
        sx = 0
        for k in range(F):
            sx += x_1d[k]
        px1 = sx / F
        px0 = 1.0 - px1

        for i in prange(P):
            sy = 0
            n11 = 0
            for k in range(F):
                v = y_2d[i, k]
                sy += v
                n11 += v & x_1d[k]
            n10 = sx - n11
            n01 = sy - n11
            n00 = F - (n11 + n10 + n01)
            py1 = sy / F
            py0 = 1.0 - py1

            mi = 0.0
            if n00 > 0 and px0 > 0.0 and py0 > 0.0:
                p = n00 / F
                mi += p * np.log(p / (px0 * py0))
            if n01 > 0 and px0 > 0.0 and py1 > 0.0:
                p = n01 / F
                mi += p * np.log(p / (px0 * py1))
            if n10 > 0 and px1 > 0.0 and py0 > 0.0:
                p = n10 / F
                mi += p * np.log(p / (px1 * py0))
            if n11 > 0 and px1 > 0.0 and py1 > 0.0:
                p = n11 / F
                mi += p * np.log(p / (px1 * py1))
            out[i] = mi
        return out

else:
    # Stub when Numba is unavailable
    def _mi_binary_vec_numba(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:  # type: ignore
        """Stub - Numba path not available."""
        raise RuntimeError("Numba path not available")


def _mi_binary_vec_accel(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
    """
    Dispatcher: use Numba if available, else NumPy.

    Parameters
    ----------
    x_1d : np.ndarray
        Shape (F,), binary values
    y_2d : np.ndarray
        Shape (P, F), binary rows

    Returns
    -------
    np.ndarray
        Shape (P,), MI values
    """
    x = np.ascontiguousarray(x_1d, dtype=np.uint8)
    Y = np.ascontiguousarray(y_2d, dtype=np.uint8)
    if _NUMBA_AVAILABLE:
        return _mi_binary_vec_numba(x, Y)
    return _mi_binary_vec_numpy(x, Y)


def _mi_binary_vec(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
    """
    Primary MI dispatcher exposed to callers.

    Uses the accelerated path when available (Numba), otherwise falls back to
    the pure NumPy implementation.
    """
    return _mi_binary_vec_accel(x_1d, y_2d)


__all__ = [
    "_mi_binary_vec_numpy",
    "_mi_binary_vec_numba",
    "_mi_binary_vec_accel",
    "_mi_binary_vec",
]
