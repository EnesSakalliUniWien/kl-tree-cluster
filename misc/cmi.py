"""
Conditional mutual information (CMI) utilities for discrete/binary data.

Standalone implementation used by hierarchy_analysis statistical tests.
No nested/inner functions; all helpers are top-level.
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


# ============================================================
# Top-level helpers (no nested functions)
# ============================================================


def _safe_mi_contrib(pxy: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Elementwise contribution pxy * log(pxy / (px * py)), masked at zeros.
    Shapes must be broadcastable and result 1-D along the batch axis.
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
    Vectorized Mutual Information I(X;Y) for binary x vs many ys (NumPy path).

    x_1d: shape (F,), values in {0,1}
    y_2d: shape (P, F), rows are different Y vectors
    returns: shape (P,), MI per row
    """
    # Keep compact storage for input, but upcast for counting to avoid overflow.
    x_u8 = np.ascontiguousarray(x_1d, dtype=np.uint8)
    Y_u8 = np.ascontiguousarray(y_2d, dtype=np.uint8)

    # >>> Critical overflow-safe upcasts for counts/matmul <<<
    x_i = x_u8.astype(np.int32, copy=False)
    Y_i = Y_u8.astype(np.int32, copy=False)

    P, F = Y_i.shape
    if F == 0:
        return np.zeros(P, dtype=float)

    sx = int(x_i.sum())  # scalar
    sy = Y_i.sum(axis=1)  # (P,)
    n11 = Y_i @ x_i  # (P,)
    n10 = sx - n11  # (P,)
    n01 = sy - n11  # (P,)
    n00 = F - (n11 + n10 + n01)  # (P,)

    Ff = float(F)
    px1 = sx / Ff
    px0 = 1.0 - px1
    py1 = sy / Ff
    py0 = 1.0 - py1

    # Probabilities
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

    def _mi_binary_vec_numba(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:  # type: ignore
        raise RuntimeError("Numba path not available")


def _mi_binary_vec_accel(x_1d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
    """
    Dispatcher: use Numba path if available, else NumPy path.
    """
    x = np.ascontiguousarray(x_1d, dtype=np.uint8)
    Y = np.ascontiguousarray(y_2d, dtype=np.uint8)
    if _NUMBA_AVAILABLE:
        return _mi_binary_vec_numba(x, Y)  # type: ignore[name-defined]
    return _mi_binary_vec_numpy(x, Y)


def _cmi_binary_vec(x_1d: np.ndarray, y_2d: np.ndarray, z_1d: np.ndarray) -> np.ndarray:
    """
    Vectorized Conditional MI I(X;Y|Z) for binary arrays.

    x_1d: (F,), binary
    y_2d: (P, F), binary rows
    z_1d: (F,), binary
    returns: (P,)
    """
    z = np.ascontiguousarray(z_1d, dtype=np.uint8)
    P = y_2d.shape[0]
    if z.size == 0:
        return np.zeros(P, dtype=float)

    m0 = z == 0
    m1 = ~m0
    n = float(z.size)
    out = np.zeros(P, dtype=float)

    n0 = int(m0.sum())
    if n0 > 0:
        x0 = np.ascontiguousarray(x_1d[m0], dtype=np.uint8)
        Y0 = np.ascontiguousarray(y_2d[:, m0], dtype=np.uint8)
        out += (n0 / n) * _mi_binary_vec_accel(x0, Y0)

    n1 = int(m1.sum())
    if n1 > 0:
        x1 = np.ascontiguousarray(x_1d[m1], dtype=np.uint8)
        Y1 = np.ascontiguousarray(y_2d[:, m1], dtype=np.uint8)
        out += (n1 / n) * _mi_binary_vec_accel(x1, Y1)

    return out


def _perm_cmi_binary_batch(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Produce K permuted CMI values in one vectorized batch.
    """
    F = y.size
    if K <= 0 or F == 0:
        return np.zeros(0, dtype=float)

    idx0 = np.flatnonzero(z == 0)
    idx1 = np.flatnonzero(z == 1)

    # Repeat y as (K, F)
    Yp = np.repeat(y[None, :], K, axis=0)

    # Permute within each stratum
    try:
        # NumPy >= 1.20
        if idx0.size > 1:
            Yp[:, idx0] = rng.permuted(Yp[:, idx0], axis=1)
        if idx1.size > 1:
            Yp[:, idx1] = rng.permuted(Yp[:, idx1], axis=1)
    except AttributeError:
        # Older NumPy fallback
        for r in range(K):
            if idx0.size > 1:
                Yp[r, idx0] = rng.permutation(Yp[r, idx0])
            if idx1.size > 1:
                Yp[r, idx1] = rng.permutation(Yp[r, idx1])

    return _cmi_binary_vec(x, Yp, z)


def _perm_test_cmi_binary(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    permutations: int = 300,
    random_state: int | None = None,
    batch_size: int = 256,
) -> tuple[float, float]:
    """
    Batched, vectorized permutation test for I(X;Y|Z).
    Returns (cmi_obs, p_value).
    """
    rng = np.random.default_rng(random_state)
    x = np.ascontiguousarray(x, dtype=np.uint8).ravel()
    y = np.ascontiguousarray(y, dtype=np.uint8).ravel()
    z = np.ascontiguousarray(z, dtype=np.uint8).ravel()

    # Observed CMI
    cmi_obs = float(_cmi_binary_vec(x, y.reshape(1, -1), z)[0])

    # Early-outs: uninformative test → p=1.0 (no dependence evidence)
    if permutations <= 0:
        return cmi_obs, 1.0
    idx0 = np.flatnonzero(z == 0)
    idx1 = np.flatnonzero(z == 1)
    if y.size == 0 or ((idx0.size <= 1) and (idx1.size <= 1)):
        return cmi_obs, 1.0

    # Batched permutations
    ge = 0
    done = 0
    bs = max(1, int(batch_size))
    while done < permutations:
        k = min(bs, permutations - done)
        cmi_perm_vec = _perm_cmi_binary_batch(x, y, z, K=k, rng=rng)
        ge += int(np.sum(cmi_perm_vec >= cmi_obs - 1e-12))
        done += k

    p_val = (1.0 + ge) / (permutations + 1.0)
    return cmi_obs, float(p_val)


def _cmi_perm_from_args(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int],
) -> tuple[float, float]:
    """
    Helper to unpack args for parallel mapping.
    args = (x, y, z, permutations, random_state, batch_size)
    """
    x, y, z, permutations, random_state, batch_size = args
    return _perm_test_cmi_binary(
        x, y, z, int(permutations), random_state, int(batch_size)
    )


__all__ = [
    "_mi_binary_vec_numpy",
    "_mi_binary_vec_numba",
    "_mi_binary_vec_accel",
    "_cmi_binary_vec",
    "_perm_cmi_binary_batch",
    "_perm_test_cmi_binary",
    "_cmi_perm_from_args",
]
