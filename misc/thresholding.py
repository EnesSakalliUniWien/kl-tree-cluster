from __future__ import annotations

import numpy as np


def compute_otsu_threshold(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold using scikit-image for 1D numeric data in [0, 1].

    - Clips input to [0, 1].
    - Returns 0.5 for empty or constant arrays.
    - Imports scikit-image on demand to keep dependency optional.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.5
    v = np.clip(v, 0.0, 1.0)
    if np.all(v == v[0]):
        return 0.5
    try:
        from skimage.filters import threshold_otsu  # type: ignore
        thr = float(threshold_otsu(v))
        return thr if np.isfinite(thr) else 0.5
    except Exception:
        # Fallback to a simple histogram-based Otsu implementation
        hist, edges = np.histogram(v, bins=256, range=(0.0, 1.0))
        hist = hist.astype(float)
        if hist.sum() <= 0:
            return 0.5
        prob = hist / hist.sum()
        omega = np.cumsum(prob)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        mu = np.cumsum(prob * bin_centers)
        mu_T = mu[-1]
        denom = omega * (1.0 - omega)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_b2 = (mu_T * omega - mu) ** 2
            mask = denom > 0
            sigma_b2[mask] = sigma_b2[mask] / denom[mask]
            sigma_b2[~mask] = 0.0
        idx = int(np.argmax(sigma_b2))
        thr = float(bin_centers[idx])
        return thr if np.isfinite(thr) else 0.5


__all__ = ["compute_otsu_threshold"]


def compute_li_threshold(values: np.ndarray) -> float:
    """
    Compute Li's threshold using scikit-image for 1D numeric data in [0, 1].

    - Clips input to [0, 1].
    - Returns 0.5 for empty or constant arrays.
    - Imports scikit-image on demand; falls back to median if unavailable.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.5
    v = np.clip(v, 0.0, 1.0)
    if np.all(v == v[0]):
        return 0.5
    try:
        from skimage.filters import threshold_li  # type: ignore
        thr = float(threshold_li(v))
        if not np.isfinite(thr):
            return 0.5
        return thr
    except Exception:
        return float(np.median(v))


__all__.append("compute_li_threshold")
