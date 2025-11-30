"""Otsu's threshold computation for 1D data."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_otsu_threshold(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold for 1D numeric data in [0, 1].

    Parameters
    ----------
    values : np.ndarray
        Input values to threshold.

    Returns
    -------
    float
        Computed threshold value, or 0.5 for empty/constant arrays.

    Notes
    -----
    Uses scikit-image if available, otherwise falls back to custom implementation.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.5
    v = np.clip(v, 0.0, 1.0)
    if np.all(v == v[0]):
        return 0.5

    try:
        from skimage.filters import threshold_otsu

        thr = float(threshold_otsu(v))
        return thr if np.isfinite(thr) else 0.5
    except Exception as e:
        logger.warning(
            "scikit-image not available, using fallback Otsu implementation. "
            "Install scikit-image for better performance: pip install scikit-image. "
            "Error: %s",
            str(e),
        )
        return _otsu_fallback(v)


def _otsu_fallback(v: np.ndarray) -> float:
    """Fallback histogram-based Otsu implementation."""
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
