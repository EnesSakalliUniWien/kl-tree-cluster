"""Li's threshold computation for 1D data."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_li_threshold(values: np.ndarray) -> float:
    """
    Compute Li's threshold for 1D numeric data in [0, 1].

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
    Uses scikit-image if available, otherwise falls back to median.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.5
    v = np.clip(v, 0.0, 1.0)
    if np.all(v == v[0]):
        return 0.5

    try:
        from skimage.filters import threshold_li

        thr = float(threshold_li(v))
        return thr if np.isfinite(thr) else 0.5
    except Exception as e:
        logger.warning(
            "scikit-image not available, using median as fallback. "
            "Install scikit-image for Li thresholding: pip install scikit-image. "
            "Error: %s",
            str(e),
        )
        return float(np.median(v))


__all__ = ["compute_li_threshold"]
