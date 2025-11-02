from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from misc.thresholding import compute_li_threshold, compute_otsu_threshold


def calculate_chi_square_test(
    kl_divergence: float, number_of_leaves: int, number_of_features: int
) -> Tuple[float, int, float]:
    """Return (χ² statistic, degrees of freedom, p-value) for 2·n·KL ~ χ²_F."""
    chi2_statistic = 2.0 * float(number_of_leaves) * float(kl_divergence)
    dof = int(number_of_features)
    p_value = float(chi2.sf(chi2_statistic, df=dof))
    return chi2_statistic, dof, p_value


def apply_benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Benjamini–Hochberg FDR control with empty-input guard."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float), float(alpha)
    reject, corrected_p_values, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_bh", is_sorted=False, returnsorted=False
    )
    return reject.astype(bool), corrected_p_values.astype(float), float(alpha)


def binary_threshold(arr: np.ndarray, thr: Union[float, str] = 0.5) -> np.ndarray:
    """Return uint8(0/1) vector for thresholded probabilities."""
    a = np.asarray(arr, dtype=float)
    if isinstance(thr, str):
        if thr == "otsu":
            thr = compute_otsu_threshold(a)
        elif thr == "li":
            thr = compute_li_threshold(a)
        else:
            raise ValueError(f"Unknown threshold method: {thr}")
    return (a >= float(thr)).astype(np.uint8, copy=False)


__all__ = [
    "calculate_chi_square_test",
    "apply_benjamini_hochberg_correction",
    "binary_threshold",
]
