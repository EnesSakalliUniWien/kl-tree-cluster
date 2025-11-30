"""Binary thresholding for probability arrays."""

from __future__ import annotations

from typing import Union

import numpy as np

from .otsu import compute_otsu_threshold
from .li import compute_li_threshold


def binary_threshold(arr: np.ndarray, thr: Union[float, str] = 0.5) -> np.ndarray:
    """
    Convert array to binary uint8 (0/1) using specified threshold.

    Parameters
    ----------
    arr : np.ndarray
        Input array to threshold.
    thr : float or str, default=0.5
        Threshold value or method name:
        - float: Use this value as threshold
        - "otsu": Compute Otsu's threshold
        - "li": Compute Li's threshold

    Returns
    -------
    np.ndarray
        Binary array with dtype uint8, values in {0, 1}.

    Raises
    ------
    ValueError
        If threshold method string is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0.1, 0.4, 0.6, 0.9])
    >>> binary_threshold(arr, 0.5)
    array([0, 0, 1, 1], dtype=uint8)
    >>> binary_threshold(arr, "otsu")  # doctest: +SKIP
    array([0, 0, 1, 1], dtype=uint8)
    """
    a = np.asarray(arr, dtype=float)

    if isinstance(thr, str):
        if thr == "otsu":
            thr = compute_otsu_threshold(a)
        elif thr == "li":
            thr = compute_li_threshold(a)
        else:
            raise ValueError(
                f"Unknown threshold method: {thr!r}. "
                f"Expected float or one of: 'otsu', 'li'"
            )

    return (a >= float(thr)).astype(np.uint8, copy=False)


__all__ = ["binary_threshold"]
