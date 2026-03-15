"""Type definitions for the inflation_correction package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class CalibrationModel:
    """Result of fitting the post-selection inflation model.

    Stores the global inflation factor c-hat estimated via continuous
    edge-weight calibration.  The model is intercept-only: a single
    constant c-hat applied uniformly to all focal sibling pairs.
    """

    method: str
    n_calibration: int  # number of pairs that contributed (weight > 0)
    global_inflation_factor: float  # the estimated c-hat
    max_observed_ratio: float = 1.0  # max(r_i) — upper bound for c-hat
    beta: Optional[np.ndarray] = None  # kept for API compat; [log(c-hat), 0, 0]
    diagnostics: Dict = field(default_factory=dict)


__all__ = ["CalibrationModel"]
