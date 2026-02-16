"""MethodRunResult dataclass for benchmarking (moved to types package)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

RunStatus = Literal["ok", "skip"]


@dataclass
class MethodRunResult:
    labels: np.ndarray | None
    found_clusters: int
    report_df: pd.DataFrame | None
    status: RunStatus
    skip_reason: str | None
    extra: dict | None = None
