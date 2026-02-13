"""MethodRunResult dataclass for benchmarking (moved to types package)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MethodRunResult:
    labels: np.ndarray | None
    found_clusters: int
    report_df: pd.DataFrame | None
    status: str
    skip_reason: str | None
    extra: dict | None = None
