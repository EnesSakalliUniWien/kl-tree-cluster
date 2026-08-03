"""Benchmark runner result contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .run_status import BenchmarkRunStatus
from .unsupported_reason import UnsupportedReason


@dataclass
class MethodRunResult:
    labels: np.ndarray | None
    found_clusters: int
    report_df: pd.DataFrame | None
    status: BenchmarkRunStatus | str
    skip_reason: str | None
    extra: dict | None = None
    unsupported_reason: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        try:
            self.status = BenchmarkRunStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark run status: {self.status!r}.") from exc
        if self.found_clusters < 0:
            raise ValueError("found_clusters must be non-negative.")

        if self.status is BenchmarkRunStatus.OK:
            if self.labels is None:
                raise ValueError("status=ok requires labels.")
            if self.skip_reason is not None:
                raise ValueError("status=ok must not include skip_reason.")
            if self.unsupported_reason is not None:
                raise ValueError("status=ok must not include unsupported_reason.")
            return

        if self.labels is not None:
            raise ValueError(f"status={self.status.value} must not include labels.")
        if self.report_df is not None:
            raise ValueError(f"status={self.status.value} must not include report_df.")
        if self.found_clusters != 0:
            raise ValueError(f"status={self.status.value} requires found_clusters=0.")

        if self.status is BenchmarkRunStatus.SKIP:
            if self.unsupported_reason is not None:
                raise ValueError("status=skip must not include unsupported_reason.")
            if self.skip_reason is None or not self.skip_reason.strip():
                raise ValueError("status=skip requires a non-empty skip_reason.")
            return

        if self.skip_reason is not None:
            raise ValueError("status=unsupported must not include skip_reason.")
        if self.unsupported_reason is None:
            raise ValueError("status=unsupported requires unsupported_reason.")
