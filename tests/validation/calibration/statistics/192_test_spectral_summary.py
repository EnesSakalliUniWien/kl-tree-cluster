import numpy as np
import pytest
from benchmarks.diagnostics.calibration.statistics.spectral_summary import effective_rank


def test_effective_rank_reports_single_dimension_for_degenerate_spectrum() -> None:
    assert effective_rank(np.array([0.0, 0.0])) == 1.0


def test_effective_rank_reports_equal_weight_dimension_count() -> None:
    assert effective_rank(np.array([2.0, 2.0, 2.0])) == pytest.approx(3.0)
