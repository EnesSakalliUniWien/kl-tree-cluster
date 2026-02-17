"""Unit tests for local KL helper utilities.

These checks ensure that Series extraction behaves predictably
for downstream statistical routines.
"""

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests import (
    get_local_kl_series,
)


def test_get_local_kl_series_handles_missing_column():
    df = pd.DataFrame({"leaf_count": [3, 4]}, index=["A", "B"])

    with pytest.raises(KeyError):
        get_local_kl_series(df)


def test_get_local_kl_series_returns_existing_series():
    df = pd.DataFrame({"kl_divergence_local": [0.1, 0.2]}, index=["A", "B"])

    series = get_local_kl_series(df)

    assert all(np.isclose(series.values, [0.1, 0.2]))
    assert list(series.index) == ["A", "B"]
