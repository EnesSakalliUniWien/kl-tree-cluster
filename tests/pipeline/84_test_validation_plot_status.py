from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks.shared.plots.summary import create_validation_plot


def test_validation_plot_uses_only_ok_values_and_reports_support_counts():
    frame = pd.DataFrame(
        {
            "method": ["tbs", "tbs", "kmeans"],
            "status": ["ok", "unsupported", "skip"],
            "true_clusters": [2, 4, 3],
            "found_clusters": [2, 0, 0],
            "ari": [0.9, np.nan, np.nan],
            "nmi": [0.8, np.nan, np.nan],
            "purity": [1.0, np.nan, np.nan],
        }
    )

    fig = create_validation_plot(frame)

    assert fig._suptitle is not None
    assert "ok=1, unsupported=1, skip=1" in fig._suptitle.get_text()
    for axis in fig.axes:
        for line in axis.lines:
            assert len(line.get_xdata()) == 1
    plt.close(fig)


def test_validation_plot_marks_quality_unavailable_without_ok_rows():
    frame = pd.DataFrame(
        {
            "method": ["tbs"],
            "status": ["unsupported"],
            "true_clusters": [4],
            "found_clusters": [0],
            "ari": [np.nan],
            "nmi": [np.nan],
            "purity": [np.nan],
        }
    )

    fig = create_validation_plot(frame)

    assert all(not axis.axison for axis in fig.axes)
    assert fig._suptitle is not None
    assert "ok=0, unsupported=1, skip=0" in fig._suptitle.get_text()
    plt.close(fig)
