"""Tests for benchmark report plot export layout choices."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks.shared.plots import export
from benchmarks.shared.result_records import ComputedResultRecord


def _record() -> ComputedResultRecord:
    labels = np.array([0, 1, 0, 1])
    return ComputedResultRecord(
        test_case_num=1,
        method="tbs",
        method_name="TBS Tree",
        params={"tree_distance_metric": "tree_distribution_kl"},
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        outlier_precision=0.0,
        outlier_recall=0.0,
        outlier_f1=0.0,
        singleton_outlier_isolated=0.0,
        grouped_outlier_cluster_recovered=0.0,
        labels=labels,
        data=pd.DataFrame(),
        meta={
            "name": "unit",
            "n_clusters": 2,
            "n_samples": 4,
            "n_features": 2,
            "generator": "unit",
            "noise": 0.0,
        },
        x_original=np.zeros((4, 2)),
        y_true=labels,
        tree=object(),
        decomposition={"cluster_assignments": {}, "num_clusters": 2},
        annotations=pd.DataFrame(),
    )


def test_tree_then_umap_exports_tree_pages_separately(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_tree_figures(*, case_num, case_results):
        calls.append("tree")
        return [plt.figure()]

    def fake_umap_figures(**kwargs):
        calls.append("umap")
        assert "extra_panels" not in kwargs
        return [plt.figure()]

    monkeypatch.setattr(export, "_create_tree_figures_for_case", fake_tree_figures)
    monkeypatch.setattr(export, "create_clustering_comparison_plots", fake_umap_figures)

    collected: list = []
    figs = export.create_tree_then_umap_plots_from_results(
        [_record()],
        tmp_path,
        verbose=False,
        save=False,
        collect=True,
        collected=collected,
    )

    assert figs is collected
    assert calls == ["tree", "umap"]
    assert len(collected) == 2

    for item in collected:
        plt.close(item["figure"])
