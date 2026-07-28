"""Tests for adult pancreas benchmark manifest timestamps."""

import importlib
import json
import re
import sys

import pandas as pd


class _FakeAdata:
    def __init__(self) -> None:
        self.obs = pd.DataFrame(
            {
                "celltype": ["alpha", "beta", "alpha"],
                "batch": ["b1", "b1", "b2"],
            },
            index=["cell0", "cell1", "cell2"],
        )
        self.n_obs = 3
        self.n_vars = 2


def _load_benchmark_module():
    module_name = "applications.scrna.pancreas_benchmark"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_pancreas_benchmark_manifest_and_summary_record_generated_timestamp(monkeypatch, tmp_path):
    module = _load_benchmark_module()
    output_dir = (
        tmp_path
        / "raw"
        / "assets"
        / "benchmark-results"
        / "pancreas_scrna_cluster_benchmark_20260623"
    )
    fake_adata = _FakeAdata()
    qc_summary = {
        "batches": 2,
        "celltype_count": 2,
        "raw_layer_present": True,
        "mitochondrial_gene_count": 0,
        "scdblfinder_status": "not_run",
        "ambient_rna_status": "not_run",
    }
    results = pd.DataFrame(
        [
            {
                "method": "baseline",
                "status": "ok",
                "significance_level": 0.01,
                "edge_alpha": 0.001,
                "n_clusters": 2,
                "overcluster_ratio": 1.0,
                "weighted_cluster_purity": 1.0,
                "weighted_label_dominant_cluster_recall": 1.0,
                "merge_error_rate": 0.0,
                "split_error_rate": 0.0,
                "weighted_effective_clusters_per_label": 1.0,
                "homogeneity": 1.0,
                "completeness": 1.0,
                "v_measure": 1.0,
                "nmi": 1.0,
                "ari": 1.0,
                "silhouette": 0.5,
                "elapsed_sec": 0.1,
                "skip_reason": "",
            }
        ]
    )

    monkeypatch.setattr(module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_prepare_adata", lambda *_args, **_kwargs: fake_adata)
    monkeypatch.setattr(module, "_write_qc_outputs", lambda *_args, **_kwargs: qc_summary)
    monkeypatch.setattr(
        module,
        "_run_benchmarks",
        lambda *_args, **_kwargs: (results, pd.DataFrame(), [], None, []),
    )
    monkeypatch.setattr(module, "_write_plots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pancreas_scrna_cluster_benchmark.py",
            "--max-cells",
            "3",
            "--n-pcs",
            "2",
            "--seed",
            "0",
        ],
    )

    module.main()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        manifest["generated_at"],
    )
    assert f"Generated at: {manifest['generated_at']}" in summary
