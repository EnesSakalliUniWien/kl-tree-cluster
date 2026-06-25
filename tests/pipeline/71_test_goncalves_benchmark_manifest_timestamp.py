"""Tests for Goncalves benchmark manifest timestamps."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd


class _FakeAdata:
    def __init__(self) -> None:
        self.obs = pd.DataFrame(
            {
                "celltype": ["trunk", "tip", "trunk"],
            },
            index=["cell0", "cell1", "cell2"],
        )
        self.n_obs = 3
        self.n_vars = 2
        self.uns = {"input_expression_kind": "processed_scaled"}
        self.layers = {"input_expression": object()}


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts/goncalves_pancreas_progenitor_benchmark.py"
    spec = importlib.util.spec_from_file_location("goncalves_pancreas_progenitor_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_goncalves_benchmark_manifest_and_summary_record_generated_timestamp(monkeypatch, tmp_path):
    module = _load_benchmark_module()
    output_dir = tmp_path / "out"
    fake_adata = _FakeAdata()
    qc_summary = {
        "celltype_count": 2,
        "batches": 1,
        "raw_layer_present": "false",
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
                "v_measure": 1.0,
                "nmi": 1.0,
                "ari": 1.0,
                "silhouette": 0.5,
                "elapsed_sec": 0.1,
                "skip_reason": "",
            }
        ]
    )

    monkeypatch.setattr(module, "_ensure_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_build_adata",
        lambda _expr, _meta, out: (out.mkdir(parents=True, exist_ok=True) or fake_adata, "test_orientation"),
    )
    monkeypatch.setattr(module, "_prepare_classical_workflow", lambda adata, *_args, **_kwargs: (adata, 2))
    monkeypatch.setattr(module, "_write_qc_outputs", lambda *_args, **_kwargs: qc_summary)
    monkeypatch.setattr(
        module,
        "_run_benchmarks",
        lambda *_args, **_kwargs: (results, pd.DataFrame(), {}, {}, []),
    )
    monkeypatch.setattr(module, "_write_plots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "goncalves_pancreas_progenitor_benchmark.py",
            "--skip-download",
            "--expr-matrix",
            str(tmp_path / "expr.tsv.gz"),
            "--metadata",
            str(tmp_path / "meta.tsv"),
            "--output-dir",
            str(output_dir),
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
