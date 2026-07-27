"""Tests for generated timestamps in scRNA audit markdown reports."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _load_script(name: str):
    script_path = (
        Path(__file__).resolve().parents[2] / "applications/scrna/analysis" / name
    )
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_path.stem] = module
    spec.loader.exec_module(module)
    return module


def _assert_generated_at(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert re.search(
        r"Generated at: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        text,
    )


def test_branch_length_audit_report_records_generated_timestamp(monkeypatch, tmp_path):
    module = _load_script("audit_branch_length_effects.py")
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    effects = pd.DataFrame(
        [
            {
                "dataset_label": "Adult pancreas",
                "geometry": "adaptive_diffusion",
                "variant": "topology_only",
                "n_clusters": 43,
                "v_measure_vs_celltype": 0.66,
                "weighted_cluster_purity": 0.93,
                "weighted_label_dominant_cluster_recall": 0.48,
                "delta_n_clusters_vs_topology": 0,
                "assignment_ari_vs_topology": 1.0,
                "edge_open_count": 42,
                "branch_length_median": 0.1,
                "branch_length_max": 1.2,
                "branch_length_optimization_method": "none",
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "dataset": "adult_pancreas",
                "dataset_label": "Adult pancreas",
                "geometry": "adaptive_diffusion",
                "length_model": "topology",
                "edge_open_count": 42,
                "n_clusters": 43,
                "weighted_cluster_purity": 0.93,
                "weighted_label_dominant_cluster_recall": 0.48,
                "v_measure": 0.66,
                "median_variance_multiplier": 1.0,
                "max_variance_multiplier": 2.0,
            }
        ]
    )

    module._write_report(effects, sensitivity, {"adult_pancreas": {"seed": 0}})

    _assert_generated_at(tmp_path / "scrna_branch_length_effect_audit.md")


def test_branch_length_audit_manifest_records_generated_timestamp(monkeypatch, tmp_path):
    module = _load_script("audit_branch_length_effects.py")
    output_root = tmp_path / "branch-audit"
    output_root.mkdir()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        module,
        "DATASETS",
        [
            module.DatasetSpec("adult_pancreas", "Adult pancreas", tmp_path / "adult"),
            module.DatasetSpec("goncalves_fetal", "Goncalves fetal pancreas", tmp_path / "goncalves"),
        ],
    )

    for name in [
        "branch_length_method_effects.csv",
        "branch_length_assignment_similarity.csv",
        "branch_time_sensitivity_combined.csv",
    ]:
        pd.DataFrame({"value": [1, 2]}).to_csv(output_root / name, index=False)
    for name in [
        "branch_length_cluster_effects.png",
        "branch_length_assignment_similarity_heatmap.png",
        "branch_time_sensitivity_effects.png",
        "scrna_branch_length_effect_audit.md",
    ]:
        (output_root / name).write_bytes(b"artifact")

    module._write_manifest("2026-06-24T20:50:00+02:00")

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_at"] == "2026-06-24T20:50:00+02:00"
    assert manifest["source_script"] == "applications/scrna/analysis/audit_branch_length_effects.py"
    assert len(manifest["artifacts"]) == 7
    method_csv = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"].endswith("branch_length_method_effects.csv")
    )
    assert method_csv["rows"] == 2
    assert method_csv["columns"] == 1


def test_distributional_action_audit_report_records_generated_timestamp(monkeypatch, tmp_path):
    module = _load_script("audit_distributional_action.py")
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    summary = pd.DataFrame(
        [
            {
                "dataset": "adult_pancreas",
                "dataset_label": "Adult pancreas",
                "geometry": "adaptive_diffusion",
                "variant": "topology_only",
                "edge_count": 1,
            },
            {
                "dataset": "adult_pancreas",
                "dataset_label": "Adult pancreas",
                "geometry": "pca_linkage",
                "variant": "topology_only",
                "edge_count": 1,
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "dataset_label": "Adult pancreas",
                "geometry": "adaptive_diffusion",
                "variant": "topology_only",
                "child_is_internal": True,
                "parent": "N1",
                "child": "N0",
                "child_leaf_count": 5,
                "standardized_delta_norm": 1.0,
                "subtree_distributional_action": 5.0,
                "branch_length": 0.2,
                "edge_test_statistic": 3.0,
                "edge_significant": True,
                "child_progenitor_interpretation": "",
                "child_top_populations": "",
            }
        ]
    )

    module._write_report(edges, summary)

    _assert_generated_at(tmp_path / "scrna_distributional_action_audit.md")


def test_distributional_action_audit_manifest_records_generated_timestamp(
    monkeypatch,
    tmp_path,
):
    module = _load_script("audit_distributional_action.py")
    output_root = tmp_path / "action-audit"
    output_root.mkdir()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        module,
        "DATASETS",
        (
            module.DatasetSpec("adult_pancreas", "Adult pancreas", tmp_path / "adult"),
            module.DatasetSpec("goncalves_fetal", "Goncalves fetal pancreas", tmp_path / "goncalves"),
        ),
    )

    for name in [
        "scrna_distributional_action_edges.csv",
        "scrna_distributional_action_method_summary.csv",
    ]:
        pd.DataFrame({"value": [1, 2, 3]}).to_csv(output_root / name, index=False)
    for name in [
        "distributional_action_vs_branch_length.png",
        "top_internal_distributional_action_edges.png",
        "distributional_action_vs_edge_statistic.png",
        "scrna_distributional_action_audit.md",
    ]:
        (output_root / name).write_bytes(b"artifact")

    module._write_manifest("2026-06-24T20:51:00+02:00")

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_at"] == "2026-06-24T20:51:00+02:00"
    assert manifest["source_script"] == "applications/scrna/analysis/audit_distributional_action.py"
    assert len(manifest["artifacts"]) == 6
    edges_csv = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"].endswith("scrna_distributional_action_edges.csv")
    )
    assert edges_csv["rows"] == 3
    assert edges_csv["columns"] == 1
