from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.result_records import ComputedResultRecord
from benchmarks.shared.result_records.factory import build_benchmark_result_row
from benchmarks.shared.tree_consensus import (
    EXPECTED_TREE_INFERENCES,
    export_tree_consensus_label_files,
    write_tree_consensus_artifacts,
)
from benchmarks.shared.util import case_execution
from benchmarks.shared.util.case_run import run_single_case


def _run_id(tree: str) -> str:
    suffix = (
        "tree_builder_neighbor_joining"
        if tree == "neighbor_joining"
        else f"tree_linkage_method_{tree}"
    )
    return (
        f"tbs_diffusion_graphtools_adaptive_nnls::graphtools_adaptive_k_tree_strategy__{suffix}__r0"
    )


def _result_rows(case_id: str, *, test_case: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_case": test_case,
                "case_id": case_id,
                "case_category": "toy",
                "source_family": "toy_source",
                "feature_representation": "toy_representation",
                "method": "tbs_diffusion_graphtools_adaptive_nnls",
                "run_id": _run_id(tree),
                "benchmark_class": "optional_gpl",
                "benchmark_grid": "graphtools_adaptive_k_tree_strategy",
                "benchmark_repeat": 0,
                "status": "ok",
                "skip_reason": "",
                "true_clusters": 2,
                "found_clusters": 2,
                "labels_length": 4,
                "ari": 0.8,
                "nmi": 0.7,
                "macro_f1": 0.9,
                "purity": 1.0,
                "silhouette_score": 0.2 + index * 0.01,
                "calinski_harabasz_index": 20.0 + index,
                "davies_bouldin_index": 2.0 + index * 0.01,
                "largest_cluster_fraction": 0.5 if tree == "single" else 0.25,
            }
            for index, tree in enumerate(EXPECTED_TREE_INFERENCES)
        ]
    )


def _write_label_files(
    labels_dir: Path, results: pd.DataFrame, *, skip_run_id: str | None = None
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for row in results.itertuples(index=False):
        if row.run_id == skip_run_id:
            continue
        frame = pd.DataFrame(
            {
                "test_case": [row.test_case] * 4,
                "case_id": [row.case_id] * 4,
                "method": [row.method] * 4,
                "run_id": [row.run_id] * 4,
                "sample_id": ["S0", "S1", "S2", "S3"],
                "cluster_label": [0, 0, 1, 1],
            }
        )
        frame.to_csv(
            labels_dir / f"{row.test_case}_{row.run_id.replace(':', '_')}.csv", index=False
        )


def test_write_tree_consensus_artifacts_outputs_expected_files(tmp_path: Path) -> None:
    results = pd.concat(
        [
            _result_rows("case_a", test_case=1),
            _result_rows("case_b", test_case=2),
        ],
        ignore_index=True,
    )
    labels_dir = tmp_path / "labels"
    _write_label_files(labels_dir, results)

    artifacts = write_tree_consensus_artifacts(
        results,
        labels_dir,
        tmp_path,
        source_path=Path("full_benchmark_comparison.csv"),
    )

    for path in [
        artifacts.label_assignments_csv,
        artifacts.pairwise_agreement_csv,
        artifacts.stability_csv,
        artifacts.rankings_csv,
        artifacts.selection_csv,
        artifacts.summary_csv,
        artifacts.report_md,
    ]:
        assert path.exists()

    summary = pd.read_csv(artifacts.summary_csv)
    assert summary.loc[0, "result_rows"] == 16
    assert summary.loc[0, "expected_result_rows"] == 16
    assert summary.loc[0, "label_integrity_pass"] is True or bool(
        summary.loc[0, "label_integrity_pass"]
    )

    selection = pd.read_csv(artifacts.selection_csv)
    assert set(selection["selector_status"]) == {"selected"}
    assert len(selection) == 2


def test_write_tree_consensus_artifacts_rejects_missing_ok_labels(tmp_path: Path) -> None:
    results = _result_rows("case_a", test_case=1)
    labels_dir = tmp_path / "labels"
    missing_run_id = str(results.iloc[0]["run_id"])
    _write_label_files(labels_dir, results, skip_run_id=missing_run_id)

    with pytest.raises(ValueError, match="Missing label assignments"):
        write_tree_consensus_artifacts(results, labels_dir, tmp_path)


def test_write_tree_consensus_artifacts_rejects_duplicate_case_run(tmp_path: Path) -> None:
    results = _result_rows("case_a", test_case=1)
    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    labels_dir = tmp_path / "labels"
    _write_label_files(labels_dir, results)

    with pytest.raises(ValueError, match="unique case_id/run_id"):
        write_tree_consensus_artifacts(duplicate, labels_dir, tmp_path)


def test_export_tree_consensus_label_files_writes_stable_columns(tmp_path: Path) -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3]}, index=["S0", "S1", "S2", "S3"])
    record = ComputedResultRecord(
        test_case_num=7,
        method="tbs_diffusion_graphtools_adaptive_nnls",
        method_name="TBS",
        run_id=_run_id("weighted"),
        benchmark_class="optional_gpl",
        benchmark_grid="graphtools_adaptive_k_tree_strategy",
        benchmark_repeat=0,
        params={},
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
        labels=np.array([0, 0, 1, 1]),
        data=data,
        meta={"name": "case_export"},
        x_original=np.zeros((4, 1)),
        y_true=np.array([0, 0, 1, 1]),
        tree=None,
        decomposition=None,
        annotations=None,
    )

    paths = export_tree_consensus_label_files([record], tmp_path)

    assert len(paths) == 1
    frame = pd.read_csv(paths[0])
    assert list(frame.columns) == [
        "test_case",
        "case_id",
        "method",
        "run_id",
        "sample_id",
        "cluster_label",
    ]
    assert frame["case_id"].unique().tolist() == ["case_export"]
    assert frame["cluster_label"].tolist() == [0, 0, 1, 1]


def test_export_tree_consensus_label_files_rejects_malformed_records(tmp_path: Path) -> None:
    record = SimpleNamespace(
        method="tbs_diffusion_graphtools_adaptive_nnls",
        run_id=_run_id("weighted"),
        benchmark_grid="graphtools_adaptive_k_tree_strategy",
        labels=None,
        data=None,
        test_case_num=7,
        meta={"name": "case_export"},
    )

    with pytest.raises(ValueError, match="labels and a data index"):
        export_tree_consensus_label_files([record], tmp_path)


def test_run_single_case_exports_tree_consensus_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3]}, index=["S0", "S1", "S2", "S3"])
    record = ComputedResultRecord(
        test_case_num=3,
        method="tbs_diffusion_graphtools_adaptive_nnls",
        method_name="TBS",
        run_id=_run_id("weighted"),
        benchmark_class="optional_gpl",
        benchmark_grid="graphtools_adaptive_k_tree_strategy",
        benchmark_repeat=0,
        params={},
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
        labels=np.array([0, 0, 1, 1]),
        data=data,
        meta={"name": "case_single"},
        x_original=np.zeros((4, 1)),
        y_true=np.array([0, 0, 1, 1]),
        tree=None,
        decomposition=None,
        annotations=None,
    )
    row = build_benchmark_result_row(
        test_case=3,
        case_id="case_single",
        case_category="toy",
        source_family="toy",
        feature_representation="toy",
        method="tbs_diffusion_graphtools_adaptive_nnls",
        run_params={},
        run_id=_run_id("weighted"),
        benchmark_class="optional_gpl",
        benchmark_grid="graphtools_adaptive_k_tree_strategy",
        benchmark_repeat=0,
        true_clusters=2,
        found_clusters=2,
        samples=4,
        features=1,
        noise=np.nan,
        ari=1.0,
        nmi=1.0,
        purity=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        worst_cluster_recall=1.0,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
        cluster_count_abs_error=0.0,
        over_split=0.0,
        under_split=0.0,
        status="ok",
        skip_reason="",
        labels_length=4,
    )

    monkeypatch.setattr(
        "benchmarks.shared.util.case_run.prepare_case_inputs",
        lambda _tc, _methods: SimpleNamespace(
            data=data,
            labels=np.array([0, 0, 1, 1]),
            original_features=np.zeros((4, 1)),
            metadata={"name": "case_single"},
            distance_matrix=None,
            distance_condensed=None,
        ),
    )

    def _fake_run_single_method_once(**_kwargs):
        return row, record, None

    monkeypatch.setattr(
        "benchmarks.shared.util.case_run.run_single_method_once",
        _fake_run_single_method_once,
    )
    label_dir = tmp_path / "labels"

    run_single_case(
        tc={"test_case_num": 3, "name": "case_single", "seed": 0},
        total_cases=1,
        selected_methods=["tbs_diffusion_graphtools_adaptive_nnls"],
        param_sets={"tbs_diffusion_graphtools_adaptive_nnls": [{}]},
        significance_level=0.01,
        edge_alpha=0.001,
        output_pdf=None,
        plots_root=tmp_path / "plots",
        matrix_audit=False,
        verbose=False,
        tree_consensus_label_dir=label_dir,
    )

    exported = sorted(label_dir.glob("*.csv"))
    assert len(exported) == 1
    assert pd.read_csv(exported[0])["case_id"].unique().tolist() == ["case_single"]


def test_isolated_worker_passes_tree_consensus_label_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_benchmark_cluster_algorithm(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"case_id": "case_worker"}]), None

    monkeypatch.setattr(
        case_execution,
        "_get_benchmark_fn",
        lambda: _fake_benchmark_cluster_algorithm,
    )

    class Queue:
        def __init__(self) -> None:
            self.payload = None

        def put(self, payload) -> None:
            self.payload = payload

    queue = Queue()
    label_dir = tmp_path / "labels"
    case_execution._run_case_worker(
        queue,
        {"name": "case_worker", "test_case_num": 1},
        ["tbs_diffusion_graphtools_adaptive_nnls"],
        False,
        False,
        False,
        None,
        False,
        {"tbs_diffusion_graphtools_adaptive_nnls": [{}]},
        str(label_dir),
    )

    assert queue.payload["ok"] is True
    assert captured["tree_consensus_label_dir"] == str(label_dir)
