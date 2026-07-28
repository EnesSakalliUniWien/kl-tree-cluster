from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.spectral_transport import (
    spectral_transport_overlap_dispatch_panel as panel,
)
from benchmarks.shared.result_records.factory import build_benchmark_result_row


def _row(
    *,
    method: str,
    found_clusters: int,
    ari: float,
    blocked_count: float,
) -> dict[str, object]:
    return {
        "schema_version": panel.SCHEMA_VERSION,
        "study_role": panel.STUDY_ROLE,
        "case_id": "toy_overlap",
        "method": method,
        "status": "ok",
        "true_clusters": 1,
        "found_clusters": found_clusters,
        "ari": ari,
        "nmi": ari,
        "ami": ari,
        "singleton_fraction": 0.0,
        "effective_cluster_count": float(found_clusters),
        "silhouette_score": 0.1 * found_clusters,
        "davies_bouldin_index": float(found_clusters),
        "calinski_harabasz_index": 10.0 / found_clusters,
        "spectral_transport_passthrough_blocked_count": blocked_count,
        "spectral_transport_bottleneck_count": blocked_count,
    }


def _benchmark_row(method: str, found_clusters: int, ari: float):
    return build_benchmark_result_row(
        test_case=1,
        case_id="toy_overlap",
        case_category="overlapping",
        source_family="binary",
        feature_representation="binary",
        method=method,
        run_params={"tree_distance_metric": "hamming"},
        true_clusters=1,
        found_clusters=found_clusters,
        samples=4,
        features=2,
        noise=0.0,
        ari=ari,
        nmi=ari,
        ami=ari,
        purity=1.0,
        homogeneity=ari,
        completeness=ari,
        v_measure=ari,
        fowlkes_mallows=ari,
        macro_recall=1.0,
        macro_f1=1.0,
        worst_cluster_recall=1.0,
        n_singleton_clusters=0.0,
        singleton_fraction=0.0,
        median_cluster_size=4.0 / found_clusters,
        largest_cluster_fraction=1.0,
        effective_cluster_count=float(found_clusters),
        cluster_size_entropy=0.0,
        cluster_size_gini=0.0,
        noise_label_fraction=0.0,
        silhouette_score=0.1 * found_clusters,
        davies_bouldin_index=float(found_clusters),
        calinski_harabasz_index=10.0 / found_clusters,
        outlier_precision=np.nan,
        outlier_recall=np.nan,
        outlier_f1=np.nan,
        singleton_outlier_isolated=np.nan,
        grouped_outlier_cluster_recovered=np.nan,
        cluster_count_abs_error=float(abs(found_clusters - 1)),
        over_split=float(found_clusters > 1),
        under_split=0.0,
        status="ok",
        skip_reason=None,
        labels_length=4,
    )


def test_pairwise_rows_label_spectral_block_less_fragmented_not_worse() -> None:
    rows = pd.DataFrame(
        [
            _row(
                method=panel.BASELINE_METHOD,
                found_clusters=3,
                ari=0.0,
                blocked_count=0.0,
            ),
            _row(
                method=panel.CANDIDATE_METHOD,
                found_clusters=1,
                ari=1.0,
                blocked_count=1.0,
            ),
        ]
    )
    computed = {
        ("toy_overlap", panel.BASELINE_METHOD): SimpleNamespace(labels=np.array([0, 0, 1, 2])),
        ("toy_overlap", panel.CANDIDATE_METHOD): SimpleNamespace(labels=np.array([0, 0, 0, 0])),
    }

    pairwise = panel.build_pairwise_rows(rows, computed)

    assert len(pairwise) == 1
    row = pairwise.iloc[0]
    assert row["delta_found_clusters_candidate_minus_baseline"] == -2
    assert row["delta_ari_candidate_minus_baseline"] == 1.0
    assert row["guard_effect_class"] == "spectral_block_less_fragmented_not_worse"
    assert np.isfinite(row["partition_ari_between_methods"])


def test_run_spectral_transport_overlap_dispatch_panel_writes_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        panel,
        "get_default_test_cases",
        lambda: [{"name": "toy_overlap", "generator": "binary", "seed": 123}],
    )
    monkeypatch.setattr(
        panel,
        "prepare_case_inputs",
        lambda _case, _methods: SimpleNamespace(
            data=pd.DataFrame(np.zeros((4, 2))),
            labels=np.zeros(4, dtype=int),
            original_features=np.zeros((4, 2)),
            metadata={},
            distance_matrix=None,
            distance_condensed=None,
        ),
    )

    def _fake_run_single_method_once(**kwargs):
        method = kwargs["method_id"]
        is_candidate = method == panel.CANDIDATE_METHOD
        found_clusters = 1 if is_candidate else 3
        ari = 1.0 if is_candidate else 0.0
        annotations = pd.DataFrame(
            {
                "Spectral_Transport_Pass_Through_Supported": [False],
                "Spectral_Transport_Pass_Through_Blocked": [is_candidate],
                "Spectral_Transport_Bottleneck": [
                    "spectral_transport_bottleneck" if is_candidate else ""
                ],
                "Spectral_Transport_Best_Descendant_Split_Path_Cost": [np.nan],
            }
        )
        labels = np.array([0, 0, 0, 0]) if is_candidate else np.array([0, 0, 1, 2])
        return (
            _benchmark_row(method, found_clusters, ari),
            SimpleNamespace(labels=labels, annotations=annotations),
            None,
        )

    monkeypatch.setattr(panel, "run_single_method_once", _fake_run_single_method_once)

    outputs = panel.run_spectral_transport_overlap_dispatch_panel(
        panel.SpectralTransportOverlapDispatchConfig(
            output_dir=tmp_path,
            case_names=("toy_overlap",),
        )
    )

    assert outputs["rows"].exists()
    assert outputs["pairwise"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    pairwise = pd.read_csv(outputs["pairwise"])
    summary = pd.read_csv(outputs["summary"])
    assert pairwise.iloc[0]["guard_effect_class"] == ("spectral_block_less_fragmented_not_worse")
    assert not summary.empty
