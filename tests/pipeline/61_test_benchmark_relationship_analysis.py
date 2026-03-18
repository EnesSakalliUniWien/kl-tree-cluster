from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.shared.relationship_analysis import (
    analyze_benchmark_relationships,
    normalize_results_dataframe,
    prepare_relationship_frame,
)


def _make_synthetic_results() -> pd.DataFrame:
    cases = [
        ("gauss_easy", "improved_gaussian", 3, 40, 20, 0.10),
        ("gauss_hard", "gaussian_extreme_noise", 4, 30, 300, 0.90),
        ("binary_easy", "improved_binary_low_noise", 4, 100, 80, 0.10),
        ("binary_noisy", "binary_noise_features", 6, 120, 500, 0.60),
        ("overlap_hard", "overlapping_binary_heavy", 8, 180, 400, 0.70),
        ("phylo_mid", "phylogenetic_dna", 5, 150, 200, 0.30),
        ("cat_mid", "categorical_clear", 4, 140, 60, 0.20),
        ("sbm_mid", "sbm_graphs", 3, 90, 70, 0.50),
    ]

    rows: list[dict[str, object]] = []
    for idx, (case_id, category, true_k, samples, features, noise) in enumerate(cases, start=1):
        rows.append(
            {
                "test_case": idx,
                "case_id": case_id,
                "case_category": category,
                "method": "kl",
                "params": "tree_distance_metric=hamming",
                "true_clusters": true_k,
                "found_clusters": [3, 1, 4, 2, 12, 3, 1, 1][idx - 1],
                "samples": samples,
                "features": features,
                "noise": noise,
                "ari": [0.92, 0.35, 0.88, 0.40, 0.20, 0.55, 0.30, 0.10][idx - 1],
                "nmi": [0.94, 0.45, 0.90, 0.48, 0.28, 0.64, 0.42, 0.18][idx - 1],
                "purity": [0.96, 0.52, 0.93, 0.60, 0.40, 0.72, 0.50, 0.33][idx - 1],
                "macro_recall": 0.5,
                "macro_f1": 0.5,
                "worst_cluster_recall": 0.3,
                "cluster_count_abs_error": abs([0, -3, 0, -4, 4, -2, -3, -2][idx - 1]),
                "over_split": float([0, 0, 0, 0, 1, 0, 0, 0][idx - 1]),
                "under_split": float([0, 1, 0, 1, 0, 1, 1, 1][idx - 1]),
                "status": "ok",
                "skip_reason": "",
                "labels_length": samples,
            }
        )
        rows.append(
            {
                "test_case": idx,
                "case_id": case_id,
                "case_category": category,
                "method": "kmeans",
                "params": "n_clusters=true",
                "true_clusters": true_k,
                "found_clusters": [3, 4, 4, 5, 7, 5, 4, 3][idx - 1],
                "samples": samples,
                "features": features,
                "noise": noise,
                "ari": [0.98, 0.82, 0.97, 0.85, 0.72, 0.88, 0.95, 0.40][idx - 1],
                "nmi": [0.98, 0.86, 0.97, 0.87, 0.78, 0.90, 0.95, 0.46][idx - 1],
                "purity": [0.99, 0.90, 0.98, 0.91, 0.85, 0.92, 0.96, 0.55][idx - 1],
                "macro_recall": 0.8,
                "macro_f1": 0.8,
                "worst_cluster_recall": 0.7,
                "cluster_count_abs_error": abs([0, 0, 0, -1, -1, 0, 0, 0][idx - 1]),
                "over_split": 0.0,
                "under_split": float([0, 0, 0, 1, 1, 0, 0, 0][idx - 1]),
                "status": "ok",
                "skip_reason": "",
                "labels_length": samples,
            }
        )

    rows.append(
        {
            "test_case": 99,
            "case_id": "skipped_case",
            "case_category": "improved_gaussian",
            "method": "dbscan",
            "params": "",
            "true_clusters": 4,
            "found_clusters": 0,
            "samples": 50,
            "features": 20,
            "noise": 0.2,
            "ari": None,
            "nmi": None,
            "purity": None,
            "macro_recall": None,
            "macro_f1": None,
            "worst_cluster_recall": None,
            "cluster_count_abs_error": None,
            "over_split": None,
            "under_split": None,
            "status": "skip",
            "skip_reason": "not available",
            "labels_length": 0,
        }
    )
    return pd.DataFrame(rows)


def _write_synthetic_audit(output_dir: Path, *, case_num: int, method_slug: str, accepted: bool) -> None:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    sibling_p = 0.002 if accepted else 0.35
    sibling_flag = accepted
    branch_scale = 0.35 if accepted else 0.12
    df = pd.DataFrame(
        [
            {
                "node_id": "N0",
                "node_label": "root",
                "leaf_count": 4,
                "is_leaf": False,
                "Child_Parent_Divergence_Significant": False,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": True,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": None,
                "Sibling_Divergence_P_Value_Corrected": None,
                "Sibling_BH_Different": False,
                "parent_node": None,
                "parent_label": "",
                "branch_length": None,
            },
            {
                "node_id": "N1",
                "node_label": "left",
                "leaf_count": 2,
                "is_leaf": False,
                "Child_Parent_Divergence_Significant": True,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": False,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": sibling_p,
                "Sibling_Divergence_P_Value_Corrected": sibling_p,
                "Sibling_BH_Different": sibling_flag,
                "parent_node": "N0",
                "parent_label": "root",
                "branch_length": branch_scale,
            },
            {
                "node_id": "N2",
                "node_label": "right",
                "leaf_count": 2,
                "is_leaf": False,
                "Child_Parent_Divergence_Significant": accepted,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": False,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": sibling_p * 1.5,
                "Sibling_Divergence_P_Value_Corrected": sibling_p * 1.5,
                "Sibling_BH_Different": sibling_flag,
                "parent_node": "N0",
                "parent_label": "root",
                "branch_length": branch_scale * 1.3,
            },
            {
                "node_id": "L1",
                "node_label": "leaf1",
                "leaf_count": 1,
                "is_leaf": True,
                "Child_Parent_Divergence_Significant": False,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": True,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": None,
                "Sibling_Divergence_P_Value_Corrected": None,
                "Sibling_BH_Different": False,
                "parent_node": "N1",
                "parent_label": "left",
                "branch_length": branch_scale * 0.6,
            },
            {
                "node_id": "L2",
                "node_label": "leaf2",
                "leaf_count": 1,
                "is_leaf": True,
                "Child_Parent_Divergence_Significant": False,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": True,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": None,
                "Sibling_Divergence_P_Value_Corrected": None,
                "Sibling_BH_Different": False,
                "parent_node": "N2",
                "parent_label": "right",
                "branch_length": branch_scale * 0.7,
            },
        ]
    )
    df.to_csv(audit_dir / f"case_{case_num}_{method_slug}_stats.csv", index=False)


def test_normalize_results_dataframe_maps_legacy_columns() -> None:
    df = pd.DataFrame(
        {
            "Test": [1],
            "Case_Name": ["legacy_case"],
            "Case_Category": ["improved_gaussian"],
            "Method": ["KL Divergence"],
            "True": [3],
            "Found": [2],
            "Samples": [30],
            "Features": [20],
            "Noise": [0.4],
            "ARI": [0.5],
            "NMI": [0.6],
            "Purity": [0.7],
            "Status": ["ok"],
        }
    )

    normalized = normalize_results_dataframe(df)

    assert list(normalized["test_case"]) == [1]
    assert list(normalized["case_id"]) == ["legacy_case"]
    assert list(normalized["method"]) == ["KL Divergence"]
    assert list(normalized["ari"]) == [0.5]
    assert list(normalized["status"]) == ["ok"]


def test_prepare_relationship_frame_derives_split_flags() -> None:
    frame = prepare_relationship_frame(_make_synthetic_results())

    kl_overlap = frame[(frame["method"] == "kl") & (frame["case_id"] == "overlap_hard")].iloc[0]
    assert kl_overlap["section"] == "overlapping"
    assert kl_overlap["over_split_flag"] == 1.0
    assert kl_overlap["under_split_flag"] == 0.0
    assert kl_overlap["exact_k"] == 0.0

    kmeans_easy = frame[(frame["method"] == "kmeans") & (frame["case_id"] == "gauss_easy")].iloc[0]
    assert kmeans_easy["exact_k"] == 1.0
    assert kmeans_easy["cluster_error_signed"] == 0.0


def test_analyze_benchmark_relationships_writes_expected_artifacts(tmp_path: Path) -> None:
    _write_synthetic_audit(tmp_path, case_num=1, method_slug="kl_divergence", accepted=False)
    _write_synthetic_audit(tmp_path, case_num=1, method_slug="k-means", accepted=True)
    _write_synthetic_audit(tmp_path, case_num=2, method_slug="kl_divergence", accepted=False)
    _write_synthetic_audit(tmp_path, case_num=2, method_slug="k-means", accepted=True)

    artifacts = analyze_benchmark_relationships(
        _make_synthetic_results(),
        tmp_path,
        source_path=tmp_path / "full_benchmark_comparison.csv",
        include_plots=True,
    )

    assert artifacts.report_md.exists()
    assert artifacts.augmented_rows_csv.exists()
    assert artifacts.method_summary_csv.exists()
    assert artifacts.section_summary_csv.exists()
    assert artifacts.method_section_summary_csv.exists()
    assert artifacts.correlation_summary_csv.exists()
    assert artifacts.pairwise_method_summary_csv.exists()
    assert artifacts.regression_ari_csv is not None and artifacts.regression_ari_csv.exists()
    assert artifacts.regression_exact_k_csv is not None and artifacts.regression_exact_k_csv.exists()
    assert artifacts.plots_pdf is not None and artifacts.plots_pdf.exists()

    augmented_rows = pd.read_csv(artifacts.augmented_rows_csv)
    assert "audit_available" in augmented_rows.columns
    assert augmented_rows["audit_available"].sum() >= 4
    assert "audit_root_sibling_neglog10_p" in augmented_rows.columns

    method_summary = pd.read_csv(artifacts.method_summary_csv)
    assert list(method_summary["method"][:2]) == ["kmeans", "kl"]
    assert method_summary.loc[method_summary["method"] == "kmeans", "mean_ari"].iloc[0] > method_summary.loc[
        method_summary["method"] == "kl", "mean_ari"
    ].iloc[0]

    section_summary = pd.read_csv(artifacts.section_summary_csv)
    assert {"gaussian", "binary", "categorical", "overlapping", "phylogenetic", "sbm"}.issubset(
        set(section_summary["section"])
    )

    ari_regression = pd.read_csv(artifacts.regression_ari_csv)
    assert "noise_z" in set(ari_regression["term"])
    assert {"log_features_z", "log_samples_per_cluster_z"}.intersection(set(ari_regression["term"]))

    report_text = artifacts.report_md.read_text()
    assert "Benchmark Relationship Report" in report_text
    assert "Method Summary" in report_text
    assert "Pairwise Method Contrasts" in report_text
    assert "Audit Factor Highlights" in report_text


def test_attach_audit_factors_does_not_cross_assign_single_method_audits(tmp_path: Path) -> None:
    _write_synthetic_audit(tmp_path, case_num=1, method_slug="kl_divergence", accepted=True)

    artifacts = analyze_benchmark_relationships(
        pd.DataFrame(
            [
                {
                    "test_case": 1,
                    "case_id": "case_1",
                    "case_category": "improved_gaussian",
                    "method": "kl",
                    "params": "",
                    "true_clusters": 2,
                    "found_clusters": 2,
                    "samples": 10,
                    "features": 5,
                    "noise": 0.1,
                    "ari": 0.8,
                    "nmi": 0.8,
                    "purity": 0.9,
                    "status": "ok",
                },
                {
                    "test_case": 1,
                    "case_id": "case_1",
                    "case_category": "improved_gaussian",
                    "method": "kmeans",
                    "params": "",
                    "true_clusters": 2,
                    "found_clusters": 2,
                    "samples": 10,
                    "features": 5,
                    "noise": 0.1,
                    "ari": 0.9,
                    "nmi": 0.9,
                    "purity": 0.95,
                    "status": "ok",
                },
            ]
        ),
        tmp_path,
        include_plots=False,
    )

    augmented_rows = pd.read_csv(artifacts.augmented_rows_csv)
    kl_row = augmented_rows.loc[augmented_rows["method"] == "kl"].iloc[0]
    kmeans_row = augmented_rows.loc[augmented_rows["method"] == "kmeans"].iloc[0]
    assert kl_row["audit_available"] == 1.0
    assert kmeans_row["audit_available"] == 0.0


def test_analyze_benchmark_relationships_handles_missing_branch_length_column(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "node_id": "N0",
                "node_label": "root",
                "leaf_count": 2,
                "is_leaf": False,
                "Child_Parent_Divergence_Significant": False,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": True,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": None,
                "Sibling_Divergence_P_Value_Corrected": None,
                "Sibling_BH_Different": False,
                "parent_node": None,
                "parent_label": "",
            },
            {
                "node_id": "N1",
                "node_label": "left",
                "leaf_count": 1,
                "is_leaf": True,
                "Child_Parent_Divergence_Significant": True,
                "Child_Parent_Divergence_Invalid": False,
                "Sibling_Divergence_Skipped": False,
                "Sibling_Divergence_Invalid": False,
                "Sibling_Divergence_P_Value": 0.01,
                "Sibling_Divergence_P_Value_Corrected": 0.01,
                "Sibling_BH_Different": True,
                "parent_node": "N0",
                "parent_label": "root",
            },
        ]
    ).to_csv(audit_dir / "case_1_kl_divergence_stats.csv", index=False)

    artifacts = analyze_benchmark_relationships(
        pd.DataFrame(
            [
                {
                    "test_case": 1,
                    "case_id": "case_1",
                    "case_category": "improved_gaussian",
                    "method": "kl",
                    "params": "",
                    "true_clusters": 2,
                    "found_clusters": 2,
                    "samples": 10,
                    "features": 5,
                    "noise": 0.1,
                    "ari": 0.8,
                    "nmi": 0.8,
                    "purity": 0.9,
                    "status": "ok",
                }
            ]
        ),
        tmp_path,
        include_plots=False,
    )

    augmented_rows = pd.read_csv(artifacts.augmented_rows_csv)
    row = augmented_rows.iloc[0]
    assert row["audit_available"] == 1.0
    assert pd.isna(row["audit_mean_branch_length"])
