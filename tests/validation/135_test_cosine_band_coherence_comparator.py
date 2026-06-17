from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.spectral.cosine_band_coherence_comparator import (
    _one_sided_enrichment_p_values,
    build_cosine_band_coherence_rows,
    cluster_biological_coherence,
    legacy_cosine_bands,
    load_comparator_matrix,
)
from scipy.stats import fisher_exact


def _coherent_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        index=[f"g{i}" for i in range(10)],
        columns=[f"go{i}" for i in range(16)],
        dtype=float,
    )


def test_legacy_cosine_bands_keep_c2ef_fixed_band_contract() -> None:
    bands = legacy_cosine_bands(rank=6)

    assert [band.block_name for band in bands] == [
        "common_mode_01",
        "variation_02_05",
        "variation_06_15",
        "broad_variation_02_35",
        "broad_variation_02_80",
        "all_modes_01_80",
    ]
    assert [(band.block_start, band.block_end) for band in bands] == [
        (1, 1),
        (2, 5),
        (6, 6),
        (2, 6),
        (2, 6),
        (1, 6),
    ]


def test_cluster_biological_coherence_detects_enriched_feature_blocks() -> None:
    data = _coherent_matrix()
    labels = pd.Series([0] * 5 + [1] * 5, index=data.index, name="cluster_id")

    coherence = cluster_biological_coherence(
        data,
        labels,
        min_significant_terms=2,
        min_prevalence_delta=0.50,
    )

    assert set(coherence["cluster_id"]) == {0, 1}
    assert coherence["coherent_by_rule"].all()
    assert coherence["n_significant_terms_q05"].min() >= 3
    assert coherence["top_term_prevalence_delta"].min() == pytest.approx(1.0)


def test_vectorized_enrichment_matches_one_sided_fisher_exact() -> None:
    cluster_counts = pd.Series([5, 3, 0], index=["a", "b", "c"])
    rest_counts = pd.Series([1, 4, 2], index=["a", "b", "c"])
    cluster_size = 6
    rest_size = 8

    observed = _one_sided_enrichment_p_values(
        cluster_counts=cluster_counts,
        rest_counts=rest_counts,
        cluster_size=cluster_size,
        rest_size=rest_size,
    )
    expected = []
    for term in cluster_counts.index:
        present_cluster = int(cluster_counts[term])
        present_rest = int(rest_counts[term])
        _, p_value = fisher_exact(
            [
                [present_cluster, cluster_size - present_cluster],
                [present_rest, rest_size - present_rest],
            ],
            alternative="greater",
        )
        expected.append(float(p_value))

    assert observed.tolist() == pytest.approx(expected)


def test_load_comparator_matrix_accepts_csv_and_tsv_inputs(tmp_path) -> None:
    data = _coherent_matrix().iloc[:3, :3]
    csv_path = tmp_path / "matrix.csv"
    tsv_path = tmp_path / "matrix.tsv"
    data.to_csv(csv_path)
    data.to_csv(tsv_path, sep="\t")

    assert load_comparator_matrix(csv_path).equals(data)
    assert load_comparator_matrix(tsv_path).equals(data)


def test_cosine_band_comparator_runs_fixed_bands_and_reports_coherence() -> None:
    data = _coherent_matrix()
    rows, coherence, spectrum = build_cosine_band_coherence_rows(
        data=data,
        weightings=("binary",),
        max_rank=6,
        min_significant_terms=2,
        min_prevalence_delta=0.50,
    )

    ok = rows[rows["status"].eq("ok")]
    assert "variation_02_05" in set(rows["block_name"])
    assert not ok.empty
    assert "diagnostic_only_not_production_calibration" in set(
        rows["production_status"]
    )
    assert ok["coherent_cluster_fraction"].max() > 0.0
    assert {"run_id", "cluster_id", "coherent_by_rule"}.issubset(coherence.columns)
    assert {"weighting", "component", "eigenvalue"}.issubset(spectrum.columns)


def test_cosine_band_comparator_writes_partial_checkpoints(tmp_path) -> None:
    data = _coherent_matrix()

    rows, _coherence, _spectrum = build_cosine_band_coherence_rows(
        data=data,
        weightings=("binary",),
        max_rank=6,
        checkpoint_dir=tmp_path,
        min_significant_terms=2,
        min_prevalence_delta=0.50,
    )

    checkpoint_rows = pd.read_csv(
        tmp_path / "cosine_band_comparator_rows.partial.csv"
    )
    assert checkpoint_rows.shape[0] == rows.shape[0]
    assert (tmp_path / "cosine_band_comparator_cluster_coherence.partial.csv").exists()
    assert (tmp_path / "cosine_band_comparator_spectrum.partial.csv").exists()


def test_cosine_band_comparator_can_run_one_named_band() -> None:
    rows, _coherence, _spectrum = build_cosine_band_coherence_rows(
        data=_coherent_matrix(),
        weightings=("binary",),
        max_rank=6,
        band_names=("variation_02_05",),
        min_significant_terms=2,
        min_prevalence_delta=0.50,
    )

    assert rows["block_name"].tolist() == ["variation_02_05"]
