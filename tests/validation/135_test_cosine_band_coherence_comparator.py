from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.spectral.cosine_band_coherence_comparator import (
    build_cosine_band_coherence_rows,
    cluster_biological_coherence,
    legacy_cosine_bands,
)


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
