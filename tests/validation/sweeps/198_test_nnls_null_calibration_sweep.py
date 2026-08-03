from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.validation.sweeps.nnls_null_calibration_sweep import (
    BRANCH_SOURCE_LINKAGE,
    BRANCH_SOURCE_NNLS,
    SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE,
    SPECTRAL_CONTEXT_LEAF_ONLY,
    hamming_squared_euclidean_embedding,
    run_nnls_null_calibration_sweep,
)
from scipy.spatial.distance import pdist


def test_hamming_squared_euclidean_embedding_matches_binary_hamming() -> None:
    data = pd.DataFrame(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        index=["a", "b", "c"],
        dtype=float,
    )

    embedding = hamming_squared_euclidean_embedding(data)

    assert np.allclose(
        pdist(embedding, metric="sqeuclidean"),
        pdist(data, metric="hamming"),
    )


def test_nnls_null_calibration_sweep_writes_cells_pairs_and_records(tmp_path) -> None:
    outputs = run_nnls_null_calibration_sweep(
        output_dir=tmp_path,
        case_names=("binary_perfect_2c",),
        branch_sources=(BRANCH_SOURCE_LINKAGE, BRANCH_SOURCE_NNLS),
        spectral_contexts=(SPECTRAL_CONTEXT_LEAF_ONLY, SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE),
        pair_sample_size=None,
    )

    assert set(outputs) == {
        "cells",
        "records",
        "parent_eigenvalues",
        "pairs",
        "spectral_pairs",
        "manifest",
    }
    cells = pd.read_csv(outputs["cells"])
    records = pd.read_csv(outputs["records"])
    parent_eigenvalues = pd.read_csv(outputs["parent_eigenvalues"])
    pairs = pd.read_csv(outputs["pairs"])
    spectral_pairs = pd.read_csv(outputs["spectral_pairs"])

    assert set(cells["branch_source"]) == {BRANCH_SOURCE_LINKAGE, BRANCH_SOURCE_NNLS}
    assert set(cells["spectral_context"]) == {
        SPECTRAL_CONTEXT_LEAF_ONLY,
        SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE,
    }
    assert cells["tree_distance_metric"].eq("hamming").all()
    assert cells["status"].eq("ok").all()
    assert {
        "tree_n_leaves",
        "tree_n_internal_nodes",
        "tree_n_nodes",
        "spectral_total_descendant_leaf_rows",
        "spectral_total_internal_distribution_rows",
        "spectral_total_matrix_rows",
        "spectral_max_internal_distribution_rows",
        "root_descendant_leaf_rows",
        "root_internal_distribution_rows",
        "root_spectral_matrix_rows",
    }.issubset(cells.columns)
    leaf_only_cells = cells[cells["spectral_context"].eq(SPECTRAL_CONTEXT_LEAF_ONLY)]
    internal_cells = cells[
        cells["spectral_context"].eq(SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE)
    ]
    assert leaf_only_cells["spectral_total_internal_distribution_rows"].eq(0).all()
    assert internal_cells["spectral_total_internal_distribution_rows"].gt(0).all()
    assert internal_cells["root_spectral_matrix_rows"].gt(
        internal_cells["root_descendant_leaf_rows"]
    ).all()
    assert records["branch_source"].isin({BRANCH_SOURCE_LINKAGE, BRANCH_SOURCE_NNLS}).all()
    assert records["spectral_context"].isin(
        {SPECTRAL_CONTEXT_LEAF_ONLY, SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE}
    ).all()
    assert not records.empty
    assert not parent_eigenvalues.empty
    assert {
        "parent",
        "eigenvalue_index",
        "eigenvalue",
        "is_positive",
        "positive_eigenvalue_share",
        "parent_descendant_leaf_rows",
        "parent_internal_distribution_rows",
        "parent_spectral_matrix_rows",
        "parent_active_feature_count",
        "parent_mp_threshold_rows",
    }.issubset(parent_eigenvalues.columns)
    assert len(pairs) == 2
    assert pairs.loc[0, "source_case_id"] == "binary_perfect_2c"
    assert len(spectral_pairs) == 2
    assert spectral_pairs["variant_spectral_context"].eq(
        SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE
    ).all()


def test_records_csv_stamps_role_support_from_the_record_rule(tmp_path) -> None:
    outputs = run_nnls_null_calibration_sweep(
        output_dir=tmp_path,
        case_names=("binary_perfect_2c",),
        branch_sources=(BRANCH_SOURCE_LINKAGE,),
        spectral_contexts=(SPECTRAL_CONTEXT_LEAF_ONLY,),
        pair_sample_size=None,
    )
    records = pd.read_csv(outputs["records"])

    assert "is_role_supported" in records.columns
    assert records["is_role_supported"].dtype == bool
    assert records["is_role_supported"].equals(
        records["is_null_like"] | records["is_edge_blocked"]
    )
