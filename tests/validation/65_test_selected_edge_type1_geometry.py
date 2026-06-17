from __future__ import annotations

from pathlib import Path

from benchmarks.validation.selected_edge_type1_geometry import (
    SelectedEdgeGeometryConfig,
    _case_contract,
    build_run_id,
    parse_alpha_grid,
    regenerate_null_case,
    run_selected_edge_replicate,
)


def test_parse_alpha_grid_rejects_invalid_values() -> None:
    assert parse_alpha_grid("0.0001,0.001") == (0.0001, 0.001)
    for raw in ("", "0", "1", "-0.1", "0.01,1.5"):
        try:
            parse_alpha_grid(raw)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid grid to fail: {raw!r}")


def test_selected_edge_config_records_output_paths(tmp_path: Path) -> None:
    config = SelectedEdgeGeometryConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        modes=("selected_tree", "fixed_tree"),
        edge_alphas=(0.0001, 0.001),
        sibling_alpha=0.01,
        replicates=3,
        base_seed=20260604,
    )

    assert build_run_id(config).startswith("selected_edge_type1_geometry__")
    assert config.edge_rows_path.name == "selected_edge_geometry_edges.csv"
    assert config.sibling_rows_path.name == "selected_edge_geometry_siblings.csv"
    assert config.final_rows_path.name == "selected_edge_geometry_final.csv"


def test_regenerate_null_case_preserves_shape_and_contract() -> None:
    data, metadata = regenerate_null_case(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        seed=7,
    )

    assert data.shape == (12, 5)
    assert metadata["true_null"] is True
    assert metadata["source_family"] == "binary_template"
    assert metadata["feature_representation"] == "binary"
    assert set(data.to_numpy().ravel()).issubset({0, 1})


def test_regenerate_categorical_null_case_uses_explicit_feature_space() -> None:
    data, metadata = regenerate_null_case(
        case_id="cat_clear_3cat_4c",
        source_family="categorical_multinomial",
        feature_representation="categorical_one_hot",
        n_samples=12,
        n_features=4,
        n_categories=3,
        seed=9,
    )

    assert data.shape == (12, 12)
    assert metadata["n_features_original"] == 4
    assert metadata["n_categories"] == 3
    assert metadata["feature_space"].family_label == "categorical"
    assert set(data.to_numpy().ravel()).issubset({0, 1})


def test_case_contract_rejects_continuous_generator_until_null_covariance_exists() -> None:
    case = {
        "name": "gauss_continuous",
        "generator": "blobs_continuous",
        "n_samples": 12,
        "n_features": 3,
    }

    try:
        _case_contract(case)
    except ValueError as exc:
        assert "continuous null regeneration" in str(exc)
        return
    raise AssertionError("continuous selected-edge null regeneration must fail explicitly")


def test_case_contract_supports_planted_hierarchy_as_binary_template() -> None:
    case = {
        "name": "traversal_deep_signal_under_same_parent",
        "generator": "planted_hierarchy_deep_signal",
        "n_samples": 24,
        "n_features": 30,
        "n_clusters": 6,
    }

    assert _case_contract(case) == (
        "traversal_deep_signal_under_same_parent",
        "binary_template",
        "binary",
        24,
        30,
        None,
    )


def test_run_selected_edge_replicate_emits_edge_rows() -> None:
    edge_rows, sibling_rows, final_rows = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=11,
        tree_seed=11,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="smoke",
    )

    assert edge_rows
    assert final_rows
    assert all(row["mode"] == "selected_tree" for row in edge_rows)
    assert all("edge_raw_p" in row for row in edge_rows)
    assert all("edge_rejected" in row for row in edge_rows)
    assert sibling_rows is not None
    assert sibling_rows
    sibling_required = {
        "sibling_raw_stat",
        "sibling_adjusted_stat",
        "sibling_raw_p",
        "covariance_inferred_df",
        "covariance_inferred_reference_scale",
        "sibling_contrast_laplacian_status",
        "parent_spectral_laplacian_status",
    }
    assert sibling_required.issubset(sibling_rows[0])


def test_run_selected_edge_replicate_supports_categorical_null() -> None:
    edge_rows, sibling_rows, final_rows = run_selected_edge_replicate(
        case_id="cat_clear_3cat_4c",
        source_family="categorical_multinomial",
        feature_representation="categorical_one_hot",
        n_samples=12,
        n_features=4,
        n_categories=3,
        replicate=0,
        data_seed=23,
        tree_seed=23,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="categorical",
    )

    assert edge_rows
    assert final_rows[0]["feature_representation"] == "categorical_one_hot"
    assert final_rows[0]["n_features"] == 12
    assert sibling_rows is not None


def test_edge_rows_include_geometry_variables() -> None:
    edge_rows, _sibling_rows, _final_rows = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=13,
        tree_seed=13,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="geometry",
    )
    required = {
        "sample_ratio",
        "edge_statistic_margin",
        "edge_bh_action",
        "tree_balance",
        "path_length_from_root",
    }
    for row in edge_rows:
        assert required.issubset(row)


def test_fixed_tree_and_selected_tree_modes_are_distinct() -> None:
    selected_edges, _selected_siblings, selected_final = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=17,
        tree_seed=17,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="selected",
    )
    fixed_edges, _fixed_siblings, fixed_final = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=19,
        tree_seed=17,
        mode="fixed_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="fixed",
    )

    assert selected_edges
    assert fixed_edges
    assert selected_final[0]["mode"] == "selected_tree"
    assert fixed_final[0]["mode"] == "fixed_tree"
