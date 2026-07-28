from __future__ import annotations

import math
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.sibling.nulls.sibling_null_prior_interpolation_audit import (
    _tree_distance_function,
    collect_reference_sets,
    compute_kernel_bandwidths,
    diagnostic_child_interpolated_prior,
    nearest_stopped_ancestor_support,
    parse_csv_list,
    run_sibling_null_prior_interpolation_audit,
)

EDGE_COLUMNS = {
    "Child_Parent_Divergence_Test_Statistic": 0.0,
    "Child_Parent_Divergence_P_Value": np.nan,
    "Child_Parent_Divergence_P_Value_BH": np.nan,
    "Child_Parent_Divergence_Significant": False,
    "Child_Parent_Divergence_df": 1.0,
    "Child_Parent_Divergence_Invalid": False,
    "Child_Parent_Divergence_Tested": False,
    "Child_Parent_Divergence_Ancestor_Blocked": False,
}


def _edge_df(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    data = {}
    for node, overrides in rows.items():
        row = EDGE_COLUMNS.copy()
        row.update(overrides)
        data[node] = row
    return pd.DataFrame.from_dict(data, orient="index")


def _tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "stopped"),
            ("stopped", "blocked_child"),
            ("root", "tested_child"),
        ]
    )
    return tree


def test_parse_csv_list_rejects_empty_case_names() -> None:
    assert parse_csv_list("a, b") == ("a", "b")
    with pytest.raises(ValueError, match="case_names"):
        parse_csv_list(" , ")


def test_direct_tested_edge_prior_uses_explicit_bh_p_value_without_dimension_fallback() -> None:
    edge_df = _edge_df(
        {
            "tested_child": {
                "Child_Parent_Divergence_P_Value_BH": 0.25,
                "Child_Parent_Divergence_Tested": True,
                "Child_Parent_Divergence_Significant": True,
                "Child_Parent_Divergence_df": 0.0,
            },
            "root": {},
        }
    )
    reference_sets = collect_reference_sets(edge_df)
    distance = _tree_distance_function(_tree())
    bandwidths = compute_kernel_bandwidths(
        tree=_tree(),
        edge_df=edge_df,
        reference_sets=reference_sets,
        tree_distance=distance,
    )

    prior = diagnostic_child_interpolated_prior(
        child="tested_child",
        tree=_tree(),
        edge_df=edge_df,
        reference_sets=reference_sets,
        kernel_bandwidths=bandwidths,
        tree_distance=distance,
    )

    assert prior.status == "direct_tested_edge_p_value"
    assert prior.prior == 0.25
    assert reference_sets.dimensionless_stable_nodes == ()


def test_blocked_child_interpolates_only_when_stopped_ancestor_is_explicit() -> None:
    tree = _tree()
    edge_df = _edge_df(
        {
            "root": {},
            "stopped": {
                "Child_Parent_Divergence_P_Value_BH": 0.6,
                "Child_Parent_Divergence_Tested": True,
                "Child_Parent_Divergence_Significant": False,
                "Child_Parent_Divergence_df": 2.0,
            },
            "blocked_child": {
                "Child_Parent_Divergence_Ancestor_Blocked": True,
                "Child_Parent_Divergence_df": 2.0,
            },
            "tested_child": {
                "Child_Parent_Divergence_P_Value_BH": 0.01,
                "Child_Parent_Divergence_Tested": True,
                "Child_Parent_Divergence_Significant": True,
                "Child_Parent_Divergence_df": 2.0,
            },
        }
    )
    distance = _tree_distance_function(tree)
    reference_sets = collect_reference_sets(edge_df)
    bandwidths = compute_kernel_bandwidths(
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        tree_distance=distance,
    )

    ancestor = nearest_stopped_ancestor_support(tree, edge_df, "blocked_child")
    prior = diagnostic_child_interpolated_prior(
        child="blocked_child",
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        kernel_bandwidths=bandwidths,
        tree_distance=distance,
    )

    assert ancestor is not None
    assert ancestor.node == "stopped"
    assert ancestor.p_value_bh == 0.6
    assert prior.status == "interpolated_from_stopped_ancestor"
    assert math.isfinite(prior.prior)
    assert 0.0 <= prior.prior <= 1.0


def test_blocked_child_without_stopped_ancestor_is_unsupported_not_neutral() -> None:
    tree = _tree()
    edge_df = _edge_df(
        {
            "root": {},
            "stopped": {
                "Child_Parent_Divergence_P_Value_BH": 0.01,
                "Child_Parent_Divergence_Tested": True,
                "Child_Parent_Divergence_Significant": True,
                "Child_Parent_Divergence_df": 2.0,
            },
            "blocked_child": {
                "Child_Parent_Divergence_Ancestor_Blocked": True,
                "Child_Parent_Divergence_df": 2.0,
            },
        }
    )
    distance = _tree_distance_function(tree)
    reference_sets = collect_reference_sets(edge_df)
    bandwidths = compute_kernel_bandwidths(
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        tree_distance=distance,
    )

    prior = diagnostic_child_interpolated_prior(
        child="blocked_child",
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        kernel_bandwidths=bandwidths,
        tree_distance=distance,
    )

    assert prior.status == "unsupported_no_stopped_ancestor"
    assert math.isnan(prior.prior)


def test_interpolation_audit_smoke_writes_diagnostic_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tables = run_sibling_null_prior_interpolation_audit(
            case_names=("binary_perfect_4c",),
            output_dir=Path(tmpdir),
        )

        assert (Path(tmpdir) / "case_summary.csv").exists()
        assert (Path(tmpdir) / "record_interpolation_audit.csv").exists()
        assert (Path(tmpdir) / "manifest.json").exists()
        assert tables.case_summary.iloc[0]["status"] == "ok"
        assert tables.case_summary.iloc[0]["diagnostic_role"].endswith("not_calibration")
        assert tables.case_summary.iloc[0]["n_selected_nonnull_positive_weight_records"] >= 0
