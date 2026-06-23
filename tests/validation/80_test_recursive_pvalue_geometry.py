from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.path_b.recursive_pvalue_geometry import (
    alpha_margin,
    build_recursive_pvalue_geometry_panels,
    chi_square_log10_residual,
    eigenvalue_geometry,
    neg_log10_pvalue,
    principal_subspace_alignment,
    summarize_recursive_pvalue_geometry,
)
from scipy.stats import chi2


class _SpectralContext:
    def __init__(self) -> None:
        self.test_projection_dimensions_by_node = {"root": 2, "L": 2, "R": 1}
        self.raw_mp_signal_counts_by_node = {"root": 1, "L": 2, "R": 0}
        self.principal_component_projections_by_node = {
            "root": np.eye(2),
            "L": np.eye(2),
            "R": np.array([[1.0, 0.0]]),
        }
        self.principal_component_eigenvalues_by_node = {
            "root": np.array([4.0, 1.0]),
            "L": np.array([3.0, 1.0]),
            "R": np.array([2.0]),
        }


def test_pvalue_transforms_and_chi_square_residual() -> None:
    stat = 6.0
    df = 2.0
    p_value = chi2.sf(stat, df=df)

    assert neg_log10_pvalue(0.01) == pytest.approx(2.0)
    assert alpha_margin(0.001, 0.01) == pytest.approx(1.0)
    assert chi_square_log10_residual(stat, df, p_value) < 1e-12


def test_principal_subspace_alignment_is_sign_invariant() -> None:
    parent = np.eye(2)
    child = -np.eye(2)

    alignment = principal_subspace_alignment(parent, child)

    assert alignment["subspace_common_dimension"] == 2
    assert alignment["subspace_largest_cosine"] == pytest.approx(1.0)
    assert alignment["subspace_mean_squared_cosine"] == pytest.approx(1.0)
    assert alignment["subspace_chordal_distance_normalized"] == pytest.approx(0.0)


def test_eigenvalue_geometry_reports_gap_and_effective_rank() -> None:
    geometry = eigenvalue_geometry(np.array([4.0, 1.0]))

    assert geometry["selected_eigenvalue_count"] == 2
    assert geometry["selected_eigenvalue_gap_ratio"] == pytest.approx(4.0)
    assert geometry["top_selected_eigenvalue_mass_fraction"] == pytest.approx(0.8)
    assert 1.0 < geometry["selected_eigenvalue_effective_rank"] < 2.0


def test_build_recursive_pvalue_geometry_panels() -> None:
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "L"), ("root", "R")])
    annotations = pd.DataFrame(
        {
            "leaf_count": [4, 2, 2],
            "Child_Parent_Divergence_Test_Statistic": [math.nan, 6.0, 2.0],
            "Child_Parent_Divergence_P_Value": [
                math.nan,
                chi2.sf(6.0, df=2),
                chi2.sf(2.0, df=1),
            ],
            "Child_Parent_Divergence_P_Value_BH": [math.nan, 0.02, 0.2],
            "Child_Parent_Divergence_df": [math.nan, 2.0, 1.0],
            "Child_Parent_Divergence_Significant": [False, True, False],
            "Sibling_Test_Statistic": [7.0, math.nan, math.nan],
            "Sibling_Degrees_of_Freedom": [2.0, math.nan, math.nan],
            "Sibling_Divergence_P_Value": [chi2.sf(7.0, df=2), math.nan, math.nan],
            "Sibling_Divergence_P_Value_Corrected": [0.03, math.nan, math.nan],
            "Sibling_BH_Different": [True, False, False],
            "Sibling_Divergence_Skipped": [False, True, True],
        },
        index=["root", "L", "R"],
    )

    edge_panel, node_panel = build_recursive_pvalue_geometry_panels(
        case_id="case",
        tree=tree,
        annotations=annotations,
        spectral_context=_SpectralContext(),
        edge_alpha=0.05,
        sibling_alpha=0.05,
    )
    summary = summarize_recursive_pvalue_geometry(edge_panel, node_panel)

    assert edge_panel.shape[0] == 2
    assert node_panel.shape[0] == 3
    assert "subspace_chordal_distance_normalized" in edge_panel.columns
    assert "edge_sibling_connectivity_score" in node_panel.columns
    assert summary["summary_id"].str.contains("pvalue").any()
