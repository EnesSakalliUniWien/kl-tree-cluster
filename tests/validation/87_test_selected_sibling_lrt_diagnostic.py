import math

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.selected_sibling_lrt_diagnostic import (
    bernoulli_sibling_deviance,
    extract_sibling_lrt_rows,
    parse_grid,
)
from kl_clustering_analysis.tree.io import tree_from_linkage


def test_bernoulli_sibling_deviance_is_zero_for_identical_children() -> None:
    out = bernoulli_sibling_deviance(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
        n_left=4,
        n_right=5,
    )

    assert math.isclose(out["bernoulli_deviance_lrt"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["bernoulli_lrt_nominal_fixed_pair_p"], 1.0, abs_tol=1e-5)


def test_bernoulli_sibling_deviance_detects_separated_binary_children() -> None:
    out = bernoulli_sibling_deviance(
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        n_left=5,
        n_right=5,
    )

    assert out["bernoulli_deviance_lrt"] > 0.0
    assert out["bernoulli_lrt_nominal_df_changed_features"] == 1.0
    assert out["bernoulli_lrt_nominal_fixed_pair_p"] < 0.05


def test_extract_sibling_lrt_rows_uses_tree_annotations() -> None:
    linkage_matrix = np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.1, 2],
            [4, 5, 1.0, 4],
        ],
        dtype=float,
    )
    data = pd.DataFrame(
        [[1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]],
        index=["a", "b", "c", "d"],
        columns=["f1", "f2", "f3"],
    )
    tree = tree_from_linkage(linkage_matrix, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    annotations = tree.annotations_df.copy()
    annotations["Child_Parent_Divergence_P_Value"] = 0.5
    annotations["Child_Parent_Divergence_P_Value_BH"] = 0.5
    annotations["Child_Parent_Divergence_Significant"] = False
    annotations["Child_Parent_Divergence_Ancestor_Blocked"] = False
    annotations["Sibling_Test_Statistic"] = 4.0
    annotations["Sibling_Degrees_of_Freedom"] = 2.0
    annotations["Sibling_Divergence_P_Value"] = 0.1
    annotations["Sibling_Divergence_P_Value_Corrected"] = 0.2
    annotations["Sibling_BH_Different"] = False
    annotations["Sibling_Divergence_Skipped"] = False
    annotations["Sibling_Projection_Dimension"] = 2

    rows = extract_sibling_lrt_rows(
        tree=tree,
        annotations=annotations,
        run_id="k15_t3_c30",
        k_neighbors=15,
        diffusion_time=3,
        n_components=30,
        edge_alpha=0.001,
        sibling_alpha=0.01,
    )

    assert len(rows) == 3
    assert set(rows["support_label_from_edge_path"]) == {"strict_null_like"}
    root_row = rows.sort_values("n_parent", ascending=False).iloc[0]
    assert root_row["bernoulli_deviance_lrt"] > 0.0
    assert root_row["sibling_selected_ratio"] == 2.0


def test_parse_grid() -> None:
    assert parse_grid("15:3,15:5,30:3") == ((15, 3), (15, 5), (30, 3))
