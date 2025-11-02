import numpy as np
import pandas as pd
import pytest

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)


def _build_small_case(n_samples=18, n_features=24, n_clusters=3, std=1.0, seed=123):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=std,
        random_state=seed,
    )
    Xb = (X > np.median(X, axis=0)).astype(int)
    df = pd.DataFrame(Xb, index=[f"S{i}" for i in range(n_samples)])
    Z = linkage(pdist(df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
    stats = calculate_hierarchy_kl_divergence(tree, df)
    return tree, stats, df


def test_annotate_nodes_with_statistical_significance_tests_basic():
    tree, stats, _ = _build_small_case()
    df = annotate_nodes_with_statistical_significance_tests(
        stats, total_number_of_features=stats.iloc[0]["distribution"].size, include_deviation_test=True
    )
    # Columns present
    assert "Are_Features_Dependent" in df.columns
    assert "BH_P_Value_Uncorrected" in df.columns
    # Booleans normalized
    assert df["Are_Features_Dependent"].dtype == bool


def test_sibling_cmi_with_li_threshold_interface():
    """
    Exercise the 'li' threshold option for sibling CMI.
    This test asserts the function accepts 'li' and produces expected columns.
    Implementation for 'li' may be pending; if so, this test will fail, guiding TDD.
    """
    tree, stats, _ = _build_small_case()
    res = annotate_child_parent_divergence(
        tree, annotate_nodes_with_statistical_significance_tests(
            stats, total_number_of_features=stats.iloc[0]["distribution"].size
        ), total_number_of_features=stats.iloc[0]["distribution"].size
    )

    # Expect no error when using 'li' and required columns in result
    out = annotate_sibling_independence_cmi(
        tree,
        res,
        binarization_threshold="li",
        permutations=20,
        random_state=0,
        parallel=False,
    )
    assert "Sibling_CMI" in out.columns
    assert "Sibling_CMI_P_Value" in out.columns

