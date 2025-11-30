import numpy as np
import pandas as pd

from kl_clustering_analysis.threshold import compute_otsu_threshold
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_sibling_independence_cmi,
)
from kl_clustering_analysis.information_metrics import (
    compute_node_divergences,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
import scipy.cluster.hierarchy as sch


def test_compute_otsu_threshold_bimodal_and_constant():
    # Bimodal distribution around 0.2 and 0.8
    rng = np.random.default_rng(42)
    a = rng.normal(0.2, 0.02, size=500)
    b = rng.normal(0.8, 0.02, size=500)
    v = np.clip(np.concatenate([a, b]), 0.0, 1.0)
    thr = compute_otsu_threshold(v)
    # Expect a threshold between the modes (allow some slack for histogram binning)
    assert 0.15 < thr < 0.85

    # Constant values → fallback to 0.5
    const = np.full(100, 0.4)
    thr_const = compute_otsu_threshold(const)
    assert thr_const == 0.5


def _make_small_tree(n_samples=12, n_features=20, n_clusters=3, seed=123):
    rng = np.random.default_rng(seed)
    # Create cluster centers and binary data around them
    centers = rng.random((n_clusters, n_features))
    X = np.vstack(
        [
            (centers[i] + 0.1 * rng.normal(size=(n_samples // n_clusters, n_features)))
            for i in range(n_clusters)
        ]
    )
    X = (X > 0.5).astype(int)
    labels = [f"S{i}" for i in range(X.shape[0])]
    df = pd.DataFrame(X, index=labels)
    Z = sch.linkage(X, method="complete", metric="hamming")
    tree = PosetTree.from_linkage(Z, leaf_names=labels)
    stats = compute_node_divergences(tree, df)
    return tree, stats


def test_sibling_cmi_with_otsu_vs_fixed_threshold():
    tree, stats = _make_small_tree()

    # With Otsu (triad-based per parent)
    res_otsu = annotate_sibling_independence_cmi(
        tree,
        stats,
        binarization_threshold="otsu",
        n_permutations=20,
        random_state=0,
        parallel=False,
    )
    # With fixed 0.5
    res_fixed = annotate_sibling_independence_cmi(
        tree,
        stats,
        binarization_threshold=0.5,
        n_permutations=20,
        random_state=0,
        parallel=False,
    )

    # Ensure expected columns exist
    for df in (res_otsu, res_fixed):
        assert "Sibling_CMI" in df.columns
        assert "Sibling_CMI_P_Value" in df.columns

    # Compare CMI values; at least one parent should differ if Otsu != 0.5
    merged = (
        res_otsu[["Sibling_CMI"]]
        .rename(columns={"Sibling_CMI": "CMI_Otsu"})
        .join(
            res_fixed[["Sibling_CMI"]].rename(columns={"Sibling_CMI": "CMI_Fixed"}),
            how="inner",
        )
    )
    # Allow for possible equality in some parents, but expect some difference overall
    diffs = np.abs((merged["CMI_Otsu"] - merged["CMI_Fixed"]).to_numpy())
    assert np.any(diffs > 1e-12)
