"""Shared helpers for the enhancement laboratory.

Every experiment imports from here to avoid duplicating tree-building,
decomposition, and metric-collection code.
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

# Ensure the project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from benchmarks.shared.generators import generate_case_data  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (  # noqa: E402
    sibling_config,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral import (  # noqa: E402
    marchenko_pastur as mp_module,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree  # noqa: E402

_MISSING = object()


@contextmanager
def temporary_attr(obj: Any, attr: str, value: Any):
    """Temporarily set ``obj.attr`` and restore the previous value on exit."""
    had_attr = hasattr(obj, attr)
    old_value = getattr(obj, attr) if had_attr else _MISSING
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if old_value is _MISSING:
            delattr(obj, attr)
        else:
            setattr(obj, attr, old_value)


@contextmanager
def temporary_attrs(*patches: tuple[Any, str, Any]):
    """Apply multiple temporary attribute overrides at once."""
    with ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(temporary_attr(obj, attr, value))
        yield


@contextmanager
def temporary_config(**overrides: Any):
    """Temporarily override ``kl_clustering_analysis.config`` values."""
    patches = tuple((config, name, value) for name, value in overrides.items())
    with temporary_attrs(*patches):
        yield


@contextmanager
def temporary_dict_values(mapping: dict[str, Any], **updates: Any):
    """Temporarily set dictionary entries and restore prior values on exit."""
    previous = {key: mapping.get(key, _MISSING) for key in updates}
    mapping.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is _MISSING:
                mapping.pop(key, None)
            else:
                mapping[key] = old_value


@contextmanager
def temporary_experiment_overrides(
    *,
    leaf_data_cache: dict[str, Any] | None = None,
    leaf_data: pd.DataFrame | None = None,
    sibling_dims: Any = _MISSING,
    sibling_pca: Any = _MISSING,
    gate2_estimator: Any = _MISSING,
    config_overrides: dict[str, Any] | None = None,
):
    """Temporarily apply the common enhancement-lab runtime overrides."""
    with ExitStack() as stack:
        if leaf_data_cache is not None and leaf_data is not None:
            stack.enter_context(temporary_dict_values(leaf_data_cache, leaf_data=leaf_data))
        if sibling_dims is not _MISSING:
            stack.enter_context(
                temporary_attr(sibling_config, "derive_sibling_spectral_dims", sibling_dims)
            )
        if sibling_pca is not _MISSING:
            stack.enter_context(
                temporary_attr(sibling_config, "derive_sibling_pca_projections", sibling_pca)
            )
        if gate2_estimator is not _MISSING:
            stack.enter_context(
                temporary_attr(
                    mp_module,
                    "estimate_k_marchenko_pastur",
                    gate2_estimator,
                )
            )
        if config_overrides:
            stack.enter_context(temporary_config(**config_overrides))
        yield


# --- Auto-generated from full_benchmark_comparison.csv ---
# Cases with ARI < 0.3 — active failures to investigate
FAILURE_CASES = [
    "sbm_hard",  # K=2/3, ARI=-0.010
    "binary_balanced_low_noise__2",  # K=1/4, ARI=0.000
    "binary_noise_feat_50i_200n",  # K=1/4, ARI=0.000
    "binary_noise_feat_80i_400n",  # K=1/6, ARI=0.000
    "cat_clear_3cat_4c",  # K=1/4, ARI=0.000
    "cat_clear_4cat_4c",  # K=1/4, ARI=0.000
    "cat_highcard_20cat_4c",  # K=1/4, ARI=0.000
    "gauss_extreme_noise_highd",  # K=1/4, ARI=0.000
    "overlap_hd_4c_1k",  # K=1/4, ARI=0.000
    "overlap_heavy_8c_large_feat",  # K=1/8, ARI=0.000
    "phylo_protein_12taxa",  # K=1/12, ARI=0.000
    "phylo_protein_4taxa",  # K=1/4, ARI=0.000
    "sbm_moderate",  # K=1/3, ARI=0.000
    "overlap_extreme_4c",  # K=7/4, ARI=0.001
    "overlap_heavy_4c_small_feat",  # K=14/4, ARI=0.051
    "phylo_conserved_8taxa",  # K=2/8, ARI=0.074
    "phylo_protein_8taxa",  # K=2/8, ARI=0.216
    "overlap_heavy_4c_med_feat",  # K=14/4, ARI=0.255
]

# Cases that must NOT regress (ARI=1.0, exact K)
REGRESSION_GUARD_CASES = [
    "binary_2clusters",  # K=2/2, ARI=1.000
    "binary_hard_8c",  # K=8/8, ARI=1.000
    "binary_low_noise_2c",  # K=2/2, ARI=1.000
    "binary_low_noise_4c",  # K=4/4, ARI=1.000
    "binary_low_noise_8c",  # K=8/8, ARI=1.000
    "binary_noise_feat_30i_500n",  # K=3/3, ARI=1.000
    "binary_null_medium",  # K=1/1, ARI=1.000
    "binary_null_small",  # K=1/1, ARI=1.000
    "binary_perfect_2c",  # K=2/2, ARI=1.000
    "binary_perfect_4c",  # K=4/4, ARI=1.000
    "binary_unbalanced_low",  # K=4/4, ARI=1.000
    "gauss_clear_large",  # K=5/5, ARI=1.000
    "gauss_clear_medium",  # K=4/4, ARI=1.000
    "gauss_moderate_3c",  # K=3/3, ARI=1.000
    "gauss_moderate_5c",  # K=5/5, ARI=1.000
    "gauss_noisy_3c",  # K=3/3, ARI=1.000
    "gauss_null_large",  # K=1/1, ARI=1.000
    "gauss_null_small",  # K=1/1, ARI=1.000
    "gauss_overlap_3c_small_q4",  # K=3/3, ARI=1.000
    "gauss_overlap_4c_med",  # K=4/4, ARI=1.000
    "gauss_overlap_6c_large",  # K=6/6, ARI=1.000
    "gauss_overlap_8c_highd",  # K=8/8, ARI=1.000
    "overlap_part_8c_large",  # K=8/8, ARI=1.000
    "phylo_divergent_8taxa",  # K=8/8, ARI=1.000
    "sparse_features_100x500",  # K=4/4, ARI=1.000
]

# Cases with 0.3 <= ARI < 0.8 — partial successes
INTERMEDIATE_CASES = [
    "cat_highcard_10cat_4c",  # K=2/4, ARI=0.327
    "phylo_dna_4taxa_low_mut",  # K=2/4, ARI=0.327
    "phylo_dna_16taxa_low_mut",  # K=6/16, ARI=0.451
    "phylo_large_32taxa",  # K=15/32, ARI=0.485
    "phylo_divergent_4taxa",  # K=2/4, ARI=0.492
    "overlap_unbal_4c_small",  # K=4/4, ARI=0.500
    "gauss_clear_small",  # K=2/3, ARI=0.554
    "gauss_extreme_noise_3c",  # K=2/3, ARI=0.554
    "cat_clear_5cat_6c",  # K=4/6, ARI=0.563
    "binary_multiscale_6c",  # K=4/6, ARI=0.564
    "gauss_extreme_noise_many",  # K=19/30, ARI=0.582
    "phylo_large_64taxa",  # K=37/64, ARI=0.588
    "phylo_dna_8taxa_low_mut",  # K=5/8, ARI=0.589
    "phylo_dna_8taxa_med_mut",  # K=5/8, ARI=0.589
    "binary_low_noise_12c",  # K=9/12, ARI=0.614
    "overlap_mod_4c_small",  # K=14/4, ARI=0.672
    "cat_highd_4cat_1000feat",  # K=4/6, ARI=0.681
    "cat_overlap_3cat_4c",  # K=3/4, ARI=0.701
    "binary_balanced_low_noise",  # K=3/4, ARI=0.705
    "sparse_features_72x72",  # K=3/4, ARI=0.705
    "binary_hard_4c",  # K=3/4, ARI=0.708
    "binary_many_features",  # K=3/4, ARI=0.708
    "phylo_conserved_4taxa",  # K=3/4, ARI=0.708
    "binary_multiscale_4c",  # K=3/4, ARI=0.709
    "cat_mod_3cat_4c",  # K=3/4, ARI=0.709
    "cat_highd_3cat_500feat",  # K=3/4, ARI=0.711
    "overlap_unbal_8c_large",  # K=6/8, ARI=0.714
    "gauss_noisy_many",  # K=6/8, ARI=0.754
    "binary_perfect_8c",  # K=6/8, ARI=0.757
    "cat_unbal_3cat_4c",  # K=3/4, ARI=0.765
]
# --- End auto-generated ---


def get_case(name: str) -> dict:
    """Look up a benchmark case by name."""
    all_cases = get_default_test_cases()
    tc = next((c for c in all_cases if c["name"] == name), None)
    if tc is None:
        available = [c["name"] for c in all_cases]
        raise KeyError(f"Case {name!r} not found. Available: {available[:20]}...")
    # Derive n_clusters from SBM 'sizes' field if missing
    if "n_clusters" not in tc and "sizes" in tc:
        tc["n_clusters"] = len(tc["sizes"])
    return tc


def build_tree_and_data(
    case_name: str,
) -> tuple[PosetTree, pd.DataFrame, np.ndarray | None, dict]:
    """Generate data, build linkage tree, return (tree, data_df, true_labels, case_dict).

    Node distributions are populated so the tree is ready for
    ``TreeDecomposition`` or ``tree.decompose()``.
    """
    tc = get_case(case_name)
    data_t, y_t, _, _ = generate_case_data(tc)
    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())
    tree.populate_node_divergences(data_t)
    return tree, data_t, y_t, tc


def run_decomposition(
    tree: PosetTree,
    data_df: pd.DataFrame,
    *,
    alpha_local: float | None = None,
    sibling_alpha: float | None = None,
) -> dict[str, Any]:
    """Run decomposition with optionally overridden alphas.

    NOTE: The benchmark runner (kl_runner.py) passes config.SIBLING_ALPHA
    for BOTH alpha_local and sibling_alpha. We match that behavior here
    so lab results are comparable to benchmark results.
    """
    return tree.decompose(
        leaf_data=data_df,
        alpha_local=alpha_local or config.SIBLING_ALPHA,
        sibling_alpha=sibling_alpha or config.SIBLING_ALPHA,
    )


def compute_ari(
    decomp: dict,
    data_df: pd.DataFrame,
    y_true: np.ndarray,
) -> float:
    """Compute ARI from decomposition result and true labels."""
    n = len(data_df)
    y_pred = np.full(n, -1, dtype=int)
    for cid, cinfo in decomp["cluster_assignments"].items():
        for leaf in cinfo["leaves"]:
            idx = data_df.index.get_loc(leaf)
            y_pred[idx] = cid
    return adjusted_rand_score(y_true, y_pred)


def quick_eval(
    case_name: str,
    *,
    alpha_local: float | None = None,
    sibling_alpha: float | None = None,
) -> dict[str, Any]:
    """Build tree, decompose, compute metrics — all in one call.

    Returns dict with: case, true_k, found_k, ari, tree, annotations_df, decomp.
    """
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df, alpha_local=alpha_local, sibling_alpha=sibling_alpha)
    true_k = tc.get("n_clusters", None)
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")
    return {
        "case": case_name,
        "true_k": true_k,
        "found_k": found_k,
        "ari": ari,
        "tree": tree,
        "annotations_df": tree.annotations_df,
        "decomp": decomp,
        "data_df": data_df,
        "y_true": y_t,
    }


def run_case_battery(
    cases: list[str],
    *,
    label: str = "",
    alpha_local: float | None = None,
    sibling_alpha: float | None = None,
) -> pd.DataFrame:
    """Run a list of cases and return a summary DataFrame.

    Columns: case, true_k, found_k, ari, delta_k (found - true).
    """
    rows = []
    for name in cases:
        try:
            result = quick_eval(name, alpha_local=alpha_local, sibling_alpha=sibling_alpha)
            rows.append(
                {
                    "case": name,
                    "true_k": result["true_k"],
                    "found_k": result["found_k"],
                    "ari": round(result["ari"], 3),
                    "delta_k": result["found_k"] - (result["true_k"] or 0),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "case": name,
                    "true_k": None,
                    "found_k": None,
                    "ari": None,
                    "delta_k": None,
                    "error": str(e),
                }
            )
    df = pd.DataFrame(rows)
    if label:
        df.insert(0, "experiment", label)
    return df


def collect_node_stats(tree: PosetTree, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-node diagnostic table for internal nodes.

    Columns: node, depth, leaf_count, gate2_pass, gate3_result,
             sib_stat, sib_df, sib_p_corr, decision.
    """
    import networkx as nx

    root = next(n for n, d in tree.in_degree() if d == 0)
    depths = nx.shortest_path_length(tree, root)
    rows = []
    for nd in tree.nodes():
        children = list(tree.successors(nd))
        if len(children) != 2:
            continue
        l, r = children
        l_sig = bool(annotations_df.loc[l, "Child_Parent_Divergence_Significant"])
        r_sig = bool(annotations_df.loc[r, "Child_Parent_Divergence_Significant"])
        g2 = l_sig or r_sig
        skipped = bool(annotations_df.loc[nd, "Sibling_Divergence_Skipped"])
        different = bool(annotations_df.loc[nd, "Sibling_BH_Different"])
        g3 = different and not skipped

        leaf_count = len(tree.compute_descendant_sets(use_labels=True).get(nd, set()))

        sib_stat = (
            annotations_df.loc[nd, "Sibling_Test_Statistic"]
            if "Sibling_Test_Statistic" in annotations_df.columns
            else np.nan
        )
        sib_df = (
            annotations_df.loc[nd, "Sibling_Degrees_of_Freedom"]
            if "Sibling_Degrees_of_Freedom" in annotations_df.columns
            else np.nan
        )
        sib_p = (
            annotations_df.loc[nd, "Sibling_Divergence_P_Value_Corrected"]
            if "Sibling_Divergence_P_Value_Corrected" in annotations_df.columns
            else np.nan
        )

        if g2 and g3:
            decision = "SPLIT"
        elif not g2:
            decision = "MERGE(G2)"
        elif skipped:
            decision = "MERGE(SKIP)"
        else:
            decision = "MERGE(G3)"

        rows.append(
            {
                "node": nd,
                "depth": depths.get(nd, -1),
                "leaf_count": leaf_count,
                "gate2_pass": g2,
                "gate3_result": "DIFF" if different else ("SKIP" if skipped else "SAME"),
                "sib_stat": sib_stat,
                "sib_df": sib_df,
                "sib_p_corr": sib_p,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values("depth")


def print_summary(df: pd.DataFrame) -> None:
    """Pretty-print a summary DataFrame."""
    print(df.to_string(index=False))
    if "ari" in df.columns:
        valid = df["ari"].dropna()
        if len(valid):
            print(f"\nMean ARI: {valid.mean():.3f}  |  Median ARI: {valid.median():.3f}")
    if "delta_k" in df.columns:
        exact = (df["delta_k"] == 0).sum()
        print(f"Exact K: {exact}/{len(df)}")
