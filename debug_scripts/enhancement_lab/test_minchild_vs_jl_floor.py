#!/usr/bin/env python3
"""Test: min-child vs jl_floor_qrt for Gate 3 projection dimension.

Reproduces exp15-19 findings with current codebase.

Compares:
  1. min_child: k = min(k_L, k_R) — current production
  2. jl_floor_qrt: k = max(min(k_L, k_R), JL/4) — exp15-19 recommendation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.edge_gate import (
    annotate_edge_gate,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.sibling_gate import (
    annotate_sibling_gate,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Sentinel cases from exp15-19 (15 cases that discriminate projection strategies)
SENTINEL_CASES = [
    "binary_balanced_low_noise",
    "binary_balanced_low_noise__2",
    "binary_perfect_8c",
    "gauss_extreme_noise_3c",
    "gauss_extreme_noise_highd",
    "gauss_clear_small",
    "gauss_overlap_4c_med",
    "overlap_heavy_4c_small_feat",
    "binary_hard_4c",
    "binary_low_noise_12c",
    "binary_noise_feat_50i_200n",
    "cat_highcard_20cat_4c",
    "sbm_moderate",
    "phylo_divergent_8taxa",
    "sparse_features_72x72",
]


def compute_jl_floor_qrt(n_samples: int, n_features: int) -> int:
    """JL/4 floor from exp15-19."""
    from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
        compute_projection_dimension_backend,
    )

    jl_k = compute_projection_dimension_backend(n_samples, n_features, eps=0.3)
    return max(2, jl_k // 4)


def derive_sibling_spectral_dims_min_child(
    tree,
    annotated_df,
) -> dict[str, int] | None:
    """Current production: min(k_L, k_R)."""
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None

    sibling_dims: dict[str, int] = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        child_ks = [
            k
            for k in (
                edge_spectral_dims.get(left, 0),
                edge_spectral_dims.get(right, 0),
            )
            if k > 0
        ]
        if child_ks:
            sibling_dims[parent] = min(child_ks)

    return sibling_dims if sibling_dims else None


def derive_sibling_spectral_dims_jl_floor_qrt(
    tree,
    annotated_df,
    leaf_data,
) -> dict[str, int] | None:
    """exp15-19 recommendation: max(min(k_L, k_R), JL/4)."""
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None

    sibling_dims: dict[str, int] = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        # Get child spectral k
        child_ks = [
            k
            for k in (
                edge_spectral_dims.get(left, 0),
                edge_spectral_dims.get(right, 0),
            )
            if k > 0
        ]

        if not child_ks:
            # Fallback to JL/4 when no child spectral k
            n_parent = tree.nodes[parent].get("leaf_count", 0)
            jl_floor = compute_jl_floor_qrt(n_parent, leaf_data.shape[1])
            sibling_dims[parent] = jl_floor
        else:
            min_child_k = min(child_ks)
            # JL/4 floor
            n_parent = tree.nodes[parent].get("leaf_count", 0)
            jl_floor = compute_jl_floor_qrt(n_parent, leaf_data.shape[1])
            sibling_dims[parent] = max(min_child_k, jl_floor)

    return sibling_dims if sibling_dims else None


def run_decomposition_custom(
    tree: PosetTree,
    leaf_data,
    spectral_dims_fn,
    alpha_local: float = 0.01,
    sibling_alpha: float = 0.01,
) -> dict:
    """Run decomposition with custom spectral_dims derivation."""
    from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
        derive_sibling_pca_projections,
    )

    # Gate 2
    annotations_df = tree.annotations_df.copy()
    edge_bundle = annotate_edge_gate(
        tree,
        annotations_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method="marchenko_pastur",
        fdr_method="tree_bh",
    )

    # Gate 3 with custom spectral_dims
    sibling_dims = spectral_dims_fn(tree, edge_bundle.annotated_df, leaf_data)
    pca_projs, pca_eigs = derive_sibling_pca_projections(edge_bundle.annotated_df, sibling_dims)

    sibling_bundle = annotate_sibling_gate(
        tree,
        edge_bundle.annotated_df,
        significance_level_alpha=sibling_alpha,
        sibling_method="cousin_adjusted_wald",
        spectral_dims=sibling_dims,
        pca_projections=pca_projs,
        pca_eigenvalues=pca_eigs,
    )

    # Decomposition
    evaluator = GateEvaluator(tree)
    cluster_roots = evaluator.decompose_with_gates(
        sibling_bundle.annotated_df,
        passthrough=True,
    )

    # Build cluster assignments
    cluster_assignments = {}
    for i, root in enumerate(sorted(cluster_roots)):
        leaves = tree.get_leaves_under(root)
        cluster_assignments[i] = {"root_node": root, "leaves": leaves, "size": len(leaves)}

    return {
        "num_clusters": len(cluster_assignments),
        "cluster_assignments": cluster_assignments,
    }


def compute_ari(results, leaf_data, y_true) -> float:
    """Compute Adjusted Rand Index."""
    n = len(leaf_data)
    y_pred = np.full(n, -1, dtype=int)
    for cid, cinfo in results["cluster_assignments"].items():
        for leaf in cinfo["leaves"]:
            idx = leaf_data.index.get_loc(leaf)
            y_pred[idx] = cid
    return adjusted_rand_score(y_true, y_pred)


def run_case(case_name: str) -> dict:
    """Run both strategies on a single case."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", "?")

    results = {}

    # Strategy 1: min_child (current production)
    t0 = time.time()
    try:
        result_min = run_decomposition_custom(tree, data_df, derive_sibling_spectral_dims_min_child)
        ari_min = compute_ari(result_min, data_df, y_true) if y_true is not None else float("nan")
        results["min_child"] = {
            "found_k": result_min["num_clusters"],
            "ari": ari_min,
            "time": time.time() - t0,
        }
    except Exception as e:
        results["min_child"] = {
            "found_k": "ERR",
            "ari": 0.0,
            "time": time.time() - t0,
            "error": str(e),
        }

    # Strategy 2: jl_floor_qrt (exp15-19 recommendation)
    t0 = time.time()
    try:
        result_jl = run_decomposition_custom(
            tree, data_df, derive_sibling_spectral_dims_jl_floor_qrt
        )
        ari_jl = compute_ari(result_jl, data_df, y_true) if y_true is not None else float("nan")
        results["jl_floor_qrt"] = {
            "found_k": result_jl["num_clusters"],
            "ari": ari_jl,
            "time": time.time() - t0,
        }
    except Exception as e:
        results["jl_floor_qrt"] = {
            "found_k": "ERR",
            "ari": 0.0,
            "time": time.time() - t0,
            "error": str(e),
        }

    results["true_k"] = true_k
    results["case"] = case_name

    return results


def build_tree_and_data(case_name: str):
    """Build tree and data for a case."""
    from lab_helpers import build_tree_and_data as _build

    return _build(case_name)


def main():
    print("=" * 90)
    print("Gate 3 Projection Dimension: min_child vs jl_floor_qrt")
    print("=" * 90)
    print(f"\nSentinel cases: {len(SENTINEL_CASES)}")
    print(
        f"Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}, SPECTRAL_METHOD={config.SPECTRAL_METHOD}"
    )
    print()

    all_results = []
    t_start = time.time()

    for i, case_name in enumerate(SENTINEL_CASES, 1):
        print(f"[{i:2d}/{len(SENTINEL_CASES)}] {case_name:<40}", end="", flush=True)
        try:
            results = run_case(case_name)
            all_results.append(results)

            ari_min = results["min_child"]["ari"]
            ari_jl = results["jl_floor_qrt"]["ari"]
            delta = ari_jl - ari_min
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
            print(f"  min={ari_min:.3f}  jl_floor={ari_jl:.3f}  [{marker}{delta:+.3f}]")
        except Exception as e:
            print(f"  ERR: {e}")
            all_results.append(
                {
                    "case": case_name,
                    "true_k": "?",
                    "min_child": {"ari": 0.0, "found_k": "ERR"},
                    "jl_floor_qrt": {"ari": 0.0, "found_k": "ERR"},
                }
            )

    total_time = time.time() - t_start

    # Aggregate
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)

    aris_min = [
        r["min_child"]["ari"] for r in all_results if isinstance(r["min_child"]["ari"], float)
    ]
    aris_jl = [
        r["jl_floor_qrt"]["ari"] for r in all_results if isinstance(r["jl_floor_qrt"]["ari"], float)
    ]

    print(f"\n{'Strategy':<20} │ {'Mean ARI':>9} │ {'Med ARI':>9} │ {'Exact K':>9} │ {'K=1':>5}")
    print("─" * 60)

    # min_child
    exact_min = sum(
        1 for r in all_results if r["min_child"]["found_k"] == r["true_k"] and r["true_k"] != "?"
    )
    k1_min = sum(1 for r in all_results if r["min_child"]["found_k"] == 1)
    print(
        f"{'min_child (prod)':<20} │ {np.mean(aris_min):9.3f} │ {np.median(aris_min):9.3f} │ {exact_min:5d}/{len(all_results):<3d} │ {k1_min:5d}"
    )

    # jl_floor_qrt
    exact_jl = sum(
        1 for r in all_results if r["jl_floor_qrt"]["found_k"] == r["true_k"] and r["true_k"] != "?"
    )
    k1_jl = sum(1 for r in all_results if r["jl_floor_qrt"]["found_k"] == 1)
    print(
        f"{'jl_floor_qrt (exp)':<20} │ {np.mean(aris_jl):9.3f} │ {np.median(aris_jl):9.3f} │ {exact_jl:5d}/{len(all_results):<3d} │ {k1_jl:5d}"
    )

    # Head-to-head
    wins_jl = sum(1 for r in all_results if r["jl_floor_qrt"]["ari"] > r["min_child"]["ari"] + 0.01)
    wins_min = sum(
        1 for r in all_results if r["min_child"]["ari"] > r["jl_floor_qrt"]["ari"] + 0.01
    )
    ties = len(all_results) - wins_jl - wins_min

    print(f"\nHead-to-head: jl_floor_qrt wins={wins_jl}, min_child wins={wins_min}, ties={ties}")
    print(
        f"Mean delta ARI (jl - min): {np.mean([r['jl_floor_qrt']['ari'] - r['min_child']['ari'] for r in all_results]):+.4f}"
    )

    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
