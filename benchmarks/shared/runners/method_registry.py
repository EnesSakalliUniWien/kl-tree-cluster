"""Method registry config for benchmarking.

Export a direct `METHOD_SPECS` mapping so callers can import it as a
configuration constant.
"""

from __future__ import annotations

import importlib

from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
)
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY,
)
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
)

from benchmarks.shared.benchmark_grid import benchmark_grid
from benchmarks.shared.types import MethodSpec


def _import_runner(module: str, attr: str):
    mod = importlib.import_module(module)
    return mod.__dict__[attr]


ADAPTIVE_PYDIFFMAP_DIFFUSION_PARAMS = {
    "diffusion_method": "adaptive_pydiffmap_diffusion",
    "k_neighbors": 10,
    "diffusion_time": 3,
    "n_components": 30,
    "metric": "hamming",
    "bandwidth_type": "-1/(d+2)",
    "epsilon": "median",
    "tree_linkage_method": "average",
}

GRAPHTOOLS_KERNEL_DIFFUSION_PARAMS = {
    "diffusion_method": "graphtools_kernel_diffusion",
    "k_neighbors": 10,
    "diffusion_time": 3,
    "n_components": 30,
    "metric": "hamming",
    "decay": 40,
    "anisotropy": 0.0,
    "kernel_symm": "+",
    "random_state": 0,
    "tree_builder": "linkage",
    "tree_rooting": "linkage_root",
    "tree_linkage_method": "average",
}

GRAPHTOOLS_ADAPTIVE_K_DIFFUSION_PARAMS = {
    **GRAPHTOOLS_KERNEL_DIFFUSION_PARAMS,
    "adaptive_neighbor_profile": "fragmentation_guard",
    "adaptive_neighbor_grid": (5, 10, 15, 25, 40, 80, 160),
}

FIXED_TOPOLOGY_NNLS_BRANCH_TIME_PARAMS = {
    "edge_branch_length_variance_policy": EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
    "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    "branch_length_optimization_target_metric": (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    "branch_length_optimization_pair_sample_size": 50_000,
    "branch_length_optimization_random_state": 0,
    "branch_length_optimization_solver_tolerance": 1e-5,
    "branch_length_optimization_max_iterations": 1000,
}

CANONICAL_BENCHMARK_CLASS = "canonical"
OPTIONAL_GPL_BENCHMARK_CLASS = "optional_gpl"
DEFAULT_METHOD_GRID = "default_methods"
GRAPHTOOLS_TREE_STRATEGY_GRID = "graphtools_adaptive_k_tree_strategy"

GRAPHTOOLS_ADAPTIVE_K_TREE_STRATEGY_PARAMS = benchmark_grid(
    benchmark_class=OPTIONAL_GPL_BENCHMARK_CLASS,
    grid_name=GRAPHTOOLS_TREE_STRATEGY_GRID,
    base_params={
        **GRAPHTOOLS_ADAPTIVE_K_DIFFUSION_PARAMS,
        **FIXED_TOPOLOGY_NNLS_BRANCH_TIME_PARAMS,
    },
    axes={
        "tree_linkage_method": (
            "average",
            "complete",
            "weighted",
            "single",
            "centroid",
            "median",
            "ward",
        ),
    },
) + benchmark_grid(
    benchmark_class=OPTIONAL_GPL_BENCHMARK_CLASS,
    grid_name=GRAPHTOOLS_TREE_STRATEGY_GRID,
    base_params={
        **GRAPHTOOLS_ADAPTIVE_K_DIFFUSION_PARAMS,
        "tree_rooting": "mad",
        **FIXED_TOPOLOGY_NNLS_BRANCH_TIME_PARAMS,
    },
    axes={"tree_builder": ("neighbor_joining",)},
)


# Note: import names are updated to point to the new benchmarks.shared.runners package.
METHOD_SPECS: dict[str, MethodSpec] = {
    "tbs": MethodSpec(
        name="TBS Divergence",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            # Default: Hamming + Average (UPGMA)
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
            },
        ],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "tbs_continuous_guarded_within_covariance": MethodSpec(
        name="TBS Continuous Guarded Within-Child Covariance",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "continuous_covariance_policy": (GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY),
                "continuous_covariance_min_child_leaf_count": (
                    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
                ),
                "continuous_sibling_gate_method": "fixed_coordinate_bh",
            },
        ],
    ),
    "tbs_complete": MethodSpec(
        name="TBS (Complete)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "complete",
            },
        ],
    ),
    "tbs_single": MethodSpec(
        name="TBS (Single)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "single",
            },
        ],
    ),
    "tbs_fixed_coordinate_bh": MethodSpec(
        name="TBS Fixed Coordinate BH",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_coordinate_bh",
            },
        ],
    ),
    "tbs_fixed_coordinate_by": MethodSpec(
        name="TBS Fixed Coordinate BY",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_coordinate_by",
            },
        ],
    ),
    "tbs_fixed_coordinate_holm": MethodSpec(
        name="TBS Fixed Coordinate Holm",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_coordinate_holm",
            },
        ],
    ),
    "tbs_fixed_coordinate_bonferroni": MethodSpec(
        name="TBS Fixed Coordinate Bonferroni",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_coordinate_bonferroni",
            },
        ],
    ),
    "tbs_fixed_block_bh": MethodSpec(
        name="TBS Fixed Block Chi-square BH",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_block_bh",
            },
        ],
    ),
    "tbs_fixed_block_simes_bh": MethodSpec(
        name="TBS Fixed Block Simes-BH",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_method": "fixed_block_simes_bh",
            },
        ],
    ),
    "tbs_conditional_topology_diagnostic": MethodSpec(
        name="TBS (Conditional Topology Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": ("fixed_coordinate_conditional_topology_diagnostic_v1"),
            },
        ],
    ),
    "tbs_global_passthrough_refined_diagnostic": MethodSpec(
        name="TBS (Global Passthrough Refined Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": ("fixed_coordinate_global_passthrough_refined_v1"),
            },
        ],
    ),
    "tbs_spectral_transport_passthrough_diagnostic": MethodSpec(
        name="TBS (Spectral Transport Passthrough Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": (
                    "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1"
                ),
            },
        ],
    ),
    "tbs_spectral_transport_passthrough": MethodSpec(
        name="TBS (Spectral Transport Passthrough)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": ("fixed_coordinate_spectral_transport_passthrough_v1"),
            },
        ],
    ),
    "tbs_internal_filter_v1": MethodSpec(
        name="TBS Internal Spectral Filter",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_internal_distribution_mode": "empirical_barycenter",
                "enforce_internal_support_thresholds": True,
            },
        ],
    ),
    "tbs_internal_filter_branch_length_v1": MethodSpec(
        name="TBS Branch-Length Internal Spectral Filter",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_internal_distribution_mode": "branch_length_state",
                "enforce_internal_support_thresholds": True,
            },
        ],
    ),
    "tbs_bandwidth_context_v1": MethodSpec(
        name="TBS Regional Bandwidth Context",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "neighborhood_bandwidth_profile": ("regional_tau_branch_length_support_only_v1"),
            },
        ],
    ),
    "tbs_neighbor_joining": MethodSpec(
        name="TBS (Neighbor Joining, MAD Root)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "tree_builder": "neighbor_joining",
                "tree_rooting": "mad",
            },
        ],
    ),
    "tbs_iqtree3": MethodSpec(
        name="TBS (IQ-TREE 3, MAD Root)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "run_tbs_on_distance"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "tree_builder": "iqtree3",
                "tree_rooting": "mad",
                "iqtree_executable": "iqtree3",
                "iqtree_model": "JC2",
                "iqtree_threads": 1,
            },
        ],
    ),
    "tbs_diffusion": MethodSpec(
        name="TBS (Hamming NN Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_method",
        ),
        param_grid=[
            {
                "diffusion_method": "hamming_nn_diffusion",
                "k_neighbors": 15,
                "diffusion_time": 3,
                "tree_linkage_method": "average",
                "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
            }
        ],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "tbs_diffusion_adaptive": MethodSpec(
        name="TBS (Adaptive pydiffmap Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_adaptive_method",
        ),
        param_grid=[
            {
                **ADAPTIVE_PYDIFFMAP_DIFFUSION_PARAMS,
                "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
            }
        ],
    ),
    "tbs_diffusion_adaptive_nnls": MethodSpec(
        name="TBS (Adaptive pydiffmap Diffusion, NNLS Branch-Time)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_adaptive_method",
        ),
        param_grid=[
            {
                **ADAPTIVE_PYDIFFMAP_DIFFUSION_PARAMS,
                **FIXED_TOPOLOGY_NNLS_BRANCH_TIME_PARAMS,
            }
        ],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "tbs_diffusion_graphtools": MethodSpec(
        name="TBS (graphtools Kernel Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_graphtools_method",
        ),
        param_grid=[
            {
                **GRAPHTOOLS_KERNEL_DIFFUSION_PARAMS,
                "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
            }
        ],
        benchmark_class=OPTIONAL_GPL_BENCHMARK_CLASS,
        benchmark_grid="graphtools_kernel_diffusion",
    ),
    "tbs_diffusion_graphtools_nnls": MethodSpec(
        name="TBS (graphtools Kernel Diffusion, NNLS Branch-Time)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_graphtools_method",
        ),
        param_grid=[
            {
                **GRAPHTOOLS_KERNEL_DIFFUSION_PARAMS,
                **FIXED_TOPOLOGY_NNLS_BRANCH_TIME_PARAMS,
            }
        ],
        benchmark_class=OPTIONAL_GPL_BENCHMARK_CLASS,
        benchmark_grid="graphtools_kernel_diffusion_nnls",
    ),
    "tbs_diffusion_graphtools_adaptive_nnls": MethodSpec(
        name="TBS (graphtools Kernel Diffusion, Adaptive-K NNLS Branch-Time)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_graphtools_method",
        ),
        param_grid=list(GRAPHTOOLS_ADAPTIVE_K_TREE_STRATEGY_PARAMS),
        benchmark_class=OPTIONAL_GPL_BENCHMARK_CLASS,
        benchmark_grid=GRAPHTOOLS_TREE_STRATEGY_GRID,
    ),
    "leiden": MethodSpec(
        name="Leiden",
        runner=_import_runner(
            "benchmarks.shared.runners.leiden_runner",
            "_run_leiden_method",
        ),
        param_grid=[{"n_neighbors": 10, "resolution": 1.0}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "louvain": MethodSpec(
        name="Louvain",
        runner=_import_runner(
            "benchmarks.shared.runners.louvain_runner",
            "_run_louvain_method",
        ),
        param_grid=[{"n_neighbors": 10, "resolution": 1.0}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "kmeans": MethodSpec(
        name="K-Means",
        runner=_import_runner(
            "benchmarks.shared.runners.kmeans_runner",
            "_run_kmeans_method",
        ),
        # Keep parity with visualization baselines by using true K per case.
        param_grid=[{"n_clusters": "true", "n_init": 10}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "spectral": MethodSpec(
        name="Spectral",
        runner=_import_runner(
            "benchmarks.shared.runners.spectral_runner",
            "_run_spectral_method",
        ),
        # Keep parity with visualization baselines by using true K per case.
        param_grid=[
            {
                "n_clusters": "true",
                "affinity": "nearest_neighbors",
                "assign_labels": "cluster_qr",
                "n_neighbors": 10,
            }
        ],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "dbscan": MethodSpec(
        name="DBSCAN",
        runner=_import_runner(
            "benchmarks.shared.runners.dbscan_runner",
            "_run_dbscan_method",
        ),
        param_grid=[{"min_samples": 5, "eps": "median_k_distance"}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "optics": MethodSpec(
        name="OPTICS",
        runner=_import_runner(
            "benchmarks.shared.runners.optics_runner",
            "_run_optics_method",
        ),
        param_grid=[{"min_samples": 5, "xi": 0.05, "min_cluster_size": 5}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
    "hdbscan": MethodSpec(
        name="HDBSCAN",
        runner=_import_runner(
            "benchmarks.shared.runners.hdbscan_runner",
            "_run_hdbscan_method",
        ),
        param_grid=[{"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.0}],
        benchmark_class=CANONICAL_BENCHMARK_CLASS,
        benchmark_grid=DEFAULT_METHOD_GRID,
    ),
}
