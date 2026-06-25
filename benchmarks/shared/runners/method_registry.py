"""Method registry config for benchmarking.

Export a direct `METHOD_SPECS` mapping so callers can import it as a
configuration constant.
"""

from __future__ import annotations

import importlib

from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY,
)

from benchmarks.shared.types import MethodSpec


def _import_runner(module: str, attr: str):
    mod = importlib.import_module(module)
    return mod.__dict__[attr]


# Note: import names are updated to point to the new benchmarks.shared.runners package.
METHOD_SPECS: dict[str, MethodSpec] = {
    "tbs": MethodSpec(
        name="TBS Divergence",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            # Default: Hamming + Average (UPGMA)
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
            },
        ],
    ),
    "tbs_continuous_guarded_within_covariance": MethodSpec(
        name="TBS Continuous Guarded Within-Child Covariance",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "continuous_covariance_policy": (
                    GUARDED_WITHIN_CHILD_CONTINUOUS_COVARIANCE_POLICY
                ),
                "continuous_covariance_min_child_leaf_count": (
                    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
                ),
                "continuous_sibling_gate_method": "fixed_coordinate_bh",
            },
        ],
    ),
    "tbs_complete": MethodSpec(
        name="TBS (Complete)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "complete",
            },
        ],
    ),
    "tbs_single": MethodSpec(
        name="TBS (Single)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "single",
            },
        ],
    ),
    "tbs_fixed_coordinate_bh": MethodSpec(
        name="TBS Fixed Coordinate BH",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": (
                    "fixed_coordinate_conditional_topology_diagnostic_v1"
                ),
            },
        ],
    ),
    "tbs_global_passthrough_refined_diagnostic": MethodSpec(
        name="TBS (Global Passthrough Refined Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": (
                    "fixed_coordinate_global_passthrough_refined_v1"
                ),
            },
        ],
    ),
    "tbs_spectral_transport_passthrough_diagnostic": MethodSpec(
        name="TBS (Spectral Transport Passthrough Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "sibling_gate_profile": (
                    "fixed_coordinate_spectral_transport_passthrough_v1"
                ),
            },
        ],
    ),
    "tbs_legacy_internal_spectral_diagnostic": MethodSpec(
        name="TBS (Legacy Internal-Node Spectral Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_mp_row_count_mode": "legacy_stacked_rows",
            },
        ],
    ),
    "tbs_internal_filter_v1": MethodSpec(
        name="TBS Internal Spectral Filter",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_internal_distribution_mode": "empirical_barycenter",
                "spectral_mp_row_count_mode": "leaf_effective_rows",
                "enforce_internal_support_thresholds": True,
            },
        ],
    ),
    "tbs_internal_filter_branch_length_v1": MethodSpec(
        name="TBS Branch-Length Internal Spectral Filter",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_internal_distribution_mode": "branch_length_state",
                "spectral_mp_row_count_mode": "leaf_effective_rows",
                "enforce_internal_support_thresholds": True,
            },
        ],
    ),
    "tbs_bandwidth_context_v1": MethodSpec(
        name="TBS Regional Bandwidth Context",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "neighborhood_bandwidth_profile": (
                    "regional_tau_branch_length_support_only_v1"
                ),
            },
        ],
    ),
    "tbs_rescued_legacy_v1": MethodSpec(
        name="TBS Guarded Legacy-Stabilized Candidate",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
                "spectral_internal_distribution_mode": "branch_length_state",
                "spectral_mp_row_count_mode": "leaf_effective_rows",
                "enforce_internal_support_thresholds": True,
                "sibling_gate_profile": (
                    "fixed_coordinate_spectral_transport_passthrough_v1"
                ),
                "neighborhood_bandwidth_profile": (
                    "regional_tau_branch_length_support_only_v1"
                ),
            },
        ],
    ),
    "tbs_legacy_c2ef9a69": MethodSpec(
        name="TBS Legacy Full Method (commit c2ef9a69)",
        runner=_import_runner(
            "benchmarks.shared.runners.legacy_commit_runner",
            "_run_legacy_c2ef9a69_tbs_method",
        ),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "tree_builder": "linkage",
                "tree_rooting": "linkage_root",
            },
        ],
    ),
    "tbs_neighbor_joining": MethodSpec(
        name="TBS (Neighbor Joining, MAD Root)",
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        runner=_import_runner("benchmarks.shared.runners.tbs_runner", "_run_tbs_method"),
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
        name="TBS (Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_method",
        ),
        param_grid=[{"k_neighbors": 15, "diffusion_time": 3}],
    ),
    "tbs_diffusion_adaptive": MethodSpec(
        name="TBS (Adaptive Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.tbs_diffusion_runner",
            "_run_tbs_diffusion_adaptive_method",
        ),
        param_grid=[
            {
                "k_neighbors": 10,
                "diffusion_time": 3,
                "n_components": 30,
                "metric": "hamming",
                "bandwidth_type": "-1/(d+2)",
                "epsilon": "median",
            }
        ],
    ),
    "leiden": MethodSpec(
        name="Leiden",
        runner=_import_runner(
            "benchmarks.shared.runners.leiden_runner",
            "_run_leiden_method",
        ),
        param_grid=[{"n_neighbors": 10, "resolution": 1.0}],
    ),
    "louvain": MethodSpec(
        name="Louvain",
        runner=_import_runner(
            "benchmarks.shared.runners.louvain_runner",
            "_run_louvain_method",
        ),
        param_grid=[{"n_neighbors": 10, "resolution": 1.0}],
    ),
    "kmeans": MethodSpec(
        name="K-Means",
        runner=_import_runner(
            "benchmarks.shared.runners.kmeans_runner",
            "_run_kmeans_method",
        ),
        # Keep parity with visualization baselines by using true K per case.
        param_grid=[{"n_clusters": "true", "n_init": 10}],
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
    ),
    "dbscan": MethodSpec(
        name="DBSCAN",
        runner=_import_runner(
            "benchmarks.shared.runners.dbscan_runner",
            "_run_dbscan_method",
        ),
        param_grid=[{"min_samples": 5, "eps": "median_k_distance"}],
    ),
    "optics": MethodSpec(
        name="OPTICS",
        runner=_import_runner(
            "benchmarks.shared.runners.optics_runner",
            "_run_optics_method",
        ),
        param_grid=[{"min_samples": 5, "xi": 0.05, "min_cluster_size": 5}],
    ),
    "hdbscan": MethodSpec(
        name="HDBSCAN",
        runner=_import_runner(
            "benchmarks.shared.runners.hdbscan_runner",
            "_run_hdbscan_method",
        ),
        param_grid=[{"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.0}],
    ),
}
