"""Method registry config for benchmarking.

Export a direct `METHOD_SPECS` mapping so callers can import it as a
configuration constant.
"""

from __future__ import annotations

import importlib

from benchmarks.shared.types import MethodSpec


def _import_runner(module: str, attr: str):
    mod = importlib.import_module(module)
    return mod.__dict__[attr]


# Note: import names are updated to point to the new benchmarks.shared.runners package.
METHOD_SPECS: dict[str, MethodSpec] = {
    "kl": MethodSpec(
        name="KL Divergence",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            # Default: Hamming + Average (UPGMA)
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
            },
        ],
    ),
    "kl_complete": MethodSpec(
        name="KL (Complete)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "complete",
            },
        ],
    ),
    "kl_single": MethodSpec(
        name="KL (Single)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "single",
            },
        ],
    ),
    "kl_conditional_topology_diagnostic": MethodSpec(
        name="KL (Conditional Topology Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
    "kl_global_passthrough_refined_diagnostic": MethodSpec(
        name="KL (Global Passthrough Refined Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
    "kl_spectral_transport_passthrough_diagnostic": MethodSpec(
        name="KL (Spectral Transport Passthrough Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
    "kl_spectral_transport_passthrough": MethodSpec(
        name="KL (Spectral Transport Passthrough)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
    "kl_legacy_internal_spectral_diagnostic": MethodSpec(
        name="KL (Legacy Internal-Node Spectral Diagnostic)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "spectral_include_internal_barycenters": True,
            },
        ],
    ),
    "kl_legacy_c2ef9a69": MethodSpec(
        name="KL Legacy Full Method (commit c2ef9a69)",
        runner=_import_runner(
            "benchmarks.shared.runners.legacy_commit_runner",
            "_run_legacy_c2ef9a69_kl_method",
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
    "kl_neighbor_joining": MethodSpec(
        name="KL (Neighbor Joining, MAD Root)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "average",
                "tree_builder": "neighbor_joining",
                "tree_rooting": "mad",
            },
        ],
    ),
    "kl_iqtree3": MethodSpec(
        name="KL (IQ-TREE 3, MAD Root)",
        runner=_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
    "kl_diffusion": MethodSpec(
        name="KL (Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.kl_diffusion_runner",
            "_run_kl_diffusion_method",
        ),
        param_grid=[{"k_neighbors": 15, "diffusion_time": 3}],
    ),
    "kl_diffusion_adaptive": MethodSpec(
        name="KL (Adaptive Diffusion)",
        runner=_import_runner(
            "benchmarks.shared.runners.kl_diffusion_runner",
            "_run_kl_diffusion_adaptive_method",
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
