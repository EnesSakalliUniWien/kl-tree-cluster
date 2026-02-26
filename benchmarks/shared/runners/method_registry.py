"""Method registry config for benchmarking.

Export a direct `METHOD_SPECS` mapping so callers can import it as a
configuration constant.
"""

from __future__ import annotations

import importlib

from benchmarks.shared.types import MethodRunResult, MethodSpec


def _safe_import_runner(module: str, attr: str):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except Exception as exc:  # pragma: no cover - optional dependencies
        # Capture `exc` in a default argument so linters and static analyzers
        # do not mark it as undefined when the nested function is inspected.
        def _skip_runner(*args, _exc=exc, **kwargs):
            return MethodRunResult(
                labels=None,
                found_clusters=0,
                report_df=None,
                status="skip",
                skip_reason=f"Missing runner {module}.{attr}: {_exc}",
                extra={},
            )

        return _skip_runner


# Note: import names are updated to point to the new benchmarks.shared.runners package.
METHOD_SPECS: dict[str, MethodSpec] = {
    "kl": MethodSpec(
        name="KL Divergence",
        runner=_safe_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
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
        runner=_safe_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "complete",
            },
        ],
    ),
    "kl_single": MethodSpec(
        name="KL (Single)",
        runner=_safe_import_runner("benchmarks.shared.runners.kl_runner", "_run_kl_method"),
        param_grid=[
            {
                "tree_distance_metric": "hamming",
                "tree_linkage_method": "single",
            },
        ],
    ),
    # "kl_ward": MethodSpec(
    #     name="KL (Ward)",
    #     runner=_safe_import_runner(
    #         "benchmarks.shared.runners.kl_runner", "_run_kl_method"
    #     ),
    #     param_grid=[
    #         {
    #             # Ward linkage requires Euclidean distance
    #             "tree_distance_metric": "euclidean",
    #             "tree_linkage_method": "ward",
    #         },
    #     ],
    # ),
    # "kl_rogerstanimoto": MethodSpec(
    #     name="KL (Rogers-Tanimoto)",
    #     runner=_safe_import_runner(
    #         "benchmarks.shared.runners.kl_runner", "_run_kl_method"
    #     ),
    #     param_grid=[
    #         # Rogers-Tanimoto double-weights mismatches
    #         {
    #             "tree_distance_metric": "rogerstanimoto",
    #             "tree_linkage_method": "average",
    #         },
    #     ],
    # ),
    # "kl_v2": MethodSpec(
    #     name="KL v2 (Signal Localization)",
    #     runner=_safe_import_runner(
    #         "benchmarks.shared.runners.kl_runner", "_run_kl_v2_method"
    #     ),
    #     param_grid=[
    #         {
    #             "tree_distance_metric": "hamming",
    #             "tree_linkage_method": "average",
    #         },
    #     ],
    # ),
    "leiden": MethodSpec(
        name="Leiden",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.leiden_runner",
            "_run_leiden_method",
        ),
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "louvain": MethodSpec(
        name="Louvain",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.louvain_runner",
            "_run_louvain_method",
        ),
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "kmeans": MethodSpec(
        name="K-Means",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.kmeans_runner",
            "_run_kmeans_method",
        ),
        # Keep parity with visualization baselines by using true K per case.
        param_grid=[{"n_clusters": "true", "n_init": 10}],
    ),
    "spectral": MethodSpec(
        name="Spectral",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.spectral_runner",
            "_run_spectral_method",
        ),
        # Keep parity with visualization baselines by using true K per case.
        param_grid=[
            {
                "n_clusters": "true",
                "affinity": "nearest_neighbors",
                "assign_labels": "cluster_qr",
                "n_neighbors": None,
            }
        ],
    ),
    "dbscan": MethodSpec(
        name="DBSCAN",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.dbscan_runner",
            "_run_dbscan_method",
        ),
        param_grid=[{"min_samples": 5, "eps": None}],
    ),
    "optics": MethodSpec(
        name="OPTICS",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.optics_runner",
            "_run_optics_method",
        ),
        param_grid=[{"min_samples": 5, "xi": 0.05, "min_cluster_size": 5}],
    ),
    "hdbscan": MethodSpec(
        name="HDBSCAN",
        runner=_safe_import_runner(
            "benchmarks.shared.runners.hdbscan_runner",
            "_run_hdbscan_method",
        ),
        param_grid=[{"min_cluster_size": 5, "min_samples": None}],
    ),
}
