"""Method registry config for benchmarking.

Export a direct `METHOD_SPECS` mapping so callers can import it as a
configuration constant.
"""

from __future__ import annotations

from kl_clustering_analysis.benchmarking.types import MethodSpec, MethodRunResult
import importlib


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


# Note: import names are intentionally matching previous runner names so
# consumers can import METHOD_SPECS directly. Runners are loaded lazily so
# optional dependencies do not prevent importing the registry.
METHOD_SPECS: dict[str, MethodSpec] = {
    "kl": MethodSpec(
        name="KL Divergence",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.kl_runner", "_run_kl_method"
        ),
        param_grid=[
            {
                "tree_distance_metric": "rogerstanimoto",
                "tree_linkage_method": "average",
            },
            {
                "tree_distance_metric": "rogerstanimoto",
                "tree_linkage_method": "complete",
            },
            # {"tree_distance_metric": "rogerstanimoto", "tree_linkage_method": "ward"},
            # {"tree_distance_metric": "jaccard", "tree_linkage_method": "average"},
            # {"tree_distance_metric": "jaccard", "tree_linkage_method": "complete"},
            # {"tree_distance_metric": "jaccard", "tree_linkage_method": "ward"},
            # {"tree_distance_metric": "euclidean", "tree_linkage_method": "average"},
            # {"tree_distance_metric": "euclidean", "tree_linkage_method": "complete"},
            # {"tree_distance_metric": "euclidean", "tree_linkage_method": "ward"},
        ],
    ),
    "leiden": MethodSpec(
        name="Leiden",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.leiden_runner",
            "_run_leiden_method",
        ),
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "louvain": MethodSpec(
        name="Louvain",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.louvain_runner",
            "_run_louvain_method",
        ),
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "dbscan": MethodSpec(
        name="DBSCAN",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.dbscan_runner",
            "_run_dbscan_method",
        ),
        param_grid=[{"min_samples": 5, "eps": None}],
    ),
    "optics": MethodSpec(
        name="OPTICS",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.optics_runner",
            "_run_optics_method",
        ),
        param_grid=[{"min_samples": 5, "xi": 0.05, "min_cluster_size": 5}],
    ),
    "hdbscan": MethodSpec(
        name="HDBSCAN",
        runner=_safe_import_runner(
            "kl_clustering_analysis.benchmarking.runners.hdbscan_runner",
            "_run_hdbscan_method",
        ),
        param_grid=[{"min_cluster_size": 5, "min_samples": None}],
    ),
}
