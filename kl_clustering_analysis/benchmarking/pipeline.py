"""
Benchmarking/validation pipeline for the clustering algorithm.

Runs configurable test cases, computes metrics, and optionally produces plots.
The implementation was previously housed in ``tests/validation_utils`` and is
kept here so it can be reused outside the test suite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import importlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
)

try:  # Prefer test config when available (repo usage)
    from tests.test_cases_config import get_default_test_cases  # type: ignore
except ImportError:
    try:  # Fallback when run from within tests package directly
        from test_cases_config import get_default_test_cases  # type: ignore
    except ImportError:  # As a last resort, use an empty default

        def get_default_test_cases():
            return []


from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from kl_clustering_analysis.benchmarking.plots import (
    create_validation_plot,
    create_umap_plots_from_results,
    create_manifold_plots_from_results,
    create_tree_plots_from_results,
)

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)


# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner: Callable[..., "MethodRunResult"]
    param_grid: list[dict[str, object]]


@dataclass
class MethodRunResult:
    labels: np.ndarray | None
    found_clusters: int
    report_df: pd.DataFrame
    status: str
    skip_reason: str | None
    extra: dict | None = None


DEFAULT_METHODS = ["kl", "leiden", "louvain", "dbscan", "optics", "hdbscan"]


def _optional_import(module: str):
    """Return imported module or None when unavailable."""
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map non-negative labels to contiguous IDs while keeping noise at -1."""
    labels_arr = np.asarray(labels, dtype=int)
    unique = sorted({int(x) for x in labels_arr if x >= 0})
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.array([mapping.get(int(x), -1) for x in labels_arr], dtype=int)


def _format_params(params: dict[str, object]) -> str:
    """Format parameters for result rows."""
    if not params:
        return ""
    parts = [f"{key}={value}" for key, value in sorted(params.items())]
    return ", ".join(parts)


def _resolve_n_neighbors(n_samples: int, n_neighbors: int | None) -> int:
    if n_samples <= 1:
        return 0
    if n_neighbors is None:
        return max(2, min(10, n_samples - 1))
    return max(1, min(int(n_neighbors), n_samples - 1))


def _knn_edge_weights(
    distance_matrix: np.ndarray, n_neighbors: int
) -> list[tuple[int, int, float]]:
    """Build undirected k-NN edge list with inverse-distance weights."""
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1 or n_neighbors <= 0:
        return []

    edge_weights: dict[tuple[int, int], float] = {}
    for i in range(n_samples):
        neighbor_idx = np.argsort(distance_matrix[i])
        neighbors = neighbor_idx[1 : n_neighbors + 1]
        for j in neighbors:
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            dist = float(distance_matrix[i, j])
            weight = 1.0 / (1.0 + dist)
            if key not in edge_weights or weight > edge_weights[key]:
                edge_weights[key] = weight

    return [(i, j, w) for (i, j), w in edge_weights.items()]


def _estimate_dbscan_eps(distance_matrix: np.ndarray, min_samples: int) -> float:
    """Estimate eps via the median k-distance heuristic."""
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        return 0.0
    k = min(max(int(min_samples), 1), n_samples - 1)
    kth = np.partition(distance_matrix, k, axis=1)[:, k]
    eps = float(np.median(kth))
    return eps if eps > 0 else 1e-9


def _create_report_dataframe_from_labels(
    labels: np.ndarray, sample_names: pd.Index
) -> pd.DataFrame:
    """Create a report dataframe from flat cluster labels."""
    if labels.size == 0:
        return pd.DataFrame(
            columns=["cluster_id", "cluster_size"],
            index=pd.Index([], name="sample_id"),
        )
    series = pd.Series(labels, index=sample_names, name="cluster_id")
    sizes = series.value_counts()
    report = pd.DataFrame(
        {
            "cluster_id": series,
            "cluster_size": series.map(sizes),
        }
    )
    report.index.name = "sample_id"
    return report


def _run_kl_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    significance_level: float,
) -> MethodRunResult:
    Z_t = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_df.index.tolist())
    decomp_t = tree_t.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=significance_level,
    )
    report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))
    labels = np.asarray(_labels_from_decomposition(decomp_t, data_df.index.tolist()))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomp_t.get("num_clusters", 0)),
        report_df=report_t,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree_t,
            "decomposition": decomp_t,
            "stats": tree_t.stats_df,
        },
    )


def _run_leiden_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None,
) -> MethodRunResult:
    igraph = _optional_import("igraph")
    leidenalg = _optional_import("leidenalg")
    if igraph is None or leidenalg is None:
        missing = [
            name
            for name, mod in [("igraph", igraph), ("leidenalg", leidenalg)]
            if mod is None
        ]
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=_create_report_dataframe_from_labels(np.array([]), pd.Index([])),
            status="skipped",
            skip_reason=f"Missing optional deps: {', '.join(missing)}",
        )

    n_samples = distance_matrix.shape[0]
    n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))  # type: ignore[arg-type]
    resolution = float(params.get("resolution", 1.0))
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    edge_list = [(i, j) for i, j, _w in edges]
    weights = [w for _i, _j, w in edges]
    graph = igraph.Graph(n=n_samples, edges=edge_list, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = _normalize_labels(np.asarray(partition.membership))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )


def _run_louvain_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None,
) -> MethodRunResult:
    n_samples = distance_matrix.shape[0]
    n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))  # type: ignore[arg-type]
    resolution = float(params.get("resolution", 1.0))
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    community = _optional_import("community")
    if community is not None:
        graph = nx.Graph()
        graph.add_nodes_from(range(n_samples))
        graph.add_weighted_edges_from(edges)
        partition = community.best_partition(
            graph, weight="weight", resolution=resolution, random_state=seed
        )
        labels = _normalize_labels(
            np.array([partition.get(i, -1) for i in range(n_samples)], dtype=int)
        )
        report_df = _create_report_dataframe_from_labels(
            labels, pd.Index(range(n_samples))
        )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
        )

    igraph = _optional_import("igraph")
    if igraph is None:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=_create_report_dataframe_from_labels(np.array([]), pd.Index([])),
            status="skipped",
            skip_reason="Missing optional deps: python-louvain or igraph",
        )

    edge_list = [(i, j) for i, j, _w in edges]
    weights = [w for _i, _j, w in edges]
    graph = igraph.Graph(n=n_samples, edges=edge_list, directed=False)
    graph.es["weight"] = weights
    partition = graph.community_multilevel(weights="weight")
    labels = _normalize_labels(np.asarray(partition.membership))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )


def _run_dbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    min_samples = int(params.get("min_samples", 5))
    eps = params.get("eps")
    if eps is None:
        eps = _estimate_dbscan_eps(distance_matrix, min_samples)
    model = DBSCAN(metric="precomputed", eps=float(eps), min_samples=min_samples)
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )


def _run_optics_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    min_samples = int(params.get("min_samples", 5))
    xi = float(params.get("xi", 0.05))
    min_cluster_size = params.get("min_cluster_size", min_samples)
    model = OPTICS(
        metric="precomputed",
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )


def _run_hdbscan_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
) -> MethodRunResult:
    hdbscan = _optional_import("hdbscan")
    if hdbscan is None:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=_create_report_dataframe_from_labels(np.array([]), pd.Index([])),
            status="skipped",
            skip_reason="Missing optional deps: hdbscan",
        )

    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    min_cluster_size = int(params.get("min_cluster_size", 5))
    min_samples = params.get("min_samples", None)
    model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = _normalize_labels(model.fit_predict(distance_matrix))
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )


METHOD_SPECS: dict[str, MethodSpec] = {
    "kl": MethodSpec(
        name="KL Divergence",
        runner=_run_kl_method,
        param_grid=[{}],
    ),
    "leiden": MethodSpec(
        name="Leiden",
        runner=_run_leiden_method,
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "louvain": MethodSpec(
        name="Louvain",
        runner=_run_louvain_method,
        param_grid=[{"n_neighbors": None, "resolution": 1.0}],
    ),
    "dbscan": MethodSpec(
        name="DBSCAN",
        runner=_run_dbscan_method,
        param_grid=[{"min_samples": 5, "eps": None}],
    ),
    "optics": MethodSpec(
        name="OPTICS",
        runner=_run_optics_method,
        param_grid=[{"min_samples": 5, "xi": 0.05, "min_cluster_size": 5}],
    ),
    "hdbscan": MethodSpec(
        name="HDBSCAN",
        runner=_run_hdbscan_method,
        param_grid=[{"min_cluster_size": 5, "min_samples": None}],
    ),
}


def _log_validation_start(n_cases: int) -> None:
    """Log the start of the validation process."""
    logger.info("=" * 80)
    logger.info("CLUSTER ALGORITHM VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Evaluating {n_cases} test cases.")


def _log_test_case_start(index: int, total: int, name: str) -> None:
    """Log the start of a specific test case."""
    logger.info(f"Running test case {index}/{total}: {name}")


def _log_validation_completion(total_runs: int, n_cases: int) -> None:
    """Log the completion of the validation process."""
    logger.info(f"Completed {total_runs} validation runs across {n_cases} test cases.")


def _log_detailed_results(df_results: pd.DataFrame) -> None:
    """Log the detailed results table."""
    logger.info("Detailed Results:")
    # Convert dataframe to string and log each line to avoid truncation issues in some loggers
    columns = [
        "Test",
        "Case_Name",
        "Method",
        "Params",
        "True",
        "Found",
        "Samples",
        "Features",
        "Noise",
        "ARI",
        "NMI",
        "Purity",
        "Status",
    ]
    available = [col for col in columns if col in df_results.columns]
    results_str = df_results[available].to_string(index=False)
    for line in results_str.split("\n"):
        logger.info(line)


def _generate_validation_plots(
    df_results: pd.DataFrame,
    computed_results: list,
    plots_root: Path,
    verbose: bool,
    plot_umap: bool,
    plot_manifold: bool,
):
    """Generate and save validation plots."""
    if df_results.empty:
        return None

    df_plot = df_results
    fig = create_validation_plot(df_plot)

    if verbose:
        fig.savefig("validation_results.png", dpi=150, bbox_inches="tight")
        logger.info("Validation plot saved to 'validation_results.png'")

        _log_detailed_results(df_results)

        logger.info("Generating tree plots...")
        plots_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_tree_plots_from_results(
            computed_results, plots_root, timestamp, verbose=False
        )
        logger.info(f"Tree plots saved to {plots_root}")

    if plot_umap:
        logger.info("Generating UMAP comparison visualizations...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_umap_plots_from_results(
            computed_results, plots_root, current_timestamp, verbose=verbose
        )

    if plot_manifold:
        logger.info("Generating manifold diagnostics (UMAP vs Isomap)...")
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_manifold_plots_from_results(
            computed_results,
            plots_root,
            current_timestamp,
            verbose=verbose,
        )

    return fig


def _generate_case_data(
    test_case: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Create a binary dataframe, true labels, original features, and metadata for a test case."""
    generator = test_case.get("generator", "blobs")
    seed = test_case.get("seed")

    if generator == "binary":
        n_rows = test_case.get("n_rows", test_case.get("n_samples"))
        n_cols = test_case.get("n_cols", test_case.get("n_features"))
        if n_rows is None or n_cols is None:
            raise ValueError(
                "Binary generator requires 'n_rows'/'n_cols' or 'n_samples'/'n_features'."
            )
        entropy = test_case.get("entropy_param", 0.5)
        balanced = test_case.get("balanced_clusters", True)
        feature_sparsity = test_case.get("feature_sparsity", None)

        data_dict, cluster_assignments = generate_random_feature_matrix(
            n_rows=n_rows,
            n_cols=n_cols,
            entropy_param=entropy,
            n_clusters=test_case["n_clusters"],
            random_seed=seed,
            balanced_clusters=balanced,
            feature_sparsity=feature_sparsity,
        )

        original_names = list(data_dict.keys())
        matrix = np.array([data_dict[name] for name in original_names], dtype=int)
        # Use original_names as index to maintain consistency with true_labels
        sample_names = original_names
        feature_names = [f"F{j}" for j in range(matrix.shape[1])]

        data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)
        true_labels = np.array(
            [cluster_assignments[name] for name in original_names],
            dtype=int,
        )

        metadata = {
            "n_samples": n_rows,
            "n_features": n_cols,
            "n_clusters": test_case["n_clusters"],
            "noise": entropy,
            "name": test_case.get("name", f"binary_{n_rows}x{n_cols}"),
            "generator": "binary",
        }

        return data_df, true_labels, matrix.astype(float), metadata

    # Default: Gaussian blobs -> binarize via median threshold
    n_samples = test_case["n_samples"]
    n_features = test_case["n_features"]
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X_bin = (X > np.median(X, axis=0)).astype(int)
    data_df = pd.DataFrame(
        X_bin,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": test_case.get("name", f"blobs_{n_samples}x{n_features}"),
        "generator": "blobs",
    }
    return data_df, y, X, metadata


def _labels_from_decomposition(
    decomposition: dict, sample_index: list[str]
) -> list[int]:
    """Extract cluster labels for each sample from a decomposition result."""
    assignments = {sample: -1 for sample in sample_index}
    for cluster_id, info in decomposition.get("cluster_assignments", {}).items():
        for leaf in info["leaves"]:
            assignments[leaf] = cluster_id
    return [assignments[sample] for sample in sample_index]


def _create_report_dataframe(cluster_assignments: dict) -> pd.DataFrame:
    """Create a report dataframe from cluster assignments."""
    if cluster_assignments:
        rows = []
        for cid, info in cluster_assignments.items():
            for leaf in info["leaves"]:
                rows.append(
                    {
                        "sample_id": leaf,
                        "cluster_id": cid,
                        "cluster_size": info["size"],
                    }
                )
        return pd.DataFrame(rows).set_index("sample_id")

    return pd.DataFrame(
        columns=["cluster_id", "cluster_size"],
        index=pd.Index([], name="sample_id"),
    )


def _calculate_ari_nmi_purity_metrics(
    num_clusters: int,
    report_df: pd.DataFrame,
    sample_names: pd.Index,
    true_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate clustering metrics (ARI, NMI, Purity)."""
    if num_clusters > 0 and not report_df.empty:
        # Create a correct mapping from sample name (report_df index) to true cluster label
        true_label_map = {name: label for name, label in zip(sample_names, true_labels)}

        # Work on a copy to avoid side effects
        df_metrics = report_df.copy()
        df_metrics["true_cluster"] = df_metrics.index.map(true_label_map)

        ari = adjusted_rand_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        nmi = normalized_mutual_info_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )

        purity = homogeneity_score(
            df_metrics["true_cluster"].values, df_metrics["cluster_id"].values
        )
        return ari, nmi, purity

    return 0.0, 0.0, 0.0


def benchmark_cluster_algorithm(
    test_cases=None,
    significance_level=config.SIBLING_ALPHA,
    verbose=True,
    plot_umap=False,
    plot_manifold=False,
    methods: list[str] | None = None,
    method_params: dict[str, list[dict[str, object]]] | None = None,
):
    """
    Benchmark the cluster decomposition algorithm across multiple test cases.

    Parameters
    ----------
    test_cases : list of dict, optional
        List of test case configurations. Each dict should contain:
        - n_samples: number of samples
        - n_features: number of features
        - n_clusters: true number of clusters
        - cluster_std: noise level (standard deviation)
        - seed: random seed
        If None, uses default test cases.
    significance_level : float, default=config.SIBLING_ALPHA
        Significance level used for sibling-independence gating in decomposition
    verbose : bool, default=True
        If True, prints progress and displays validation results
    plot_umap : bool, default=False
        If True, generates t-SNE plots comparing KL clustering with K-means and spectral clustering
    plot_manifold : bool, default=False
        If True, saves UMAP-vs-Isomap manifold diagnostics for each test case

    methods : list of str, optional
        Clustering methods to run (defaults to the full registry).
    method_params : dict, optional
        Optional per-method parameter grids. Values are lists of param dicts.

    Returns
    -------
    df_results : pd.DataFrame
        Results dataframe with columns: Test, True, Found, Samples, Features,
        Noise, ARI, NMI, Purity
    fig : None
        Plotting has been removed; always returns None.
    """

    # Default test cases
    project_root = Path(__file__).resolve().parents[2]
    plots_root = project_root / "cluster_tree_plots"
    if test_cases is None:
        test_cases = get_default_test_cases()

    if verbose:
        _log_validation_start(len(test_cases))

    results_data = []

    # Store computed results to avoid recalculation
    computed_results = []

    # Run test cases
    selected_methods = methods or DEFAULT_METHODS
    method_params = method_params or {}
    for method_id in selected_methods:
        if method_id not in METHOD_SPECS:
            raise ValueError(f"Unknown method: {method_id}")

    param_sets = {
        method_id: (method_params.get(method_id) or METHOD_SPECS[method_id].param_grid)
        for method_id in selected_methods
    }
    total_cases = len(test_cases)
    total_runs = total_cases * sum(len(params) for params in param_sets.values())

    for i, tc in enumerate(test_cases, 1):
        case_name = tc.get("name", f"Case {i}")
        if verbose:
            _log_test_case_start(i, total_cases, case_name)

        # Generate and process data
        data_t, y_t, X_original, meta = _generate_case_data(tc)

        needs_distance_matrix = any(
            method_id in {"leiden", "louvain", "dbscan", "optics", "hdbscan"}
            for method_id in selected_methods
        )
        needs_distance_condensed = "kl" in selected_methods or needs_distance_matrix
        distance_condensed = None
        distance_matrix = None
        if needs_distance_condensed:
            distance_condensed = pdist(
                data_t.values, metric=config.TREE_DISTANCE_METRIC
            )
        if needs_distance_matrix:
            distance_matrix = squareform(distance_condensed)

        for method_id in selected_methods:
            spec = METHOD_SPECS[method_id]
            params_list = param_sets[method_id]
            for params in params_list:
                meta_run = meta.copy()
                if method_id == "kl":
                    result = spec.runner(
                        data_t,
                        distance_condensed,
                        significance_level,
                    )
                elif method_id in {"leiden", "louvain"}:
                    result = spec.runner(
                        distance_matrix,
                        params,
                        tc.get("seed"),
                    )
                else:
                    result = spec.runner(distance_matrix, params)

                if result.status == "ok" and result.labels is not None:
                    labels = result.labels
                    report_df = _create_report_dataframe_from_labels(
                        labels, data_t.index
                    )
                    found_clusters = result.found_clusters
                    labels_len = len(labels)
                    ari, nmi, purity = _calculate_ari_nmi_purity_metrics(
                        found_clusters, report_df, data_t.index, y_t
                    )
                else:
                    labels_len = 0
                    found_clusters = 0
                    ari, nmi, purity = np.nan, np.nan, np.nan

                results_data.append(
                    {
                        "Test": i,
                        "Case_Name": case_name,
                        "Method": spec.name,
                        "Params": _format_params(params),
                        "True": meta["n_clusters"],
                        "Found": found_clusters,
                        "Samples": meta["n_samples"],
                        "Features": meta["n_features"],
                        "Noise": meta["noise"],
                        "ARI": ari,
                        "NMI": nmi,
                        "Purity": purity,
                        "Status": result.status,
                        "Skip_Reason": result.skip_reason or "",
                        "Labels_Length": labels_len,
                    }
                )

                if result.status == "ok" and result.labels is not None:
                    meta_run["found_clusters"] = found_clusters
                    computed_results.append(
                        {
                            "test_case_num": i,
                            "method_name": spec.name,
                            "labels": result.labels,
                            "data": data_t,
                            "meta": meta_run,
                            "X_original": X_original,
                            "y_true": y_t,
                            "tree": result.extra.get("tree") if result.extra else None,
                            "decomposition": result.extra.get("decomposition")
                            if result.extra
                            else None,
                            "stats": result.extra.get("stats")
                            if result.extra
                            else None,
                        }
                    )

    if verbose:
        _log_validation_completion(total_runs, len(test_cases))

    df_results = pd.DataFrame(results_data)

    if not verbose and not plot_umap and not plot_manifold:
        return df_results, None

    fig = _generate_validation_plots(
        df_results, computed_results, plots_root, verbose, plot_umap, plot_manifold
    )
    if not verbose:
        return df_results, None
    return df_results, fig


__all__ = [
    "benchmark_cluster_algorithm",
    "_labels_from_decomposition",
]
