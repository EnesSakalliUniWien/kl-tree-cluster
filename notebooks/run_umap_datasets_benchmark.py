"""
UMAP Datasets Benchmark for KL Divergence Clustering.

Benchmarks the KL clustering algorithm on the datasets used in the UMAP documentation:
1. Palmer Penguins - 333 samples, 4 features, 3 species
2. Sklearn Digits - 1797 samples, 64 features (8x8 images), 10 classes

Reference: https://umap-learn.readthedocs.io/en/latest/basic_usage.html
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import matplotlib.pyplot as plt

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def load_penguins() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the Palmer Penguins dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, 4)
        Numerical features: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    y : ndarray of shape (n_samples,)
        Species labels (0=Adelie, 1=Chinstrap, 2=Gentoo)
    species_names : list
        Names of the species
    """
    print("\n" + "=" * 60)
    print("Loading Palmer Penguins dataset...")
    print("=" * 60)

    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
    df = pd.read_csv(url)

    # Drop rows with missing values
    df = df.dropna()

    # Extract numerical features
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    X = df[feature_cols].values

    # Convert species to numeric labels
    species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    y = df["species"].map(species_map).values
    species_names = list(species_map.keys())

    print(f"  Shape: {X.shape}")
    print(f"  Species distribution: {dict(zip(species_names, np.bincount(y)))}")

    return X, y, species_names


def load_sklearn_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the sklearn digits dataset.

    Returns
    -------
    X : ndarray of shape (1797, 64)
        Pixel values (8x8 images flattened)
    y : ndarray of shape (1797,)
        Digit labels (0-9)
    images : ndarray of shape (1797, 8, 8)
        Original 8x8 digit images
    """
    print("\n" + "=" * 60)
    print("Loading Sklearn Digits dataset...")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target
    images = digits.images  # Keep original 8x8 images for visualization

    print(f"  Shape: {X.shape}")
    print(f"  Digit distribution: {np.bincount(y)}")

    return X, y, images


def discretize_continuous(
    X: np.ndarray, n_bins: int = 5, strategy: str = "quantile"
) -> np.ndarray:
    """Discretize continuous features into bins.

    Parameters
    ----------
    X : ndarray
        Continuous data
    n_bins : int
        Number of bins per feature
    strategy : str
        Binning strategy: 'uniform', 'quantile', or 'kmeans'

    Returns
    -------
    X_discrete : ndarray
        Discretized data (integer labels per feature)
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    X_discrete = discretizer.fit_transform(X).astype(int)
    return X_discrete


def binarize_data(X: np.ndarray, threshold: float = None) -> np.ndarray:
    """Binarize data using a threshold.

    Parameters
    ----------
    X : ndarray
        Input data
    threshold : float, optional
        Threshold for binarization. If None, uses median per feature.

    Returns
    -------
    X_binary : ndarray
        Binary data (0/1)
    """
    if threshold is None:
        # Use median per feature
        thresholds = np.median(X, axis=0)
        X_binary = (X > thresholds).astype(int)
    else:
        X_binary = (X > threshold).astype(int)
    return X_binary


def run_kl_clustering(
    X: np.ndarray,
    sample_names: list[str] = None,
    distance_metric: str = "rogerstanimoto",
    linkage_method: str = "weighted",
    significance_level: float = 0.05,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Run KL divergence clustering on data.

    Parameters
    ----------
    X : ndarray
        Input data matrix (already preprocessed/binarized)
    sample_names : list, optional
        Sample identifiers
    distance_metric : str
        Distance metric for hierarchical clustering
    linkage_method : str
        Linkage method for hierarchical clustering
    significance_level : float
        Alpha level for statistical tests
    verbose : bool
        Whether to print progress

    Returns
    -------
    labels : ndarray
        Cluster assignments (-1 for unclustered)
    results : dict
        Full decomposition results
    """
    if sample_names is None:
        sample_names = [f"S{i}" for i in range(X.shape[0])]

    # Create DataFrame
    data = pd.DataFrame(X, index=sample_names)

    if verbose:
        print(f"  Building hierarchy with {distance_metric} + {linkage_method}...")

    # Build hierarchical tree
    Z = linkage(
        pdist(data.values, metric=distance_metric),
        method=linkage_method,
    )

    # Create PosetTree
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Decompose
    results = tree.decompose(
        leaf_data=data,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    n_clusters = results.get("num_clusters", 0)
    cluster_assignments = results.get("cluster_assignments", {})

    if verbose:
        print(f"  Found {n_clusters} clusters")

    # Convert to label array
    label_map = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id

    labels = np.array([label_map.get(name, -1) for name in sample_names])

    return labels, results


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute clustering metrics.

    Returns
    -------
    metrics : dict
        ARI, NMI, and cluster counts
    """
    # Filter out unclustered points for metrics
    mask = y_pred >= 0
    if mask.sum() == 0:
        return {"ari": 0.0, "nmi": 0.0, "n_clusters": 0, "n_unclustered": len(y_pred)}

    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    n_clusters = len(np.unique(y_pred[mask]))
    n_unclustered = (~mask).sum()

    return {
        "ari": ari,
        "nmi": nmi,
        "n_clusters": n_clusters,
        "n_true_clusters": len(np.unique(y_true)),
        "n_unclustered": n_unclustered,
    }


def plot_umap_embedding(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    title: str,
    ax_true,
    ax_pred,
    class_names: list = None,
):
    """Plot UMAP embedding colored by true and predicted labels."""
    try:
        import umap
    except ImportError:
        print("  UMAP not installed, skipping visualization")
        return

    # Fit UMAP
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    # Plot true labels
    scatter1 = ax_true.scatter(
        embedding[:, 0], embedding[:, 1], c=y, cmap="Spectral", s=10, alpha=0.7
    )
    ax_true.set_title(f"{title} - True Labels")
    ax_true.set_aspect("equal", "datalim")

    # Plot predicted labels
    # Handle unclustered points
    mask_clustered = labels >= 0
    if mask_clustered.any():
        ax_pred.scatter(
            embedding[mask_clustered, 0],
            embedding[mask_clustered, 1],
            c=labels[mask_clustered],
            cmap="tab20",
            s=10,
            alpha=0.7,
        )
    if (~mask_clustered).any():
        ax_pred.scatter(
            embedding[~mask_clustered, 0],
            embedding[~mask_clustered, 1],
            c="gray",
            s=5,
            alpha=0.3,
            marker="x",
            label="Unclustered",
        )
    ax_pred.set_title(
        f"{title} - KL Clusters ({len(np.unique(labels[mask_clustered]))})"
    )
    ax_pred.set_aspect("equal", "datalim")


def create_bokeh_digits_plot(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray,
    output_path: Path,
):
    """Create interactive Bokeh plot with digit image hover tooltips.

    Parameters
    ----------
    X : ndarray
        Feature data for UMAP embedding
    y : ndarray
        True digit labels (0-9)
    labels : ndarray
        Predicted cluster labels
    images : ndarray
        Original 8x8 digit images
    output_path : Path
        Where to save the HTML file
    """
    try:
        import umap
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
        from bokeh.palettes import Spectral10, Category20
        from bokeh.layouts import row
    except ImportError as e:
        print(f"  Missing dependency for Bokeh visualization: {e}")
        return

    from io import BytesIO
    from PIL import Image
    import base64

    def embeddable_image(data):
        """Convert digit array to base64-encoded PNG for embedding in HTML."""
        img_data = 255 - 15 * data.astype(np.uint8)
        image = Image.fromarray(img_data, mode="L").resize(
            (64, 64), Image.Resampling.BICUBIC
        )
        buffer = BytesIO()
        image.save(buffer, format="png")
        for_encoding = buffer.getvalue()
        return "data:image/png;base64," + base64.b64encode(for_encoding).decode()

    print("  Computing UMAP embedding for Bokeh plot...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    # Create embedded images
    print("  Encoding digit images...")
    embedded_images = [embeddable_image(img) for img in images]

    # Create DataFrame for Bokeh
    digits_df = pd.DataFrame(embedding, columns=("x", "y"))
    digits_df["digit"] = [str(d) for d in y]
    digits_df["cluster"] = [str(c) for c in labels]
    digits_df["image"] = embedded_images

    datasource = ColumnDataSource(digits_df)

    # Color mapper for true digits
    digit_color_mapping = CategoricalColorMapper(
        factors=[str(x) for x in range(10)], palette=Spectral10
    )

    # Color mapper for clusters (use Category20 for more colors)
    unique_clusters = sorted(set(str(c) for c in labels))
    n_clusters = len(unique_clusters)
    if n_clusters <= 20:
        cluster_palette = Category20[max(3, min(20, n_clusters))]
    else:
        # Repeat palette for many clusters
        cluster_palette = (Category20[20] * ((n_clusters // 20) + 1))[:n_clusters]

    cluster_color_mapping = CategoricalColorMapper(
        factors=unique_clusters, palette=cluster_palette
    )

    # Create figure for TRUE labels
    plot_true = figure(
        title="UMAP - True Digit Labels",
        width=600,
        height=600,
        tools=("pan, wheel_zoom, reset"),
    )

    plot_true.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>True Digit:</span>
            <span style='font-size: 18px'>@digit</span>
        </div>
        <div>
            <span style='font-size: 14px; color: #666'>Cluster:</span>
            <span style='font-size: 14px'>@cluster</span>
        </div>
    </div>
    """
        )
    )

    plot_true.scatter(
        "x",
        "y",
        source=datasource,
        color=dict(field="digit", transform=digit_color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=5,
    )

    # Create figure for PREDICTED clusters
    plot_pred = figure(
        title=f"UMAP - KL Clusters ({n_clusters} clusters)",
        width=600,
        height=600,
        tools=("pan, wheel_zoom, reset"),
        x_range=plot_true.x_range,  # Share axes
        y_range=plot_true.y_range,
    )

    plot_pred.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Cluster:</span>
            <span style='font-size: 18px'>@cluster</span>
        </div>
        <div>
            <span style='font-size: 14px; color: #666'>True Digit:</span>
            <span style='font-size: 14px'>@digit</span>
        </div>
    </div>
    """
        )
    )

    plot_pred.scatter(
        "x",
        "y",
        source=datasource,
        color=dict(field="cluster", transform=cluster_color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=5,
    )

    # Combine plots side by side
    layout = row(plot_true, plot_pred)

    # Save to HTML
    output_file(str(output_path), title="Digits Clustering - Interactive")
    save(layout)
    print(f"  Interactive Bokeh plot saved to: {output_path}")


def benchmark_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    preprocessing_methods: list[dict],
    tree_methods: list[dict],
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark on a single dataset with multiple configurations.

    Parameters
    ----------
    name : str
        Dataset name
    X : ndarray
        Raw feature data
    y : ndarray
        True labels
    preprocessing_methods : list of dict
        Each dict has 'name' and 'transform' function
    tree_methods : list of dict
        Each dict has 'distance' and 'linkage' keys
    verbose : bool
        Print progress

    Returns
    -------
    results_df : pd.DataFrame
        Results for all configurations
    """
    results = []

    for preproc in preprocessing_methods:
        preproc_name = preproc["name"]
        X_processed = preproc["transform"](X)

        if verbose:
            print(f"\n  Preprocessing: {preproc_name}")
            print(f"    Processed shape: {X_processed.shape}")

        for tree_config in tree_methods:
            dist = tree_config["distance"]
            link = tree_config["linkage"]

            if verbose:
                print(f"    Tree: {dist} + {link}...", end=" ")

            try:
                labels, _ = run_kl_clustering(
                    X_processed,
                    distance_metric=dist,
                    linkage_method=link,
                    verbose=False,
                )
                metrics = evaluate_clustering(y, labels)

                if verbose:
                    print(
                        f"ARI={metrics['ari']:.3f}, NMI={metrics['nmi']:.3f}, "
                        f"clusters={metrics['n_clusters']}/{metrics['n_true_clusters']}"
                    )

                results.append(
                    {
                        "dataset": name,
                        "preprocessing": preproc_name,
                        "distance": dist,
                        "linkage": link,
                        **metrics,
                    }
                )
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                results.append(
                    {
                        "dataset": name,
                        "preprocessing": preproc_name,
                        "distance": dist,
                        "linkage": link,
                        "ari": np.nan,
                        "nmi": np.nan,
                        "n_clusters": 0,
                        "n_true_clusters": len(np.unique(y)),
                        "n_unclustered": len(y),
                        "error": str(e),
                    }
                )

    return pd.DataFrame(results)


def main():
    """Run the UMAP datasets benchmark."""
    print("\n" + "=" * 70)
    print("UMAP DATASETS BENCHMARK - KL Divergence Clustering")
    print("=" * 70)

    # Define preprocessing methods
    preproc_continuous = [
        {
            "name": "scaled_binarized_median",
            "transform": lambda X: binarize_data(
                StandardScaler().fit_transform(X), threshold=0.0
            ),
        },
        {
            "name": "discretized_5bins",
            "transform": lambda X: discretize_continuous(
                X, n_bins=5, strategy="quantile"
            ),
        },
    ]

    preproc_image = [
        {
            "name": "binarized_0.5",
            "transform": lambda X: binarize_data(
                X / 16.0, threshold=0.5
            ),  # Digits are 0-16
        },
        {
            "name": "binarized_0.1",
            "transform": lambda X: binarize_data(X / 16.0, threshold=0.1),
        },
    ]

    # Define tree construction methods
    tree_methods = [
        {"distance": "hamming", "linkage": "weighted"},
        {"distance": "hamming", "linkage": "complete"},
        {"distance": "jaccard", "linkage": "weighted"},
        {"distance": "jaccard", "linkage": "complete"},
        {"distance": "rogerstanimoto", "linkage": "weighted"},
    ]

    all_results = []
    best_configs = {}

    # =========================================================================
    # Dataset 1: Palmer Penguins
    # =========================================================================
    X_penguins, y_penguins, species_names = load_penguins()

    df_penguins = benchmark_dataset(
        name="Penguins",
        X=X_penguins,
        y=y_penguins,
        preprocessing_methods=preproc_continuous,
        tree_methods=tree_methods,
        verbose=True,
    )
    all_results.append(df_penguins)

    # Find best config for penguins
    best_idx = df_penguins["ari"].idxmax()
    best_configs["Penguins"] = df_penguins.loc[best_idx].to_dict()

    # =========================================================================
    # Dataset 2: Sklearn Digits
    # =========================================================================
    X_digits, y_digits, digit_images = load_sklearn_digits()

    df_digits = benchmark_dataset(
        name="Digits",
        X=X_digits,
        y=y_digits,
        preprocessing_methods=preproc_image,
        tree_methods=tree_methods,
        verbose=True,
    )
    all_results.append(df_digits)

    # Find best config for digits
    best_idx = df_digits["ari"].idxmax()
    best_configs["Digits"] = df_digits.loc[best_idx].to_dict()

    # =========================================================================
    # Combine and save results
    # =========================================================================
    df_all = pd.concat(all_results, ignore_index=True)

    # Save results
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = results_dir / f"umap_datasets_benchmark_{timestamp}.csv"
    df_all.to_csv(results_file, index=False)

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for dataset_name, best in best_configs.items():
        print(f"\n{dataset_name}:")
        print(
            f"  Best config: {best['preprocessing']} + {best['distance']} + {best['linkage']}"
        )
        print(f"  ARI: {best['ari']:.4f}")
        print(f"  NMI: {best['nmi']:.4f}")
        print(f"  Clusters: {best['n_clusters']} (true: {best['n_true_clusters']})")

    print(f"\n\nResults saved to: {results_file}")

    # =========================================================================
    # Create visualization with best configs
    # =========================================================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS...")
    print("=" * 70)

    try:
        import umap

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Penguins visualization
        best_peng = best_configs["Penguins"]
        X_peng_proc = [
            p for p in preproc_continuous if p["name"] == best_peng["preprocessing"]
        ][0]["transform"](X_penguins)
        labels_peng, _ = run_kl_clustering(
            X_peng_proc,
            distance_metric=best_peng["distance"],
            linkage_method=best_peng["linkage"],
            verbose=False,
        )
        plot_umap_embedding(
            X_penguins, y_penguins, labels_peng, "Penguins", axes[0, 0], axes[0, 1]
        )

        # Digits visualization
        best_dig = best_configs["Digits"]
        X_dig_proc = [
            p for p in preproc_image if p["name"] == best_dig["preprocessing"]
        ][0]["transform"](X_digits)
        labels_dig, _ = run_kl_clustering(
            X_dig_proc,
            distance_metric=best_dig["distance"],
            linkage_method=best_dig["linkage"],
            verbose=False,
        )
        plot_umap_embedding(
            X_digits, y_digits, labels_dig, "Digits", axes[1, 0], axes[1, 1]
        )

        plt.tight_layout()

        # Save figure
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plots_dir / f"umap_datasets_benchmark_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {fig_path}")
        plt.show()

        # Create interactive Bokeh plot for digits with hover tooltips
        print("\n  Creating interactive Bokeh plot with digit hover...")
        bokeh_path = plots_dir / f"digits_interactive_{timestamp}.html"
        create_bokeh_digits_plot(
            X=X_digits,
            y=y_digits,
            labels=labels_dig,
            images=digit_images,
            output_path=bokeh_path,
        )

    except ImportError:
        print("UMAP not installed - skipping visualization")
        print("Install with: pip install umap-learn")

    # =========================================================================
    # Print full results table
    # =========================================================================
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)

    # Sort by dataset then ARI descending
    df_sorted = df_all.sort_values(["dataset", "ari"], ascending=[True, False])
    print(df_sorted.to_string(index=False))

    return df_all


if __name__ == "__main__":
    df_results = main()
