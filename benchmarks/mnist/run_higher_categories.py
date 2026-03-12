"""
MNIST Higher-Category Benchmark for KL Divergence Clustering.

Runs the existing MNIST KL clustering pipeline and maps digit labels
(0-9) into coarser, higher-level categories.

Usage:
    python benchmarks/mnist/run_higher_categories.py
    python benchmarks/mnist/run_higher_categories.py --scheme parity_2
    python benchmarks/mnist/run_higher_categories.py --pixel-categorization bayesian_gmm
"""

from __future__ import annotations

import argparse
import base64
import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Load shared path bootstrap helper from benchmarks root.
_script_path = Path(__file__).resolve()
_benchmarks_root = (
    _script_path.parent if _script_path.parent.name == "benchmarks" else _script_path.parents[1]
)
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))
from _bootstrap import ensure_repo_root_on_path

repo_root = ensure_repo_root_on_path(__file__)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import BayesianGaussianMixture

from run import load_mnist_subset, run_kl_clustering
from kl_clustering_analysis.tree.poset_tree import PosetTree


HIGHER_CATEGORY_SCHEMES: dict[str, dict[int, str]] = {
    "shape_3": {
        0: "looped",
        1: "linear",
        2: "arc_mixed",
        3: "arc_mixed",
        4: "linear",
        5: "arc_mixed",
        6: "looped",
        7: "linear",
        8: "looped",
        9: "looped",
    },
    "parity_2": {
        0: "even",
        1: "odd",
        2: "even",
        3: "odd",
        4: "even",
        5: "odd",
        6: "even",
        7: "odd",
        8: "even",
        9: "odd",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST higher-category benchmark")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of MNIST samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scheme",
        type=str,
        default="shape_3",
        choices=sorted(HIGHER_CATEGORY_SCHEMES.keys()),
        help="Higher-category mapping scheme",
    )
    parser.add_argument(
        "--binarize-threshold",
        type=float,
        default=0.0,
        help="Binarization threshold passed to run_kl_clustering",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="rogerstanimoto",
        help="Distance metric used for linkage",
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="weighted",
        help="Linkage method",
    )
    parser.add_argument(
        "--with-umap-html",
        action="store_true",
        help="Create interactive UMAP HTML visualization",
    )
    parser.add_argument(
        "--with-umap-html-3d",
        action="store_true",
        help="Create interactive 3D UMAP HTML visualization",
    )
    parser.add_argument(
        "--pixel-categorization",
        type=str,
        default="binary_threshold",
        choices=["binary_threshold", "bayesian_gmm"],
        help="Per-pixel categorization strategy before tree decomposition",
    )
    parser.add_argument(
        "--bgm-max-components",
        type=int,
        default=8,
        help="Max mixture components per pixel for BayesianGaussianMixture",
    )
    parser.add_argument(
        "--bgm-weight-threshold",
        type=float,
        default=1e-3,
        help="Component weight threshold used to count active categories",
    )
    parser.add_argument(
        "--bgm-fit-samples",
        type=int,
        default=1500,
        help="Subsample size used to fit per-pixel mixture models (0 uses all samples)",
    )
    parser.add_argument(
        "--bgm-max-iter",
        type=int,
        default=400,
        help="Max iterations for per-pixel BayesianGaussianMixture fits",
    )
    return parser.parse_args()


def encode_higher_categories(
    digit_labels: np.ndarray,
    digit_to_category_name: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    category_names = sorted(set(digit_to_category_name.values()))
    category_name_to_id = {category_name: category_id for category_id, category_name in enumerate(category_names)}
    category_ids = np.array(
        [category_name_to_id[digit_to_category_name[int(digit_label)]] for digit_label in digit_labels],
        dtype=int,
    )
    return category_ids, category_names


def map_clusters_to_categories(
    cluster_labels: np.ndarray,
    true_higher_category_ids: np.ndarray,
) -> dict[int, int]:
    cluster_to_higher_category_id: dict[int, int] = {}
    for cluster_id in np.unique(cluster_labels):
        is_cluster_member = cluster_labels == cluster_id
        if is_cluster_member.sum() == 0:
            continue
        category_ids_in_cluster, category_counts = np.unique(
            true_higher_category_ids[is_cluster_member], return_counts=True
        )
        dominant_category_id = int(category_ids_in_cluster[np.argmax(category_counts)])
        cluster_to_higher_category_id[int(cluster_id)] = dominant_category_id
    return cluster_to_higher_category_id


def run_tree_decomposition_on_preprocessed_data(
    preprocessed_matrix: np.ndarray,
    *,
    distance_metric: str,
    linkage_method: str,
    significance_level: float,
    verbose: bool,
) -> np.ndarray:
    """Run tree decomposition from a preprocessed integer feature matrix."""
    sample_names = [f"Sample_{sample_index}" for sample_index in range(preprocessed_matrix.shape[0])]
    annotations_df = pd.DataFrame(preprocessed_matrix, index=sample_names)

    if verbose:
        print(f"  Building hierarchy with {distance_metric} + {linkage_method}")

    linkage_matrix = linkage(
        pdist(annotations_df.values, metric=distance_metric),
        method=linkage_method,
    )
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=sample_names)
    decomposition_results = tree.decompose(
        leaf_data=annotations_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    cluster_assignments = decomposition_results.get("cluster_assignments", {})
    label_map: dict[str, int] = {}
    for cluster_identifier, cluster_metadata in cluster_assignments.items():
        for leaf_name in cluster_metadata["leaves"]:
            label_map[leaf_name] = int(cluster_identifier)
    return np.array([label_map.get(sample_name, -1) for sample_name in sample_names], dtype=int)


def _fit_bayesian_gmm_for_pixel(
    pixel_values_all_samples: np.ndarray,
    pixel_values_fit_subset: np.ndarray,
    *,
    random_seed: int,
    max_components: int,
    max_iter: int,
    weight_threshold: float,
) -> tuple[np.ndarray, int, list[float], list[float]]:
    """Fit Bayesian GMM on one pixel and return ordinal category IDs."""
    if np.allclose(pixel_values_fit_subset, pixel_values_fit_subset[0]):
        constant_category_ids = np.zeros(pixel_values_all_samples.shape[0], dtype=np.int16)
        constant_value = float(pixel_values_fit_subset[0, 0])
        return constant_category_ids, 1, [1.0], [constant_value]

    model = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type="dirichlet_process",
        covariance_type="full",
        random_state=random_seed,
        max_iter=max_iter,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(pixel_values_fit_subset)

    component_weights = np.asarray(model.weights_, dtype=float)
    component_means = np.asarray(model.means_, dtype=float).reshape(-1)

    active_component_indices = np.flatnonzero(component_weights > weight_threshold)
    if active_component_indices.size == 0:
        active_component_indices = np.array([int(np.argmax(component_weights))], dtype=int)

    sorted_active_component_indices = active_component_indices[
        np.argsort(component_means[active_component_indices])
    ]

    predicted_component_indices = model.predict(pixel_values_all_samples)
    active_component_set = set(int(index) for index in sorted_active_component_indices.tolist())
    if len(active_component_set) < component_weights.shape[0]:
        component_means_active = component_means[sorted_active_component_indices]
        for sample_index, predicted_component in enumerate(predicted_component_indices.tolist()):
            if int(predicted_component) in active_component_set:
                continue
            nearest_active_index = int(
                sorted_active_component_indices[
                    int(np.argmin(np.abs(component_means_active - component_means[int(predicted_component)])))
                ]
            )
            predicted_component_indices[sample_index] = nearest_active_index

    component_index_to_category_id = np.full(component_weights.shape[0], fill_value=-1, dtype=np.int16)
    for category_id, component_index in enumerate(sorted_active_component_indices.tolist()):
        component_index_to_category_id[int(component_index)] = int(category_id)

    category_ids = component_index_to_category_id[predicted_component_indices]
    if (category_ids < 0).any():
        category_ids = np.maximum(category_ids, 0)

    return (
        category_ids.astype(np.int16, copy=False),
        int(sorted_active_component_indices.size),
        [float(component_weights[index]) for index in sorted_active_component_indices.tolist()],
        [float(component_means[index]) for index in sorted_active_component_indices.tolist()],
    )


def infer_pixel_categories_with_bayesian_gmm(
    feature_matrix: np.ndarray,
    *,
    random_seed: int,
    max_components: int,
    weight_threshold: float,
    fit_samples: int,
    max_iter: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Infer per-pixel category IDs and category-count summary via Bayesian GMM."""
    n_samples, n_pixels = feature_matrix.shape
    random_generator = np.random.default_rng(random_seed)

    if fit_samples > 0 and fit_samples < n_samples:
        fit_sample_indices = np.sort(random_generator.choice(n_samples, size=fit_samples, replace=False))
    else:
        fit_sample_indices = np.arange(n_samples)

    categorical_feature_matrix = np.zeros((n_samples, n_pixels), dtype=np.int16)
    summary_rows: list[dict[str, object]] = []

    print(
        "Inferring per-pixel categories with BayesianGaussianMixture "
        f"(max_components={max_components}, fit_samples={len(fit_sample_indices)})..."
    )

    for pixel_index in range(n_pixels):
        if pixel_index % 64 == 0 or pixel_index == n_pixels - 1:
            print(f"  Pixel {pixel_index + 1}/{n_pixels}")

        pixel_values_all_samples = feature_matrix[:, pixel_index].reshape(-1, 1)
        pixel_values_fit_subset = pixel_values_all_samples[fit_sample_indices]

        (
            category_ids,
            inferred_category_count,
            active_component_weights,
            active_component_means,
        ) = _fit_bayesian_gmm_for_pixel(
            pixel_values_all_samples=pixel_values_all_samples,
            pixel_values_fit_subset=pixel_values_fit_subset,
            random_seed=random_seed,
            max_components=max_components,
            max_iter=max_iter,
            weight_threshold=weight_threshold,
        )

        categorical_feature_matrix[:, pixel_index] = category_ids
        summary_rows.append(
            {
                "pixel_index": pixel_index,
                "inferred_category_count": inferred_category_count,
                "realized_category_count": int(np.unique(category_ids).size),
                "active_component_weights": ";".join(f"{weight:.6f}" for weight in active_component_weights),
                "active_component_means": ";".join(f"{mean:.6f}" for mean in active_component_means),
            }
        )

    category_summary_dataframe = pd.DataFrame(summary_rows)
    return categorical_feature_matrix, category_summary_dataframe


def expand_pixel_categories_to_one_hot(
    categorical_feature_matrix: np.ndarray,
) -> np.ndarray:
    """Expand per-pixel category IDs to binary one-hot indicator features."""
    n_samples, n_pixels = categorical_feature_matrix.shape
    one_hot_feature_blocks: list[np.ndarray] = []

    for pixel_index in range(n_pixels):
        pixel_category_ids = categorical_feature_matrix[:, pixel_index].astype(int, copy=False)
        n_pixel_categories = int(pixel_category_ids.max()) + 1
        one_hot_block = np.zeros((n_samples, n_pixel_categories), dtype=np.int8)
        one_hot_block[np.arange(n_samples), pixel_category_ids] = 1
        one_hot_feature_blocks.append(one_hot_block)

    return np.concatenate(one_hot_feature_blocks, axis=1)


def create_bokeh_higher_category_plot(
    feature_matrix: np.ndarray,
    digit_labels: np.ndarray,
    cluster_labels: np.ndarray,
    true_higher_category_names: np.ndarray,
    predicted_higher_category_names: np.ndarray,
    output_path: Path,
) -> None:
    """Create interactive UMAP HTML with image hover and higher-category labels."""
    try:
        import umap
        from PIL import Image
        from bokeh.layouts import row
        from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
        from bokeh.palettes import Category10, Category20
        from bokeh.plotting import figure, output_file, save
        from bokeh.resources import INLINE
    except ImportError as import_error:
        print(f"Skipping UMAP HTML output: missing dependency ({import_error})")
        return

    def select_palette(category_count: int) -> list[str]:
        if category_count <= 10:
            return list(Category10[10][:category_count])
        if category_count <= 20:
            return list(Category20[20][:category_count])
        return list((Category20[20] * ((category_count // 20) + 1))[:category_count])

    def encode_digit_image(flattened_pixels: np.ndarray) -> str:
        feature_count = flattened_pixels.shape[0]
        image_side = int(np.sqrt(feature_count))
        if image_side * image_side != feature_count:
            image_matrix = np.expand_dims(flattened_pixels, axis=0)
        else:
            image_matrix = flattened_pixels.reshape(image_side, image_side)
        image_matrix = (255.0 * (1.0 - np.clip(image_matrix, 0.0, 1.0))).astype(np.uint8)
        image = Image.fromarray(image_matrix, mode="L").resize((64, 64), Image.Resampling.BICUBIC)
        image_buffer = BytesIO()
        image.save(image_buffer, format="png")
        return "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode()

    print("\nCreating interactive UMAP HTML...")
    umap_reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = umap_reducer.fit_transform(feature_matrix)

    embedded_images = [encode_digit_image(row) for row in feature_matrix]

    plot_dataframe = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "digit": [str(digit_label) for digit_label in digit_labels],
            "cluster": [str(cluster_label) for cluster_label in cluster_labels],
            "true_higher_category": [str(name) for name in true_higher_category_names],
            "pred_higher_category": [str(name) for name in predicted_higher_category_names],
            "image": embedded_images,
        }
    )
    data_source = ColumnDataSource(plot_dataframe)

    true_factors = sorted(plot_dataframe["true_higher_category"].unique().tolist())
    pred_factors = sorted(plot_dataframe["pred_higher_category"].unique().tolist())
    cluster_factors = sorted(plot_dataframe["cluster"].unique().tolist(), key=lambda value: int(value))

    true_color_mapper = CategoricalColorMapper(
        factors=true_factors,
        palette=select_palette(len(true_factors)),
    )
    pred_color_mapper = CategoricalColorMapper(
        factors=pred_factors,
        palette=select_palette(len(pred_factors)),
    )
    cluster_color_mapper = CategoricalColorMapper(
        factors=cluster_factors,
        palette=select_palette(len(cluster_factors)),
    )

    true_plot = figure(
        title="UMAP - True Higher Categories",
        width=700,
        height=620,
        tools=("pan,wheel_zoom,reset"),
    )
    true_plot.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div><img src='@image' style='float: left; margin: 5px 5px 5px 5px'/></div>
        <div><span style='font-size: 16px; color: #224499'>True Higher Category:</span>
             <span style='font-size: 16px'>@true_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>Pred Higher Category:</span>
             <span style='font-size: 14px'>@pred_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>Digit:</span>
             <span style='font-size: 14px'>@digit</span></div>
        <div><span style='font-size: 14px; color: #666'>Cluster:</span>
             <span style='font-size: 14px'>@cluster</span></div>
    </div>
    """
        )
    )
    true_plot.scatter(
        "x",
        "y",
        source=data_source,
        color={"field": "true_higher_category", "transform": true_color_mapper},
        line_alpha=0.6,
        fill_alpha=0.65,
        size=5,
    )

    pred_plot = figure(
        title="UMAP - Predicted Higher Categories",
        width=700,
        height=620,
        tools=("pan,wheel_zoom,reset"),
        x_range=true_plot.x_range,
        y_range=true_plot.y_range,
    )
    pred_plot.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div><img src='@image' style='float: left; margin: 5px 5px 5px 5px'/></div>
        <div><span style='font-size: 16px; color: #224499'>Pred Higher Category:</span>
             <span style='font-size: 16px'>@pred_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>True Higher Category:</span>
             <span style='font-size: 14px'>@true_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>Digit:</span>
             <span style='font-size: 14px'>@digit</span></div>
        <div><span style='font-size: 14px; color: #666'>Cluster:</span>
             <span style='font-size: 14px'>@cluster</span></div>
    </div>
    """
        )
    )
    pred_plot.scatter(
        "x",
        "y",
        source=data_source,
        color={"field": "pred_higher_category", "transform": pred_color_mapper},
        line_alpha=0.6,
        fill_alpha=0.65,
        size=5,
    )

    cluster_plot = figure(
        title=f"UMAP - Raw Tree Clusters ({len(cluster_factors)})",
        width=700,
        height=620,
        tools=("pan,wheel_zoom,reset"),
        x_range=true_plot.x_range,
        y_range=true_plot.y_range,
    )
    cluster_plot.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div><img src='@image' style='float: left; margin: 5px 5px 5px 5px'/></div>
        <div><span style='font-size: 16px; color: #224499'>Tree Cluster:</span>
             <span style='font-size: 16px'>@cluster</span></div>
        <div><span style='font-size: 14px; color: #666'>Pred Higher Category:</span>
             <span style='font-size: 14px'>@pred_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>True Higher Category:</span>
             <span style='font-size: 14px'>@true_higher_category</span></div>
        <div><span style='font-size: 14px; color: #666'>Digit:</span>
             <span style='font-size: 14px'>@digit</span></div>
    </div>
    """
        )
    )
    cluster_plot.scatter(
        "x",
        "y",
        source=data_source,
        color={"field": "cluster", "transform": cluster_color_mapper},
        line_alpha=0.6,
        fill_alpha=0.65,
        size=5,
    )

    output_file(
        str(output_path),
        title="MNIST Higher Categories - Interactive UMAP",
        mode="inline",
    )
    save(row(true_plot, pred_plot, cluster_plot), resources=INLINE)
    print(f"Saved interactive HTML: {output_path}")


def create_plotly_higher_category_plot_3d(
    feature_matrix: np.ndarray,
    digit_labels: np.ndarray,
    cluster_labels: np.ndarray,
    true_higher_category_names: np.ndarray,
    predicted_higher_category_names: np.ndarray,
    output_path: Path,
) -> None:
    """Create interactive 3D UMAP HTML with panels for true/predicted/cluster labels."""
    try:
        import umap
        import plotly.graph_objects as go
        import plotly.offline as plotly_offline
        from plotly.subplots import make_subplots
    except ImportError as import_error:
        print(f"Skipping 3D UMAP HTML output: missing dependency ({import_error})")
        return

    print("\nCreating interactive 3D UMAP HTML...")
    umap_reducer = umap.UMAP(
        n_components=3,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
    )
    embedding_3d = umap_reducer.fit_transform(feature_matrix)

    plot_dataframe = pd.DataFrame(
        {
            "x": embedding_3d[:, 0],
            "y": embedding_3d[:, 1],
            "z": embedding_3d[:, 2],
            "digit": [str(digit_label) for digit_label in digit_labels],
            "cluster": [str(cluster_label) for cluster_label in cluster_labels],
            "true_higher_category": [str(name) for name in true_higher_category_names],
            "pred_higher_category": [str(name) for name in predicted_higher_category_names],
        }
    )

    def palette_for_count(category_count: int) -> list[str]:
        base_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#393b79",
            "#637939",
            "#8c6d31",
            "#843c39",
            "#7b4173",
            "#3182bd",
            "#31a354",
            "#756bb1",
            "#636363",
            "#e6550d",
        ]
        if category_count <= len(base_palette):
            return base_palette[:category_count]
        repeats = (category_count // len(base_palette)) + 1
        return (base_palette * repeats)[:category_count]

    def add_colored_panel(
        figure: go.Figure,
        *,
        row: int,
        col: int,
        color_column: str,
        panel_title: str,
    ) -> None:
        factors = sorted(plot_dataframe[color_column].unique().tolist(), key=str)
        panel_palette = palette_for_count(len(factors))
        color_by_factor = {factor: panel_palette[index] for index, factor in enumerate(factors)}

        for factor in factors:
            factor_mask = plot_dataframe[color_column] == factor
            panel_points = plot_dataframe[factor_mask]
            hover_text = [
                (
                    f"Digit: {digit}<br>"
                    f"Tree Cluster: {cluster}<br>"
                    f"True Higher Category: {true_category}<br>"
                    f"Pred Higher Category: {pred_category}"
                )
                for digit, cluster, true_category, pred_category in zip(
                    panel_points["digit"],
                    panel_points["cluster"],
                    panel_points["true_higher_category"],
                    panel_points["pred_higher_category"],
                )
            ]
            figure.add_trace(
                go.Scatter3d(
                    x=panel_points["x"],
                    y=panel_points["y"],
                    z=panel_points["z"],
                    mode="markers",
                    name=factor,
                    legendgroup=f"{color_column}_{factor}",
                    showlegend=(col == 1),
                    marker={
                        "size": 3,
                        "opacity": 0.78,
                        "color": color_by_factor[factor],
                    },
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text,
                ),
                row=row,
                col=col,
            )

        scene_key = "scene" if col == 1 else f"scene{col}"
        figure.update_layout(
            **{
                scene_key: {
                    "xaxis_title": "UMAP-1",
                    "yaxis_title": "UMAP-2",
                    "zaxis_title": "UMAP-3",
                }
            }
        )
        figure.layout.annotations[col - 1].text = panel_title

    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "3D UMAP - True Higher Categories",
            "3D UMAP - Predicted Higher Categories",
            f"3D UMAP - Raw Tree Clusters ({plot_dataframe['cluster'].nunique()})",
        ),
        horizontal_spacing=0.02,
    )

    add_colored_panel(
        figure,
        row=1,
        col=1,
        color_column="true_higher_category",
        panel_title="3D UMAP - True Higher Categories",
    )
    add_colored_panel(
        figure,
        row=1,
        col=2,
        color_column="pred_higher_category",
        panel_title="3D UMAP - Predicted Higher Categories",
    )
    add_colored_panel(
        figure,
        row=1,
        col=3,
        color_column="cluster",
        panel_title=f"3D UMAP - Raw Tree Clusters ({plot_dataframe['cluster'].nunique()})",
    )

    figure.update_layout(
        title="MNIST Higher Categories - Interactive 3D UMAP",
        width=2100,
        height=720,
        legend_title="Panel 1 Categories",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )

    plotly_offline.plot(figure, filename=str(output_path), auto_open=False, include_plotlyjs="inline")
    print(f"Saved interactive 3D HTML: {output_path}")


def main() -> None:
    arguments = parse_arguments()

    print("=" * 72)
    print("MNIST Higher-Category Benchmark")
    print("=" * 72)
    print(f"scheme={arguments.scheme}")
    print(f"pixel_categorization={arguments.pixel_categorization}")
    print(
        f"clustering={arguments.distance_metric}+{arguments.linkage_method}, "
        f"binarize_threshold={arguments.binarize_threshold}"
    )

    feature_matrix, digit_labels = load_mnist_subset(
        n_samples=arguments.n_samples,
        seed=arguments.seed,
        use_pca=False,
    )

    pixel_category_summary_dataframe: pd.DataFrame | None = None
    clustering_distance_metric = arguments.distance_metric

    if arguments.pixel_categorization == "binary_threshold":
        cluster_labels = run_kl_clustering(
            feature_matrix,
            verbose=True,
            binarize_threshold=arguments.binarize_threshold,
            distance_metric=arguments.distance_metric,
            linkage_method=arguments.linkage_method,
        )
    else:
        categorical_feature_matrix, pixel_category_summary_dataframe = infer_pixel_categories_with_bayesian_gmm(
            feature_matrix,
            random_seed=arguments.seed,
            max_components=arguments.bgm_max_components,
            weight_threshold=arguments.bgm_weight_threshold,
            fit_samples=arguments.bgm_fit_samples,
            max_iter=arguments.bgm_max_iter,
        )

        one_hot_feature_matrix = expand_pixel_categories_to_one_hot(categorical_feature_matrix)
        print(
            "Expanded pixel categories to one-hot binary features: "
            f"{categorical_feature_matrix.shape[1]} pixels -> {one_hot_feature_matrix.shape[1]} features"
        )

        cluster_labels = run_tree_decomposition_on_preprocessed_data(
            one_hot_feature_matrix,
            distance_metric=clustering_distance_metric,
            linkage_method=arguments.linkage_method,
            significance_level=0.05,
            verbose=True,
        )

    digit_to_category_name = HIGHER_CATEGORY_SCHEMES[arguments.scheme]
    true_higher_category_ids, higher_category_names = encode_higher_categories(
        digit_labels,
        digit_to_category_name,
    )

    ari_against_higher_categories = adjusted_rand_score(true_higher_category_ids, cluster_labels)
    nmi_against_higher_categories = normalized_mutual_info_score(
        true_higher_category_ids,
        cluster_labels,
    )

    cluster_to_higher_category_id = map_clusters_to_categories(
        cluster_labels,
        true_higher_category_ids,
    )
    predicted_higher_category_ids = np.array(
        [cluster_to_higher_category_id[int(cluster_label)] for cluster_label in cluster_labels],
        dtype=int,
    )
    majority_vote_accuracy = accuracy_score(
        true_higher_category_ids,
        predicted_higher_category_ids,
    )

    output_dataframe = pd.DataFrame(
        {
            "sample_id": [f"Sample_{sample_index}" for sample_index in range(arguments.n_samples)],
            "digit_label": digit_labels,
            "true_higher_category": [
                higher_category_names[category_id] for category_id in true_higher_category_ids
            ],
            "pred_cluster": cluster_labels,
            "pred_higher_category": [
                higher_category_names[category_id] for category_id in predicted_higher_category_ids
            ],
        }
    )

    confusion_matrix = pd.crosstab(
        output_dataframe["true_higher_category"],
        output_dataframe["pred_higher_category"],
        rownames=["true"],
        colnames=["predicted"],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = repo_root / "benchmarks" / "results"
    output_directory.mkdir(exist_ok=True)
    output_csv_file = output_directory / f"mnist_higher_categories_{arguments.scheme}_{timestamp}.csv"
    output_dataframe.to_csv(output_csv_file, index=False)
    output_html_file = output_directory / f"mnist_higher_categories_{arguments.scheme}_{timestamp}.html"
    output_html_3d_file = output_directory / f"mnist_higher_categories_{arguments.scheme}_{timestamp}_3d.html"
    pixel_summary_file = None
    if pixel_category_summary_dataframe is not None:
        pixel_summary_file = (
            output_directory / f"mnist_pixel_categories_bayesian_gmm_{timestamp}.csv"
        )
        pixel_category_summary_dataframe.to_csv(pixel_summary_file, index=False)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"n_samples: {arguments.n_samples}")
    print(f"n_pred_clusters: {len(np.unique(cluster_labels))}")
    print(f"ARI vs higher categories: {ari_against_higher_categories:.4f}")
    print(f"NMI vs higher categories: {nmi_against_higher_categories:.4f}")
    print(f"Majority-vote higher-category accuracy: {majority_vote_accuracy:.4f}")
    print(f"distance_metric_used: {clustering_distance_metric}")

    if pixel_category_summary_dataframe is not None:
        inferred_categories_per_pixel = pixel_category_summary_dataframe["inferred_category_count"]
        print(
            "Per-pixel inferred categories (Bayesian GMM): "
            f"min={int(inferred_categories_per_pixel.min())}, "
            f"median={float(inferred_categories_per_pixel.median()):.1f}, "
            f"mean={float(inferred_categories_per_pixel.mean()):.2f}, "
            f"max={int(inferred_categories_per_pixel.max())}"
        )

    print("\nHigher-category distribution:")
    for category_id, category_name in enumerate(higher_category_names):
        category_count = int((true_higher_category_ids == category_id).sum())
        print(f"  {category_name:12s}: {category_count}")

    print("\nTrue vs predicted higher-category confusion matrix:")
    print(confusion_matrix.to_string())

    print(f"\nSaved assignments: {output_csv_file}")
    if pixel_summary_file is not None:
        print(f"Saved per-pixel category summary: {pixel_summary_file}")

    if arguments.with_umap_html:
        create_bokeh_higher_category_plot(
            feature_matrix=feature_matrix,
            digit_labels=digit_labels,
            cluster_labels=cluster_labels,
            true_higher_category_names=output_dataframe["true_higher_category"].to_numpy(),
            predicted_higher_category_names=output_dataframe["pred_higher_category"].to_numpy(),
            output_path=output_html_file,
        )

    if arguments.with_umap_html_3d:
        create_plotly_higher_category_plot_3d(
            feature_matrix=feature_matrix,
            digit_labels=digit_labels,
            cluster_labels=cluster_labels,
            true_higher_category_names=output_dataframe["true_higher_category"].to_numpy(),
            predicted_higher_category_names=output_dataframe["pred_higher_category"].to_numpy(),
            output_path=output_html_3d_file,
        )


if __name__ == "__main__":
    main()
