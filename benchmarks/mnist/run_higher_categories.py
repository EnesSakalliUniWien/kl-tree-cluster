"""
MNIST Higher-Category Benchmark for KL Divergence Clustering.

Runs the existing MNIST KL clustering pipeline and maps digit labels
(0-9) into coarser, higher-level categories.

Usage:
    python benchmarks/mnist/run_higher_categories.py
    python benchmarks/mnist/run_higher_categories.py --scheme parity_2
"""

from __future__ import annotations

import argparse
import base64
import sys
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
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from run import load_mnist_subset, run_kl_clustering


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


def main() -> None:
    arguments = parse_arguments()

    print("=" * 72)
    print("MNIST Higher-Category Benchmark")
    print("=" * 72)
    print(f"scheme={arguments.scheme}")
    print(
        f"clustering={arguments.distance_metric}+{arguments.linkage_method}, "
        f"binarize_threshold={arguments.binarize_threshold}"
    )

    feature_matrix, digit_labels = load_mnist_subset(
        n_samples=arguments.n_samples,
        seed=arguments.seed,
        use_pca=False,
    )

    cluster_labels = run_kl_clustering(
        feature_matrix,
        verbose=True,
        binarize_threshold=arguments.binarize_threshold,
        distance_metric=arguments.distance_metric,
        linkage_method=arguments.linkage_method,
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

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"n_samples: {arguments.n_samples}")
    print(f"n_pred_clusters: {len(np.unique(cluster_labels))}")
    print(f"ARI vs higher categories: {ari_against_higher_categories:.4f}")
    print(f"NMI vs higher categories: {nmi_against_higher_categories:.4f}")
    print(f"Majority-vote higher-category accuracy: {majority_vote_accuracy:.4f}")

    print("\nHigher-category distribution:")
    for category_id, category_name in enumerate(higher_category_names):
        category_count = int((true_higher_category_ids == category_id).sum())
        print(f"  {category_name:12s}: {category_count}")

    print("\nTrue vs predicted higher-category confusion matrix:")
    print(confusion_matrix.to_string())

    print(f"\nSaved assignments: {output_csv_file}")

    if arguments.with_umap_html:
        create_bokeh_higher_category_plot(
            feature_matrix=feature_matrix,
            digit_labels=digit_labels,
            cluster_labels=cluster_labels,
            true_higher_category_names=output_dataframe["true_higher_category"].to_numpy(),
            predicted_higher_category_names=output_dataframe["pred_higher_category"].to_numpy(),
            output_path=output_html_file,
        )


if __name__ == "__main__":
    main()
