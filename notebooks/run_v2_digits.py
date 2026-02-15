"""
V2 Benchmark on Sklearn Digits Dataset.
Tests the new signal localization logic and generates a Bokeh plot.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)


def load_sklearn_digits():
    print("Loading Sklearn Digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    images = digits.images
    return X, y, images


def binarize_data(X, threshold=0.0):
    return (X > threshold).astype(int)


def run_kl_clustering_v2(X, sample_names=None):
    if sample_names is None:
        sample_names = [f"S{i}" for i in range(X.shape[0])]

    data = pd.DataFrame(X, index=sample_names)

    # Build tree
    print("  Building tree (Jaccard + Complete)...")
    Z = linkage(pdist(data.values, metric="jaccard"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Populate stats
    tree.populate_node_divergences(data)

    from kl_clustering_analysis.hierarchy_analysis.statistics import (
        annotate_child_parent_divergence,
        annotate_sibling_divergence,
    )

    # Annotate and UPDATE stats_df
    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )

    # DEBUG: Inspect top nodes stats
    print("\n[DEBUG] Top Node Statistics:")
    cols = [
        "leaf_count",
        "Child_Parent_Divergence_P_Value",
        "Child_Parent_Divergence_P_Value_BH",
        "Child_Parent_Divergence_Significant",
        "Sibling_BH_Different",
        "Sibling_Divergence_P_Value",
    ]
    # Filter for nodes with high leaf count (top of tree)
    top_nodes = tree.stats_df[tree.stats_df["leaf_count"] > 1000].sort_values(
        "leaf_count", ascending=False
    )
    print(top_nodes[cols].head(10))

    # DEBUG: Deep Inspection
    print(f"\n[DEBUG] DF Index Unique? {tree.stats_df.index.is_unique}")
    if "N3592" in tree.stats_df.index:
        raw_val = tree.stats_df.loc["N3592", "Sibling_BH_Different"]
        print(f"[DEBUG] Raw DF val for N3592: {raw_val} (type {type(raw_val)})")
        if hasattr(raw_val, "values"):  # Series if duplicates
            print(f"[DEBUG] Values if duplicate: {raw_val.values}")

    # DEBUG: Inspect Branch Lengths
    print("\n[DEBUG] Branch Length Analysis:")
    edge_lens = [
        d["branch_length"] for u, v, d in tree.edges(data=True) if "branch_length" in d
    ]
    if edge_lens:
        mean_bl = np.mean(edge_lens)
        max_bl = np.max(edge_lens)
        print(f"  Mean Branch Length: {mean_bl:.4f}")
        print(f"  Max Branch Length:  {max_bl:.4f}")

        # Check root's children edges
        root = tree.root()
        for child in tree.successors(root):
            if tree.has_edge(root, child):
                bl = tree.edges[root, child].get("branch_length", "N/A")
                print(f"  Edge {root}->{child}: len={bl}")
    else:
        print("  No branch lengths found.")
    print("-" * 60)

    decomposer = TreeDecomposition(
        tree=tree,
        results_df=tree.stats_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
        use_signal_localization=True,  # Enable localization
        localization_max_depth=5,  # Set depth limit
    )

    print("  Running decompose_tree_v2()...")
    results = decomposer.decompose_tree_v2()

    # Localization stats
    loc_results = results.get("localization_results", {})
    soft_boundaries = sum(1 for r in loc_results.values() if r.has_soft_boundaries)
    print(f"  Split points with soft boundaries: {soft_boundaries}/{len(loc_results)}")

    cluster_assignments = results.get("cluster_assignments", {})
    label_map = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id

    labels = np.array([label_map.get(name, -1) for name in sample_names])
    return labels


def create_bokeh_digits_plot(X, y, labels, images, output_path):
    """Create interactive Bokeh plot with digit image hover tooltips."""
    try:
        import umap
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
        from bokeh.palettes import Spectral10, Category20
        from bokeh.layouts import row
        from io import BytesIO
        from PIL import Image
        import base64
    except ImportError as e:
        print(f"  Missing dependency for Bokeh visualization: {e}")
        return

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

    print("  Computing UMAP for plot...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    print("  Encoding images...")
    embedded_images = [embeddable_image(img) for img in images]

    digits_df = pd.DataFrame(embedding, columns=("x", "y"))
    digits_df["digit"] = [str(d) for d in y]
    digits_df["cluster"] = [str(c) for c in labels]
    digits_df["image"] = embedded_images

    datasource = ColumnDataSource(digits_df)

    # TRUE labels
    digit_color_mapping = CategoricalColorMapper(
        factors=[str(x) for x in range(10)], palette=Spectral10
    )

    # PREDICTED clusters
    unique_clusters = sorted(set(str(c) for c in labels))
    n_clusters = len(unique_clusters)
    if n_clusters <= 20:
        cluster_palette = Category20[max(3, min(20, n_clusters))]
    else:
        cluster_palette = (Category20[20] * ((n_clusters // 20) + 1))[:n_clusters]

    cluster_color_mapping = CategoricalColorMapper(
        factors=unique_clusters, palette=cluster_palette
    )

    # Plot True
    plot_true = figure(
        title="UMAP - True Digits",
        width=600,
        height=600,
        tools=("pan, wheel_zoom, reset"),
    )
    plot_true.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div><img src='@image' style='float: left; margin: 5px'/></div>
        <div><span style='font-size: 16px; color: #224499'>True:</span> <span style='font-size: 18px'>@digit</span></div>
        <div><span style='font-size: 14px; color: #666'>Cluster:</span> <span style='font-size: 14px'>@cluster</span></div>
    </div>
    """
        )
    )
    plot_true.scatter(
        "x",
        "y",
        source=datasource,
        color=dict(field="digit", transform=digit_color_mapping),
        alpha=0.6,
        size=5,
    )

    # Plot Predicted
    plot_pred = figure(
        title="UMAP - KL (v2) Clusters",
        width=600,
        height=600,
        tools=("pan, wheel_zoom, reset"),
        x_range=plot_true.x_range,
        y_range=plot_true.y_range,
    )
    plot_pred.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div><img src='@image' style='float: left; margin: 5px'/></div>
        <div><span style='font-size: 16px; color: #224499'>Cluster:</span> <span style='font-size: 18px'>@cluster</span></div>
        <div><span style='font-size: 14px; color: #666'>True:</span> <span style='font-size: 14px'>@digit</span></div>
    </div>
    """
        )
    )
    plot_pred.scatter(
        "x",
        "y",
        source=datasource,
        color=dict(field="cluster", transform=cluster_color_mapping),
        alpha=0.6,
        size=5,
    )

    layout = row(plot_true, plot_pred)
    output_file(str(output_path), title="Digits Clustering (V2) - Interactive")
    save(layout)
    print(f"  Interactive plot saved to: {output_path}")


def main():
    X, y, images = load_sklearn_digits()
    X_proc = binarize_data(X / 16.0, threshold=0.5)

    print(f"\nRunning V2 Benchmark on Digits ({len(X)} samples)...")
    try:
        labels = run_kl_clustering_v2(X_proc)

        mask = labels >= 0
        if mask.any():
            ari = adjusted_rand_score(y[mask], labels[mask])
            nmi = normalized_mutual_info_score(y[mask], labels[mask])
            n_clusters = len(np.unique(labels[mask]))
            print(f"\nResults:")
            print(f"  ARI: {ari:.4f}")
            print(f"  NMI: {nmi:.4f}")
            print(f"  Clusters Found: {n_clusters}")

            output_plot = Path(repo_root) / "digits_interactive_v2.html"
            create_bokeh_digits_plot(X, y, labels, images, output_plot)

        else:
            print("No clusters found.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
