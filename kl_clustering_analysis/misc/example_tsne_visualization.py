#!/usr/bin/env python3
"""
Example script demonstrating t-SNE visualization of clustering comparison.

This script shows how the local KL divergence clustering compares to
traditional methods like K-means and Spectral clustering using t-SNE
dimensionality reduction for visualization.
"""

from tests.test_cluster_validation import validate_cluster_algorithm


def main():
    """Run clustering comparison with visualization."""

    # Example test case: 6 clusters, moderate complexity
    test_cases = [
        {
            "n_samples": 60,
            "n_features": 60,
            "n_clusters": 6,
            "cluster_std": 1.8,
            "seed": 45,
        }
    ]

    print("🔬 Running KL Divergence Clustering with t-SNE Visualization")
    print("=" * 60)

    # Run validation with t-SNE plotting enabled
    df_results, _ = validate_cluster_algorithm(
        test_cases=test_cases,
        verbose=True,
        plot_umap=True,  # This creates t-SNE plots
    )

    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    result = df_results.iloc[0]
    success = "✅ SUCCESS" if result["True"] == result["Found"] else "❌ FAILED"

    print(f"Test Case: {int(result['Test'])}")
    print(f"Expected Clusters: {int(result['True'])}")
    print(f"Found Clusters: {int(result['Found'])}")
    print(f"Status: {success}")
    print(f"ARI Score: {result['ARI']:.3f}")
    print(f"NMI Score: {result['NMI']:.3f}")
    print(f"Purity Score: {result['Purity']:.3f}")

    print("\n📈 VISUALIZATION CREATED")
    print("t-SNE comparison plot saved as: tsne_comparison_test_1.png")
    print("\nThe plot shows 6 panels comparing:")
    print("• Ground Truth: True cluster labels")
    print("• KL Divergence: Our algorithm's results")
    print("• K-Means: Traditional centroid-based clustering")
    print("• Spectral: Graph-based clustering")
    print("• t-SNE X/Y: Raw dimensionality reduction (no clustering)")

    print("\n💡 INSIGHTS TO LOOK FOR:")
    print("• How well does KL clustering match the ground truth?")
    print("• How do different methods group the same data points?")
    print("• Are there patterns that one method captures better than others?")
    print("• How does the data look in the raw t-SNE projection?")


if __name__ == "__main__":
    main()
