#!/usr/bin/env python3
"""
Comprehensive profiling script for KL-divergence clustering algorithm.

This script profiles the clustering pipeline across different dataset sizes
and generates detailed performance reports and visualizations.
"""

import cProfile
import pstats
import io
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import tracemalloc
import psutil
import os
from contextlib import contextmanager

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    current, peak = tracemalloc.get_traced_memory()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    tracemalloc.stop()

    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")


def profile_clustering_performance():
    """Profile clustering algorithm across different dataset sizes."""

    # Test configurations
    test_configs = [
        {"n_samples": 30, "n_features": 30, "n_clusters": 3, "name": "Small"},
        {"n_samples": 50, "n_features": 50, "n_clusters": 3, "name": "Medium"},
        {"n_samples": 100, "n_features": 100, "n_clusters": 5, "name": "Large"},
        {"n_samples": 200, "n_features": 200, "n_clusters": 8, "name": "XL"},
    ]

    results = []

    for config in test_configs:
        print(f"\n{'=' * 60}")
        print(
            f"Profiling {config['name']} dataset: {config['n_samples']} samples, "
            f"{config['n_features']} features, {config['n_clusters']} clusters"
        )
        print(f"{'=' * 60}")

        # Create test case
        test_case = {
            "n_samples": config["n_samples"],
            "n_features": config["n_features"],
            "n_clusters": config["n_clusters"],
            "cluster_std": 1.0,
            "seed": 42,
        }

        # Profile the clustering
        pr = cProfile.Profile()

        with memory_monitor():
            start_time = time.time()
            pr.enable()

            try:
                from tests.test_cluster_validation import validate_cluster_algorithm

                df_results, fig = validate_cluster_algorithm(
                    test_cases=[test_case],
                    correlation_threshold=0.1000,
                    significance_level=0.05,
                    verbose=False,
                    plot_umap=False,
                )

                pr.disable()
                end_time = time.time()

                # Get profiling stats
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
                ps.print_stats(20)
                profile_output = s.getvalue()

                # Extract key metrics
                total_time = end_time - start_time
                clusters_found = (
                    df_results.iloc[0]["Found"] if not df_results.empty else 0
                )
                ari_score = df_results.iloc[0]["ARI"] if not df_results.empty else 0
                nmi_score = df_results.iloc[0]["NMI"] if not df_results.empty else 0

                # Extract function call counts
                lines = profile_output.split("\n")
                function_calls = {}
                for line in lines:
                    if line.strip() and not line.startswith(" "):
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                ncalls = int(
                                    parts[0].split("/")[0]
                                )  # Handle recursive calls
                                function_name = " ".join(parts[5:])
                                function_calls[function_name] = ncalls
                            except (ValueError, IndexError):
                                continue

                results.append(
                    {
                        "config": config["name"],
                        "n_samples": config["n_samples"],
                        "n_features": config["n_features"],
                        "n_clusters": config["n_clusters"],
                        "total_time": total_time,
                        "clusters_found": clusters_found,
                        "ari_score": ari_score,
                        "nmi_score": nmi_score,
                        "function_calls": function_calls,
                        "profile_output": profile_output,
                    }
                )

                print(".3f")
                print(f"Clusters found: {clusters_found}")
                print(".3f")
                print(".3f")

            except Exception as e:
                pr.disable()
                print(f"Error profiling {config['name']}: {e}")
                results.append({"config": config["name"], "error": str(e)})

    return results


def create_performance_plots(results):
    """Create performance visualization plots."""

    # Filter out failed runs
    successful_results = [r for r in results if "error" not in r]

    if not successful_results:
        print("No successful profiling results to plot")
        return

    # Prepare data for plotting
    plot_data = []
    for result in successful_results:
        plot_data.append(
            {
                "Dataset Size": result["config"],
                "Samples": result["n_samples"],
                "Total Time (s)": result["total_time"],
                "Clusters Found": result["clusters_found"],
                "ARI Score": result["ari_score"],
                "NMI Score": result["nmi_score"],
            }
        )

    df_plot = pd.DataFrame(plot_data)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "KL-Divergence Clustering Performance Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Time vs Dataset Size
    axes[0, 0].plot(
        df_plot["Samples"],
        df_plot["Total Time (s)"],
        "o-",
        linewidth=3,
        markersize=10,
        color="#2E86AB",
    )
    axes[0, 0].set_xlabel("Number of Samples", fontsize=12)
    axes[0, 0].set_ylabel("Total Time (seconds)", fontsize=12)
    axes[0, 0].set_title(
        "Computation Time vs Dataset Size", fontsize=14, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3)

    # Add sample size labels
    for i, row in df_plot.iterrows():
        axes[0, 0].annotate(
            f"{row['Dataset Size']}\n({row['Samples']} samples)",
            (row["Samples"], row["Total Time (s)"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    # 2. Quality Metrics
    x = np.arange(len(df_plot))
    width = 0.35

    axes[0, 1].bar(
        x - width / 2,
        df_plot["ARI Score"],
        width,
        label="ARI",
        color="#A23B72",
        alpha=0.8,
    )
    axes[0, 1].bar(
        x + width / 2,
        df_plot["NMI Score"],
        width,
        label="NMI",
        color="#F18F01",
        alpha=0.8,
    )

    axes[0, 1].set_xlabel("Dataset Configuration", fontsize=12)
    axes[0, 1].set_ylabel("Score", fontsize=12)
    axes[0, 1].set_title("Clustering Quality Metrics", fontsize=14, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_plot["Dataset Size"])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Time per Sample
    time_per_sample = df_plot["Total Time (s)"] / df_plot["Samples"]
    axes[1, 0].plot(
        df_plot["Samples"],
        time_per_sample,
        "s-",
        linewidth=3,
        markersize=10,
        color="#C73E1D",
    )
    axes[1, 0].set_xlabel("Number of Samples", fontsize=12)
    axes[1, 0].set_ylabel("Time per Sample (seconds)", fontsize=12)
    axes[1, 0].set_title("Scaling Efficiency", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Function Call Analysis (if available)
    if successful_results and "function_calls" in successful_results[0]:
        # Get top 10 most called functions across all runs
        all_functions = {}
        for result in successful_results:
            if "function_calls" in result:
                for func, calls in result["function_calls"].items():
                    if func in all_functions:
                        all_functions[func] += calls
                    else:
                        all_functions[func] = calls

        # Sort and get top 10
        sorted_functions = sorted(
            all_functions.items(), key=lambda x: x[1], reverse=True
        )[:10]

        if sorted_functions:
            funcs, calls = zip(*sorted_functions)
            y_pos = np.arange(len(funcs))

            axes[1, 1].barh(y_pos, calls, color="#6B73A6", alpha=0.7)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(
                [f[:50] + "..." if len(f) > 50 else f for f in funcs]
            )
            axes[1, 1].set_xlabel("Number of Calls", fontsize=12)
            axes[1, 1].set_title(
                "Most Called Functions", fontsize=14, fontweight="bold"
            )
            axes[1, 1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def save_detailed_report(results, timestamp):
    """Save detailed profiling report to file."""

    report_path = f"profiling_report_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("KL-DIVERGENCE CLUSTERING ALGORITHM PROFILING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in results:
            if "error" in result:
                f.write(f"ERROR in {result['config']}: {result['error']}\n\n")
                continue

            f.write(
                f"DATASET: {result['config']} "
                f"({result['n_samples']} samples, {result['n_features']} features, "
                f"{result['n_clusters']} clusters)\n"
            )
            f.write("-" * 60 + "\n")
            f.write(f"Total Time: {result['total_time']:.3f} seconds\n")
            f.write(f"Clusters Found: {result['clusters_found']}\n")
            f.write(f"ARI Score: {result['ari_score']:.3f}\n")
            f.write(f"NMI Score: {result['nmi_score']:.3f}\n")
            f.write("\nDETAILED PROFILING OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(result["profile_output"])
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Detailed report saved to: {report_path}")
    return report_path


def main():
    """Main profiling execution."""
    print("Starting comprehensive profiling of KL-divergence clustering algorithm...")
    print("This may take several minutes depending on dataset sizes...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run profiling
    results = profile_clustering_performance()

    # Create and save plots
    fig = create_performance_plots(results)
    if fig:
        plot_path = f"profiling_plots_{timestamp}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Performance plots saved to: {plot_path}")
        plt.close(fig)

    # Save detailed report
    report_path = save_detailed_report(results, timestamp)

    # Summary
    successful_runs = len([r for r in results if "error" not in r])
    print("\nProfiling complete!")
    print(f"Successful runs: {successful_runs}/{len(results)}")
    print(f"Report: {report_path}")
    if fig:
        print(f"Plots: {plot_path}")


if __name__ == "__main__":
    main()
