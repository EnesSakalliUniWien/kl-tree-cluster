#!/usr/bin/env python3
"""
KL-TE Clustering Performance Profiler

A comprehensive profiling tool for the KL-TE clustering framework that identifies
and analyzes performance bottlenecks across the entire pipeline.

Usage
-----
    # Profile default benchmark case
    python debug_scripts/kl_clustering_profiler.py

    # Profile specific test case
    python debug_scripts/kl_clustering_profiler.py --case binary_perfect_4c

    # Profile with custom data size
    python debug_scripts/kl_clustering_profiler.py --n-samples 200 --n-features 150

    # Save detailed trace to file
    python debug_scripts/kl_clustering_profiler.py --output profiler_results.csv

    # Run with cProfile for function-level breakdown
    python debug_scripts/kl_clustering_profiler.py --profile-level function

Output
------
    - Console: Summary table with timing breakdown and optimization recommendations
    - CSV (optional): Detailed timing data for all components
    - PNG (optional): Visual timing breakdown chart

The profiler measures:
    1. Tree construction (linkage)
    2. Distribution population
    3. Spectral decomposition (eigendecomposition)
    4. Gate 2 annotation (child-parent divergence)
    5. Gate 3 annotation (sibling divergence)
    6. Tree decomposition traversal
    7. Total pipeline time
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.tree.poset_tree import PosetTree

# =============================================================================
# TIMING INFRASTRUCTURE
# =============================================================================


@dataclass
class TimingResult:
    """Stores timing information for a single component."""

    component: str
    description: str
    elapsed_sec: float
    call_count: int = 1
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_sec * 1000

    def percentage(self, total: float) -> float:
        if total <= 0:
            return 0.0
        return (self.elapsed_sec / total) * 100


class ProfilerTimer:
    """Context manager for timing code blocks."""

    def __init__(self, results: list[TimingResult], component: str, description: str):
        self.results = results
        self.component = component
        self.description = description
        self.start_time: float = 0.0
        self.result: Optional[TimingResult] = None

    def __enter__(self) -> "ProfilerTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.result = TimingResult(
            component=self.component,
            description=self.description,
            elapsed_sec=elapsed,
        )
        self.results.append(self.result)


class KLProfiler:
    """
    Comprehensive profiler for the KL-TE clustering pipeline.

    Profiles each major component and provides optimization recommendations.
    """

    def __init__(
        self,
        n_samples: int = 80,
        n_features: int = 100,
        n_clusters: int = 4,
        random_seed: int = 2001,
        output_path: Optional[str] = None,
        profile_level: str = "component",
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.output_path = output_path
        self.profile_level = profile_level

        self.timing_results: list[TimingResult] = []
        self.data: Optional[pd.DataFrame] = None
        self.tree: Optional[PosetTree] = None
        self.annotations_df: Optional[pd.DataFrame] = None
        self.decomposition_result: Optional[dict] = None

    @contextmanager
    def _time(self, component: str, description: str):
        """Context manager for timing code blocks."""
        timer = ProfilerTimer(self.timing_results, component, description)
        with timer:
            yield timer

    def _generate_data(self) -> pd.DataFrame:
        """Generate synthetic benchmark data."""
        with self._time("data_generation", "Generate synthetic feature matrix"):
            leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
                n_rows=self.n_samples,
                n_cols=self.n_features,
                entropy_param=0.05,
                n_clusters=self.n_clusters,
                random_seed=self.random_seed,
                balanced_clusters=True,
                feature_sparsity=0.05,
            )

            leaf_names = sorted(leaf_matrix_dict.keys())
            X = np.array([leaf_matrix_dict[n] for n in leaf_names], dtype=float)
            self.data = pd.DataFrame(
                X,
                index=leaf_names,
                columns=[f"F{j}" for j in range(X.shape[1])],
            )
            return self.data

    def _build_tree(self) -> PosetTree:
        """Build hierarchical clustering tree."""
        with self._time("tree_construction", "Compute pairwise distances and linkage"):
            distances = pdist(self.data.values, metric=config.TREE_DISTANCE_METRIC)
            Z = linkage(distances, method=config.TREE_LINKAGE_METHOD)
            self.tree = PosetTree.from_linkage(Z, leaf_names=self.data.index.tolist())
        return self.tree

    def _populate_distributions(self):
        """Populate node distributions in the tree."""
        with self._time(
            "distribution_population", "Compute node distributions (Bernoulli parameters)"
        ):
            populate_distributions(self.tree, self.data)
        return self.tree

    def _annotate_gate2(self):
        """Annotate tree with Gate 2 (child-parent divergence) statistics."""
        with self._time(
            "gate2_annotation",
            "Gate 2: Child-parent divergence (spectral + Wald tests)",
        ):
            # Prepare annotations DataFrame with required columns
            annotations_df = self._prepare_annotations_df()
            self.annotations_df = annotate_child_parent_divergence(
                tree=self.tree,
                annotations_df=annotations_df,
                significance_level_alpha=config.EDGE_ALPHA,
                leaf_data=self.data,
            )
        return self.annotations_df

    def _annotate_gate3(self):
        """Annotate tree with Gate 3 (sibling divergence) statistics."""
        with self._time(
            "gate3_annotation",
            "Gate 3: Sibling divergence (FDR-corrected tests)",
        ):
            # Gate 3 uses spectral dims and PCA from Gate 2 annotations
            # Extract these from the annotations DataFrame
            from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
                collect_parent_principal_component_inputs_for_sibling_tests,
            )
            from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
                derive_sibling_projection_dimensions_from_child_edge_comparisons,
            )

            spectral_dims = derive_sibling_projection_dimensions_from_child_edge_comparisons(self.tree, self.annotations_df)
            pca_projections, pca_eigenvalues = collect_parent_principal_component_inputs_for_sibling_tests(
                self.annotations_df, spectral_dims
            )

            self.annotations_df = annotate_sibling_divergence(
                tree=self.tree,
                annotations_df=self.annotations_df,
                significance_level_alpha=config.SIBLING_ALPHA,
                spectral_dims=spectral_dims,
                pca_projections=pca_projections,
                pca_eigenvalues=pca_eigenvalues,
                    )
        return self.annotations_df

    def _run_decomposition(self) -> dict:
        """Run tree decomposition to form clusters."""
        with self._time(
            "tree_decomposition",
            "Top-down traversal with statistical gate evaluation",
        ):
            decomp = self.tree.decompose(
                leaf_data=self.data,
                alpha_local=config.EDGE_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
            self.decomposition_result = decomp
        return decomp

    def _prepare_annotations_df(self) -> pd.DataFrame:
        """Prepare an empty annotations DataFrame with required columns."""
        # Collect leaf counts from tree using NetworkX helpers
        leaf_counts = {}
        for node in self.tree.nodes():
            if self.tree.out_degree(node) == 0:
                # Leaf node
                leaf_counts[node] = 1
            else:
                # Internal node - count descendant leaves using NetworkX
                descendants = nx.descendants(self.tree, node)
                leaf_nodes = [d for d in descendants if self.tree.out_degree(d) == 0]
                leaf_counts[node] = len(leaf_nodes)

        # Create DataFrame with required columns
        annotations_df = pd.DataFrame(
            {
                "leaf_count": leaf_counts,
            }
        )
        return annotations_df

    def _run_full_pipeline(self) -> dict:
        """Run the complete annotation pipeline (alternative to separate gate calls)."""
        with self._time(
            "full_annotation_pipeline",
            "Combined Gate 2 + Gate 3 annotation (orchestrator)",
        ):
            # Prepare annotations DataFrame with required columns
            annotations_df = self._prepare_annotations_df()
            result = run_gate_annotation_pipeline(
                tree=self.tree,
                annotations_df=annotations_df,
                alpha_local=config.EDGE_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
                leaf_data=self.data,
                        )
            self.annotations_df = result.annotated_df
        return self.annotations_df

    def profile(self, use_orchestrator: bool = True) -> list[TimingResult]:
        """
        Run the full profiling pipeline.

        Parameters
        ----------
        use_orchestrator : bool
            If True, use the combined annotation pipeline.
            If False, run Gate 2 and Gate 3 separately for detailed breakdown.
        """
        print("=" * 80)
        print("KL-TE CLUSTERING PROFILER")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Samples (n):        {self.n_samples}")
        print(f"  Features (d):       {self.n_features}")
        print(f"  True clusters (K):  {self.n_clusters}")
        print(f"  Random seed:        {self.random_seed}")
        print(f"  Annotation mode:    {'orchestrator' if use_orchestrator else 'separate gates'}")
        print()

        # Generate data
        self._generate_data()
        print(f"  Data shape:         {self.data.shape}")

        # Build tree
        self._build_tree()
        n_internal = sum(1 for node in self.tree.nodes() if self.tree.out_degree(node) > 0)
        print(f"  Tree nodes:         {self.tree.number_of_nodes()} ({n_internal} internal)")

        # Populate distributions
        self._populate_distributions()

        # Annotate (choose method)
        if use_orchestrator:
            self._run_full_pipeline()
        else:
            self._annotate_gate2()
            self._annotate_gate3()

        # Run decomposition
        decomp = self._run_decomposition()
        print(f"  Found clusters:     {decomp['num_clusters']}")
        print()

        return self.timing_results

    def run_cprofile(self, func: Callable, *args, **kwargs) -> tuple[Any, pstats.Stats]:
        """
        Run cProfile on a function for detailed function-level profiling.

        Returns the function result and cProfile statistics.
        """
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # Convert to pstats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")

        return result, stats

    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed timing report as a DataFrame."""
        if not self.timing_results:
            raise ValueError("No timing results available. Run profile() first.")

        total_time = sum(r.elapsed_sec for r in self.timing_results)

        report_data = []
        for result in self.timing_results:
            report_data.append(
                {
                    "Component": result.component,
                    "Description": result.description,
                    "Time (ms)": result.elapsed_ms,
                    "Time (s)": result.elapsed_sec,
                    "Percentage (%)": result.percentage(total_time),
                    "Call Count": result.call_count,
                    "Time per Call (ms)": result.elapsed_ms / result.call_count,
                }
            )

        df = pd.DataFrame(report_data)
        df = df.sort_values("Time (ms)", ascending=False).reset_index(drop=True)
        return df

    def print_report(self):
        """Print a formatted timing report to console."""
        df = self.generate_report()
        total_time = df["Time (s)"].sum()

        print("=" * 80)
        print("PROFILING RESULTS")
        print("=" * 80)
        print()

        # Summary table
        print("Timing Breakdown:")
        print("-" * 80)
        print(f"{'Component':<25} {'Time (ms)':>12} {'% Total':>10} {'Description':<35}")
        print("-" * 80)

        for _, row in df.iterrows():
            print(
                f"{row['Component']:<25} {row['Time (ms)']:>12.2f} {row['Percentage (%)']:>9.1f}% "
                f"{row['Description'][:35]:<35}"
            )

        print("-" * 80)
        print(f"{'TOTAL':<25} {total_time * 1000:>12.2f} {100.0:>9.1f}%")
        print()

        # Optimization recommendations
        self._print_recommendations(df)

    def _print_recommendations(self, df: pd.DataFrame):
        """Print optimization recommendations based on profiling results."""
        print("=" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        print()

        recommendations = []

        # Analyze each component
        for _, row in df.iterrows():
            component = row["Component"]
            time_pct = row["Percentage (%)"]
            time_ms = row["Time (ms)"]

            if component == "tree_construction" and time_pct > 30:
                recommendations.append(
                    {
                        "component": component,
                        "issue": "High tree construction time",
                        "recommendation": "Consider using faster linkage methods (e.g., 'single' or 'complete' instead of 'average'). "
                        "For large n, consider approximate methods like HDBSCAN or pre-clustering.",
                        "priority": "HIGH",
                    }
                )

            elif component == "distribution_population" and time_pct > 20:
                recommendations.append(
                    {
                        "component": component,
                        "issue": "Slow distribution population",
                        "recommendation": "Distributions are computed bottom-up. For large trees, consider caching or lazy evaluation. "
                        "Vectorize the computation using NumPy operations on all leaves simultaneously.",
                        "priority": "MEDIUM",
                    }
                )

            elif component in ["gate2_annotation", "full_annotation_pipeline"] and time_pct > 40:
                recommendations.append(
                    {
                        "component": component,
                        "issue": "Gate 2 annotation is the bottleneck",
                        "recommendation": "Spectral decomposition (eigendecomposition) is O(min(n,d)³). "
                        "Consider: (1) Using dual formulation when n < d, (2) Truncated eigendecomposition (scipy.sparse.linalg.eigsh), "
                        "(3) Randomized SVD for approximate eigenvalues, (4) Caching eigen decompositions for overlapping node sets.",
                        "priority": "HIGH",
                    }
                )

            elif component == "gate3_annotation" and time_pct > 30:
                recommendations.append(
                    {
                        "component": component,
                        "issue": "Gate 3 annotation is slow",
                        "recommendation": "Sibling divergence uses Benjamini-Hochberg FDR correction which requires sorting all p-values. "
                        "Consider: (1) Vectorizing the Wald statistic computation, (2) Using sparse operations for high-dimensional data, "
                        "(3) Parallel processing across sibling pairs.",
                        "priority": "MEDIUM",
                    }
                )

            elif component == "tree_decomposition" and time_pct > 20:
                recommendations.append(
                    {
                        "component": component,
                        "issue": "Decomposition traversal is slow",
                        "recommendation": "The top-down traversal is O(n) but gate evaluation happens at each node. "
                        "Consider: (1) Pre-computing has_descendant_split flags (already implemented), "
                        "(2) Early stopping when subtree is homogeneous, (3) Parallel traversal for independent subtrees.",
                        "priority": "LOW",
                    }
                )

        # General recommendations based on data size
        if self.n_samples > 500:
            recommendations.append(
                {
                    "component": "general",
                    "issue": f"Large dataset (n={self.n_samples})",
                    "recommendation": "For n > 500, consider: (1) Subsampling for initial tree construction, "
                    "(2) Using approximate nearest neighbors for distance computation, "
                    "(3) Incremental/online clustering methods.",
                    "priority": "HIGH",
                }
            )

        if self.n_features > 500:
            recommendations.append(
                {
                    "component": "general",
                    "issue": f"High dimensionality (d={self.n_features})",
                    "recommendation": "For d > 500, consider: (1) Feature selection or dimensionality reduction (PCA, autoencoders), "
                    "(2) Sparse data structures if data is sparse, (3) Random projection for faster distance computation.",
                    "priority": "MEDIUM",
                }
            )

        # Print recommendations
        if not recommendations:
            print("No critical performance issues detected.")
            print("The pipeline is well-balanced for the current configuration.")
        else:
            # Sort by priority
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['priority']}] {rec['component'].upper()}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Recommendation: {rec['recommendation']}")
                print()

    def save_csv(self, filepath: str):
        """Save detailed timing results to CSV."""
        df = self.generate_report()

        # Add metadata
        metadata = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_clusters": self.n_clusters,
            "random_seed": self.random_seed,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Save with metadata in comments
        with open(filepath, "w") as f:
            f.write("# KL-TE Clustering Profiler Results\n")
            f.write(f"# Generated: {metadata['timestamp']}\n")
            f.write(
                f"# n_samples: {metadata['n_samples']}, n_features: {metadata['n_features']}, "
                f"n_clusters: {metadata['n_clusters']}\n"
            )
            f.write("#\n")
            df.to_csv(f, index=False)

        print(f"Results saved to: {filepath}")

    def run_function_profile(self, func: Callable, *args, **kwargs):
        """Run cProfile on a specific function and print results."""
        print(f"\nProfiling function: {func.__name__}")
        print("-" * 80)

        _, stats = self.run_cprofile(func, *args, **kwargs)

        # Print top 20 functions by cumulative time
        print("\nTop 20 functions by cumulative time:")
        stats.print_stats(20)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KL-TE Clustering Performance Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=80,
        help="Number of samples (default: 80)",
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=100,
        help="Number of features (default: 100)",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of true clusters (default: 4)",
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Use a predefined test case by name (overrides n-samples, n-features, n-clusters)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2001,
        help="Random seed (default: 2001)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: None, print to console only)",
    )

    parser.add_argument(
        "--profile-level",
        type=str,
        choices=["component", "function"],
        default="component",
        help="Profiling granularity (default: component)",
    )

    parser.add_argument(
        "--separate-gates",
        action="store_true",
        help="Profile Gate 2 and Gate 3 separately (default: use combined orchestrator)",
    )

    parser.add_argument(
        "--profile-specific",
        type=str,
        choices=["tree", "gate2", "gate3", "decomposition", "full"],
        default=None,
        help="Profile only a specific component (default: profile all)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the profiler."""
    args = parse_args()

    # Handle predefined test case
    if args.case:
        from benchmarks.shared.cases import get_default_test_cases

        cases = get_default_test_cases()
        tc = next((c for c in cases if c["name"] == args.case), None)
        if tc is None:
            print(f"ERROR: Case '{args.case}' not found.")
            available = [c["name"] for c in cases[:10]]
            print(f"Available cases: {available}...")
            sys.exit(1)

        # Extract parameters from test case
        args.n_samples = tc.get("n_samples", args.n_samples)
        args.n_features = tc.get("n_features", args.n_features)
        args.n_clusters = tc.get("n_clusters", args.n_clusters)

    # Create profiler
    profiler = KLProfiler(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        random_seed=args.seed,
        output_path=args.output,
        profile_level=args.profile_level,
    )

    # Run profiling
    if args.profile_specific == "tree":
        profiler._generate_data()
        profiler._build_tree()
    elif args.profile_specific == "gate2":
        profiler._generate_data()
        profiler._build_tree()
        profiler._populate_distributions()
        profiler._annotate_gate2()
    elif args.profile_specific == "gate3":
        profiler._generate_data()
        profiler._build_tree()
        profiler._populate_distributions()
        profiler._annotate_gate2()
        profiler._annotate_gate3()
    elif args.profile_specific == "decomposition":
        profiler._generate_data()
        profiler._build_tree()
        profiler._populate_distributions()
        profiler._run_full_pipeline()
        profiler._run_decomposition()
    else:
        # Full pipeline
        profiler.profile(use_orchestrator=not args.separate_gates)

    # Print report
    profiler.print_report()

    # Save to CSV if requested
    if args.output:
        profiler.save_csv(args.output)

    # Run function-level profiling if requested
    if args.profile_level == "function":
        print("\n" + "=" * 80)
        print("FUNCTION-LEVEL PROFILING")
        print("=" * 80)

        # Profile spectral decomposition
        from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
            eigendecompose_correlation_backend,
        )

        profiler.run_function_profile(
            eigendecompose_correlation_backend,
            profiler.data.values,
            compute_eigenvectors=False,
        )


if __name__ == "__main__":
    main()
    main()
