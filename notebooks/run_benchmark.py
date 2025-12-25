# Setup Python path to import from project root
import sys
import os
from datetime import datetime
from pathlib import Path
from kl_clustering_analysis.benchmarking import benchmark_cluster_algorithm


# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
project_root = str(repo_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run validation with SMALL test cases and UMAP plotting enabled, producing PDFs only
    df_results, fig = benchmark_cluster_algorithm(
        significance_level=0.05,
        verbose=True,
        plot_umap=True,
        plot_manifold=False,
        concat_plots_pdf=True,
        save_individual_plots=False,
    )

    # Save validation results to results folder
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"validation_results_{current_date}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"Validation results saved to {results_file}")
    print(f"PDF output written under {repo_root / 'results' / 'plots'}")
