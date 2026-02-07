import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
repo_root = Path(__file__).resolve().parents[0]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.cases.overlapping import get_overlapping_cases


def run_debug_segfault():
    print("Fetching overlapping cases...")
    all_cases = get_overlapping_cases()
    target_name = "overlap_part_10c_highd"
    case = next((c for c in all_cases if c["name"] == target_name), None)

    if not case:
        print(f"Case {target_name} not found!")
        return

    print(f"Running case: {target_name}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = repo_root / "debug_segfault" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    case["test_case_num"] = 999

    # Run WITHOUT UMAP/Isomap first to see if it's the plotting
    benchmark_cluster_algorithm(
        test_cases=[case],
        methods=["kl"],
        verbose=True,
        save_individual_plots=False,
        matrix_audit=True,
        concat_output=str(run_dir / "debug_plots.pdf"),
        plot_umap=False,
        plot_manifold=False,
    )


if __name__ == "__main__":
    run_debug_segfault()
