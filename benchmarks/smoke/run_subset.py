#!/usr/bin/env python3
"""
Quick subset benchmark — picks ~15 representative cases across categories
and runs them with plots.
"""

import os

from benchmarks.shared.benchmark_runs.runtime import apply_single_thread_runtime_defaults
from benchmarks.shared.benchmark_runs.smoke import (
    print_smoke_case_manifest,
    print_tbs_smoke_summary,
    select_smoke_cases,
)
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm

# Default to single-threaded spectral decomposition workers to avoid
# thread oversubscription. Users can override by setting TBS_N_JOBS.
apply_single_thread_runtime_defaults()
spectral_jobs = os.environ["TBS_N_JOBS"]

all_cases = get_default_test_cases()
subset = select_smoke_cases(all_cases)
print_smoke_case_manifest(subset, total_cases=len(all_cases), spectral_jobs=spectral_jobs)

df_results, fig = benchmark_cluster_algorithm(
    test_cases=subset,
    verbose=True,
    plot_umap=True,
    concat_plots_pdf=True,
    methods=["tbs"],
)

print_tbs_smoke_summary(df_results)
