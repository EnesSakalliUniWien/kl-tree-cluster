# Repository Organization Audit - 2026-04-27

## Scope

This audit checks repository-level organization after the `benchmarks/results` cleanup. It focuses on structural inconsistencies, misplaced generated artifacts, duplicate result roots, and tracked files that look like outputs rather than source.

## Cleanup Status

`benchmarks/results/` is now grouped by result type:

- `00_current_20260427_blob_analysis/`
- `01_mnist_digits_umap/`
- `02_hc_cms_go_runs/`
- `03_gene_go_feature_matrix_runs/`
- `04_generic_benchmark_runs/`
- `05_diagnostics_validation_audits/`
- `06_method_experiments_sweeps/`
- `07_reports_tables_logs/`
- `08_visualizations_interactive/`
- `09_utilities_system_misc/`
- `10_analysis_pipeline_runs/`

The current Blob 15 analysis is physically organized under:

`benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/`

MNIST and digits artifacts are physically organized under:

`benchmarks/results/01_mnist_digits_umap/`

Additional cleanup completed after the initial audit:

- Root feature matrices moved to `data/feature_matrices/`.
- Endotype/reference data moved to `data/reference/`.
- Profiling CSVs moved to `reports/profiling/`.
- Logs moved to `reports/logs/`.
- Notebook-generated PNGs moved to `reports/notebook_figures/`.
- `analysis/peps.h5mu` moved to `local_data/analysis/`.
- `analysis/results/` moved to `benchmarks/results/10_analysis_pipeline_runs/`.
- Top-level `results/` Swiss-roll outputs moved to `benchmarks/results/06_method_experiments_sweeps/swiss_roll_diffusion/`.
- Historical diagnostic outputs moved to `benchmarks/results/05_diagnostics_validation_audits/`.
- Manuscript root build outputs moved to `manuscript/build/root_outputs/`.
- `.DS_Store` files were removed from the working tree.
- Directory READMEs were added for `data/`, feature matrices, references, reports, profiling, logs, notebook figures, and moved alpha runs.

## Resolved High-Priority Inconsistencies

### 1. Root Directory Data and Generated Outputs

The repository root was cleared of misplaced data and generated outputs. The former root artifacts now live at:

- `data/feature_matrices/feature_matrix.tsv` - tracked, used as the default dataset by scripts and benchmarks.
- `data/feature_matrices/feature_matrix_julia_GOBP.tsv` - untracked generated/input matrix.
- `data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv` - untracked generated/input matrix.
- `data/reference/adg6375_File_S7_endotypes_julia.txt` - untracked reference/data file.
- `reports/profiling/profiler_results.csv` - tracked profiling output.
- `reports/profiling/profiler_results_detailed.csv` - untracked profiling output.
- `reports/profiling/profiler_results_large.csv` - untracked profiling output.
- `reports/logs/comparison_output.log`, `reports/logs/main.log` - log files.

Script defaults and references were updated to the new feature-matrix and profiler locations.

### 2. Multiple Result Roots

Generated outputs have been consolidated under `benchmarks/results/` where practical:

- Current blob analysis: `benchmarks/results/00_current_20260427_blob_analysis/`
- HC/CMS alpha runs: `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/`
- Analysis pipeline runs: `benchmarks/results/10_analysis_pipeline_runs/`
- Debug diagnostics: `benchmarks/results/05_diagnostics_validation_audits/`
- Swiss-roll experiments: `benchmarks/results/06_method_experiments_sweeps/swiss_roll_diffusion/`

### 3. Large Local Analysis Data

The large local-only file was moved from `analysis/peps.h5mu` to:

- `local_data/analysis/peps.h5mu`

`local_data/` is ignored.

### 4. Manuscript Build Artifacts

Root-level manuscript build outputs were moved to:

- `manuscript/build/root_outputs/`

The existing `manuscript/build/` remains the manuscript build output area.

### 5. Notebook Images

Tracked notebook PNG outputs were moved from `notebooks/` to:

- `reports/notebook_figures/`

## Remaining Inconsistencies

### 1. Entrypoints Are Spread Across Multiple Locations

Runnable scripts exist at root, `scripts/`, `benchmarks/`, `notebooks`, and
`analysis`. One-off benchmark diagnostics now live under
`benchmarks/diagnostics/`.

Examples:

- `quick_start.py`
- `run_benchmark.py`
- `scripts/run_feature_matrix_with_umap.py`
- `scripts/run_full_benchmark_isolated.py`
- `benchmarks/run_subset.py`
- `benchmarks/run_regression_gate.py`
- `notebooks/run_benchmark.py`
- `analysis/run_all.py`

Recommendation:

- Define entrypoint categories:
  - `scripts/` for user-facing utilities.
  - `benchmarks/` for benchmark runners.
  - `benchmarks/diagnostics/` for benchmark investigation utilities.
  - `notebooks/` for notebooks only, not reusable scripts.
- Move or document root entrypoints.

### 2. Empty Directories May Remain

Examples include empty result/run folders in:

- `benchmarks/results/03_gene_go_feature_matrix_runs/`
- `benchmarks/results/04_generic_benchmark_runs/*/plots`
- `benchmarks/results/06_method_experiments_sweeps/swiss_roll_diffusion/swiss_roll_diffusion_20260323_164908`

Recommendation:

- Remove empty generated folders.
- Keep placeholder directories only if they contain a `.gitkeep` and a README explaining their purpose.

### 3. `.gitignore` and Tracked Generated Files Need a Policy Decision

`.gitignore` ignores many generated paths:

- `results/`
- `benchmarks/**/results/`
- `*.log`
- `analysis/`

But tracked generated artifacts still exist, including:

- `data/feature_matrices/feature_matrix.tsv`
- `reports/profiling/profiler_results.csv`
- `reports/logs/comparison_output.log`
- PNGs under `reports/notebook_figures/`

Recommendation:

- Decide which tracked generated artifacts are canonical fixtures.
- Move canonical fixtures into a named fixture directory.
- Remove or untrack non-canonical outputs.

### 4. Benchmark Diagnostics Need Clear Ownership

Reusable diagnostics are now part of `benchmarks/diagnostics/`. New
investigation scripts should either be promoted there with tests and explicit
inputs, or kept out of the tracked repository.

## Suggested Next Cleanup Order

1. Add a short path policy to `README.md`.
2. Decide whether canonical fixture data should remain tracked under `data/`.
3. Decide whether `reports/notebook_figures/` should be tracked or treated as generated output.
4. Remove empty generated folders.
5. Add or refresh README maps for `scripts/` and `benchmarks/diagnostics/`.

## Commands Used For Audit

- top-level inventory with `find . -maxdepth 2`
- generated artifact scan for CSV/TSV/PNG/PDF/LOG/ZIP
- empty-directory scan
- tracked generated artifact scan with `git ls-files`
- stale reference scan with `rg`
