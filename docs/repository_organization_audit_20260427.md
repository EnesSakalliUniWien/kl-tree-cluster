# Repository Organization Audit - 2026-04-27

> Historical audit. The 2026-07-27 organization pass supersedes its
> `scripts/analysis/` recommendation: maintained dataset commands now live in
> `applications/`, reusable space separation in
> `tree_break_selection/space_separation/`, and reusable plotting engines in
> `tree_break_selection/plot/`.

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
- Profiling CSVs, logs, and notebook-generated PNGs were removed from version
  control unless they are deliberately retained evidence.
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
- `data/feature_matrices/feature_matrix_julia_GOBP.tsv` - tracked generated/input matrix.
- `data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv` - tracked generated/input matrix.
- `data/reference/adg6375_File_S7_endotypes_julia.txt` - untracked reference/data file.
- `reports/README.md` - policy marker for retained evidence; generated logs,
  profiling CSVs, and notebook figures are not tracked by default.

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

Tracked notebook PNG outputs were removed from version control. Notebook image
exports should be regenerated unless a specific file is cited as retained
evidence.

- `reports/README.md`

## Remaining Inconsistencies

### 1. Entrypoints Are Categorized By Domain

Runnable commands are grouped by domain. User-facing matrix-analysis utilities
now live under `applications/endotypes/`; benchmark runners and diagnostics
live under `benchmarks/`; test orchestration lives under
`scripts/run_tests_ordered.py`. Notebook-only Python helpers were removed from
`notebooks/`.

Examples:

- `quick_start.py`
- `benchmarks/smoke/run_subset.py`
- `benchmarks/regression/run_gate.py`
- `applications/endotypes/pipelines/run_feature_matrix_with_umap.py`
- `applications/endotypes/analysis/analyze_hc_clusters.py`
- `applications/endotypes/analysis/assess_method_correctness.py`

Recommendation:

- Keep this separation: `applications/` for dataset commands,
  `tree_break_selection/` for reusable methods and plotting engines,
  `benchmarks/` for benchmark execution, `benchmarks/diagnostics/` for
  investigation commands, and `notebooks/` for notebooks only.

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

Tracked generated-like artifacts are limited to canonical feature matrices and
explicit documentation/evidence files.

Recommendation:

- Keep canonical feature matrices in `data/feature_matrices/`.
- Keep generated reports untracked unless they are cited retained evidence.
- Remove or untrack non-canonical outputs.

### 4. Benchmark Diagnostics Need Clear Ownership

Reusable diagnostics are now part of `benchmarks/diagnostics/`. New
investigation scripts should either be promoted there with tests and explicit
inputs, or kept out of the tracked repository.

## Suggested Next Cleanup Order

1. Add a short path policy to `README.md`.
2. Remove empty generated folders.
3. Add or refresh README maps for `scripts/` and `benchmarks/diagnostics/`.

## Commands Used For Audit

- top-level inventory with `find . -maxdepth 2`
- generated artifact scan for CSV/TSV/PNG/PDF/LOG/ZIP
- empty-directory scan
- tracked generated artifact scan with `git ls-files`
- stale reference scan with `rg`
