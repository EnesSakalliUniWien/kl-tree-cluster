# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2025-12-24

### Changed
- Remove deprecated `n_permutations` parameter from the decomposition API; callers passing it will no longer be accepted. Tests updated accordingly.
- Pipeline plotting behavior: default no longer writes intermediate PNGs; when `concat_plots_pdf=True` the pipeline collects Figures and writes categorized PDFs (`k_distance_plots.pdf`, `tree_plots.pdf`, `umap_plots.pdf`) instead of emitting PNGs.
- PDF utilities: improved diagnosis, figure classification (manifold plots grouped with UMAP), and robust headless handling (Agg backend).

### Removed
- `kl_clustering_analysis.threshold` package removed — helper functionality that was required by debug scripts is now replaced with a small conservative fallback (debug-only) or should be implemented separately where needed.

### Fixed
- Various import-time and optional-dependency issues (runners now import optional dependencies lazily and return skip results when missing) to improve test/CI stability.
- Resolved multiple test failures and tightened integration behavior for PDF generation.


(For full history, add older entries here in reverse chronological order.)