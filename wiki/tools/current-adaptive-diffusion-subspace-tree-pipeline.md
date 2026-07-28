---
title: Current Adaptive Diffusion Subspace Tree Pipeline
type: tool
status: reviewed
updated: 2026-06-18
sources:
  - applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py
tags:
  - pipeline
  - allgo
  - adaptive-diffusion
  - subspace
---

# Current Adaptive Diffusion Subspace Tree Pipeline

## Summary

Use `applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py`
as the full current-method pipeline for GO feature matrices. It must create
the same directory structure for every input matrix: a timestamped experiment
root under `results/analyses/`, method-separated PDFs, connected manifests,
rankings, and one artifact-complete directory per cosine subspace.

## Usage

Run the pipeline with an explicit feature matrix:

```bash
MPLBACKEND=Agg python applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py \
  --input /path/to/feature_matrix.tsv
```

The default output root is:

```text
results/analyses/<matrix-stem-without-feature_matrix_>_current_adaptive_diffusion_subspace_tree_<timestamp>/
```

Every full run should write these top-level artifacts:

- `rankings/current_adaptive_diffusion_subspace_tree_ranking.csv`
- `rankings/current_adaptive_diffusion_subspace_tree_specificity_aware_ranking.csv`
- `rankings/current_adaptive_diffusion_subspace_tree_subspace_blocks.csv`
- `rankings/current_adaptive_diffusion_subspace_tree_spectrum.csv`
- `ARTIFACT_INDEX.md`
- `artifact_index.csv`
- `subspace_plot_index.csv`
- `connected_results_manifest.json`
- `<artifact-prefix>_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/`
- `<artifact-prefix>_quality_aware_go_ic_plots/`
- `subspaces/<weighting>/<block_name>/`

Each `subspaces/<weighting>/<block_name>/` directory should contain the
subspace coordinates, diffusion metadata, linkage tree, cluster assignments or
explicit failure status, GO-IC quality summary, coherence tables, TF-IDF
quality tables, axis term-loading CSVs, per-axis term plots, combined term
plot, and plain plus term-annotated embeddings.

The preferred reading order is `specificity_aware_rank`: quality tier,
specificity score, specific-cluster fraction, weighted specificity delta, then
GO-BIC active per gene. `old_display_rank` preserves the previous
quality-tiered GO-IC order, and `raw_go_ic_rank` preserves the raw GO-IC-only
order for audit.

## Evidence

- `applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py`
  derives the experiment root and artifact prefix from the input matrix name
  and writes the full connected artifact structure in one run.

## Links

- [[julia-allgo-new-current-adaptive-diffusion-subspace-tree-20260618]]
- [[allgo-new-interactome-current-adaptive-diffusion-subspace-tree-20260618]]
