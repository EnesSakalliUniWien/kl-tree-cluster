---
title: Categorical Adaptive Diffusion Focus Audit 2026-06-26
type: analysis
status: draft
updated: 2026-06-26
sources:
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh_results.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/focus_case_ablation_summary.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/gauss_overlap_3c_small_q3__assignments.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/gauss_overlap_3c_small_q4__assignments.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/gauss_overlap_3c_small_q5__assignments.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/phylo_large_32taxa__assignments.csv
  - benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/phylo_large_64taxa__assignments.csv
  - benchmarks/results/run_20260624_153337Z_categorical/categorical_benchmark_comparison.csv
  - benchmarks/shared/generators/gaussian_cases.py
  - benchmarks/shared/generators/categorical_cases.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
tags:
  - analysis
  - benchmarks
  - categorical
  - diffusion
---

# Categorical Adaptive Diffusion Focus Audit 2026-06-26

## Summary

The categorical adaptive-diffusion fixed-coordinate row is weaker on the
quantized Gaussian overlap cases and `phylo_large` because the selected
clusters are already determined by the adaptive-diffusion topology. Recomputed
fixed-topology NNLS branch lengths do not change the partition on these five
focus cases.

Quantization is implicated for the Gaussian overlap cases: the quantile
one-hot representation creates dense within-class strata that the adaptive tree
keeps as pure fragments. The phylogenetic large cases show the opposite
failure mode: adaptive diffusion merges nearby taxa, while raw Hamming
fixed-coordinate TBS keeps more taxon boundaries.

## Details

The big categorical run used pydiffmap adaptive diffusion with Euclidean
neighbors, variable bandwidth `-1/(d+2)`, median epsilon, `k=15`,
diffusion time `3`, and `30` components before average-linkage TBS. The TBS
row then used fixed-coordinate BH, passthrough traversal, and NNLS branch-time
refitting.

On `gauss_overlap_3c_small_q3`, `q4`, and `q5`, the adaptive row found `8`,
`6`, and `11` clusters for a true `K=3`. Homogeneity was high
(`0.970`, `1.000`, `1.000`) while completeness was lower (`0.627`, `0.719`,
`0.584`), so the loss is mostly pure over-fragmentation. Assignment inspection
shows the same pattern: q4 splits one true class into four sizeable predicted
clusters, and q5 splits all three true classes while keeping predicted clusters
pure.

On `phylo_large_32taxa` and `phylo_large_64taxa`, the adaptive row found `30`
and `48` clusters for true `K=32` and `K=64`. Completeness stayed near one
(`0.994`, `0.998`) while homogeneity fell (`0.963`, `0.915`), so the loss is
mostly taxon merging. The 32-taxon assignment has three mixed predicted
clusters, each joining two true taxa. The 64-taxon assignment has seventeen
mixed predicted clusters, mostly two-taxon merges.

The ablation confirms the branch-time layer is not the cause on these rows.
For every focus case, `adaptive_t3_c30_topology_only` and
`adaptive_t3_c30_nnls` returned identical cluster counts and scores. Raw
Hamming fixed-coordinate TBS, by contrast, gets `phylo_large_32taxa` exactly
right and keeps `phylo_large_64taxa` much closer (`59` clusters,
`ARI=0.9168`) than the adaptive row (`48` clusters, `ARI=0.7707`).

## Evidence

- `benchmarks/shared/generators/gaussian_cases.py` defines
  `generate_blobs_quantile_case`, which creates Gaussian blobs, discretizes
  each continuous feature by quantiles, then one-hot encodes the categories.
- `benchmarks/shared/generators/categorical_cases.py` defines
  `generate_phylogenetic_case`, which one-hot encodes simulated categorical
  sequence leaves and within-taxon samples.
- `benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh_results.csv`
  records the five focus-case metrics and the adaptive diffusion metadata.
- `benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/focus_case_ablation_summary.csv`
  records that topology-only and NNLS adaptive rows are identical on all five
  focus cases.
- The assignment CSVs under
  `benchmarks/results/run_20260626_133650_categorical_big_adaptive_diffusion_nnls_fixed_coordinate_bh/assignments/`
  record the pure-fragment Gaussian pattern and mixed-taxon phylogenetic
  pattern.
- `benchmarks/results/run_20260624_153337Z_categorical/categorical_benchmark_comparison.csv`
  records the older categorical baselines, including raw Hamming
  fixed-coordinate TBS outperforming adaptive diffusion on both phylo-large
  rows.

## Links

- [[tree-break-selection]]
- [[scrna-branch-length-effect-audit-20260624]]
- [[benchmark-pipeline-contract]]

## Open Questions

- Should adaptive diffusion be disabled for quantile one-hot Gaussian cases
  unless a categorical-aware kernel preserves coarse class margins?
- Should phylogenetic one-hot sequence cases use raw Hamming or a
  phylogenetic/tree-aware distance instead of adaptive Euclidean diffusion?
- Should the categorical benchmark report topology-only and NNLS branch-time
  rows separately when assignments are identical but NNLS diagnostics differ?
