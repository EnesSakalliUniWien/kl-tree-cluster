---
title: graphtools Diffusion Focus Benchmark 2026-06-27
type: source
status: reviewed
updated: 2026-06-27
sources:
  - benchmarks/results/focus_graphtools_diffusion_20260627/focused_benchmark_results.csv
  - benchmarks/results/focus_graphtools_diffusion_20260627/graphtools_diagnostic_summary.csv
  - benchmarks/results/focus_graphtools_diffusion_20260627/phylo_large_32taxa/graphtools_final_boundary_summary.csv
  - benchmarks/results/focus_graphtools_diffusion_20260627/phylo_large_32taxa/distance_label_margins.csv
  - benchmarks/results/focus_graphtools_diffusion_20260627/phylo_large_64taxa/graphtools_final_boundary_summary.csv
  - benchmarks/results/focus_graphtools_diffusion_20260627/phylo_large_64taxa/distance_label_margins.csv
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - pyproject.toml
tags:
  - source
  - benchmarks
  - diffusion
  - phylogenetic
---

# graphtools Diffusion Focus Benchmark 2026-06-27

## Summary

An optional `tbs_diffusion_graphtools` runner was added as
`TBS (graphtools Kernel Diffusion)`. The dependency is isolated under the
`experimental-gpl` extra because `graphtools` is GPL-licensed. The runner uses
the graphtools Hamming kNN kernel as the diffusion graph backend and keeps the
existing TBS average-linkage and gate pipeline unchanged.

The focused benchmark compared `tbs_diffusion`, `tbs_diffusion_adaptive`, and
`tbs_diffusion_graphtools` on the quantized Gaussian overlap q3/q4/q5 cases
and the two `phylo_large` cases.

## Key Points

- `tbs_diffusion_graphtools` recovers all three quantized Gaussian overlap
  q3/q4/q5 rows exactly with ARI `1.000000` and `3` clusters.
- On `phylo_large_32taxa`, graphtools returns ARI `0.902655` with `29`
  clusters. Its diffusion distance preserves strict true-taxon sample gaps for
  all `32` taxa, but TBS still merges three taxon pairs. The mixed boundaries
  have corrected sibling p-values just above alpha, including `0.014286` for
  taxa `9/14` and `0.012663` for taxa `20/22`.
- On `phylo_large_64taxa`, graphtools improves over pydiffmap adaptive
  diffusion but remains partial: ARI `0.709257`, `102` clusters, homogeneity
  `0.941181`, completeness `0.909484`, and V-measure `0.925061`.
- `phylo_large_64taxa` still loses raw taxon geometry under the graphtools
  diffusion distance. Raw Hamming has strict sample gaps for all `64` taxa, but
  graphtools diffusion keeps strict gaps for only `33` taxa and drops k=5
  nearest-neighbor same-label fraction to `0.831641`.
- The result supports keeping `graphtools` as an optional benchmark backend,
  not promoting it to the canonical TBS diffusion method.

## Evidence

- `benchmarks/results/focus_graphtools_diffusion_20260627/focused_benchmark_results.csv`
  records the focused method comparison.
- `benchmarks/results/focus_graphtools_diffusion_20260627/graphtools_diagnostic_summary.csv`
  records mixed-cluster, fragmentation, strict-gap, and nearest-neighbor
  purity diagnostics.
- `benchmarks/shared/runners/tbs_diffusion_runner.py`,
  `benchmarks/shared/runners/method_registry.py`, and
  `benchmarks/shared/runners/dispatch.py` implement and register the optional
  graphtools runner.
- `pyproject.toml` records the `experimental-gpl` optional dependency group.

## Links

- [[phylo-large-adaptive-pydiffmap-focus-audit-20260627]]
- [[full-adaptive-pydiffmap-benchmark-run-20260627]]
- [[categorical-adaptive-diffusion-focus-audit-20260626]]
