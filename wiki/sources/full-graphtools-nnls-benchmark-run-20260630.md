---
title: Full Graphtools NNLS Benchmark Run 2026-06-30
type: source
status: reviewed
updated: 2026-06-30
sources:
  - benchmarks/results/run_20260630_130921Z_full/full_benchmark_comparison.csv
  - benchmarks/results/run_20260630_130921Z_full/failure_report.md
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/README.md
tags:
  - source
  - benchmarks
  - diffusion
  - graphtools
  - nnls
---

# Full Graphtools NNLS Benchmark Run 2026-06-30

## Summary

The full `121`-case benchmark was rerun with the optional GPL
`tbs_diffusion_graphtools_nnls` method after installing the project
`experimental-gpl` extra and registering the method as a named benchmark
method. Plots and relationship analysis were disabled. The runner also included
plain `tbs` because the full runner requires it for the standard tree contract.

## Key Points

- Results were written under
  `benchmarks/results/run_20260630_130921Z_full/`.
- `tbs_diffusion_graphtools_nnls` produced `117` ok rows and `4` skips, versus
  `106` ok rows and `15` skips for the previous pydiffmap
  `tbs_diffusion_adaptive_nnls` full run.
- Mean ok-row metrics for graphtools NNLS were ARI `0.741645`, NMI `0.761942`,
  and macro F1 `0.811544`. The previous pydiffmap NNLS run had mean ok-row ARI
  `0.736107`, NMI `0.748786`, and macro F1 `0.793331`.
- On `104` paired finite rows versus pydiffmap NNLS, graphtools NNLS won `15`,
  tied `70`, and lost `19` by ARI. The mean paired ARI delta was `-0.005328`
  and the median was `0.0`.
- Strong graphtools NNLS repairs include `cat_highd_3cat_500feat` (`4`
  clusters, ARI/NMI `1.0`), `bar_binary_balanced_4c`, `gauss_overlap_8c_highd`,
  several high-cardinality categorical rows, and large phylogenetic rows.
- Graphtools NNLS losses include `gauss_null_small`, `gauss_extreme_noise_many`,
  `overlap_hd_4c_1k`, `cat_clear_3cat_4c`, `gauss_overlap_3c_small`,
  `sbm_clear_small`, and `dim_consolidated_4c_24f`.
- The four graphtools NNLS skips are strict sibling-inflation support failures,
  not backend import failures.
- The run emitted graphtools warnings on duplicate/tied rows and expensive
  high-dimensional kNN graph construction, so the method remains an optional
  backend candidate rather than a default method.

## Evidence

- `benchmarks/results/run_20260630_130921Z_full/full_benchmark_comparison.csv`
  records the full run.
- `benchmarks/results/run_20260630_130921Z_full/failure_report.md` records the
  failure diagnosis.
- `benchmarks/shared/runners/method_registry.py` registers
  `tbs_diffusion_graphtools_nnls`.
- `benchmarks/shared/runners/dispatch.py` routes the named graphtools NNLS
  method through the graphtools diffusion runner.

## Links

- [[full-adaptive-nnls-benchmark-run-20260628]]
- [[adaptive-nnls-regression-skip-analysis-20260628]]
- [[graphtools-diffusion-focus-benchmark-20260627]]
