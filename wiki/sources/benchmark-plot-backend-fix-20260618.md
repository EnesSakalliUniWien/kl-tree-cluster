---
title: Benchmark Plot Backend Fix 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - benchmarks/shared/plots/backend.py
  - benchmarks/shared/plots/__init__.py
  - benchmarks/full/run.py
  - benchmarks/shared/util/case_execution.py
  - tests/pipeline/60_test_case_execution_pdf_cover_mode.py
  - tests/pipeline/58_test_pipeline_pdf_behavior.py
tags:
  - source
  - benchmarks
  - plots
  - umap
---

# Benchmark Plot Backend Fix 2026-06-18

## Summary

The plotted full benchmark crash was traced to Matplotlib backend selection,
not to UMAP embedding itself. A direct one-case reproducer aborted in
`matplotlib.pyplot.subplots()` while loading the macOS GUI backend
`matplotlib.backends._macosx`. The UMAP flag triggered the plotting path, but
the native abort occurred before UMAP-specific rendering was the active frame.

Benchmark plotting now defaults to the noninteractive `Agg` backend before any
benchmark module imports `pyplot`. This makes file/PDF rendering safe in
command-line benchmark runs and in spawned case workers.

## Key Points

- `benchmarks.shared.plots.backend.configure_matplotlib_backend()` sets
  `MPLBACKEND=Agg` unless the user explicitly provides another backend.
- The shared plot package calls that backend configuration before applying
  Matplotlib PDF font defaults.
- `benchmarks/full/run.py` configures the backend before importing
  `matplotlib.pyplot`.
- The isolated case worker configures the backend before importing the shared
  benchmark pipeline inside the spawned process.
- The previous direct plotting reproducer now writes a per-case PDF instead of
  aborting.
- A plotted `method_proof` full-run suite completed all `11` cases with
  `isolate_umap_cases=True` and merged the PDF report.

## Evidence

- `tests/pipeline/60_test_case_execution_pdf_cover_mode.py` covers the worker
  defaulting to the file-safe backend.
- `tests/pipeline/58_test_pipeline_pdf_behavior.py` continues to cover PDF
  streaming behavior.
- The verification run with `TBS_CASE_SUITE=method_proof`,
  `TBS_ENABLE_PLOTS=1`, and `TBS_METHODS=tbs` completed the plotted
  isolated subprocess path that previously failed with subprocess exit `-6`.

## Links

- [[branch-length-candidate-full-big-20260618]]
- [[benchmark-runner-guarded-contract-fix-20260618]]
