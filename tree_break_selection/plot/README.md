# Plotting Engines

This package owns reusable plotting primitives for Tree-Break Selection:

- discrete cluster-color specifications;
- rectangular and radial tree rendering with gate annotations;
- noninteractive Matplotlib backend selection;
- consistent image panels for report composition.
- multi-scale region overlays on existing UMAP coordinates.

Application-specific page composition belongs beside the application under
`applications/`. Benchmark covers, metric summaries, and full benchmark report
assembly remain under `benchmarks/shared/plots/` because their interface uses
benchmark result records and case metadata.
