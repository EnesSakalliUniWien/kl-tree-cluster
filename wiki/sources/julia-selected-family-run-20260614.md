---
title: Julia Selected Family Run 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/family/run_selected_family_matrix.py
  - benchmarks/diagnostics/calibration/selected/family/multiscale_umap.py
  - raw/assets/benchmark-results/julia_selected_family_20260614/manifest.json
  - raw/assets/benchmark-results/julia_selected_family_20260614/cluster_assignments.csv
  - raw/assets/benchmark-results/julia_selected_family_20260614/multiscale_node_decisions.csv
  - raw/assets/benchmark-results/julia_selected_family_20260614/selected_family_guard_rows.csv
  - raw/assets/benchmark-results/julia_selected_family_20260614/umap_overlay/multiscale_umap_overlay.png
tags:
  - source
  - julia
  - traversal
  - selection
---

# Julia Selected Family Run 2026-06-14

## Summary

The combined Julia GOCC/GOBP/GOMF binary matrix was run through
`fixed_coordinate_global_passthrough_refined_v1` using the real-matrix
selected-family diagnostic runner. The run produced multi-scale traversal
tables and a UMAP overlay under
`raw/assets/benchmark-results/julia_selected_family_20260614/`.

## Key Points

- Input matrix: `703` genes by `14766` binary GO features.
- The refined selected-family profile returned `401` flat clusters, down from
  the earlier baseline TBS run's `670` clusters.
- The run produced `306` singleton clusters and `397` genes in non-singleton
  clusters. Baseline TBS had `648` singletons and only `55` genes in
  non-singleton clusters.
- The traversal had `8` selected-family/root guard rows, and all `8` were
  blocked. The dominant selected-root guard zone `zone_N1404` covered `665`
  genes, so the UMAP overlay annotates it rather than outlining nearly every
  point.
- Reference recovery on the `262` matched Julia endotype genes was ARI
  `0.064157` and NMI `0.618847`. This improves ARI over baseline TBS
  (`0.001812`) but is close to the older tree-estimator NMI range
  (`0.616780`--`0.633958`).
- UMAP compactness moved toward the neighbor-joining regime: weighted mean
  UMAP radius was `0.355163`, compared with baseline TBS `0.181714` and
  neighbor joining `0.425424`.
- Feature coherence was intermediate: mean active-feature Jaccard was
  `0.363638`, below baseline TBS `0.595212` but above neighbor joining
  `0.318604`.

## Evidence

- `manifest.json` records the selected-family profile, alpha settings, input
  dimensions, cluster count, guard count, and output paths.
- `clustering_diagnostics/cluster_size_summary.csv`,
  `umap_compactness_summary.csv`, `feature_coherence_summary.csv`, and
  `reference_recovery_summary.csv` record the Julia diagnostics.
- `umap_overlay/multiscale_umap_overlay.png` renders stable regions first and
  overlays non-dominant guard zones.

## Links

- [[julia-tree-estimator-run-20260614]]
- [[selected-family-traversal-panel-20260614]]
- [[fixed-sibling-gate-profile-validation-20260613]]
- [[open-mathematical-questions]]
