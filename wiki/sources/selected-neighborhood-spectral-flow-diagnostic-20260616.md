---
title: Selected Neighborhood Spectral Flow Diagnostic 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_spectral_flow.py
  - tests/validation/146_test_selected_neighborhood_spectral_flow.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_spectral_flow_overlap_three_case
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_spectral_flow_overlap_three_case_conditional_topology
tags:
  - source
  - diagnostics
  - spectral
  - marchenko-pastur
  - neighborhood
---

# Selected Neighborhood Spectral Flow Diagnostic 2026-06-16

## Summary

`selected_neighborhood_spectral_flow.py` tests whether selected-neighborhood
parent-child edges carry coherent Marchenko--Pastur-supported eigenspaces. It
uses raw MP signal counts to separate MP-certified modes from floor-only Wald
projection dimensions, then compares neighboring row-basis eigenspaces by
principal angles and neighboring eigenvalue spectra by log drift. The extended
mode-transport layer groups MP outlier eigenvalues into multiplicity blocks,
stores each block's spectral projector and normalized characteristic
polynomial, and matches blocks across edges by an optimal transport cost. The
output is diagnostic-only and does not change traversal, calibration, or
production p-values.

## Key Points

- The three-case overlap run covers `overlap_part_4c_small`,
  `overlap_mod_4c_small`, and `overlap_heavy_4c_small_feat` under selected-null
  and signal roles for the refined global pass-through profile. It writes
  `5,194` node rows and `5,188` parent-child edge rows.
- The paired conditional-topology profile run gives the same spectral-flow
  counts and metric summaries on these three cases. This means the diagnostic
  is measuring the shared selected-tree/data spectral geometry underneath the
  profiles; the observed method differences still live in gate/traversal
  actions and cluster assignments.
- The multiplicity-aware extension writes `1,599` MP block rows and `5,188`
  mode-edge rows for each profile. MP outlier blocks are almost all singleton:
  selected-null has `875` multiplicity-1 blocks and `2` multiplicity-2 blocks;
  signal has `720` multiplicity-1 blocks and `2` multiplicity-2 blocks.
- Mode transport has the same weak direction as the simpler spectral-flow
  panel. `mode_transport_affinity` gives AUC `0.536`; lower connection
  residual gives inverse AUC about `0.536`; lower projector distance gives
  inverse AUC about `0.556`; lower log-eigenvalue center drift gives inverse
  AUC about `0.580`.
- Multiplicity and polynomial terms are now represented, but they are not
  active separators in this overlap run. Median multiplicity distance is `0`,
  and median normalized characteristic-polynomial distance is effectively `0`
  because almost all matched MP blocks are singleton blocks.
- MP-certified modes are sparse. Selected-null rows have median raw MP signal
  count `0` and mean `0.338`; signal rows have median `0` and mean `0.279`.
  The diagnostic therefore marks most edges as
  `floor_only_no_mp_certified_mode`.
- Edge support counts are `2,187` floor-only and `407` MP-supported selected-null
  edges, versus `2,344` floor-only and `250` MP-supported signal edges.
- Among MP-supported edges, signal rows show slightly more coherent spectral
  flow: median MP subspace chordal distance is `0.937` for signal versus
  `0.956` for selected-null, and median log eigenvalue drift is `0.144` versus
  `0.199`.
- The separation is weak but directionally useful. `spectral_flow_affinity`
  gives signal-vs-selected-null AUC `0.584`, MP mean squared cosine gives AUC
  `0.552`, and lower MP chordal distance corresponds to inverse AUC about
  `0.552`.
- The observed effect supports MP eigenvector/eigenvalue flow as a bottleneck
  localizer and stratum variable. It is not strong enough to act as a standalone
  rescue rule or split threshold.
- The implementation compares eigenspaces with singular values of
  `U_parent @ U_child.T`, so individual eigenvector sign flips do not create
  artificial discontinuities. Real rotations of the MP-supported subspace still
  appear as large chordal distance.

## Evidence

- `benchmarks/diagnostics/calibration/selected_neighborhood_spectral_flow.py`
  implements row-level node and edge panels, MP support status separation,
  sign-invariant subspace comparison, spectral-barrier scores, multiplicity
  block construction, normalized characteristic-polynomial fingerprints,
  optimal mode matching, connection-residual summaries, and
  signal-vs-selected-null AUC summaries.
- `tests/validation/146_test_selected_neighborhood_spectral_flow.py` validates
  sign invariance, top-eigenvector rotation detection, coherent-mode barrier
  ordering, MP-supported versus floor-only edge labeling, repeated-eigenvalue
  block construction, within-block rotation invariance, multiplicity/polynomial
  penalties, mode-transport row generation, and AUC summary generation.
- The refined and conditional three-case output directories record manifests,
  node CSVs, edge CSVs, block CSVs, mode-edge CSVs, summary CSVs, and
  separation CSVs used for the counts and AUC values above.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-pvalue-interpolation-comparison-20260616]]
- [[mp-projection-dimension-behavior-sweeps-20260605]]
