---
title: Selected Neighborhood Internal Spectral Flow Panel 2026-06-17
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/neighborhood/selected_neighborhood_internal_spectral_flow_panel.py
  - tests/validation/calibration/selected/neighborhood/188_test_selected_neighborhood_internal_spectral_flow_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_overlap_three_case
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_branch_length_overlap_four_case
tags:
  - source
  - diagnostics
  - spectral
  - marchenko-pastur
  - internal-barycenters
  - branch-lengths
  - neighborhood
---

# Selected Neighborhood Internal Spectral Flow Panel 2026-06-17

## Summary

`selected_neighborhood_internal_spectral_flow_panel.py` compares leaf-only
selected-neighborhood spectral flow against the opt-in internal-barycenter
spectral context on identical generated overlap data. The panel keeps traversal
unchanged in production and treats internal distributions as deterministic
tree-filtered support evidence, not as independent Marchenko--Pastur samples.

The three-case overlap run confirms both sides of the current hypothesis.
Internal barycenters substantially increase MP-supported spectral-flow edges,
but they do so almost equally on selected-null and signal rows. This makes
internal distributions useful as a coarse spectral diagnostic and unsafe as an
unconditional split or p-value rescue.

## Key Points

- The run covers `overlap_part_4c_small`, `overlap_mod_4c_small`, and
  `overlap_heavy_4c_small_feat` for `selected_null` and `signal` roles under
  `fixed_coordinate_global_passthrough_refined_v1`.
- It writes `10,388` node rows, `10,376` spectral-flow edge rows, `5,194`
  leaf-versus-internal node-pairwise rows, and `5,188` edge-pairwise rows.
- Leaf-only MP-supported edges are sparse: `430/2,594` selected-null edges and
  `267/2,594` signal edges.
- Internal barycenters increase MP-supported edges to `1,140/2,594`
  selected-null edges and `969/2,594` signal edges.
- The internal channel creates MP support on `710` selected-null edges and
  `702` signal edges. Multiplicity-mode support creation has the same counts.
- Node-level raw MP signal counts also rise similarly: selected-null nodes have
  `445` positive deltas and total delta `581`; signal nodes have `443`
  positive deltas and total delta `557`.
- Newly supported internal edges are not cleanly coherent. Their median
  internal MP chordal distance is about `0.983` for selected null and `0.975`
  for signal, with median internal spectral barriers around `1.145` and
  `1.118`.
- Among already MP-supported edges, internal barycenters improve some barriers
  but degrade more: selected null has `129` improved and `301` degraded edges;
  signal has `91` improved and `175` degraded edges.
- The angular/radial node-pairwise panel separates spike creation from
  rotation and radius change. Internal-only spike creation appears on `315`
  selected-null nodes and `385` signal nodes. Stable angle/stable radius
  matches appear on `834` selected-null nodes and `666` signal nodes.
- Whole-object rotation is not observed in this run. Single-mode rotation is
  rare: `4` selected-null nodes and `3` signal nodes. Mixed angular/radial
  shifts appear on `63` selected-null nodes and `48` signal nodes.
- The neighborhood-energy table aggregates selected parent-child edges into
  graph Dirichlet-style angle/radius energies. It treats parent-child edges as
  unit-weight topology neighbors and separates own-variant MP support from
  strict shared MP support and internal-only MP support.
- Strict shared MP support remains sparse and not cleanly beneficial. Across
  the three cases the strict shared joint-transport delta is `-0.040`,
  `0.028`, and `0.027` for selected null and `-0.022`, `0.085`, and `0.043`
  for signal, so internal filtering smooths one case and degrades two.
- Internal-only support remains the warning channel. Internal-only MP-supported
  edges total `710` for selected null and `702` for signal, and their
  joint-transport energies are similar rather than cleanly separated.
- The summary statuses are
  `diagnostic_warn_selected_null_internal_support_gain` for selected null and
  `diagnostic_signal_internal_mp_support_gain_observed` for signal.

## Evidence

- `selected_neighborhood_internal_spectral_flow_panel.py` implements the
  leaf-only/internal-barycenter variant run, node angular/radial transport,
  edge and mode pairwise joins, support-gain classification, summaries,
  manifest writing, and CLI.
- `tests/validation/calibration/selected/neighborhood/188_test_selected_neighborhood_internal_spectral_flow_panel.py`
  covers edge support creation, barrier improvement, mode-block support
  creation, node angular/radial classification, summary warning status, and
  output writing.
- The output manifest records the two variants:
  `leaf_only_spectral_flow` with
  `spectral_include_internal_barycenters = false`, and
  `internal_barycenter_spectral_flow` with
  `spectral_include_internal_barycenters = true`.
- The summary table records selected-null support creation and signal support
  creation at nearly equal scale, so this diagnostic supports internal
  barycenters as a spectral filter channel but not as a production rescue rule.
- The neighborhood-energy table records the same conclusion in graph-filter
  language: strict shared support can measure local smoothing, but internal-only
  support is mirrored on selected-null rows and must remain fail-closed.

## Branch-Length Internal State Rerun

The panel now has a third diagnostic variant:
`branch_length_internal_state_spectral_flow`. It keeps traversal unchanged but
passes `spectral_internal_distribution_mode = branch_length_state` through the
spectral estimator. This mode does not overwrite stored node distributions.
Instead it appends branch-length-aware latent internal states as spectral rows,
with child precision proportional to descendant leaf support divided by edge
length.

The four-case overlap rerun covers `overlap_extreme_4c`,
`overlap_part_4c_small`, `overlap_mod_4c_small`, and
`overlap_heavy_4c_small_feat`. It writes `22,776` node rows, `22,752`
edge rows, `15,184` node-pairwise rows, `15,168` edge-pairwise rows, and
`16` neighborhood-energy rows. The manifest records three variants:
leaf-only, empirical internal barycenter, and branch-length internal state.

The branch-length internal state remains diagnostic-only. It increases
internal MP-supported edges slightly more than empirical barycenters, but the
increase appears on both selected-null and signal roles:

- selected null: branch-length internal support `1,744/3,792` edges versus
  empirical internal support `1,734/3,792`, with `756` versus `746`
  internal-created support edges.
- signal: branch-length internal support `1,600/3,792` edges versus empirical
  internal support `1,565/3,792`, with `785` versus `750` internal-created
  support edges.

On the hard negative `overlap_extreme_4c`, branch-length internal state again
fails as a rescue rule. Selected-null has `40` internal-only MP-supported edges
under branch-length state versus `36` under empirical barycenters. Signal has
`50` versus `48`. Strict shared transport improves more under branch lengths
than empirical barycenters, but the same selected-null internal-only warning
remains. The status is therefore still
`diagnostic_warn_selected_null_internal_only_transport_energy` for selected
null and diagnostic-only smoothing for signal.

This supports the narrow conclusion: branch lengths are useful for spectral
localization and for separating empirical barycenter rows from latent
tree-state rows, but they do not by themselves turn internal distributions into
valid conditional rescue evidence.

## Links

- [[selected-neighborhood-spectral-flow-diagnostic-20260616]]
- [[selected-neighborhood-bottleneck-law]]
- [[local-marchenko-pastur-rule]]
- [[selected-pca-projected-wald-validation]]
