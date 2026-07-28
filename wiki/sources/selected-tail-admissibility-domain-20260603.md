---
title: Selected Tail Admissibility Domain 2026-06-03
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_geometry_covariates.py
  - benchmarks/diagnostics/calibration/selected/tail/selected_tail_admissibility_domain.py
  - tests/validation/calibration/selected/tail/57_test_selected_tail_admissibility_domain.py
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/context_admissibility_domain.csv
  - raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/admissibility_summary_by_run_family.csv
  - raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/nearest_boundary_contexts.csv
tags:
  - source
  - calibration
  - selection
  - tail-law
---

# Selected Tail Admissibility Domain 2026-06-03

## Summary

This diagnostic records the current admissibility domain for external
selected-tail calibration. It combines the broad 200-replicate selected-tail
run, the focused 300-replicate run, and a new 500-replicate boundary expansion
over comparable Gaussian and categorical source families, plus a 600-replicate
binary boundary expansion. It is diagnostic only: admissible rows identify
contexts where the selected-tail support contract is met, while all
non-admissible contexts remain undefined for production external calibration.

## Key Points

- The 500-replicate boundary run used `gauss_null_large`,
  `gauss_clear_medium`, `cat_highcard_20cat_4c`, and
  `cat_highd_3cat_500feat`.
- The 600-replicate binary boundary run used `binary_low_noise_4c` and
  `overlap_heavy_4c_med_feat`.
- Five contexts are production-admissible under the diagnostic support
  contract:
  `gaussian_blobs` and `categorical_multinomial` in `small_0_0.25`,
  `edge_action_ge8`, with sibling projection dimensions `1` and `2`, plus
  `binary_template` in `small_0_0.25`, `edge_action_ge8`, with sibling
  projection dimension `1`.
- The admissible Gaussian contexts have `929` and `928` independent matching
  simulations, held-out exceedance rates `0.009982` and `0.010093`, and
  held-out standard errors `0.000371` and `0.000648`.
- The admissible categorical contexts have `776` and `689` independent
  matching simulations, held-out exceedance rates `0.009987` and `0.010005`,
  and held-out standard errors `0.000358` and `0.000679`.
- The admissible binary context has `606` independent matching simulations,
  `67,831` records, held-out exceedance `0.010025`, and held-out standard
  error `0.000383`. The corresponding binary projection-2 small-parent,
  high-edge context remains support-limited at `445/499` simulations.
- Non-small-parent contexts remain non-admissible even when support is large:
  categorical root/high-edge/projection-2 has `701` matching simulations but
  fails the held-out standard-error contract, and categorical
  medium-parent/high-edge/projection-2 reaches `499` simulations but also
  fails held-out precision.
- The current admissibility domain is therefore not "Gaussian only." It is
  small-parent, high-edge-action selected-tail contexts for Gaussian,
  categorical, and binary projection-1 settings in this targeted panel.
  Continuous, root, medium-parent, large-parent, lower-edge-action, binary
  projection-2, and precomputed-distance contexts remain outside the validated
  production domain.
- The selected-ratio tail-law CSVs now carry the explicit
  `max_exceedance_standard_error` contract column. The admissibility-domain
  table computes `tail_precision_margin` from that recorded value rather than
  from an implicit default.

## Evidence

- `benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_geometry_covariates.py`
  generated the 500-replicate boundary selected-ratio tail-law table.
- `benchmarks/diagnostics/calibration/selected/tail/selected_tail_admissibility_domain.py`
  combines selected-tail runs into context-level admissibility and boundary
  tables.
- `tests/validation/calibration/selected/tail/57_test_selected_tail_admissibility_domain.py` verifies
  the domain classification, recorded precision-contract margin, and output
  writer.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/selected_ratio_tail_law.csv`
  records the 500-replicate boundary context table.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv`
  records the 600-replicate binary boundary context table.
- `raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/context_admissibility_domain.csv`
  records admissibility status for broad, focused, categorical/Gaussian
  boundary, and binary boundary runs.
- `raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/nearest_boundary_contexts.csv`
  records contexts nearest the support and precision boundaries.

## Links

- [[selected-ratio-tail-law-diagnostic-20260602]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-geometric-law-map]]
- [[open-mathematical-questions]]
