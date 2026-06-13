---
title: Barycentric Method Literature Request 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - raw/inbox/barycentric-method-literature-request-20260604.txt
  - manuscript/sections/method/representation.tex
  - manuscript/sections/method/notation.tex
  - manuscript/sections/method/assumptions_validation.tex
  - manuscript/references.bib
  - benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py
  - tests/validation/68_test_selected_tail_law_q5_validation.py
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_summary.csv
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_validation.csv
tags:
  - source
  - barycentric
  - method
  - calibration
---

# Barycentric Method Literature Request 2026-06-04

## Summary

This source captures the request to make the barycentric layer explicit in the
KL-TE method and to use that algebra in later result-finding. The code already
stores internal node distributions as leaf-count weighted empirical subtree
barycenters and existing diagnostics verify the edge/sibling barycentric
z-identity. The follow-up adds manuscript literature framing, explicit
barycentric leverage notation, and an expanded Q5 selected-tail diagnostic.

## Key Points

- The method text now states the parent barycenter
  \(\theta_u=\beta_u\theta_L+(1-\beta_u)\theta_R\) with
  \(\beta_u=n_L/n_u\), and the raw identities
  \(\theta_L-\theta_u=(1-\beta_u)(\theta_L-\theta_R)\) and
  \(\theta_R-\theta_u=-\beta_u(\theta_L-\theta_R)\).
- Under the implemented nested child-parent and sibling variance scales, a
  shared local covariance baseline, no branch-length scaling, and the same
  whitening chart, the manuscript records
  \(z_{L\to u}^{\mathrm{edge}}=z_u^{\mathrm{sib}}\) and
  \(z_{R\to u}^{\mathrm{edge}}=-z_u^{\mathrm{sib}}\).
- The bibliography now includes Bregman centroid and agglomerative references
  for the current coordinatewise barycenter, Wasserstein references for future
  distribution-valued nodes, and Frechet/Karcher/BHV tree-space references for
  future phylogenetic or topological extensions.
- Q5 now derives `barycentric_balance`, `log_barycentric_leverage`, and
  `log_sampling_variance_scale` from left/right child sample sizes and adds
  them to three barycentric selected-tail model variants.
- The expanded Q5 rerun says barycentric leverage improves the low-dimensional
  edge/spectral tail model. `q5_barycentric_edge_spectral` has median
  residual-tail absolute error `0.002386`, compared with `0.007755` for the
  prior edge/spectral model. It is still diagnostic-only because the full
  all-variable laws remain unstable across parent-size and feature-family
  transfer.

## Evidence

- `raw/inbox/barycentric-method-literature-request-20260604.txt` is the
  ingested request.
- `manuscript/sections/method/representation.tex` contains the barycentric
  method insertion, edge/sibling identity, selected-tail leverage variables,
  and future Wasserstein/Frechet/BHV scope boundary.
- `manuscript/sections/method/notation.tex` defines the new barycentric
  symbols.
- `manuscript/references.bib` contains the added Bregman, Wasserstein,
  Frechet/Karcher, phylogenetic tree-space, and Ward references.
- `q5_selected_tail_law_summary.csv` records the expanded six-model Q5 summary.

## Links

- [[selected-tail-law-q5-validation-20260604]]
- [[open-mathematical-questions]]
- [[selected-hierarchy-selection-geometry]]
- [[root-selected-region-model]]
