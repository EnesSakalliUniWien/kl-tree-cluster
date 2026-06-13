---
title: Calibration Contract Enhancement Request 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - raw/inbox/calibration-contract-enhancement-request-20260604.txt
tags:
  - source
  - calibration
  - selection
  - manuscript
---

# Calibration Contract Enhancement Request 2026-06-04

## Summary

This request reviews the active KL-TE calibration path and argues that the
method should preserve its fail-closed internal empirical-null contract while
making calibration statuses explicit. It also identifies a manuscript mismatch:
the production spectral basis uses descendant leaf rows only, while
`manuscript/sections/method/edge_test.tex` still described adding internal
descendant distributions to the local spectral matrix.

## Key Points

- Production sibling calibration should not use external Gaussian diagnostics
  as a fallback. External selected-tail calibration belongs in production only
  after a predeclared admissible context and held-out precision checks.
- Runtime code should expose a typed calibration decision with status, scalar
  factor, adjusted p-value, support metrics, exact context, and descriptive
  strata so unsupported states are visible at the use site.
- Internal support and focal prediction support should be separated. A model
  can have global internal support while a target feature family or local
  context remains unsupported.
- The current strict rejection of selected non-null positive-weight records is
  scientifically correct and should not be weakened into a neutral or external
  fallback.
- The edge-test manuscript should state the current leaf-only inferential PCA
  basis rather than the historical internal-row spectral matrix.

## Evidence

- `raw/inbox/calibration-contract-enhancement-request-20260604.txt` is the
  ingested review and enhancement request.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-tail-admissibility-domain-20260603]]
- [[selected-pca-projected-wald-validation]]
- [[local-marchenko-pastur-rule]]
