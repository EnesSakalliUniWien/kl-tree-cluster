---
title: Projected-Wald Statistic
type: concept
status: reviewed
updated: 2026-05-24
sources:
  - manuscript/guides/full_method_logic_map.md
  - manuscript/sections/method/edge_test.tex
  - manuscript/sections/method/sibling_test.tex
tags:
  - method
  - statistics
---

# Projected-Wald Statistic

## Summary

The projected-Wald statistic is the projected quadratic form used to combine a
standardized high-dimensional contrast into a lower-dimensional test statistic
with a chi-square reference under the fixed-subspace approximation.

## Details

For edge tests, the raw child-parent contrast is standardized coordinatewise
with the nested variance model. For sibling tests, the left-right contrast is
standardized with the two-sample Bernoulli variance model. In both cases, the
standardized vector is projected into parent-local spectral directions before
summing squared projected coordinates.

The manuscript terminology distinguishes the implemented orthonormal
projected-Wald reference from alternative whitening statistics. Eigenvalues
select the local subspace, while the implemented reference compares the raw
orthonormal projected quadratic against a chi-square distribution with the
selected projection dimension.

## Evidence

- `manuscript/sections/method/edge_test.tex` defines the nested edge contrast,
  local spectral summary, and projected edge p-value.
- `manuscript/sections/method/sibling_test.tex` defines the sibling contrast,
  sibling projection dimension, and orthonormal reference law.
- `manuscript/guides/full_method_logic_map.md` gives the canonical
  terminology and assumptions.

## Links

- [[kl-te-method]]
- [[top-down-traversal]]
- [[tree-decomposition]]

## Open Questions

- Does the final manuscript present a proof under fixed projection, or clearly
  label the data-dependent projection as a working approximation?
- Which sensitivity analysis justifies the minimum spectral dimension of `2`?
