---
title: Selected PCA Projected-Wald Validation
type: analysis
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/validation/statistics/selected_pca_projected_wald_calibration.py
  - benchmarks/validation/manifests/selected_pca_projected_wald_validation_manifest.json
  - raw/assets/selected-pca-projected-wald-validation/20260601-selected-pca-projected-wald-calibration.json
  - raw/assets/selected-pca-projected-wald-validation/20260601-selected-pca-projected-wald-calibration.csv
  - benchmarks/results/validation/selected_pca_projected_wald_calibration_check_20260605.json
  - benchmarks/results/validation/selected_pca_projected_wald_calibration_check_20260605.csv
  - tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py
  - tree_break_selection/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
tags:
  - method
  - projection
  - validation
---

# Selected PCA Projected-Wald Validation

## Summary

The fixed-subspace projected-Wald reference behaves as expected when PCA rows
are selected from leaf-only Gaussian null rows in the local sibling-null
diagnostic, but it fails badly when deterministic child-mean rows are included
in the same local spectral matrix. The locked run used commit
`492c8520809e6cfad9e4853e91c89a887eac5a21`, seed `20260601`, 1000 replicates
per setting, and an empty recorded worktree status.

This is not full-pipeline evidence. It validates one local question: whether
the chi-square projected-Wald reference remains calibrated when PCA rows and the
MP dimension are selected from the same fixed-membership Gaussian null context.
It does not validate hierarchy construction, tree-selected sibling pairs,
sibling FDR, traversal, empirical-null inflation, categorical blocks, or
real-data misspecification.

A fresh 2026-06-05 rerun on the current worktree reproduces the locked
pattern: leaf-only rejection rates at nominal `0.05` are `0.046`, `0.048`, and
`0.059`, while the corresponding child-mean-row settings reject at `0.671`,
`0.992`, and `1.000`.

## Details

The diagnostic simulates two sibling samples from the same Gaussian
distribution. For each replicate, it computes the production
empirical-Gaussian sibling Wald contrast, maps the local rows into the
null-whitened tangent coordinates used by the spectral backend, selects PCA rows
and projection dimension through the production MP path, and evaluates the
production projected-Wald kernel.

The paired validation grid compares leaf-only local spectral rows against the
same setting with child-mean rows appended as deterministic internal rows:

| setting | row mode | rejection rate at 0.05 | 95% Wilson interval | mean p-value | KS p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| 8-dimensional identity Gaussian, 80x80 | leaves | 0.047 | 0.0355--0.0619 | 0.5023 | 0.9673 |
| 8-dimensional identity Gaussian, 80x80 | child means | 0.648 | 0.6179--0.6770 | 0.0625 | 0 |
| 16-dimensional AR(1) Gaussian, 60x60 | leaves | 0.045 | 0.0338--0.0597 | 0.5003 | 0.6877 |
| 16-dimensional AR(1) Gaussian, 60x60 | child means | 0.988 | 0.9791--0.9931 | 0.0036 | 0 |
| 32-dimensional ill-conditioned Gaussian, 50x50 | leaves | 0.050 | 0.0381--0.0653 | 0.4861 | 0.1987 |
| 32-dimensional ill-conditioned Gaussian, 50x50 | child means | 1.000 | 0.9962--1.0000 | 0.0000053 | 0 |

All six settings used mean projection dimension 2 and mean raw MP signal count
0. The anti-conservative behavior is therefore not caused by the MP rule
counting many false spikes in this run. It arises even when the selected
dimension is only the configured spectral floor. The leaf-only rows are
compatible with the fixed-subspace approximation in this diagnostic; the
child-mean rows are not.

This means Q12 and Q13 should be kept separate. Q12's fixed-projection
chi-square statement is valid only under fixed or conditionally independent
valid projections. Q13's selected-PCA question is not globally negative:
leaf-only fixed-membership Gaussian selected PCA is calibrated in this
diagnostic. The failure is the leakage mode where deterministic summaries of
the tested sibling means enter the spectral row set, plus the broader
selected-tree setting where the hierarchy and focal sibling are selected using
the same evidence.

The mathematical interpretation is direct. In the leaf-only Gaussian null, the
selected PCA rows are random but selected from independent null variation
around the parent mean. The fixed-subspace chi-square reference remains
approximately calibrated in the tested settings. When child means are appended,
the spectral matrix contains deterministic summaries of exactly the two groups
used by the sibling contrast. The selected subspace can then align with the
tested left-right contrast, so conditioning on the supplied projection rows as
fixed no longer approximates the actual selected reference law.

## Evidence

- `raw/assets/selected-pca-projected-wald-validation/20260601-selected-pca-projected-wald-calibration.json`
  records the locked JSON report, including commit, command, seed, worktree
  status, grid, confidence intervals, p-value uniformity summaries, and
  limitations.
- `raw/assets/selected-pca-projected-wald-validation/20260601-selected-pca-projected-wald-calibration.csv`
  records the row-level summary table.
- `benchmarks/results/validation/selected_pca_projected_wald_calibration_check_20260605.csv`
  records the current-worktree rerun used to check that the locked conclusion
  still reproduces.
- `benchmarks/validation/statistics/selected_pca_projected_wald_calibration.py` defines
  the scaffold and validation contract.
- `tree_break_selection/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py`
  documents the current production leaf-only spectral orchestration. The
  child-mean/internal-row mode is preserved only in the locked validation
  artifact as the anti-conservative comparison arm.

## Links

- [[open-mathematical-questions]]
- [[local-marchenko-pastur-rule]]
- [[projected-wald-statistic]]

## Open Questions

Production now excludes internal spectral rows from the projected-Wald PCA
basis. The remaining method question exposed by that change is sibling
calibration support: high-dimensional and high-cardinality contexts can have no
strict empirical-null calibration records even when the raw sibling evidence is
strong. That requires either a validated external/null calibration object or an
explicit unsupported status; it should not be patched by reintroducing internal
summary rows into the inferential basis.
