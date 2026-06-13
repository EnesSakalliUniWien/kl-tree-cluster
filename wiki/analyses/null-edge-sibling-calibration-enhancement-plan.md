---
title: Null Edge Sibling Calibration Enhancement Plan
type: analysis
status: reviewed
updated: 2026-06-06
sources:
  - docs/superpowers/specs/2026-06-06-path-conditioned-barycentric-action-diagnostics-design.md
  - wiki/sources/path-conditioned-barycentric-action-diagnostics-20260606.md
  - wiki/sources/barycentric-action-equation-diagnostic-20260606.md
  - wiki/sources/root-selected-region-margins-20260603.md
  - wiki/sources/selected-tail-promotion-gate-debug-20260605.md
  - wiki/sources/internal-calibration-q9-q10-q11-debug-20260605.md
tags:
  - calibration
  - edge
  - sibling
  - diagnostics
  - plan
---

# Null Edge Sibling Calibration Enhancement Plan

## Summary

The path-conditioned barycentric action analyses are integrated as diagnostic
evidence, not as production calibration. The next calibration work should
separate edge null calibration, sibling null calibration, and traversal
geometry. The KAK radius/angle/action result is useful as a traversal
stratifier and guard-panel candidate, but not as a selected-tail p-value
correction.

## Details

The method has three coupled but distinct mathematical objects:

1. Edge null calibration: the law of child-parent edge statistics under fixed
   or selected hierarchy/projection contexts.
2. Sibling null calibration: the law of sibling statistics after the same
   barycentric direction has also participated in edge-path opening and
   traversal selection.
3. Traversal geometry: radius, angle, independent shell mass, and action
   budget as predictors of fragmentation or pass-through behavior.

The implemented diagnostics cover these objects only descriptively. The exact
barycentric identity is now traced; KAK radius/angle/action is reproduced with
median AUC `0.892810`; and the high-action angular-shell guard panel identifies
candidate thresholds such as `action_ge_0.9__angle_ge_75__ind_ge_0.85`. These
results should drive validation panels rather than production rules.

The calibration enhancement plan therefore proposes four phases:

- an edge null diagnostic panel with fixed-tree null, selected-tree null, and
  selected-tree signal rows;
- a sibling null diagnostic panel with strict-null, stopped-edge, selected-
  non-null-only, and externally selected-tail contexts;
- a traversal geometry guard validation panel using action budget and angular
  shell variables;
- a production-admissible decision contract with explicit fail-closed statuses.

The production rule remains strict: unsupported sibling calibration contexts
must fail closed, and external selected-tail calibration may not be used unless
support, precision, and held-out tail validation pass in predeclared contexts.

## Evidence

- [[path-conditioned-barycentric-action-diagnostics-20260606]] records the
  implemented diagnostic outputs and guard-panel result.
- [[barycentric-action-equation-diagnostic-20260606]] separates selected-tail
  calibration from KAK traversal fragmentation.
- [[root-selected-region-margins-20260603]] records edge-opening and selected
  root-region diagnostic objects.
- [[selected-tail-promotion-gate-debug-20260605]] records why external
  selected-tail promotion remains blocked.
- [[internal-calibration-q9-q10-q11-debug-20260605]] records the current
  internal support-threshold and weight-leakage status.

## Links

- [[open-mathematical-questions]]
- [[selected-hierarchy-null-support-contract]]
- [[top-down-traversal]]
- [[projected-wald-statistic]]

## Open Questions

- Which selected-edge context variables are sufficient for held-out edge null
  calibration?
- Can sibling external selected-tail support be reached without sparse or
  hidden context borrowing?
- Does a high-action angular-shell traversal guard reduce pure-fragment splits
  without blocking true mixed-parent signal?
- Which selected-basis validation is required before KAK or cosine coordinates
  can enter any production traversal guard?
