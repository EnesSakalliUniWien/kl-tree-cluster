---
title: Selected Root Pass-Through Null Fixture 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - raw/inbox/selected-root-pass-through-null-fixture-20260614.md
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/README.md
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/manifest.json
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/feature_matrix.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/labels.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/decision_path.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/node_annotations.csv
tags:
  - source
  - fixture
  - selection
  - traversal
  - null
---

# Selected Root Pass-Through Null Fixture 2026-06-14

## Summary

`raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/`
materializes a compact selected-null failure fixture. The matrix is iid
Bernoulli null data with `400` samples and `80` binary features, regenerated
from the `overlap_mod_4c_small` case dimensions using seed `20310054`. The true
cluster count is one, but replaying
`fixed_coordinate_global_passthrough_refined_v1` returns three clusters.

## Key Points

- The fixture is a clean global null: all truth labels are `0`.
- The raw root sibling p-value is `6.721334448174447e-06`, showing the
  selected-root anti-conservative symptom.
- The root does not open after guards. The root stability guard blocks it, with
  root subsample mean ARI `0.015626276502294405`.
- The false split happens below the closed root. Exactly one descendant node,
  `N797`, remains sibling-open after guards with sibling p-value
  `5.212803085628001e-07`.
- The compact decision table therefore separates the two effects: selected-root
  raw significance is corrected by the root guard, but pass-through traversal
  still exposes a selected descendant under the null.
- The fixture intentionally uses the fixed-coordinate profile. It isolates the
  selected-root/pass-through null failure and does not demonstrate the separate
  adaptive PCA/Marchenko--Pastur projected-Wald failure.

## Evidence

- `feature_matrix.csv` stores the binary feature matrix.
- `labels.csv` stores all-zero truth labels and the replay predicted labels.
- `decision_path.csv` stores the root plus the one open descendant after
  guards.
- `node_annotations.csv` stores the full node-level gate annotations.
- `manifest.json` records the generation seed, profile, observed root fields,
  and replay command.
- `fixed_sibling_gate_profile_validation_rows.csv` records the public runner
  row verifying the same failure.

## Links

- [[fixed-sibling-gate-profile-validation-20260613]]
- [[selected-root-selected-family-traversal-literature-20260614]]
- [[root-selection-literature-20260614]]
- [[open-mathematical-questions]]
