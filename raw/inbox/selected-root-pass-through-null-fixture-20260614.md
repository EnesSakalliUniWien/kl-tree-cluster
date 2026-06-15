---
title: Selected Root Pass-Through Null Fixture 2026-06-14
captured: 2026-06-14
artifact_paths:
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/README.md
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/manifest.json
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/feature_matrix.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/labels.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/decision_path.csv
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/node_annotations.csv
---

# Selected Root Pass-Through Null Fixture 2026-06-14

This fixture materializes a clean selected-null KL-TE failure as durable raw
evidence. It uses the `overlap_mod_4c_small` benchmark dimensions only:
`400` samples and `80` binary features. The data are regenerated as iid
Bernoulli null data with seed `20310054`, so the true cluster count is one.

Replay profile:

- `fixed_coordinate_global_passthrough_refined_v1`
- edge alpha `0.001`
- sibling alpha `0.01`
- Hamming distance
- average linkage

Observed result:

- true clusters: `1`
- found clusters: `3`
- ARI: `0.0`
- false split: `true`
- root node: `N798`
- root raw sibling p-value: `6.721334448174447e-06`
- root sibling open after guards: `false`
- root stability guard blocked: `true`
- root stability subsample mean ARI: `0.015626276502294405`
- root selected permutation p-value: `0.46`
- open descendant sibling nodes after guards: `N797`

The compact decision path is:

```text
N797: descendant open, sibling p = 5.212803085628001e-07
N798: root closed, raw root sibling p = 6.721334448174447e-06, root stability blocked
```

Interpretation:

The fixture isolates the selected-root/pass-through null failure. The raw root
statistic is anti-conservative under selected topology, but the root guard
closes it. The final false split happens because traversal still reaches one
open descendant below the closed root. This fixture intentionally uses the
fixed-coordinate profile, so it does not demonstrate the separate adaptive
PCA/MP projected-Wald failure.
