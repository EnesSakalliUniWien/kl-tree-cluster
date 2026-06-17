---
title: Legacy c2ef9a69 Method Package 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/METADATA.md
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/__init__.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/tree/poset_tree.py
  - benchmarks/shared/runners/legacy_commit_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/util/method_sets.py
  - tests/pipeline/51_test_dispatch_contract.py
tags:
  - source
  - diagnostics
  - legacy
  - benchmark
---

# Legacy c2ef9a69 Method Package 2026-06-16

## Summary

The full old `kl_clustering_analysis` method package from commit
`c2ef9a69e0888168950bdee4a41ae8ab9996e32f` is now available as an isolated
nested package:

`kl_clustering_analysis.legacy_methods.commit_c2ef9a69.kl_clustering_analysis`

This is a whole-package method snapshot, not only the internal-node spectral
patch. Its absolute imports were mechanically rewritten into the nested
namespace so the old implementation can be imported in the same Python process
as the current code.

## Key Points

- The package provenance is recorded in
  `kl_clustering_analysis/legacy_methods/commit_c2ef9a69/METADATA.md`.
- The standard benchmark method id is `kl_legacy_c2ef9a69`.
- The runner in `legacy_commit_runner.py` calls the old `PosetTree` and
  old `TreeDecomposition` directly.
- The old method intentionally ignores modern-only options such as typed
  `FeatureSpace`, phylogenetic builders, root stability guards, and spectral
  transport. Using those would turn the snapshot into a hybrid rather than the
  full old method.
- The dispatcher smoke test confirms that `kl_legacy_c2ef9a69` runs through the
  current benchmark contract and records the legacy commit hash in result
  metadata.

## Evidence

- `legacy_commit_runner.py` exposes `_run_legacy_c2ef9a69_kl_method`.
- `method_registry.py` registers `kl_legacy_c2ef9a69` with hamming distance,
  average linkage, and linkage-root tree construction.
- `method_sets.py` adds the method id to the KL runner method set.
- `51_test_dispatch_contract.py` verifies package import, registry exposure,
  dispatcher argument flow, and a small real-run smoke through
  `run_clustering_result`.

## Links

- [[legacy-internal-spectral-comparison-panel-20260616]]
- [[old-vs-current-method-stack-comparison-20260615]]
- [[local-marchenko-pastur-rule]]
