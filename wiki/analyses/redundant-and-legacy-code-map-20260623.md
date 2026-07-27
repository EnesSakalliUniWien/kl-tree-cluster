---
title: Redundant and Legacy Code Map 2026-06-23
type: analysis
status: reviewed
updated: 2026-07-27
sources:
  - tree_break_selection/space_separation/diffusion.py
  - tree_break_selection/space_separation/adaptive_cosine.py
  - tree_break_selection/space_separation/invariant_equivariant.py
  - tree_break_selection/tree/io.py
  - tree_break_selection/hierarchy_analysis/bootstrap_consensus.py
  - tree_break_selection/plot/image_panel.py
  - applications/endotypes/_shared.py
  - applications/mnist/_shared.py
  - benchmarks/shared/audit_utils.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/tbs_tree_context.py
  - benchmarks/validation/feature_covariance_calibration.py
  - benchmarks/validation/selected_pca_projected_wald_calibration.py
tags:
  - code-audit
  - legacy
  - redundancy
---

# Redundant and Legacy Code Map 2026-06-23

## Summary

The former importable c2ef9a69 legacy package and its registry bridges were
retired on 2026-06-25. The 2026-07-27 recheck found no high-confidence dead
code in project-owned package, application, script, or benchmark Python files.
It removed the clearest method- and application-level duplication while
leaving broad diagnostic-panel plumbing as a documented maintenance surface.

## Details

The adaptive cosine weighting, eigendecomposition, segmentation, and block
coordinate logic previously lived inside a benchmark probe and was consumed by
applications. It now has one implementation in
`tree_break_selection/space_separation/adaptive_cosine.py`. The scRNA
invariant/equivariant decomposition likewise moved from a dataset script into
`tree_break_selection/space_separation/invariant_equivariant.py`.
The fixed Hamming-neighbor diffusion engine is now public beside adaptive and
block diffusion in `tree_break_selection/space_separation/diffusion.py`, so
applications and experiments no longer import a private benchmark-runner
function.

Two copied image-panel renderers were replaced by
`tree_break_selection/plot/image_panel.py`. Repeated endotype filename and
matrix-slug rules now use `applications/endotypes/_shared.py`; duplicated MNIST
summary-selection and compact digit-count parsing use
`applications/mnist/_shared.py`.

The tree-construction recheck found no second implementation of neighbor
joining, IQ-TREE import, MAD rooting, or branch-length NNLS. It did find a
residual repeated sequence—condensed distance, SciPy `linkage`, then
`PosetTree.from_linkage`—in the production runner, bootstrap analysis,
diagnostic tree context, experiments, and application adapters. Those callers
have different data contracts and output needs, so they were mapped rather
than bulk-rewritten. `PosetTree.from_agglomerative` and
`PosetTree.from_undirected_edges` have no live in-repository caller outside
tests; both remain documented public representation adapters rather than being
deleted as dead code. See [[tree-construction-method-map]].

A structural AST comparison still finds repeated small helpers across the
large calibration-diagnostic surface, especially `_require_columns`,
`_finite_float`, `_string_value`, `_json_default`, CLI parsers, and shard
plumbing. Some shared implementations already exist in
`benchmarks/shared/audit_utils.py`. Migrating dozens of active research panels
without contract tests would create more risk than the duplication currently
does, so this pass records the seam instead of applying a bulk rewrite.

The two validation programs
`benchmarks/validation/feature_covariance_calibration.py` and
`benchmarks/validation/selected_pca_projected_wald_calibration.py` still share
report-contract and interval helpers. This is the highest-value remaining
consolidation once their output schemas are covered by focused tests.

## Evidence

- Vulture at 90% confidence reported no unused project-owned code after the
  reorganization.
- Ruff passes over the relocated library, application, benchmark, and test
  surfaces.
- Focused method, endotype, scRNA, MNIST, and benchmark-interface tests pass.
- Exact function-body comparison identified the remaining repeated diagnostic
  helpers described above; it found and motivated the endotype and MNIST
  application helper consolidation.
- Exact constructor search separated the three live registered topology
  builders from repeated linkage call sites and two test-only public adapters.

## Links

- [[project-overview]]
- [[repository-hygiene-and-completion-audit-20260727]]
- [[method-application-and-plot-seams-20260727]]
- [[tree-construction-method-map]]

## Open Questions

- Which diagnostic helper contracts are stable enough to move into
  `benchmarks/shared/` without masking panel-specific validation semantics?
- Should the two calibration validation reports first receive golden-schema
  tests, then share one report-contract module?
- After application output contracts are covered, should direct
  distance/linkage/`PosetTree` sequences share one deep construction interface
  in `tree_break_selection/tree/`?
