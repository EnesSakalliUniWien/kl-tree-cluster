---
title: Project Overview
type: project
status: reviewed
updated: 2026-05-25
sources:
  - README.md
  - manuscript/guides/full_method_logic_map.md
  - manuscript/sections/method/overview.tex
tags:
  - project
  - kl-te
---

# Project Overview

## Summary

This repository implements and documents KL-TE, a clustering workflow that
starts from a discrete sample-feature matrix and a rooted binary candidate
hierarchy, evaluates local splits with child-parent and sibling tests, and
returns terminal clusters through a top-down traversal.

## Details

The source code centers on `kl_clustering_analysis/`, with tree structures in
`kl_clustering_analysis/tree/`, decomposition logic in
`kl_clustering_analysis/hierarchy_analysis/`, statistical tests under
`kl_clustering_analysis/hierarchy_analysis/statistics/`, benchmark and
diagnostic harnesses under `benchmarks/`, user-facing real-data commands under
`scripts/analysis/`, and validation coverage under `tests/`.

Tracked feature matrices now have one canonical domain:
`data/feature_matrices/`. External reference tables live under
`data/reference/`. Generated reports, logs, profiling outputs, notebook figure
exports, and large local datasets are not tracked by default unless they are
explicitly retained as evidence.

The manuscript describes the inferential objective: the hierarchy proposes
candidate splits, but the statistical method decides whether each split has
enough evidence to remain in the final partition. The durable method concepts
are [[kl-te-method]], [[projected-wald-statistic]], and
[[top-down-traversal]].

The main implementation entities are [[poset-tree]] and
[[tree-decomposition]].

## Evidence

- `README.md` maps the package, pipeline workflow, and testing entry points.
- `data/README.md` and `reports/README.md` define the current data/report
  storage contract.
- `manuscript/guides/full_method_logic_map.md` states the mathematical
  contract, assumptions, constants, and submission gaps.
- `manuscript/sections/method/overview.tex` defines the inferential role of
  edge and sibling tests.

## Links

- [[kl-te-method]]
- [[poset-tree]]
- [[tree-decomposition]]
- [[projected-wald-statistic]]
- [[top-down-traversal]]

## Open Questions

- Which validation outputs should become the locked manuscript evidence set?
- Which method constants should be promoted from implementation defaults to
  justified manuscript choices?
