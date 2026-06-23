---
title: Project Overview
type: project
status: reviewed
updated: 2026-06-23
sources:
  - README.md
  - docs/onboarding.md
  - manuscript/guides/full_method_logic_map.md
  - manuscript/sections/method/overview.tex
tags:
  - project
  - tree-break-selection
---

# Project Overview

## Summary

This repository implements and documents Tree-Break Selection, a clustering workflow that
starts from a typed sample-feature matrix and a rooted binary candidate
hierarchy, evaluates local splits with child-parent and sibling tests, and
returns terminal clusters through a top-down traversal. The active feature-space
contract covers Bernoulli coordinates, categorical one-hot blocks with
drop-last multinomial covariance, and explicit continuous empirical-Gaussian
blocks.

## Details

The source code centers on `tree_break_selection/`, with tree structures in
`tree_break_selection/tree/`, decomposition logic in
`tree_break_selection/hierarchy_analysis/`, statistical tests under
`tree_break_selection/hierarchy_analysis/statistics/`, benchmark and
diagnostic harnesses under `benchmarks/`, user-facing real-data commands under
`scripts/analysis/`, and validation coverage under `tests/`.

Tracked feature matrices now have one canonical domain:
`data/feature_matrices/`. External reference tables live under
`data/reference/`. Generated reports, logs, profiling outputs, notebook figure
exports, and large local datasets are not tracked by default unless they are
explicitly retained as evidence.

New contributors should start with `docs/onboarding.md` for the first-pass
route through package code, benchmarks, tests, wiki, and manuscript context.

The manuscript describes the inferential objective: the hierarchy proposes
candidate splits, but the statistical method decides whether each split has
enough evidence to remain in the final partition. The durable method concepts
are [[tree-break-selection]], [[projected-wald-statistic]], and
[[top-down-traversal]].

The main implementation entities are [[poset-tree]] and
[[tree-decomposition]].

## Evidence

- `README.md` maps the package, pipeline workflow, and testing entry points.
- `docs/onboarding.md` defines the new-contributor route and the main code,
  benchmark, and testing paths.
- `data/README.md` and `reports/README.md` define the current data/report
  storage contract.
- `manuscript/guides/full_method_logic_map.md` states the mathematical
  contract, assumptions, constants, and submission gaps.
- `manuscript/sections/method/overview.tex` defines the inferential role of
  edge and sibling tests.

## Links

- [[tree-break-selection]]
- [[poset-tree]]
- [[tree-decomposition]]
- [[projected-wald-statistic]]
- [[top-down-traversal]]

## Open Questions

- Which validation outputs should become the locked manuscript evidence set?
- Which method constants should be promoted from implementation defaults to
  justified manuscript choices?
