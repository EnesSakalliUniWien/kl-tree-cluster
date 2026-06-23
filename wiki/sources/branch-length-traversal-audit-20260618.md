---
title: Branch Length Traversal Audit 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/manifest.json
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/traversal_tuples.csv
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/traversal_edges.csv
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/traversal_case_summary.csv
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/method_rows.csv
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/run.log
  - raw/assets/benchmark-results/branch_length_traversal_audit_20260618/verification.log
tags:
  - source
  - benchmarks
  - traversal
  - branch-length
  - guarded
---

# Branch Length Traversal Audit 2026-06-18

## Summary

This audit records fixed-candidate traversal evidence for current `tbs` and
`tbs_internal_filter_branch_length_v1`. It does not define an adaptive routing
policy. For each successful method row it saves the live traversal counters and
an edge-reachable traversal that walks from the root until child-parent edge
tests close, along with parent/left/right tuples and branch lengths.

## Key Points

- The run covers `16` selected cases, `2` methods, and `32` method rows.
- Current `tbs` has `9` OK rows and `7` skips; branch-length internal filtering
  has `11` OK rows and `5` skips on this targeted panel.
- The diagnostic writes `3032` traversal tuple rows and `3352` traversal edge
  rows.
- Current `tbs` contributes `477` live visited nodes and `1713` full edge-
  reachable nodes across OK rows.
- Branch-length internal filtering contributes `277` live visited nodes and
  `1319` full edge-reachable nodes across OK rows.
- No branch lengths are missing on recorded binary traversal tuples in this
  run.

## Interpretation

The edge-reachable trace separates two objects that should not be conflated:
the clustering traversal that actually forms clusters, and the larger
edge-supported traversal map that shows where the selected tree still has
child-parent edge evidence. This is the correct next audit surface for the
branch-length candidate because it localizes traversal stops, skipped rows, and
branch-length evidence without choosing methods after observing outcomes.

## Evidence

- `traversal_tuples.csv` stores one row per edge-reachable node visit, including
  live-visit status, live decision, child edge-test states, sibling gate state,
  parent/left/right tuple ids, and left/right branch lengths.
- `traversal_edges.csv` maps tuple rows into parent-child edge rows with
  branch length and child-edge status.
- `traversal_case_summary.csv` stores method status, ARI where available, and
  live/full traversal counters per case and method.
- `run.log` records command context and per-case method progress.
- `verification.log` records targeted tests, CSV sanity checks, `git diff
  --check`, and the current `make wiki-lint` result.

## Links

- [[branch-length-candidate-promotion-audit-20260618]]
- [[branch-length-candidate-full-big-20260618]]
- [[top-down-traversal]]
