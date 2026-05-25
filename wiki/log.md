---
title: Wiki Log
type: control
status: reviewed
updated: 2026-05-25
sources:
  - AGENTS.md
  - raw/inbox/wiki-construction-brief.md
tags:
  - wiki
  - log
---

# Wiki Log

## Summary

Append meaningful scaffold, ingest, query, analysis, implementation, review,
verification, and maintenance events here in chronological order.

## Details

### 2026-05-24

- Scaffolded the docs-as-code wiki structure with `raw/`, `wiki/`, source,
  concept, entity, analysis, question, tool, candidate, and template
  directories.
- Added the root operating guide in `AGENTS.md`.
- Added [[schema]], [[maintenance]], [[wiki-search]], templates, and the
  `make wiki-lint` validation path.
- Captured the wiki construction brief in `raw/inbox/` and summarized it as
  [[wiki-construction-brief]].
- Seeded project and method pages: [[project-overview]], [[kl-te-method]],
  [[poset-tree]], [[tree-decomposition]], [[projected-wald-statistic]], and
  [[top-down-traversal]].
- Recorded wiki design references in [[github-wiki-structure-research]].
- Opened maintenance questions: [[generated-index]] and
  [[markdown-validation]].
- Added [[oracle-gate-path-diagnostic]] to record the mathematical
  recoverability oracle, corrected failure classes, gate-path trace evidence,
  and method implications from the current full KL benchmark diagnostics.
- Updated [[oracle-gate-path-diagnostic]] with the sibling-inflation follow-up:
  leave-one-out inflation still blocks the high-dimensional binary Gaussian
  case, the continuous Gaussian case lacks positive calibration support, and
  binary/categorical blockers are sibling-FDR failures rather than inflation
  failures. Recorded the next method step as a calibration-support contract
  plus a separate sibling-FDR analysis.
- Defined the sibling calibration-support contract in
  [[oracle-gate-path-diagnostic]]: strict empirical-null support,
  stopped-or-null support, selected non-null context only, and unsupported
  without target. Recorded that unsupported high-dimensional Gaussian contexts
  require an explicit calibration-data error or a separately validated external
  calibration model rather than a neutral fallback.
- Ran the support-status diagnostic on the two high-dimensional Gaussian
  blockers. Both lack admissible internal empirical-null calibration support;
  the binary case has selected non-null context only, and the continuous case
  lacks positive non-focal local calibration weight.
- Added the fixed-subspace Gaussian sibling-null diagnostic. The two
  high-dimensional Gaussian blockers have external mean-over-reference ratios
  near one, while runtime empirical inflation remains in the thousands. This
  points to a selection-conditioned calibration problem, not a failure of the
  projected-Wald chi-square reference.
- Added the level-1 local edge-selection sibling-null diagnostic. Conditioning
  on the child-parent edge gate raises the null ratio only to about 2.8--3.4
  for the two high-dimensional Gaussian blockers, still far below the runtime
  empirical inflation factors.
- Implemented the strict production sibling-calibration contract: the fitted
  inflation model now admits only strict-null or edge-blocked/stopped
  positive-weight records. Selected non-null records are rejected as
  calibration support instead of being used to estimate empirical-null
  inflation.
- Added the fixed-tree root Tree-BH selection diagnostic. For the two
  high-dimensional Gaussian blockers, the Tree-BH root edge-path event is
  equivalent to local edge selection and gives \(c\) about 2.8--3.5; production
  calibration is now reported as unsupported.

### 2026-05-25

- Tightened the calibration-support interpretation after the root Tree-BH
  diagnostic: current evidence does not justify a named external
  selection-conditioned production model. Missing internal empirical-null
  support remains a fail-closed production error; deeper full-selection
  conditioning is a research diagnostic, not a fallback.
- Consolidated repository structure: user-facing analysis commands now live in
  `scripts/analysis/`, benchmark diagnostics remain under
  `benchmarks/diagnostics/`, canonical feature matrices live under
  `data/feature_matrices/`, duplicate HC/CMS matrix aliases were removed, and
  generated report/log/notebook/profiling artifacts were removed from tracked
  source control.
- Added a new-contributor onboarding route in `docs/onboarding.md`, filled the
  statistics and benchmark-shared directory maps, and corrected stale package
  metadata/test documentation so a cold-start reader has one route through the
  project.

## Evidence

- `raw/inbox/wiki-construction-brief.md` records the requested scaffold.
- `AGENTS.md` records the operating guide.

## Links

- [[project-overview]]
- [[wiki-construction]]
- [[schema]]
- [[maintenance]]
- [[wiki-search]]
