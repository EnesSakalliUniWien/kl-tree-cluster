---
title: Repository Hygiene and Completion Audit 2026-07-27
type: analysis
status: reviewed
updated: 2026-07-28
sources:
  - docs/repository_hygiene_audit_20260727.md
  - README.md
  - pyproject.toml
  - tests/README.md
  - wiki/questions/open-mathematical-questions.md
tags:
  - repository
  - hygiene
  - release-readiness
---

# Repository Hygiene and Completion Audit 2026-07-27

## Summary

The current checkout is the published `dev` development line. Local and remote
branch state has been reduced to `main` and `dev`; the former topology,
analysis-results, and publication branches and their worktrees were explicitly
removed. The reorganized repository passed Ruff, wiki lint, the complete
1,251-test suite, package build, quick start, benchmark smoke, and focused
benchmark checks.

It is still not a finished release: release metadata and CI are absent, the
descriptive regression runner enforces no acceptance thresholds, and the
maintained method record contains unresolved production and publication
blockers.

## Details

The cleanup establishes one project-health interface through `make check`, one
Pytest configuration in `pyproject.toml`, complete ordered-test coverage, an
explicit MIT license, truthful research-software status, and a root lint seam
that treats the pinned BranchArchitect submodule as external implementation.
The follow-up responsibility pass separates importable methods, dataset
applications, benchmark diagnostics, manuscript figure composition, and
maintenance commands into named directories with README entry maps.

The organization work is committed and published on `dev`. `main` remains the
older release line. There is no remaining divergent topic branch to reconcile,
and the primary worktree is clean. The full integration still carries a broad
retained-evidence history, including historically tracked `results/` files, so
branch consolidation does not settle the separate evidence-retention policy.

The largest remaining repository risk is retained evidence volume. Generated
and captured evidence spans `raw/`, `reports/`, and an ignored but historically
tracked `results/` tree. Future large outputs need a single explicit retention
policy; historical rewriting is a separate destructive decision.

Completion has two meanings here. `dev` is now the single canonical development
line and contains `main`. It is not a finished release until open statistical
promotion gaps are resolved or scoped, benchmark acceptance criteria are
explicit, CI/release automation exists, and a release is versioned and tagged.

## Evidence

- `docs/repository_hygiene_audit_20260727.md` records Git ancestry, remote
  state, executable checks, cleanup, and ranked follow-up work.
- `README.md` now states the active-research status and exposes `make check`.
- `pyproject.toml` owns package, Pytest, license, and Ruff configuration.
- `tests/README.md` and `scripts/run_tests_ordered.py` cover every test surface.
- `wiki/questions/open-mathematical-questions.md` records the unresolved
  calibration and promotion gaps that prevent a publication-ready claim.
- An isolated temporary index assembled every organization change without
  changing the real index and proved `dev` is a fast-forward ancestor.

## Links

- [[project-overview]]
- [[repository-execution-and-benchmark-map]]
- [[open-mathematical-questions]]
- [[maintenance]]

## Open Questions

- Which evidence bundles should remain in Git versus Git LFS or release
  storage?
- What explicit statistical acceptance criteria define version `1.0.0`?
