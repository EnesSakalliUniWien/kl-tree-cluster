---
title: Repository Hygiene and Completion Audit 2026-07-27
type: analysis
status: reviewed
updated: 2026-07-27
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

The current checkout is the newest coherent local development line and passes
the complete 1,252-test suite, package build, Ruff, quick start, and wiki lint.
It is not a finished release: the branch is unpublished, remote development is
not fully consolidated, release metadata and CI are absent, and the maintained
method record still contains production and publication blockers.

## Details

The cleanup establishes one project-health interface through `make check`, one
Pytest configuration in `pyproject.toml`, complete ordered-test coverage, an
explicit MIT license, truthful research-software status, and a root lint seam
that treats the pinned BranchArchitect submodule as external implementation.
The follow-up responsibility pass separates importable methods, dataset
applications, benchmark diagnostics, manuscript figure composition, and
maintenance commands into named directories with README entry maps.

The organization candidate can fast-forward to `dev` after it is committed:
local and remote `dev` share the exact candidate merge base, and an isolated
index check found no source conflict. Merging the present `HEAD` alone would be
incorrect because it would omit the 189-file uncommitted organization pass.
The full integration also carries a much broader retained-evidence history,
including 865 tracked `results/` files, so technical mergeability and evidence
retention approval are separate decisions.

The largest remaining repository risk is retained evidence volume. Generated
and captured evidence spans `raw/`, `reports/`, and an ignored but historically
tracked `results/` tree. Future large outputs need a single explicit retention
policy; historical rewriting is a separate destructive decision.

Completion has two meanings here. The current branch is the strongest local
candidate for continued work because it contains `main` and `dev`. It is not a
canonical final repository until divergent work is reconciled, the branch is
published and merged, open statistical promotion gaps are resolved or scoped,
and a release is versioned and tagged.

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
- [[open-mathematical-questions]]
- [[maintenance]]

## Open Questions

- Which divergent branch commits belong in the canonical integration line?
- Which evidence bundles should remain in Git versus Git LFS or release
  storage?
- What explicit statistical acceptance criteria define version `1.0.0`?
