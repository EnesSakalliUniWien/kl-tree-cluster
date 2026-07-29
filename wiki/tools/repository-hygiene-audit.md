---
title: Repository Hygiene Audit
type: tool
status: reviewed
updated: 2026-07-29
sources:
  - tools/repository_audit/README.md
  - tools/repository_audit/pyproject.toml
  - tools/repository_audit/src/tbs_repo_audit/cli.py
  - tools/repository_audit/src/tbs_repo_audit/coverage_evidence.py
  - tools/repository_audit/src/tbs_repo_audit/field_lineage.py
  - tools/repository_audit/src/tbs_repo_audit/inventory.py
  - tools/repository_audit/tests/test_cli.py
  - tools/repository_audit/tests/test_coverage_evidence.py
  - tools/repository_audit/tests/test_field_lineage.py
  - tools/repository_audit/tests/test_inventory.py
tags:
  - hygiene
  - diagnostics
  - testing
  - dead-code
---

# Repository Hygiene Audit

## Summary

`tbs-audit` is the globally reachable interface for evidence-based repository
hygiene work. It keeps static import mapping, string-field tracing, calibration
coverage contexts, duplication checks, mutation testing, and Git history behind
one interface. Findings are candidates for review, not automatic deletion
instructions.

## Usage

Verify the installed adapters from any directory:

```bash
tbs-audit --doctor
```

Within this worktree, build the inexpensive evidence map:

```bash
tbs-audit --mode map
```

Run the static tools, clone scan, and dead-fixture collection:

```bash
tbs-audit --mode quick
```

The dead-fixture collector runs pytest through the repository's locked
application environment and adds only `pytest-deadfixtures` as a transient
overlay. This prevents the audit tool's isolated Python environment from
mistaking application dependencies for missing packages.

The higher-cost modes are explicit:

```bash
tbs-audit --mode fields
tbs-audit --mode evidence
tbs-audit --mode mutation --mutation-target tree_break_selection/tree
```

`fields` builds a LibCST-backed field/function lineage map and exports JSON,
Markdown, and NetworkX GraphML. This mode is the static source cleanup path for
pandas and dictionary result fields. OpenLineage is intentionally not part of
this mode because its standard role is runtime job, dataset, and run lineage
metadata for executed pipelines.

`evidence` adds per-test calibration coverage contexts through a transient
pytest-cov overlay on the repository's locked environment. It records the
calibration-minus-non-calibration line set and deletes the much larger raw
context exports. `mutation` requires an explicit in-repository target,
additionally runs mutmut, and can be substantially slower. Its generated
`mutants/` worktree is ignored by Git. The default generated snapshot is
`reports/audits/generated/repository-hygiene.json`, which is ignored by Git
because it contains current-worktree paths and timestamps.

From another directory or repository, use an explicit root:

```bash
tbs-audit --repo /absolute/path/to/repository --mode map
```

The repository also exposes `make audit`, `make audit-quick`, and
`make audit-test`.

## Evidence

- `tools/repository_audit/src/tbs_repo_audit/inventory.py` combines an AST-wide
  import map with Grimp, traces string subscript and mapping-method accesses,
  recognizes standalone `__main__` entry points, records documentation
  references, and attaches Git history.
- `tools/repository_audit/src/tbs_repo_audit/field_lineage.py` uses LibCST for
  lossless static Python parsing and NetworkX GraphML export so pandas and
  dictionary fields can be reviewed by writer scope, reader scope, schema
  declaration, and surface before deletion.
- Calibration package initializers are excluded from study-module counts.
- Statically unimported files are separated from unresolved files because
  command runners and documented reproducibility artifacts can be valid without
  an importer.
- `tools/repository_audit/tests/` tests the public command interface, repository
  discovery, test-only and non-test imports, package-initializer exclusion,
  standalone documented runners, read/write classification, parse errors, and
  report writing.
- The 2026-07-29 installation exposed `tbs-audit` through `uv tool install`,
  with Coverage.py, Grimp, mutmut, pytest-cov, pytest-deadfixtures,
  pytest-testmon, Ruff, Semgrep, and Vulture isolated in its environment.
  jscpd is installed globally; Git, ripgrep, Node, npm, and uv were already
  available.
- The first full map found 132 calibration implementation modules: 76 were
  imported only by tests and 8 were statically unimported, but documentation or
  standalone-entry evidence meant none qualified as unresolved automatically.
- The first full calibration-versus-non-calibration comparison identified 79
  calibration-only production lines. Moving the selected-family UMAP renderer
  and calibration-only effective-rank summary to their diagnostic owners, then
  replacing the sibling-calibration catch-all with exhaustive labels, reduced
  that count to 2. The two retained lines choose sequential or all-core
  execution in the production spectral worker resolver. The remaining 21,936
  calibration-only lines are confined to benchmark and diagnostic modules.
- The tool package has 24 focused tests, including compact coverage-set
  comparison, punctuation-insensitive documentation matching, and LibCST
  field-lineage read/write classification.
- The first responsibility-aware clone tranche split application/production
  and benchmark scans before editing. The combined scan fell from 492 groups
  and 8,753 duplicated lines (3.71%) to 473 groups and 8,178 lines (3.47%).
  Production/application duplication fell to 35 groups and 507 lines (1.16%);
  benchmark duplication fell to 353 groups and 6,139 lines (4.26%). The next
  benchmark-only helper-contract tranche moved selected-root parsing,
  support-role classification, spectral/log-action transforms, selected
  generator manifest writing, and method-proof trace columns to their existing
  owners; the benchmark slice then fell to 345 groups and 5,808 lines (4.03%).
  These counts remain evidence for subsequent tranches, not a mandate to merge
  domain-distinct code with coincidentally similar syntax.

## Links

- [[redundant-and-legacy-code-map-20260623]]
- [[repository-hygiene-and-completion-audit-20260727]]
- [[wiki-search]]
