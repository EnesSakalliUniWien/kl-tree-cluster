# Repository Audit

`tbs-audit` combines static reachability, string-field tracing, test coverage
contexts, duplication checks, mutation testing, and Git history into one
repository-hygiene interface.

The modes deliberately escalate in cost:

- `tbs-audit --mode map` builds read-only import, field-use, and history evidence.
- `tbs-audit --mode fields` builds LibCST field/function lineage and exports a
  NetworkX GraphML graph for pandas/dictionary field cleanup review.
- `tbs-audit --mode quick` also runs Ruff, Vulture, jscpd, and
  pytest-deadfixtures. Test collection runs through the repository's locked
  application environment, with the plugin added as a transient overlay, so
  application dependencies are available without changing the project lock.
- `tbs-audit --mode evidence` also executes calibration tests with per-test
  coverage contexts through a transient pytest-cov overlay on the repository's
  locked environment. Raw per-test coverage exports are deleted after the
  compact calibration-only set difference is recorded.
- `tbs-audit --mode mutation --mutation-target PATH` additionally runs mutmut
  against an explicit production target and can be very slow. The explicit
  target prevents accidental whole-repository mutation runs.

The default report is
`reports/audits/generated/repository-hygiene.json`. Generated snapshots are
ignored by Git because they contain timestamps and current-worktree evidence.
The field-lineage mode writes
`reports/audits/generated/field-function-lineage.json`,
`reports/audits/generated/field-function-lineage.graphml`, and
`reports/audits/generated/field-function-lineage.md` by default. OpenLineage is
not used for static source cleanup; it belongs to runtime job/dataset/run
lineage if the project later needs executed-pipeline metadata.
Run
`tbs-audit --doctor` to verify that all adapters are reachable.

From another repository, pass `--repo /absolute/path`. Within a Git worktree,
the repository root is detected automatically.
