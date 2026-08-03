# Repository Audit

`tbs-audit` combines static reachability, string-field tracing, test coverage
contexts, duplication checks, mutation testing, and Git history into one
repository-hygiene interface.

The modes deliberately escalate in cost:

- `tbs-audit --mode map` builds read-only import, field-use, and history evidence.
- `tbs-audit --mode fields` builds LibCST field/function lineage and exports a
  NetworkX GraphML graph for pandas/dictionary field cleanup review. It keeps
  raw read/write reuse evidence separate from cleanup classification, excluding
  graph, environment, configuration, and known output-schema keys from true
  dead-write candidates.
- `tbs-audit --mode duplicates --scope applications` runs jscpd for a focused
  path and writes a classified duplicate-cleanup report. Scopes are repeatable,
  default to `applications`, and can be used from another worktree with
  `--repo /absolute/path/to/repo`.
- `tbs-audit --mode generators` runs the benchmark generator-geometry audit in
  the repository environment. It records resolved feature families, duplicate
  burden, within/between distance signal, nearest-neighbor label purity, and
  scientific-assumption warnings before benchmark method results are interpreted.
- `tbs-audit --mode calibration-contract --calibration-records PATH` separates
  tested-null sibling records, stopped-subtree frontiers, and nested blocked
  descendants. It reports the production node-record support alongside a
  structural frontier count so their difference is explicit. Support is read
  from the `is_role_supported` column stamped by the producing sweep, not
  re-derived here, so the audit cannot disagree with the production rule.
- `tbs-audit --mode quick` also runs Ruff, Vulture, jscpd, and
  pytest-deadfixtures. Test collection runs through the repository's locked
  application environment, with the plugin added as a transient overlay, so
  application dependencies are available without changing the project lock. It
  also records a LibCST field-lineage summary so terminal output separates raw
  write-only string keys from filtered cleanup candidates after graph,
  environment, configuration, and output-schema keys are excluded.
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
The duplicate-cleanup mode writes
`reports/audits/generated/duplicate-cleanup.json` and
`reports/audits/generated/duplicate-cleanup.md` by default. Clone groups are
classified as production, benchmark, plot/report, analysis-methodological,
test-helper, bootstrap/structural, application, or uncategorized, with a risk
label and cleanup recommendation for each group.
The generator-geometry mode writes
`reports/audits/generated/generator-geometry.csv` and
`reports/audits/generated/generator-geometry.md` by default.
The calibration-contract mode writes
`reports/audits/generated/calibration-contract.json` and
`reports/audits/generated/calibration-contract.md` by default.
Run
`tbs-audit --doctor` to verify that all adapters are reachable.

From another repository, pass `--repo /absolute/path`. Within a Git worktree,
the repository root is detected automatically.
