---
title: Wiki Log
type: control
status: reviewed
updated: 2026-06-01
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
- Split benchmark diagnostics into purpose-named subdirectories:
  `oracle/`, `calibration/`, `failure/`, and `analysis/`. Updated imports,
  tests, benchmark docs, and the oracle gate-path wiki evidence paths without
  keeping old-path compatibility wrappers.
- Removed local workspace clutter from the repository root by moving large
  ignored artifacts to `/Users/berksakalli/Projects/kl-te-cluster-local-artifacts`
  and deleting cache/build products. Promoted only the small benchmark CSV/MD
  files cited by wiki or diagnostic notes into `raw/assets/benchmark-results/`
  so evidence paths remain durable without restoring `benchmarks/results/`.
- Added [[manuscript-life-science-readiness]] after checking the
  scientific-writing and life-science routing constraints. The page records
  that the draft is methods-coherent but cannot make a biological application
  claim until one real-data track has a locked manifest, outputs, figures, and
  interpretation table.
- Added [[open-mathematical-questions]] as the consolidated method backlog from
  the conversation and current manuscript/wiki gap markers. It separates
  calibration hierarchy, external conditional nulls, projected-Wald/PCA
  assumptions, feature-space covariance, sibling FDR, traversal, recoverability,
  and manuscript-evidence questions.
- Added [[local-marchenko-pastur-rule]] and a raw controlled-spectrum evidence
  note for the MP dimension-rule audit. The analysis records that the
  production MP edge is algebraically correct for the backend eigenvalue scale,
  but the finite-sample edge, minimum dimension floor, internal spectral rows,
  and unit-scale whitening assumption still need validation before being
  treated as calibrated defaults.
- Split the MP dimension contract in code: raw MP signal count, projected-Wald
  test dimension, effective independent row count, and MP threshold row count
  are now separate typed outputs. The spectral worker records descendant leaf
  count separately from the former augmented-row MP threshold when internal
  distributions are stacked for PCA directions. Added
  `compare_mp_dimension_contracts.py` plus subset and finite-null smoke CSV
  evidence under `raw/assets/mp-dimension-rule-analysis/`.

### 2026-05-26

- Ran a targeted MP threshold-policy smoke on high-cardinality categorical,
  dimensional Gaussian, high-dimensional binary, and heavy-overlap binary
  cases. The tested finite-null threshold is not a drop-in production
  replacement: it matched the current rule on `binary_many_features`, worsened
  `cat_highcard_20cat_4c`, and under-split dimensional Gaussian cases. Updated
  [[local-marchenko-pastur-rule]] and [[open-mathematical-questions]] to keep
  selection-aware MP calibration as an open research question rather than a
  production fallback.
- Added benchmark case-suite selection by mathematical input contract. The
  full runner now accepts `KL_TE_CASE_SUITE` with `binary`, `categorical`,
  `continuous`, `discretized_gaussian`, `graph`, or `full`, so native Bernoulli
  results can be interpreted separately from experimental continuous
  empirical-Gaussian results and historical discretized-Gaussian stress cases.
- Systematized generated benchmark metadata: every generated case now names a
  `source_family` and a `feature_representation`, and generator dispatch uses a
  single registry instead of a long conditional chain. This separates the
  stochastic source process from the matrix representation consumed by KL-TE;
  benchmark result rows preserve both fields.
- Split benchmark case-data generation by family while preserving
  `generate_case_data()` as the stable dispatch entry point. The continuous
  suite now contains 9 selected representation-forwarding examples, not one
  continuous clone per historical Gaussian stress case; the full suite now
  resolves to 110 cases.
- Added [[dimensional-gaussian-representation-diagnostic]] after the full
  benchmark showed that continuous representation improves consolidated
  dimensional Gaussian cases but not the diffuse case. The analysis records
  that consolidated continuous trees are recoverable because Euclidean distance
  preserves a large block-mean signal, while diffuse continuous trees remain
  unrecoverable because weak spread-out signal is dominated by noise dimensions.
- Added [[benchmark-pipeline-contract]] and tightened benchmark docs around the
  active execution order: full runner orchestration, optional subprocess
  isolation, shared one-case pipeline, generated-data validation, method
  dispatch, result-row construction, and report assembly. The page also records
  the strict precomputed KL tree-distance contract.
- Moved standalone benchmark experiments under `benchmarks/experiments/`, moved
  the subset and regression runners under `benchmarks/smoke/` and
  `benchmarks/regression/`, split active MP diagnostics into
  `benchmarks/diagnostics/spectral/`, and removed unused ad hoc diagnostics for
  old MP-switch, passthrough, and alpha-sweep investigations.
- Moved phylogenetic benchmark cases into `benchmarks/shared/cases/`, edge
  calibration diagnostics into `benchmarks/diagnostics/calibration/`, and the
  method-constant manifest into `benchmarks/validation/manifests/` so the
  benchmark root exposes only active domains.
- Added [[spectral-backend-runtime-diagnostic]] and
  `benchmarks/diagnostics/spectral/profile_spectral_backends.py`. The
  representative profile shows that spectral runtime is dominated by
  null-whitened tangent matrix materialization, not SciPy eigendecomposition;
  exact diagonal vectorization matched current matrices and was about
  \(194\times\) to \(654\times\) faster on Bernoulli/continuous diagonal
  feature spaces.
- Installed `ruff` as a project development dependency and into the active
  Python environment so both `ruff` and `python -m ruff` use an explicit
  dependency. Ran a broader 18-additional-case spectral backend profile,
  confirming the same materialization bottleneck in Gaussian, binary,
  continuous, outlier, SBM, categorical, phylogenetic, overlap, and quantile
  one-hot cases. The remaining non-diagonal speed target is grouped
  multinomial/simplex whitening for categorical and phylogenetic blocks.
- Implemented exact vectorized diagonal null-whitening in
  `contrast_covariance.py` for pure one-dimensional Bernoulli and continuous
  empirical-Gaussian feature spaces. The repair preserves the same whitening
  formulas while avoiding generic per-block Cholesky work and generic
  per-block validation. Post-repair profiles reduced `binary_many_features`
  from about 3.25 s to 0.11 s and `gauss_extreme_noise_highd_continuous` from
  about 91.10 s to 1.05 s in the spectral worker. Updated
  [[spectral-backend-runtime-diagnostic]] with the post-change evidence.
- Implemented exact grouped multinomial null-whitening for pure categorical
  and phylogenetic one-hot feature spaces in `contrast_covariance.py`. The
  repair preserves the same drop-last simplex covariance map while batching
  Cholesky/solve work by category count. A production A/B comparison matched
  the generic block path exactly on selected categorical and phylogenetic
  cases; post-repair profiles reduced `cat_highd_3cat_500feat` from about
  14.86 s to 0.31 s, `phylo_dna_16taxa_low_mut` from about 21.83 s to 0.46 s,
  and `phylo_large_32taxa` from about 27.32 s to 0.70 s in the spectral worker.
- Removed stale numeric gate terminology from active code, benchmark timing
  fields, tests, scripts, and method notes. The canonical split contract is
  now the binary structure prerequisite, edge-divergence gate, and
  sibling-divergence gate; node distributions are documented as empirical
  subtree barycenters where that weighted-center construction is actually
  used.

### 2026-06-01

- Settled the active feature covariance contract. Generated continuous
  benchmark inputs now use one empirical-Gaussian block spanning the raw
  continuous columns, categorical one-hot blocks are explicitly
  `multinomial_drop_last`, and the old continuous diagonal whitening shortcut
  was removed from active covariance code. Updated
  [[open-mathematical-questions]] to mark covariance shape as settled while
  preserving validation gaps for high-cardinality categorical and continuous
  finite-sample calibration.
- Added `benchmarks/validation/feature_covariance_calibration.py` as a strict
  evidence generator for the two remaining feature-covariance validation
  targets:
  `categorical_multinomial_drop_last_covariance` and
  `continuous_empirical_gaussian_covariance`. The scaffold records local
  sibling-null full-tangent Wald calibration metrics, confidence intervals,
  p-value uniformity diagnostics, seed, commit, dirty-worktree status, command,
  and limitations; it does not claim to validate selected projections, MP
  selection, sibling FDR, traversal, tree construction, or empirical-null
  inflation.
- Audited installation and benchmark-entrypoint hygiene. The canonical setup
  path is now `uv venv --python 3.11 .venv` followed by
  `uv sync --extra dev --extra benchmark --extra viz --locked`; the
  `requirements.txt` fallback, direct benchmark path-bootstrap helper, pytest
  import path injection, and bare `pytest`/direct benchmark documentation were
  removed from active project surfaces.
- Added a preflight implementation contract for dense empirical-Gaussian
  continuous covariance. The exact dense path now raises a clear unsupported
  error for oversized \(p \gg n\) continuous blocks instead of allowing the
  process to be killed by scatter-matrix allocation. The 2026-06-01 full
  benchmark completed with plots disabled after the 20,000-feature continuous
  stress case was recorded as an explicit KL skip; a validated low-rank or
  regularized continuous covariance model remains open.
- Corrected manuscript and wiki method text that still described KL-TE as a
  discrete-only or discretize-continuous pipeline. The manuscript now describes
  the active typed feature-space contract: Bernoulli coordinates, categorical
  one-hot blocks with drop-last multinomial covariance, and explicit continuous
  empirical-Gaussian blocks. Discretized Gaussian inputs are now described only
  as benchmark variants.
- Added a selected-PCA projected-Wald validation scaffold. The scaffold isolates
  the fixed-membership Gaussian sibling-null question where PCA rows and MP
  projection dimension are selected from the same local null-whitened rows used
  by the tested contrast; it explicitly does not validate hierarchy selection,
  sibling FDR, traversal, or empirical-null inflation.
- Ran the selected-PCA projected-Wald validation scaffold at commit
  `492c8520809e6cfad9e4853e91c89a887eac5a21` with seed `20260601` and 1000
  replicates per setting. Leaf-only Gaussian selected PCA was calibrated in the
  tested settings, while child-mean internal spectral rows caused severe
  anti-conservative rejection rates. The locked JSON/CSV evidence is stored
  under `raw/assets/selected-pca-projected-wald-validation/`.
- Removed the production internal-spectral-row configuration and made the
  inferential projected-Wald spectral basis descendant-leaf-only. The
  regression-gate benchmark completed under this stricter contract with mean
  ARI `0.6017`, median ARI `0.6803`, exact \(K\) in 4 of 17 rows, and six
  explicit unsupported-calibration skips. A floor diagnostic showed that
  \(k_{\min}=1\) reduces skips but lowers aggregate ARI, so the exposed problem
  is missing empirical-null calibration support rather than an internal-row or
  floor fallback. Updated [[local-marchenko-pastur-rule]],
  [[selected-pca-projected-wald-validation]], and
  [[open-mathematical-questions]] with the benchmark reaction.
- Ran the full KL-only benchmark with plots and relationship analysis disabled
  under the leaf-only spectral contract. The run completed 110 cases with 85
  `ok` rows, 25 explicit `skip` rows, valid-row mean ARI `0.8612`, valid-row
  median ARI `1.0`, and exact \(K\) in 63 of 110 rows. The full CSV and failure
  report are stored under `raw/assets/mp-dimension-rule-analysis/`.
- Decomposed the unsupported sibling-calibration problem to its upstream edge
  source. A current support audit found that 24 of 25 full-benchmark skipped
  rows rebuild through edge and sibling-record collection, but every one has
  zero strict-null or stopped-edge empirical-null calibration records. The
  edge-selection null audit then showed that pure null data rejects about
  `99%` of child-parent edges when the hierarchy is selected from the same data,
  while fixed-tree feature permutations have median rejection rate `0.0`.
  Added [[edge-selection-null-audit-20260601]] and updated
  [[oracle-gate-path-diagnostic]] and [[open-mathematical-questions]].
- Added `benchmarks/diagnostics/calibration/sample_split_selection_audit.py`.
  The script explicitly rejects literal sample splitting because KL-TE uses a
  sample-leaf hierarchy and held-out samples have no canonical training-tree
  node membership. It implements the valid feature-split cross-fit diagnostic:
  build the tree from one feature block, test gates on a held-out feature block,
  and compare against in-sample and fixed-tree feature-permutation regimes.
  The first run is stored under
  `raw/assets/benchmark-results/sample_split_selection_audit_20260601/` and
  summarized in [[feature-split-selection-audit-20260601]].
- Added `benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py`
  to estimate the same-data selected-hierarchy sibling null directly. The
  first Bernoulli/categorical run used 20 null replicates on four
  representative cases. Selected-hierarchy correction factors were large
  (`28.8` to `58.5`), blocked the null case, and still rejected the three
  signal examples under the mean-scaled chi-square summary. The evidence is
  stored under
  `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/` and
  summarized in [[selected-hierarchy-null-audit-20260601]].
- Added [[selected-hierarchy-selection-geometry]] to consolidate the geometric
  explanation behind the selected-hierarchy calibration problem. The page
  separates fixed projected-Wald tangent geometry from same-data hierarchy
  selection, where internal node barycenters, child-parent edges, and focal
  sibling contexts are selected high-contrast objects rather than ordinary
  fixed-tree null contrasts.
- Tightened `selected_hierarchy_null_audit.py` so selected records require an
  open child-parent edge path, added explicit non-root target modes, and added
  projection/parent-size/depth context matching. The 100-replicate richer root
  rerun still blocks `gauss_null_large` and leaves the three signal examples
  significant. The non-root strongest rerun has matched Gaussian support but
  zero matched binary/categorical support under the strict context, marking a
  diagnostic support limitation rather than a calibration estimate or fallback
  decision.
- Added selected-hierarchy diagnostic precision fields: record-level and
  simulation-level standard errors for \(c\), empirical-tail standard errors,
  and tail-resolution diagnostics. The 100-replicate richer runs describe the
  large selected-hierarchy scale effect, but their empirical-tail resolution is
  too coarse to be a production external-calibration validation at
  `SIBLING_ALPHA = 0.01`.
- Ran 500-replicate descriptive selected-hierarchy precision studies for root
  and non-root strongest targets, plus a non-root context-relaxation ladder for
  binary and categorical rows. The strict root and Gaussian non-root rows keep
  \(c\) in the tens with improved precision. Non-root binary/categorical support
  is sparse under exact depth matching but returns when depth and parent-size
  are relaxed, showing a support-geometry phenomenon rather than a production
  borrowing rule.
- Added [[selected-hierarchy-null-support-contract]] to formalize the
  diagnostic support states, no-fallback rule, current exact matching variables,
  descriptive stratifiers, and Monte Carlo precision targets for
  selected-hierarchy null studies.
- Added `selected_hierarchy_stratification_diagnostic.py` and ran a
  500-replicate stratification study over the four representative selected-null
  cases. The diagnostic groups selected edge-path-open sibling records by
  parent depth and parent-size bins. It shows that small selected parent nodes
  often have higher selected-null scale than root-like selected nodes, while
  exact depth is too sparse to promote to a validated exact conditioning
  variable. Added
  [[selected-hierarchy-stratification-diagnostic-20260602]].
- Extended the selected-hierarchy stratification evidence to record the
  selected-ratio law \(R=W/(a\nu)\), raw statistic quantiles, reference
  expectations, and standard projected-Wald rejection rates inside each
  descriptive stratum. The compact 500-replicate output still writes no
  per-record dump and remains diagnostic-only, not a calibration fallback.

## Evidence

- `raw/inbox/wiki-construction-brief.md` records the requested scaffold.
- `AGENTS.md` records the operating guide.

## Links

- [[project-overview]]
- [[wiki-construction]]
- [[schema]]
- [[maintenance]]
- [[wiki-search]]
