---
title: Wiki Log
type: control
status: reviewed
updated: 2026-06-14
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
- Added `selected_hierarchy_external_calibration_contract.py` and ran a
  500-replicate external selected-hierarchy calibration contract diagnostic.
  The production tail-resolution rule requires `499` independent matching
  simulations at `SIBLING_ALPHA = 0.01`; no current row is production
  admissible. Scalar mean scaling is conservative in reliable rows but not
  distribution-calibrated, so future external calibration would need a
  selected-ratio tail law rather than a scalar fallback. Added
  [[selected-hierarchy-external-calibration-contract-20260602]].

### 2026-06-02

- Added `selected_hierarchy_geometry_covariates.py` and ran a 100-replicate
  diagnostic over four representative Bernoulli/categorical selected-null
  cases. The run records row-level tree geometry, edge p-values, eigenvalue
  summaries, and angular alignment between the selected sibling contrast and
  the selected PCA subspace. Edge-selection strength is the largest recorded
  correlate of log selected-ratio; spectral and angular variables add smaller
  descriptive structure. Added
  [[selected-hierarchy-geometry-covariates-20260602]] and kept the result
  explicitly diagnostic-only, not a production external calibration model.
- Added [[selected-hierarchy-geometric-law-map]] to separate actual method laws
  from physical analogies for the selected-hierarchy variables. The map links
  projected-Wald energy to the chi-square law, edge-selection strength to a
  large-deviation/action coordinate, eigenvalues to Marchenko--Pastur spectral
  modes, angles to Pythagorean projection, parent size/balance to sampling
  variance, branch lengths to phylogenetic covariance context, and node
  distributions to barycentric coarse-graining.
- Extended the selected-hierarchy geometry diagnostic with candidate-equation
  scoring. The compact edge-plus-spectral equation nearly matches the full
  descriptive equation for top-10% selected-ratio tail discrimination, while
  the full equation fits mean log-ratio better. The result defines the next
  validation target, not a production law.
- Added candidate-equation holdout scoring to
  `selected_hierarchy_geometry_covariates.py` and reran the same 100-replicate
  four-case panel. Replicate-fold holdout preserves strong tail ranking for
  the compact edge-plus-spectral equation, and leave-one-case-out transfer
  makes it more stable than the larger full descriptive candidate for tail
  discrimination. Updated [[selected-hierarchy-geometry-covariates-20260602]]
  and [[selected-hierarchy-geometric-law-map]] while keeping the result
  diagnostic-only.
- Added source-family holdout to the geometry diagnostic and ran a broad
  200-replicate panel over Gaussian, dimensional Gaussian, binary,
  high-cardinality/high-dimensional categorical, heavy-overlap binary,
  phylogenetic, and SBM boundary cases. Eight cases completed; `sbm_moderate`
  was an explicit precomputed-distance null-generator boundary skip. The
  broad evidence separates global top-tail ranking from calibrated selected
  scale: simple edge/spectral scores rank tails well, while absolute
  selected-ratio scale varies strongly by family and still needs a selected
  tail-law target rather than a production fallback.
- Added a within-context selected-ratio tail-law diagnostic to the same
  selected-hierarchy geometry tool. The context uses source family, feature
  family, parent-size bin, sibling projection dimension, and binned edge
  action. A broad 200-replicate run produced descriptive held-out tail-law
  rows but no production-admissible context under the strict `499` matching
  simulation, `499` matched record, and `0.002` held-out SE contract. Added
  [[selected-ratio-tail-law-diagnostic-20260602]].
- Tightened selected-ratio tail-law support counting to use the explicit
  `selected_hierarchy_simulation_id` case-replicate unit rather than bare
  `replicate_index`. Regenerating the broad 200-replicate result leaves the
  production conclusion unchanged: `0` of `104` contexts are admissible, and
  the largest context has `376` independent matching simulations.
- Ran a focused 300-replicate selected-tail support study over multi-case
  `gaussian_blobs`, `binary_template`, and `categorical_multinomial` source
  families. Two exact small-parent, high-edge-action Gaussian contexts become
  production-admissible under the diagnostic support contract. Categorical
  contexts approach but do not reach the `499` simulation threshold, and binary
  contexts remain support-fragmented.
- Added [[selected-geometry-mp-integral-literature-20260602]] after checking
  selective-inference and random-matrix references for the no-bootstrap
  analytic direction. The wiki now records signed distance, curvature,
  tangent-cone selected-region geometry as the retained selective-inference
  object, and the Silverstein--Choi/Ledoit--Wolf Stieltjes-transform integral
  route as the analytic generalization of the current identity
  Marchenko--Pastur edge.
- Added `local_mp_identity_law_diagnostic.py` and
  [[local-mp-identity-law-diagnostic-20260602]]. The representative screen
  shows frequent near-edge or above-edge identity-MP spikes in selected
  Bernoulli/discretized spectra, mixed categorical behavior, and continuous
  empirical-covariance spectra far below the identity-MP positive support. A
  self-whitening check explains the continuous result: the positive eigenvalues
  match the finite-rank \((m_u-1)/m_u\) covariance scale. The result supports a
  deformed/effective local spectral-law question, not a bootstrap threshold
  fallback.
- Extended the local MP identity-law diagnostic with a finite identity-null
  top-edge comparison. The split is now sharper: selected
  Gaussian-null/high-dimensional Bernoulli/diffuse discretized spectra exceed
  the finite-null 95% envelope in roughly `55%`--`72%` of evaluated nodes,
  while high-cardinality/high-dimensional categorical screens are closer to
  ordinary finite fluctuation at about `7%`--`12%`.
- Extended the same diagnostic with selected-tree spectral-law covariates and
  categorical extreme-node output. The rerun keeps the Bernoulli/discretized
  interpretation as broad selected spectral inflation, while categorical
  behavior separates into selected extreme nodes: `cat_highcard_20cat_4c` is
  strongly tied to node size/aspect ratio, and `cat_highd_3cat_500feat` mixes
  root/half-tree extremes with many tiny selected extreme nodes.
- Added [[hierarchy-gate-separation-20260603]] after rerunning oracle
  recoverability on the `85` runnable rows from the latest strict KL-only full
  benchmark and tracing the five runnable gate failures. The current full
  separation is `62` solved, `24` calibration-support-undefined skips, `14`
  tree/metric unrecoverable rows, `4` gate over-splits, `4`
  oracle-matched-below-solved rows, `1` continuous covariance boundary, and
  `1` gate under-split.
- Added [[method-proof-web]] to connect the fixed-object proof spine
  (feature-space covariance, whitening, fixed projection, projected-Wald
  chi-square law, MP dimension rule) with the selected-hierarchy proof gap
  (selected region, selected-ratio tail law, admissible empirical/external
  calibration support, and oracle recoverability separation).
- Added [[root-selected-region-model]] as the first simplified selected-region
  target: root sibling selection with fixed hierarchy-construction procedure,
  merge inequalities, root edge-opening inequalities, and explicit
  differential-geometric objects for signed distance, tangent cone, and
  curvature.
- Added `benchmarks/diagnostics/calibration/root_selected_region_margins.py`
  and [[root-selected-region-margins-20260603]]. The diagnostic replays
  average-linkage merge-selection inequalities for observed root contexts,
  joins them to root edge/sibling/spectral quantities, and records that
  representative Hamming/discretized/categorical root cells are tie-heavy while
  the continuous Euclidean representative has positive construction margins.
- Extended the root selected-region margin diagnostic to schema `v2`: smooth
  Euclidean average-linkage constraints now record the first-order signed
  distance \(m_t/\|\nabla g_t\|\), while Hamming/discrete cases are explicitly
  classified as discrete tie-cell geometry. The representative continuous case
  has minimum first-order signed distance about `2.24e-4`; curvature and
  null-whitened distance remain open.
- Extended the same diagnostic to schema `v3` with root empirical-Gaussian
  null-whitened first-order merge distance and a relationship table. In the
  supported eight-case continuous panel, edge action has Spearman correlation
  `1.0` with log root sibling selected ratio, while raw merge margin,
  ambient signed distance, and null-whitened merge distance have weak
  relationships. The next selected-region object is therefore the edge-opening
  boundary/action, not more merge-margin tuning.
- Extended the root selected-region diagnostic to schema `v4` with explicit
  fixed-subspace projected-Wald edge-opening geometry. The active edge
  annotation contract now preserves `Child_Parent_Divergence_Test_Statistic`;
  the diagnostic records edge-path radial distance
  \(\sqrt Q-\sqrt{q_{1-\alpha,k}}\), statistic margin, and Tree-BH action.
  In the supported eight-case continuous panel, all three edge-opening
  coordinates have Spearman correlation `1.0` with log root sibling selected
  ratio, while merge-boundary distances remain weak.
- Extended the root selected-region diagnostic to schema `v5` with the
  edge/sibling Wald relationship. The diagnostic now verifies that, under the
  active no-branch-scaling barycentric parent contract, root child-parent edge
  z-vectors equal \(z_{L,R}\) and \(-z_{L,R}\) up to numerical residuals. The
  supported continuous root panel has zero extra edge projection energy beyond
  the sibling projection, so local edge and sibling statistics use the same
  projected barycentric energy before thresholds, Tree-BH, sibling
  FDR/inflation, and selected conditioning are applied.
- Extended the root selected-region diagnostic to schema `v6` with the
  fixed-projection sibling tail conditional only on root edge opening, plus
  the current internal empirical-inflation sibling tail or explicit
  unsupported status. The diagnostic shows that edge conditioning alone does
  not explain diffuse-dimensional root blockers: raw and edge-conditioned
  sibling p-values remain significant, while the empirical-inflation layer can
  block. The next mathematical target is therefore the empirical-inflation
  versus fuller selected-hierarchy law, not another root edge-opening
  truncation formula.
- Checked the selective-inference literature framing against the manuscript
  assumptions section. `assumptions_validation.tex`, `sibling_test.tex`, the
  notation table, and the method logic guide now cite the selection-event
  conditioning literature where the assumption is stated and describe the
  active empirical-null inflation contract as joint child-edge null-evidence
  weights plus feature-family/context-vector localization, not the older
  geometric-mean scalar-context description. The wiki oracle gate-path page now
  records the same active support-set notation.
- Added
  `benchmarks/diagnostics/calibration/internal_vs_selected_hierarchy_inflation.py`
  and [[internal-vs-selected-hierarchy-inflation-20260603]]. The diagnostic
  compares active internal empirical-null inflation with regenerated
  selected-hierarchy descriptive scale for the same root sibling target without
  installing an external fallback. In `dim_diffuse_6c_136f`, the internal and
  selected-hierarchy scales agree (`85.27` versus `86.23`); in
  `gauss_null_large` and `cat_highcard_20cat_4c`, selected-hierarchy support
  exists descriptively while internal production support remains absent.
- Clarified the manuscript logic map, method validation table, and wiki index
  wording for selected-tail calibration. The focused selected-ratio tail-law
  study did find two narrow admissible Gaussian small-parent, high-edge-action
  contexts; the external calibration law remains undefined outside such
  predeclared admissible contexts and is not a general fallback.
- Ran a 500-replicate selected-tail admissibility boundary study over
  comparable Gaussian and categorical source families and added
  [[selected-tail-admissibility-domain-20260603]]. After that run, the
  admissible domain included Gaussian and categorical `small_0_0.25`,
  `edge_action_ge8` contexts with sibling projection dimension `1` and `2`.
  Root, medium-parent, large-parent, lower-edge-action, binary, continuous, and
  precomputed-distance contexts remain outside the current admissible
  production domain.
- Ran a 600-replicate binary selected-tail boundary study and regenerated the
  admissibility-domain table. `binary_template`, `small_0_0.25`,
  `edge_action_ge8`, sibling projection dimension `1` is now admissible with
  `606` independent matching simulations; the corresponding binary
  projection-2 context remains support-limited at `445/499` simulations.
- Added [[phylogenetic-ml-topological-selected-tail-literature-20260603]] after
  checking phylogenetic, machine-learning selective-inference, distance-based
  inference, and topological graph literature. The durable conclusion is that
  medium/large parent contexts likely lack tail homogeneity when support
  exists but held-out precision fails; topology, balance, merge persistence,
  covariance condition, ancestor edge action, and spectral alignment are
  diagnostic coordinates, not fallback calibration rules.
- Added `selected_tail_topology_refinement.py`, topology fields to the
  row-level selected-hierarchy geometry diagnostic, and
  [[selected-tail-topology-refinement-20260603]]. A 300-replicate
  Gaussian/categorical panel shows that exact balance, topology,
  merge-persistence, edge-path, spectral-alignment, and compact combined
  refinements do not restore medium/large selected-tail production
  admissibility. The result points toward a lower-dimensional modeled
  selected-tail law rather than exact multiway topology matching.
- Debugged the topology-refinement admissibility wording after noticing that
  data-adaptive refinement bins were being summarized with production
  admissibility language. The diagnostic now separates
  `diagnostic_tail_law_contract_passed` from
  `production_tail_law_admissible`; only predeclared base contexts can be
  production-admissible. Regenerated
  `raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/`
  and updated [[selected-tail-topology-refinement-20260603]],
  [[selected-hierarchy-null-support-contract]], and
  [[open-mathematical-questions]].
- Debugged the selected-tail artifact contract after finding that several
  `selected_ratio_tail_law.csv` files relied on the manifest-only precision
  threshold while downstream summaries used a hidden `0.002` constant. The
  tail-law evaluator now writes `max_exceedance_standard_error` into every
  row, topology and admissibility-domain summaries read that column, and the
  affected benchmark CSVs plus
  `raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/`
  were regenerated or schema-migrated from their recorded manifests.
- Rechecked the other mathematical analyses against their verification
  artifacts. Selected-PCA, local MP, root selected-region, internal-vs-selected
  inflation, and hierarchy-gate numeric claims match the cited CSV/JSON files.
  Corrected stale selected-PCA evidence wording to the current leaf-only
  production spectral contract. Added the 300-replicate topology-input
  candidate-equation transfer result: edge-sampling and edge-spectral equations
  transfer across source families, while selected-energy and full descriptive
  equations fail that split and remain variable-screening diagnostics only.
- Added `benchmarks/cloud/aws_selected_tail_equation_study.py`,
  `benchmarks/cloud/aws/Dockerfile`, and
  [[aws-selected-tail-equation-study]] so the selected-tail equation study can
  run as AWS Batch shards and merge row-level records into one recomputed
  diagnostic output. During this cleanup, geometry case summaries were moved
  to the explicit `selected_hierarchy_simulation_id` contract so cloud shards
  cannot undercount independent regenerations by reusing local replicate
  indices.
- Deployed the AWS Batch selected-tail stack in account `067744548702`, pushed
  `067744548702.dkr.ecr.us-east-1.amazonaws.com/kl-te-selected-tail:latest`,
  ran a successful one-shard smoke job, then ran the 20-shard selected-tail
  cloud diagnostic and merge job. Synced compact merged outputs into
  `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/`
  and added [[selected-tail-equation-cloud-run-20260603]]. The row-level
  merged table remains in S3 because it is `261.5 MiB`. Added
  `local_deployment_context.json` because the container did not include `.git`
  and therefore could not record commit/branch inside its generated manifest.
- Rebuilt and pushed the AWS Batch image after adding explicit
  `KL_TE_GIT_COMMIT` and `KL_TE_GIT_BRANCH` build arguments plus `.dockerignore`
  pruning. The post-run rebuilt smoke job
  `5d72447b-9921-4163-8249-df4b922f912a` succeeded and its shard manifest
  records `build_commit=a61f1cfe81ded8923b4f11cc16697036f0ed6220` and
  `build_branch=dev`.
- Reran the AWS selected-tail equation experiment with the rebuilt image,
  base seed `20260604`, array job
  `dfbd5d03-5862-47fa-9966-db1c5301c187`, and merge job
  `1b869dc6-b4cd-43f3-9467-c95f66d24154`. Synced compact merged outputs into
  `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/`
  and added [[selected-tail-equation-cloud-run-20260604]]. The rerun reproduced
  the same seven admissible contexts as the 2026-06-03 run and no
  admissibility-status changes.
- Debugged the 2026-06-04 cloud run contract. The AWS runner evaluated the six
  predeclared descriptive candidate functions from
  `selected_hierarchy_geometry_covariates.py` and did not introduce additional
  fitted law families. The `271,801` selected-record count differs from the
  `271,798` candidate-equation row count because three selected Gaussian rows
  have zero selected-hierarchy ratio and therefore undefined log response.
- Audited selected-tail result weighting for the 2026-06-04 cloud run. The
  run is arithmetically consistent, but candidate-equation scores are
  selected-node weighted: `gauss_null_large` contributes `63.6%` of rows versus
  `30.8%` of independent selected simulations. Equal-simulation weighting
  keeps `edge_spectral_modes` as the strongest tested coordinate, but with a
  lower top-tail AUC of about `0.927`, so the result remains a search
  direction rather than a calibrated production law.
- Debugged benchmark creation and representation contracts. A generation audit
  over the 110 default cases found no case-recipe geometry, label-length,
  value-domain, forwarded-representation, or precomputed-distance metadata
  mismatch. The actual skew was method compatibility: `kl_diffusion` used
  Hamming diffusion on continuous float matrices and could return `ok` rows
  with meaningless ARI. The Hamming diffusion runner now rejects continuous
  `FeatureSpace` inputs and non-binary values, and the full-run console summary
  labels mean ARI as ok-row-only while printing method status counts.
- Added a canonical statistical decision trace derived from the gate-path
  diagnostic. The trace records edge raw/BH p-values, traversal-aligned sibling
  BH state, empirical-inflation status, calibration-support status, and final
  traversal decision in one stable table. Removed the optional sibling-FDR
  early-return helper and updated active documentation to describe the current
  traversal-aligned sibling BH contract rather than stale flat-BH wording.
- Added [[sibling-null-prior-interpolation-audit-20260604]] and
  `benchmarks/diagnostics/calibration/sibling_null_prior_interpolation_audit.py`
  to reconstruct the old tree-neighborhood sibling-null prior score as a
  diagnostic-only table. The four-case run shows positive interpolated weights
  on selected non-null records while strict internal support remains absent,
  confirming that the old score describes support failure but is not admissible
  empirical-null calibration. Verified with
  `tests/validation/60_test_sibling_null_prior_interpolation_audit.py` and
  wrote compact outputs under
  `raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/`.
- Ran the same interpolation audit over the 24 KL calibration-support skips
  from the complete `run_20260604_115308Z_full` benchmark. All 24 remain
  `no_strict_internal_support`; 23 receive positive diagnostic interpolated
  selected-nonnull weights, while `gauss_extreme_noise_highd` receives none.
  Recorded outputs under
  `raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_full_skips_20260604/`.
- Removed statistical alpha thresholds from mutable `config.py` and introduced
  `kl_clustering_analysis/hierarchy_analysis/statistics/alpha_contract.py` as
  the canonical source for `DEFAULT_EDGE_ALPHA = 0.001` and
  `DEFAULT_SIBLING_ALPHA = 0.01`. Benchmark KL rows now record both alpha
  values in their parameter contract, and decomposition output reports both
  edge and sibling alpha values.
- Removed the disabled branch-length variance-scaling config path from the
  active projected-Wald kernel. Branch lengths remain valid tree metadata and
  diagnostic/topological covariates, but they no longer enter edge or sibling
  Wald variance through a hidden runtime switch.
- Added [[alpha-grid-full-20260604]] after running the AWS full-suite alpha
  grid over `25` edge/sibling alpha pairs. The current default pair
  `edge_alpha = 0.001` and `sibling_alpha = 0.01` had the best mean ARI in
  the grid, while lower edge alpha had more exact cluster-count hits. Recorded
  this as benchmark evidence, not a selected-tree Type-I proof.
- Added `benchmarks/validation/selected_edge_type1_geometry.py`,
  `benchmarks/cloud/aws_selected_edge_type1_geometry.py`, and
  `benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py` for a
  strict selected-edge Type-I geometry diagnostic. Ran local direct,
  local shard/merge, Docker shard/merge, and AWS Batch pilot checks. The AWS
  pilot used four shards, two binary global-null cases, two edge alpha values,
  and `40` replicates, producing `40,960` edge rows. Same-data selected-tree
  mode rejected about `99.6%` to `99.8%` of Tree-BH-tested frontier edges,
  while strict sibling-calibration support failures remained explicit. Added
  [[selected-edge-type1-geometry-pilot-20260604]] and updated
  [[open-mathematical-questions]].
- Extended the selected-edge diagnostic to direct categorical multinomial
  nulls with an explicit categorical `FeatureSpace`, while leaving continuous
  null regeneration as a hard unsupported state. Ran the AWS binary/categorical
  pilot with four cases, four shards, and `40` replicates, producing
  `136,320` edge rows. The run confirms selected-tree frontier rejection near
  one across binary and categorical nulls, and exposes high-cardinality
  categorical fixed-tree frontier inflation as a separate open calibration
  problem. Added [[selected-edge-binary-categorical-pilot-20260604]].
- Added `benchmarks/validation/traversal_sibling_fdr_null.py` and
  `benchmarks/cloud/aws_traversal_sibling_fdr_null.py` to separate sibling FDR
  algorithm behavior from fixed-tree projected-Wald calibration, same-data
  selected-tree calibration, and empirical-inflation support. The local smoke
  recorded mean FDP `0.03` for synthetic valid p-values at sibling alpha
  `0.01`, mean FDP `0.25` for fixed-tree raw Wald on `binary_2clusters`, mean
  FDP `0.95` for selected-tree raw Wald, and `18/20` support failures for the
  active inflated selected-tree layer. Added
  [[traversal-sibling-fdr-smoke-20260604]] and kept the result classified as
  smoke evidence, not final FDR calibration.
- Ingested
  `raw/inbox/calibration-contract-enhancement-request-20260604.txt` and added
  [[calibration-contract-enhancement-request-20260604]]. Implemented explicit
  internal `CalibrationDecision` statuses for sibling empirical-null
  calibration, preserved scalar prediction as an admissibility-checked
  compatibility helper, and corrected `manuscript/sections/method/edge_test.tex`
  from historical internal-row spectral matrices to the production leaf-only
  inferential PCA contract.
- Updated [[open-mathematical-questions]] to separate the new code-level
  `CalibrationDecision` status contract from still-open mathematical validation
  questions about effective internal support thresholds, leave-one-target
  stability, and external selected-tail admissibility.
- Added `benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py`
  and [[selected-tail-law-q5-validation-20260604]] for Q5. The diagnostic fits
  residual-tail models using edge severity, parent size, sibling projection
  dimension, feature family, parent-size bin, and spectral geometry on the
  300-replicate row-level selected-geometry table. The full Q5 model validates
  on replicate holdout but fails case/parent-size extrapolation, while the
  edge-plus-spectral model transfers best; the result remains diagnostic and
  not a production external calibration law.
- Ingested `raw/inbox/recursive-method-program-20260604.txt` and added
  [[recursive-method-followups-20260604]]. Implemented Q9/Q11
  support-threshold reporting and opt-in sparse-context enforcement, added the
  Q10 sibling empirical-null weight-rule diagnostic grid, ran the Q14/Q15
  MP/minimum-dimension smoke, and added the Q17 sibling projection-dimension
  rule-grid diagnostic. Updated [[open-mathematical-questions]] so these items
  are marked as partially answered by code or diagnostics while Q19--Q44 remain
  larger idea and validation tracks.
- Ingested `raw/inbox/barycentric-method-literature-request-20260604.txt` and
  added [[barycentric-method-literature-request-20260604]]. Expanded the
  manuscript method with the barycentric parent formula, the edge/sibling
  whitened z-identity, barycentric leverage variables, Bregman/Wasserstein/
  Frechet/BHV literature, and a validation-status row. Extended the Q5
  selected-tail diagnostic with barycentric balance, log leverage, sampling
  scale, and `log(p/n_u)` predictors; the expanded rerun finds that the
  barycentric edge-plus-spectral model reduces median residual-tail absolute
  error to `0.002386` while remaining diagnostic-only.
- Added `raw/assets/benchmark-results/open_question_diagnostic_audit_20260604/open_question_diagnostic_audit.csv`
  and [[open-question-diagnostic-audit-20260604]]. Reran the focused diagnostic
  validation gate, Q10 sibling null-weight rule diagnostic, and Q17 sibling
  projection-dimension rule grid. Updated [[open-mathematical-questions]] with
  a compact all-44-question diagnostic status ledger.
- Added `benchmarks/diagnostics/open_questions/full_diagnostic_contract.py`,
  `tests/validation/71_test_open_question_full_diagnostic_contract.py`, and
  [[open-question-full-diagnostic-contract-20260604]]. Generated
  `raw/assets/benchmark-results/open_question_full_diagnostic_contract_20260604/`
  so all `Q1`--`Q44` now have fully specified diagnostic contracts while
  remaining explicitly non-promoted method claims.
- Added `benchmarks/diagnostics/calibration/selected_tail_promotion_gate.py`
  and [[selected-tail-promotion-gate-20260604]]. The strict Q1/Q5/Q7/Q8 gate
  over the 2026-06-04 1000-replicate selected-tail run evaluates `79`
  contexts: `69` are `undefined_support_failure`, `10` are
  `external_diagnostic_only`, and `0` are `external_admissible`. The seven
  context-level selected-tail admissible rows remain diagnostic-only because
  Q5 fails leave-one-parent-size-bin transfer and relative c-hat precision is
  not available in the selected-tail context table.
- Added
  `benchmarks/diagnostics/calibration/selected_tail_parent_size_balance_stability.py`
  and [[selected-tail-parent-size-balance-stability-20260604]]. On the
  300-replicate row-level selected-geometry table, the diagnostic finds `3`
  balance-scoped, parent-size-stable external-candidate contexts, all
  Gaussian Bernoulli projection-2 high-edge-action rows with balance bins
  `balance_0.1_0.25`, `balance_0.25_0.4`, and `balance_0.4_0.5`; `51`
  contexts remain support failures and `2` lack parent-size holdout coverage.
- Ingested `raw/inbox/traceable-benchmark-suite-request-20260604.txt` and
  added [[traceable-benchmark-suite-request-20260604]] and
  [[traceable-mathematical-benchmark-suite]]. Implemented the first
  `method_proof` benchmark suite with ten traceable cases, lightweight
  generators for the new binary/categorical/continuous/phylogenetic case
  families, and `benchmarks/diagnostics/math_trace/infer_benchmark_math.py`
  for deterministic row-level mathematical failure attribution.
- Ran the canonical full benchmark on 2026-06-05 and added
  [[full-benchmark-run-20260605]]. The run completed `120` cases and `9`
  methods with `1080` result rows in
  `benchmarks/results/run_20260605_084136Z_full/`. Added generator geometry
  coverage for the method-proof cases, a `KL_TE_RUN_DIR` resume/report
  override for the full runner, and a `method_proof` report section so the
  completed CSV could be post-processed into relationship, failure, and PDF
  reports.
- Ran the canonical MNIST example and added
  [[mnist-benchmark-run-20260605]]. The run sampled `2000` MNIST images and
  tested `20` distance/linkage configurations; four single-linkage
  configurations completed with heavy over-splitting and low ARI, while the
  remaining configurations failed closed under the strict sibling empirical-null
  support contract.
- Ran a continuous MNIST PCA50 probe and added
  [[mnist-continuous-pca50-run-20260605]]. The probe retained `83.13%` PCA
  variance, used Euclidean tree distances, and found the best ARI under
  complete linkage (`0.258244`) while single and average linkage collapsed to
  one cluster.
- Ran a focused continuous MNIST PCA50 alpha sweep and added
  [[mnist-continuous-alpha-sweep-20260605]]. The `75`-row grid completed
  without calibration-support or dense-covariance failures; strict Ward
  (`edge_alpha=0.0001`, `sibling_alpha=0.0001`) improved to ARI `0.514854`
  with `11` clusters, while higher alphas over-split the Ward tree.
- Debugged the strict selected-tail promotion gate and added
  [[selected-tail-promotion-gate-debug-20260605]]. The zero-admissible result
  decomposes into `69` support failures plus global Q5 parent-size transfer
  failure and missing c-hat precision metadata; six context-law candidates
  would pass only if both global blockers were relaxed, while the later
  balance-conditioned diagnostic exposes three narrower diagnostic candidates.
- Debugged the Q9/Q10/Q11 internal calibration status and added
  [[internal-calibration-q9-q10-q11-debug-20260605]]. Q9/Q11 support metadata
  and opt-in sparse-context enforcement are implemented. A follow-up threaded
  the enforcement flag and custom thresholds through sibling adjustment,
  sibling annotation, gate annotation, and `TreeDecomposition` while keeping
  defaults off. Q10 remains diagnostic-only because the current
  selected-geometry input lacks internal-support labels.
- Added `benchmarks/diagnostics/calibration/internal_support_threshold_validation.py`
  and extended `sibling_null_weight_rule_validation.py` so mixed null/signal
  panels can score permissive/current/strict internal support thresholds and
  Q10 can report selected-nonnull weight leakage when true support labels are
  supplied. No production-scale mixed panel is promoted; the result is
  diagnostic infrastructure for Q9/Q10/Q11 validation.
- Ran mixed internal-calibration sweeps and added
  [[mixed-internal-calibration-sweeps-20260605]]. The method-proof, binary,
  and categorical panels produce `163100`, `225639`, and `58861` Q10 sibling
  records, respectively. Q10 current product-BH selected-nonnull leakage is
  near zero in all three panels, but Q9/Q11 thresholds remain unpromoted
  because categorical admissible null false splits remain above nominal alpha
  and signal retention is low.
- Added `benchmarks/diagnostics/spectral/mp_projection_dimension_behavior_sweep.py`,
  ran method-proof, binary, and categorical Q14/Q15/Q17 behavior sweeps, and
  added [[mp-projection-dimension-behavior-sweeps-20260605]]. Raw parent MP
  count often chooses `k=0` and reduces false splits, but in the above-BBP MP
  spike case it also suppresses all signal; floor rules recover signal while
  leaving selected-tree raw sibling p-values anti-conservative. Recorded
  Gavish-Donoho hard singular-value thresholding as diagnostic rank-selection
  input rather than production calibration evidence.
- Debugged the Q14/Q15/Q17 behavior sweep failure mode with post-hoc
  attribution tables. The raw-MP improvement comes primarily from the
  zero-dimensional no-test branch; conditional on `k>0`, null-labeled selected
  sibling tests reject around `99%` of contexts. Required-alpha summaries show
  that nominal `1%` control would require raw p-value cutoffs on the order of
  `1e-113` or smaller, so simple alpha tuning is not a credible production
  calibration.
- Checked Q12/Q13 selected-PCA evidence by rerunning
  `benchmarks.validation.selected_pca_projected_wald_calibration` at 1000
  replicates per setting. The rerun reproduces the locked conclusion:
  leaf-only fixed-membership Gaussian selected PCA is calibrated near nominal
  (`0.046`, `0.048`, `0.059` rejection at alpha `0.05`), while child-mean-row
  spectral bases reject at `0.671`, `0.992`, and `1.000`. Updated
  [[selected-pca-projected-wald-validation]] and Q12/Q13 wording accordingly.

### 2026-06-05

- Recovered historical root-level KAK/cosine spectral scripts from git history
  after confirming they are absent from the current working tree. Added
  [[historical-kak-spectral-pipelines-20260605]] to record that
  `adaptive_cosine_spectral_blocks.py`, `kak_mp_tree_method_test.py`, and
  `cosine_subspace_tree_sweep.py` selected tree topology from spectral
  coordinates before running `TreeDecomposition` gates on the original feature
  matrix, while `kak_signal_adaptive_umap_tree_page.py` was visualization-only.
- Added and ran a current-compatible adaptive cosine/KAK benchmark probe using
  the production gate-bundle path. The method-proof run produced `37` ok and
  `5` fail-closed spectral-block rows; the full run produced `369` ok, `46`
  failed-gate, and `3` failed-spectrum rows. Added
  [[adaptive-cosine-kak-benchmark-probe-20260605]] to record that early
  non-common spectral intervals, especially `2-5` and `2-6`, carry most
  recoverable signal, while common-mode or high-energy block selection is only
  diagnostic without selected-tree calibration.
- Extended the adaptive cosine/KAK probe with explicit internal support
  threshold enforcement and a full-data matrix runner. Support enforcement
  reduced the full benchmark KAK probe from `369` to `248` ok rows and reduced
  cases with an ok block from `110` to `96`. On the combined Julia GO matrix,
  default blocks over-split into `101` to `650` clusters, while all `15`
  binary/TF-IDF blocks failed closed under support-threshold enforcement.

### 2026-06-06

- Restored `scripts/kak_signal_adaptive_umap_tree_page.py` as a
  current-compatible diagnostic page generator for the adaptive cosine/KAK
  matrix probe. The script now writes global/block embeddings, per-gene
  radius/angle/invariant-axis geometry, per-block pages, an interactive 3D
  embedding, and a per-internal-merge tree geometry table.
- Ran the restored diagnostic on the default combined Julia GO matrix KAK
  output. The full render wrote `11` ok block pages and `7722` internal merge
  rows under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/`.
  The restored internal geometry reinforces the diagnostic-only conclusion:
  non-common KAK blocks mostly live at high angle to the leading block axis,
  have high independent-radius fractions, and still over-split into many local
  pure fragments without support-enforced selected-tree calibration.
- Tested a candidate path-conditioned barycentric action equation and added
  [[barycentric-action-equation-diagnostic-20260606]]. The selected-tail side
  keeps `q5_barycentric_edge_spectral` as the best current compact equation;
  angle-augmented variants are unstable under feature-family transfer. The KAK
  traversal side shows that radius/angle/internal-axis variables improve
  leave-one-block-out explanation of current local pure fragments, so they are
  traversal diagnostics or future stratifiers rather than direct calibration
  replacements.
- Reran the canonical full benchmark and added [[full-benchmark-run-20260606]].
  The run completed `120` cases and `1080` method rows in
  `benchmarks/results/run_20260605_224254Z_full_big/`, with plots,
  relationship analysis, failure diagnosis, and a merged PDF report. The mean
  ARI leaderboard and KL skip profile match the previous full run: `kmeans`
  leads ok-row mean ARI, `kl` has `92` ok rows and `28` strict-support or dense
  covariance skips, and low-ARI ok `kl` rows are dominated by root-rejection
  under-splits.
- Added [[path-conditioned-barycentric-action-diagnostics-20260606]] after
  implementing cached-output diagnostics for the path-conditioned barycentric
  action trace. The diagnostic reproduces the KAK `radius_angle_action` median
  AUC `0.892810`, reruns the full benchmark in
  `benchmarks/results/run_20260606_path_action_full/`, and ranks angular-shell,
  action-budget, and traversal-survival paths as validation-panel candidates
  while keeping external selected-tail calibration fail-closed.
- Extended the path-conditioned barycentric action diagnostic with row-level
  radius/angle/action annotations and a high-action angular-shell guard panel.
  The panel shows that pure-fragment KAK contexts have higher median capped
  action-budget proxy than mixed contexts, and the best tested guard
  `action_ge_0.9__angle_ge_75__ind_ge_0.85` has pure-fragment precision
  `0.713537` with mixed-context flag rate `0.086202`. This remains a
  post-hoc diagnostic validation-panel candidate, not a production traversal
  rule.
- Added [[null-edge-sibling-calibration-enhancement-plan]] to integrate the
  path-conditioned diagnostics into a null edge and null sibling calibration
  roadmap. The plan keeps edge selected-null calibration, sibling
  selected-tail/internal support calibration, and traversal geometry as
  separate objects with fail-closed production defaults.
- Added a cost-sensitive utility curve to the path-conditioned barycentric
  action diagnostic and wrote
  `benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_utility/`.
  The equal-cost best guard is
  `action_ge_0.75__angle_ge_60__ind_ge_0.85` with net utility `744`, while the
  two-times mixed-context-cost best guard is
  `action_ge_0.9__angle_ge_60__ind_ge_0.85` with net utility `197`. No tested
  guard remains net-positive once delaying a mixed context costs `3x` or more
  than preventing a pure fragment.
- Added [[phase1-path-b-foundation-20260606]] after implementing
  `benchmarks/diagnostics/path_b/phase1_path_b_foundation.py`. The full KL-only
  Phase 1 Path B sweep wrote `960` rows over `k_min in {0,1,2,3}` crossed with
  pass-through on/off. The penalized full-suite optimum is `k_min=1` with
  pass-through enabled: penalized mean ARI `0.728301`, ok-row mean ARI
  `0.794510`, skip rate `0.083333`, and exact-k rate `0.736364`. The Q5
  selected-tail geometry gain panel again selects
  `q5_barycentric_edge_spectral`, reducing median residual-tail error by
  `0.260780` versus the no-spectral baseline.
- Added [[recursive-pvalue-geometry-20260606]] after implementing
  `benchmarks/diagnostics/path_b/recursive_pvalue_geometry.py`. The full
  recursive p-value geometry diagnostic completed `110` ok KL cases and `10`
  expected skips. It found connected but asymmetric p-value fields: raw
  edge-vs-parent-sibling coupling `0.392907`, raw recursive sibling continuity
  `0.443386`, raw subspace-rotation-vs-gap coupling `0.254220`, and
  eigen-gap-vs-tail-sensitivity `0.027132`. Raw edge alpha margins pass on
  `0.8372` of edges, while raw sibling margins pass on only `0.0822` of
  sibling contexts, reinforcing that edge traversal and sibling calibration
  must remain separate equations.
- Added [[mixed-null-signal-geometry-validation-20260606]] after implementing
  `benchmarks/diagnostics/path_b/mixed_null_signal_geometry_validation.py`.
  The full labeled panel completed `110` ok KL cases with `26369` sibling
  contexts: `22599` `null_context`, `2366` `mixed_context`, and `1404`
  `signal_context`. Corrected sibling splitting is conservative on clean
  truth-null contexts (`0.002611`) but weak on truth-signal contexts
  (`0.141026`). After adding row-aligned KAK-style radius/angle/action
  covariates in the root selected PCA frame, `kak_radius_angle_action` leads
  the full held-out signal-vs-null panel with median AUC `0.957892`, a gain of
  `0.051982` over `chi_square_only`; the method-proof suite also selects KAK
  with median AUC `0.897334` and gain `0.330447`. These terms remain
  diagnostic-only and are not a production selected-tail calibration rule.
- Extended [[mixed-null-signal-geometry-validation-20260606]] with
  scikit-learn continuous edge/sibling/KAK surfaces and replaced the
  process-randomized `case_hash_modulo_5` split with a stable CRC32 fold id.
  Re-evaluating the cached full labeled panel gives
  `sklearn_hist_gradient_edge_sibling_kak_surface` median AUC `0.985084`,
  `sklearn_logistic_edge_sibling_kak_surface` median AUC `0.966521`, and the
  original row-aligned `kak_radius_angle_action` median AUC `0.948342`, versus
  stable-fold `chi_square_only` median AUC `0.895775`. This supports modeling
  the edge/sibling/KAK relationship as a continuous diagnostic surface before
  any threshold guard or production calibration rule.
- Added [[diagnostic-framework-github-scan-20260606]] after scanning GitHub for
  analytical and machine learning frameworks that could strengthen KL-TE
  calibration diagnostics. No open GitHub issues or PRs were found in the
  project remote for calibration, selected-tail, KAK, FDR, or sibling search
  terms. The useful framework stack is selective-inference formalization,
  simulation-based inference for amortized selected-tail diagnostics, conformal
  risk control for held-out guards, and calibration/observability tooling for
  monitoring and interpretation.
- Debugged KL clustering failure attribution and added
  [[clustering-root-audit-debug-20260606]]. The root sibling decision is stored
  on the root audit row, so the failure and relationship analyzers now read the
  root row instead of root children. Corrected artifacts show `79` accepted-root
  ok KL rows with mean ARI `0.901264`, `13` rejected-root rows with mean ARI
  `0.307692`, and four low-ARI accepted-root post-root sibling traversal
  stalls where deeper sibling split rate is `0.0` despite persistent edge
  action.

### 2026-06-07

- Regenerated the canonical derived reports in
  `benchmarks/results/run_20260605_224254Z_full_big/` after the root-audit
  analyzer fix. The refreshed `benchmark_relationship_*` artifacts,
  `failure_report.md`, `benchmark_relationship_plots.pdf`, and
  `full_benchmark_report.pdf` now use the root-row sibling decision. The
  canonical low-ARI KL diagnosis now separates nine true root rejections from
  four accepted-root post-root sibling traversal stalls.

### 2026-06-10

- Updated [[adaptive-cosine-kak-benchmark-probe-20260605]] with a diagnostic
  visualization rule: when reference labels are available, KAK/cosine tree
  visualizations should order displayed trees or block pages by cluster-level
  NMI so the most informative decompositions are inspected first. This is
  presentation-only and does not change tree construction, gates, traversal, or
  calibration.
- Implemented the NMI-ordering diagnostic in
  `scripts/kak_signal_adaptive_umap_tree_page.py` and reran the restored
  KAK/cosine visualization on the Julia GO matrix with Julia endotype labels.
  The run wrote the sorted page, per-block pages, reference-label table,
  reference NMI/ARI table, and NMI ranking plot under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/`.
  The top-NMI block was `binary__adaptive_modes_16_19` with NMI `0.632591`,
  ARI `0.002297`, and `650` clusters, confirming that the view is useful for
  inspecting geometry but still reflects severe fragmentation rather than a
  production-selected decomposition.
- Recreated the historical full-matrix comparators with
  `scripts/analysis/run_feature_matrix_with_umap.py` after updating it to the
  current gate-annotation bundle path. Against the same `262` saved Julia
  reference-label matches, fixed diffusion with gates led the recreated
  comparators by ARI (`0.062392`, `54` clusters), followed by paper cosine
  complete `K=20` (`0.058671`, `20` clusters). The high-NMI KAK/KL gate rows
  were near-zero ARI because they split the `703` genes into roughly `650` to
  `670` clusters. The comparison artifacts are under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/`.
- Added and ran
  `benchmarks/diagnostics/spectral/adaptive_cosine_kak_diffusion_matrix_probe.py`
  to test fixed diffusion inside the separated KAK/cosine spectral spaces on
  the Julia GO matrix. The run wrote
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/`.
  It produced `12` ok rows and `3` calibration-support fail-closed rows. The
  best row, `binary__adaptive_modes_02_05`, improved Julia-reference ARI to
  `0.085175` with NMI `0.498510`, but still produced `153` clusters and
  singleton fraction `0.568627`, so separated-space diffusion is a useful
  diagnostic direction rather than a production clustering rule.
- Extended the separated-space diffusion diagnostic with
  `--diffusion-mode adaptive` and reran the Julia GO matrix under the pydiffmap
  variable-bandwidth backend. The adaptive run wrote
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/`.
  Its best row was again `binary__adaptive_modes_02_05`, with `130` clusters,
  singleton fraction `0.261538`, NMI `0.516169`, and ARI `0.062483`. Adaptive
  diffusion reduced singleton fragmentation in that block but did not beat the
  fixed separated-space ARI `0.085175`.
- Reran the Julia GO tree comparison under an internal context-quality framing
  that intentionally ignores external Julia ARI. The context report under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/`
  compares non-singleton coverage, medium-size context counts, within-context
  GO Jaccard, and Fisher/BH enrichment fractions. Full adaptive diffusion is
  the strongest broad context tree among tested candidates (`63` clusters,
  `54` non-singleton contexts, `40` medium-size contexts, gene-weighted
  Jaccard `0.143262`, enrichment fraction `0.925926`), while separated
  adaptive TF-IDF `14-30` gives finer but more fragmented high-coherence
  subcontexts.
- Added a conditional subspace-lens analysis that treats full adaptive diffusion
  as the main context tree and compares KAK/cosine block assignments as
  overlays. The report under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/`
  shows that raw KAK `tfidf__adaptive_modes_14_30`, separated fixed diffusion
  `tfidf__adaptive_modes_14_30`, and raw KAK `binary__adaptive_modes_10_15`
  are the strongest refinement lenses, especially for main adaptive context
  `60`.
- Generated the plot-list page under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/14_main_context_subspace_lens_plot_list_20260610/`
  with the main adaptive diffusion embedding/tree, context-quality plots,
  lens-refinement scatter, raw KAK geometry overview, and top subspace-lens
  panels. Updated [[adaptive-cosine-kak-benchmark-probe-20260605]] to record
  that the current decomposition is cosine spectral/KAK-inspired diagnostic
  geometry, not a formal Cartan decomposition unless the symmetry group,
  invariant axis, and selected-subspace calibration law are specified.
- Added `benchmarks/diagnostics/spectral/kak_lens_alpha_sweep.py` and ran a
  focused edge/sibling alpha sweep over six useful KAK/cosine diagnostic
  lenses while holding full adaptive diffusion fixed as the main context tree.
  The output under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/`
  contains `54` rows: `48` ok and `6` fail-closed gate rows. The sweep shows
  sibling alpha is the main lens-resolution knob, while edge alpha is mostly
  secondary except that `edge_alpha=0.003` breaks empirical-null support for
  two separated-space lenses.

### 2026-06-11

- Extended `benchmarks/diagnostics/spectral/kak_lens_alpha_sweep.py` with
  average-versus-Ward linkage comparisons for KAK/cosine diagnostic lenses.
  Ward is constrained to Euclidean lens coordinates, not Hamming distances.
  The raw KAK binary `10-15` smoke at `edge_alpha=0.001` and
  `sibling_alpha=0.01` showed average linkage as the useful overlay
  (`101` clusters, singleton-gene fraction `0.022760`, lens score `0.877331`)
  and Ward-Euclidean as an over-fragmenting tree (`638` clusters,
  singleton-gene fraction `0.846373`, lens score `0.190178`).
- Added `benchmarks/cloud/aws_kak_lens_linkage_alpha_sweep.py`, updated the AWS
  benchmark Docker image dependencies, documented the Batch commands in
  `benchmarks/cloud/aws/README.md`, and added
  `tests/validation/82_test_aws_kak_lens_linkage_alpha_sweep.py`. The default
  cloud contract initially covered six lenses, average plus Ward-Euclidean, and
  a `5 x 5` edge/sibling alpha grid. Local two-shard run/merge smoke passed; AWS
  submission remained pending because the local AWS session had expired and
  `aws login` was waiting for browser authentication.
- Added complete linkage to the KAK/cosine lens linkage comparison and reran
  the raw KAK binary `10-15` local smoke with average, complete, and Ward at
  `edge_alpha=0.001`, `sibling_alpha=0.01`. Complete behaved like Ward on this
  lens: `638` clusters, singleton-gene fraction `0.846373`, and lens score
  `0.199089`, compared with average linkage at `101` clusters and score
  `0.877331`. The AWS default contract now covers six lenses, three linkage
  methods, and a `5 x 5` alpha grid for `450` candidate rows.
- Added `benchmarks/diagnostics/spectral/covariance_axis_stability.py` and
  `tests/validation/83_test_covariance_axis_stability.py` to test the
  covariance/PCA analogue of the candidate invariant axis. The diagnostic keeps
  genes aligned, resamples feature columns, recomputes centered dual covariance
  sample axes, and aligns replicates by sign and Procrustes. On the Julia
  matrix with `30` replicates and `80%` feature subsampling, median axis-1
  absolute cosine was `0.999692` for binary and `0.999702` for TF-IDF; median
  top-6 subspace canonical correlation was `0.998592` for binary and
  `0.998271` for TF-IDF. Recorded this as stable covariance-axis diagnostic
  evidence, not as a formal Cartan invariant-axis proof.

### 2026-06-12

- Added `benchmarks/diagnostics/spectral/kak_lens_feature_axis_clustering.py`
  and `tests/validation/84_test_kak_lens_feature_axis_clustering.py` to map
  raw KAK/cosine lens axes back to GO feature loadings using
  \(q_j = Z^T u_j / \sqrt{\lambda_j}\), then cluster genes through the normal
  KL-TE gate path. The diagnostic intentionally accepts only `raw_kak` lenses;
  separated diffusion lenses need separate nonlinear attribution. The Julia
  run under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/`
  reproduced the raw binary `10-15` `101`-cluster lens with singleton fraction
  `0.022760` and produced a more fragmented raw TF-IDF `14-30` lens with
  `311` clusters and singleton fraction `0.220484`. In both lenses, positive
  regulation of DNA-templated transcription (`GO:0045893`) was the top
  common-to-variant feature bridge.
- Debugged the raw KAK lens feature-axis mapping by adding reconstruction and
  orthogonality audit metrics, then reran the Julia binary `10-15` and TF-IDF
  `14-30` lens mappings under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/21_kak_lens_feature_axis_clustering_debug_20260612/`.
  The largest full feature-axis reconstruction error was `6.55e-15`, so the
  feature-axis map is numerically consistent and the remaining fragmentation is
  traversal/geometric behavior rather than a normalization mismatch.
- Added `benchmarks/diagnostics/spectral/tree_strategy_semantic_panel.py`,
  `benchmarks/cloud/aws_tree_strategy_semantic_panel.py`, and
  `tests/validation/85_test_tree_strategy_semantic_panel.py` to join Julia
  tree/lens diagnostics into the requested semantic panel. The local smoke
  under `benchmarks/results/diagnostics/tree_strategy_semantic_panel_smoke_20260612/`
  wrote `59` rows: `2` main-context trees, `37` diagnostic-lens rows, `19`
  fragmentation-lens rows, and `1` coarse baseline row. AWS submission was not
  completed because the local AWS CLI could not reach the AWS sign-in endpoint.
- Corrected the main rectangular tree renderer in
  `kl_clustering_analysis/plot/cluster_tree_visualization.py` to draw routed
  elbow edges instead of straight parent-child diagonals. The crossing-edge
  artifact was a visualization routing problem in deep unbalanced trees, not a
  graph or clustering change. The Plotly subembedding white lines were
  identified as scene grid/axis lines rather than tree edges.
- Completed the full AWS KAK/cosine lens linkage alpha sweep after retrying
  missing shard `0005`; the original array failure was an ECR pull timeout.
  The merged `450`-row output is under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/22_aws_kak_lens_linkage_alpha_sweep_full_20260612/merged/`.
  The regenerated semantic panel is under
  `benchmarks/results/diagnostics/tree_strategy_semantic_panel_aws_full_kak_20260612/`
  and classifies the full joined evidence as `2` main context trees, `149`
  diagnostic lenses, `185` fragmentation lenses, `2` fine lenses, and `1`
  coarse baseline row.
- Added `benchmarks/diagnostics/spectral/kak_feature_subspace_clustering.py`
  and `tests/validation/86_test_kak_feature_subspace_clustering.py` to test
  the feature-side interpretation of the KAK route. The diagnostic embeds
  binary GO features, not genes, in each selected raw KAK block as
  `Q_B sqrt(Lambda_B)`. The first Julia binary run under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/23_kak_binary_feature_subspace_clustering_20260612/`
  wrote feature coordinates for all eight adaptive blocks but all blocks failed
  closed in the current gate layer, exposing the missing feature-side
  edge/sibling calibration equations.
- Generated a classical UMAP-only clustering panel for the Julia full-data
  coordinates under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/`.
  On the stored two-dimensional UMAP plane, complete linkage `K=20` had the
  highest reference ARI (`0.145744`, NMI `0.403709`), with Ward, average,
  GMM, and k-means close behind. This was recorded as a visual manifold
  baseline, not as calibrated tree-context evidence.
- Added `benchmarks/diagnostics/calibration/selected_sibling_lrt_diagnostic.py`
  and `tests/validation/87_test_selected_sibling_lrt_diagnostic.py` to compare
  Bernoulli sibling deviance against the existing projected-Wald sibling
  statistic on whole-space fixed-diffusion Julia trees. The diagnostic run
  under
  `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/`
  shows that the fragmented `k=30,t=3` regime raises median projected-Wald
  selected ratio to `18.603097` while median deviance per changed feature stays
  near `2.010217`, separating topology/projection amplification from raw
  child-distribution contrast.
- Wired the first production-side external selected-tail calibration branch.
  `ExternalSelectedTailCalibrationModel` stores exact pre-promoted contexts and
  returns `external_admissible_scalar` only on exact context matches; otherwise
  it returns `undefined_external_not_admissible`. The sibling adjustment path
  can now use this external model when internal support thresholds fail or when
  no internal empirical-null model can be fit, and gate metadata records whether
  an external selected-tail rule set was active. Focused verification passed:
  `pytest tests/statistics/35_test_empirical_null_inflation_estimation.py -q`,
  `pytest tests/core/test_gate_annotation_reuse.py -q`, and
  `pytest tests/validation/72_test_selected_tail_promotion_gate.py tests/validation/73_test_selected_tail_parent_size_balance_stability.py -q`.
  This is runtime plumbing only; production-scale row-level evidence is still
  required before diagnostic balance-scoped contexts should be promoted.
- Added executable diagnostics for all four null edge/sibling calibration
  roadmap phases: `edge_null_calibration_panel.py`,
  `sibling_null_calibration_panel.py`, `traversal_guard_validation_panel.py`,
  and `production_admissibility_contract.py`, with validation tests
  `88`--`91` and source pages [[edge-null-calibration-panel-20260613]],
  [[sibling-null-calibration-panel-20260613]],
  [[traversal-guard-validation-panel-20260613]], and
  [[production-admissibility-contract-20260613]]. The package separates
  fixed-tree edge nulls, selected-tree edge nulls, sibling strict/stopped/null
  roles, selected-nonnull sibling rows, external selected-tail contexts,
  traversal guard candidates, and final production-admissibility decisions.
  All outputs are diagnostic-only unless a downstream predeclared evidence
  panel supplies production-ready component statuses.
- Added `benchmarks/diagnostics/calibration/selected_edge_sibling_null_equation.py`,
  `tests/validation/92_test_selected_edge_sibling_null_equation.py`, and
  [[selected-edge-sibling-null-equation-20260613]]. The diagnostic implements
  the practical conditional equation
  \(P(T_u^{\mathrm{sib}}\ge t\mid f_u,k_u,A_u^{\mathrm{edge}},b_u,\mathrm{edge\ path\ open})\)
  by exact matched empirical-null contexts and add-one tail probabilities.
  Unsupported or sparse matched contexts fail closed rather than producing a
  calibrated p-value.
- Checked the selected edge+sibling equation against current benchmark
  evidence. The stored 2026-06-04 selected-edge pilot sibling artifact cannot
  run the equation because `sibling_raw_stat` and `sibling_raw_p` are entirely
  absent, while live method-proof examples produce conditional empirical
  p-values only in contexts with exact matched strict-null support. The
  production-admissibility contract now treats those finite equation p-values
  as diagnostic-only and sparse/unmatched equation contexts as fail-closed.
- Added `benchmarks/diagnostics/calibration/statistic_distribution_shape_panel.py`,
  `tests/validation/93_test_statistic_distribution_shape_panel.py`, and
  [[statistic-distribution-shape-panel-20260613]]. The panel compares empirical
  statistic skew/tails against the chi-square skew implied by current df and an
  optional covariance-inferred Satterthwaite df/scale reference. Live
  method-proof sibling rows showed covariance-inferred effective df almost
  unchanged from current projection df, but large covariance scales that reduce
  tail rates without restoring nominal calibration. This makes df count alone
  unlikely to explain the observed distribution mismatch.
- Added `benchmarks/diagnostics/calibration/covariance_laplacian_panel.py`,
  `tests/validation/94_test_covariance_laplacian_panel.py`, and
  [[covariance-laplacian-panel-20260613]]. The panel turns covariance matrices
  into absolute-correlation Laplacian graphs. Live method-proof checks showed
  all sampled Bernoulli sibling contrast covariance matrices are diagonal
  Laplacian graphs, while reconstructed parent spectral covariance matrices are
  dense and mostly connected. This records a wiki coverage gap: prior pages
  covered MP eigenvalues, selected-tail spectral covariates, and covariance
  validation, but not covariance graph connectivity or the separation between
  sibling whitening covariance and parent spectral covariance.
- Updated `benchmarks/validation/selected_edge_type1_geometry.py` so future
  selected-edge sibling artifacts export raw sibling statistic, raw p-value,
  adjusted statistic/p-value when available, covariance-inferred df/scale, and
  bounded Laplacian summaries for sibling contrast and parent spectral
  covariance. Raw sibling rows are collected before decomposition adjustment,
  so calibration-support failures can still be inspected by the distribution
  and selected edge+sibling equation panels.
- Added `benchmarks/diagnostics/calibration/selected_edge_sibling_postrun_analysis.py`,
  `tests/validation/95_test_selected_edge_sibling_postrun_analysis.py`, and
  [[selected-edge-sibling-postrun-analysis-20260613]]. A local two-replicate
  `binary_2clusters` selected-tree smoke run produced `196` edge rows, `98`
  sibling rows, and `2` final rows. Current chi-square sibling tails remained
  inflated at tail rate `1.0` in both df bins; covariance-inferred alternate
  tail rates dropped to about `0.158` and `0.864`; the selected edge+sibling
  equation produced `95` supported conditional empirical p-values and withheld
  `3` rows for insufficient null support.
- Extended the selected-edge sibling post-run analyzer so it writes
  `production_admissibility_components.csv` and
  `production_admissibility_summary.csv` through the same conservative
  production-admissibility contract. On the local two-replicate
  `binary_2clusters` smoke artifact, the summary has `7` required components,
  `3` fail-closed blockers, and `4` diagnostic-only components, with final
  decision `fail_closed_undefined`. The blockers are the two
  `skew_exceeds_df_reference` distribution-shape bins and
  `selected_edge_sibling_equation:insufficient_null_support`.
- Added `benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py`,
  `tests/validation/96_test_differential_statistic_validity_panel.py`, and
  [[differential-statistic-validity-panel-20260613]]. The panel reconstructs
  the implemented whitened sibling contrast and parent PCA projection, then
  reports Fisher/Wald boundary geometry, fixed-projection finite differences,
  recomputed-projection sensitivity, eigengap instability, and nonsmooth
  Hamming selection stability. A one-replicate `binary_2clusters` fixed-tree
  plus selected-tree smoke produced `98` rows: `81`
  `wald_metric_boundary_unstable`, `11` `fixed_subspace_candidate`, and `6`
  `nonsmooth_selection_geometry`. The production summary had `4` required
  components, all fail-closed, with final decision `fail_closed_undefined`.
- Added `benchmarks/diagnostics/calibration/regularized_wald_statistic_panel.py`,
  `tests/validation/97_test_regularized_wald_statistic_panel.py`, and
  [[regularized-wald-statistic-panel-20260613]]. The panel compares plug-in,
  Jeffreys-smoothed, Dirichlet-smoothed, and root-shrunk sibling
  projected-Wald statistics before any production promotion. In a five-
  replicate `binary_2clusters` fixed-tree smoke, smoothing reduced boundary
  instability to zero for Jeffreys, Dirichlet, and root-shrink variants, but
  fixed-tree chi-square tail rates still ranged from `0.500` to `0.933`.
  The selected-tree smoke had smoothed tail rates `0.969`--`1.000`. Both
  production summaries remained `fail_closed_undefined`.
- Added `benchmarks/diagnostics/calibration/null_law_decomposition_panel.py`,
  `tests/validation/98_test_null_law_decomposition_panel.py`, and
  [[null-law-decomposition-panel-20260613]]. The panel separates fixed-topology
  sibling null behavior into same-sample adaptive projection, independent
  tree-sample projection, and random fixed orthonormal projection. In a
  20-replicate `binary_2clusters` smoke, same-sample adaptive projection had
  tail rates `0.916` and `0.745` across the two df bins, while independent
  tree-sample projection had `0.053` and `0.076`, and random fixed projection
  had `0.063` and `0.036`. Projection row orthonormality and operator weights
  stayed at the fixed-subspace chi-square values, so the null-law failure is
  same-sample adaptive projection/dimension selection rather than the
  quadratic form itself. The production summary remains
  `fail_closed_undefined`.
- Recorded the user constraint that cross-fit/sample-split methods are not the
  intended next production path. Added `raw/inbox/toomanycells-method-notes-20260613.md`
  and [[toomanycells-method-20260613]] after checking the TooManyCells public
  documentation and publication references. The note is relational, not a
  direct comparator: TooManyCells is recorded as a tree-first divisive
  hierarchical spectral clustering method with Newman-Girvan modularity
  stopping. It is not a repair for the current KL-TE projected-Wald null law and
  not a proposed replacement for the KL-TE gate.
- Added `benchmarks/diagnostics/calibration/data_independent_sibling_gate_panel.py`,
  `tests/validation/99_test_data_independent_sibling_gate_panel.py`,
  `raw/inbox/data-independent-sibling-gate-smoke-20260613.md`, and
  [[data-independent-sibling-gate-panel-20260613]]. The panel implements a
  same-data, non-cross-fit candidate that removes adaptive parent PCA/dimension
  selection from the sibling statistic and uses fixed coordinate-wise
  Bonferroni/BH p-values with a selected-topology penalty. In the
  `binary_2clusters` 50-replicate selected-topology smoke, null effective-alpha
  rejection was `0.007755` against target `0.01`, signal rejection was
  `0.175510` to `0.194694`, and large-parent signal rejection was `0.738019`.
  Production admissibility remains `diagnostic_only`.
- Extended the data-independent sibling gate panel to evaluate a
  selected-topology penalty grid, support direct categorical cases with
  `FeatureSpace` metadata, emit
  `data_independent_sibling_gate_penalty_transfer_summary.csv`, and scope
  production-admissibility contracts by method, penalty, topology mode, and
  feature family. Added `raw/inbox/data-independent-sibling-gate-transfer-20260613.md`.
  In a three-case binary transfer smoke, `coordinate_bh` with penalty `10` was
  the only grid candidate controlling all selected-null cases while retaining
  all signal cases under the all-parent threshold. A direct categorical smoke
  remained null-conservative but signal-weak, leaving categorical aggregation
  and traversal-target power as open work.
- Added `block_bonferroni` and `block_bh` candidate methods to the
  data-independent sibling gate panel. These aggregate fixed feature-block
  chi-square p-values instead of individual coordinates, using the categorical
  simplex chart dimension as degrees of freedom. A categorical block-gate smoke
  over `cat_clear_3cat_4c`, `cat_mod_3cat_4c`, and
  `cat_highcard_10cat_4c` stayed null-conservative but signal-weak; the best
  minimum all-parent signal rejection was `0.033166` for block BH and
  `0.049580` for coordinate BH. This rules out simple categorical block
  aggregation as the missing fix.
- Added `benchmarks/diagnostics/calibration/data_independent_sibling_gate_traversal_panel.py`,
  `tests/validation/100_test_data_independent_sibling_gate_traversal_panel.py`,
  `raw/inbox/data-independent-sibling-gate-traversal-20260613.md`, and
  [[data-independent-sibling-gate-traversal-panel-20260613]]. The panel runs
  fixed data-independent sibling gates through the actual top-down traversal.
  In an eight-replicate six-case smoke, fixed `coordinate_bh` gates recovered
  strong signal ARI for binary and direct categorical cases, but selected-null
  traversal still false-split above the diagnostic threshold. The remaining
  blocker is selected topology plus edge-reachable traversal null control, not
  same-sample adaptive sibling projection.
- Extended the data-independent sibling gate traversal panel with split-geometry
  diagnostics and
  `data_independent_sibling_gate_traversal_transfer_summary.csv`. A mixed
  16-replicate smoke with edge alpha `0.0001` and penalties `500` and `1000`
  showed binary transfer candidates with max null false-split rate `0.0000`
  and minimum signal mean ARI at least `0.987067`; direct categorical transfer
  still failed because high-cardinality null false splits persisted and
  moderate categorical signal weakened at stronger penalties. A 128-replicate
  `cat_highcard_10cat_4c` geometry follow-up estimated false splitting at
  `7/128 = 0.0546875`, all at the selected root with first child size `9`--`44`
  out of `200`. The matching signal run had mean ARI `0.755123` and first root
  min-child size at least `48`, but other valid signal cases have smaller first
  splits, so the evidence supports selective/adaptive traversal-null
  conditioning rather than a universal hard balance guard.
- Added an optional selected-root permutation diagnostic to the same traversal
  panel. It preserves Bernoulli/categorical feature-block margins, reruns
  selected tree construction on permuted null samples, and reports a
  Monte-Carlo selected-root p-value for the fixed sibling gate. In a
  four-replicate `cat_highcard_10cat_4c` smoke with `9` permutation draws, the
  known false root changed from raw p-value `1.707e-6` to selected-root
  p-value `0.1`, but true signal roots also had selected-root p-values
  `0.1`--`0.2`. This makes permutation conditioning a useful selected-null
  diagnostic but not yet a power-preserving production stopping statistic.
- Extended the traversal panel to emit
  `production_admissibility_components.csv` and
  `production_admissibility_summary.csv` from traversal transfer evidence. A
  small contract smoke over `binary_2clusters` and `cat_highcard_10cat_4c` at
  penalty `1000` produced the intended conservative boundary: binary
  fixed-coordinate transfer is represented as `diagnostic_only`, while
  categorical high-cardinality transfer remains `fail_closed_undefined`. This
  turns the non-cross-fit repair into an explicit domain-scoped candidate
  without promoting it beyond the available evidence.
- Added optional selected-root feature-subsample stability diagnostics and a
  root-stability guard to the data-independent traversal panel. The diagnostic
  rebuilds selected trees on feature-block subsamples and compares root
  bipartitions by ARI. In a 16-replicate `cat_highcard_10cat_4c` smoke, a
  threshold of `0.08` blocked the one null false root, reduced false-split rate
  from `0.0625` to `0.0`, and blocked no signal roots; signal mean ARI stayed
  `0.824357`. In a six-case mixed smoke the guard controlled all tested nulls
  and preserved binary transfer, but categorical transfer remained signal-weak
  because `cat_mod_3cat_4c` stayed below the `0.75` signal threshold. The best
  categorical penalty/edge follow-up reached minimum signal mean ARI `0.747419`
  while keeping null false-split rate `0.0`.
- Ran a follow-up threshold sweep for the root-stability guard. With fixed
  coordinate BH, edge alpha `0.001`, selected-topology penalty `50`,
  feature-subsample fraction `0.8`, `12` stability subsamples, and
  root-stability threshold `0.15`, the 16-replicate six-case mixed smoke
  transferred across binary and direct categorical cases. Binary max null
  false-split rate was `0.0` with minimum signal mean ARI `0.865005`;
  categorical max null false-split rate was `0.0` with minimum signal mean ARI
  `0.771764`. Production summaries remain `diagnostic_only`, but the current
  method candidate is now explicit: fixed coordinate BH plus selected-topology
  penalty plus selected-root feature-subsample stability.
- Added confidence-bound transfer diagnostics to the data-independent traversal
  panel. Case summaries now include Wilson upper confidence bounds for null
  false-split rates and one-sided t lower confidence bounds for signal mean
  ARI; production-admissibility rows include separate point-transfer and
  confidence-bound components. In the 16-replicate candidate smoke, point
  estimates still transferred, but both binary and categorical confidence
  components failed closed because zero observed false splits still produced a
  null false-split upper confidence bound of `0.193608`, above the `0.05`
  target. This keeps the candidate diagnostic-only until a larger validation
  run or analytic selected-null bound tightens the uncertainty.
- Added validation-support sizing to traversal transfer summaries. For the
  current 95% Wilson bound and a `0.05` false-split target, zero observed false
  splits require `73` null replicates per case. The 16-replicate candidate
  smoke therefore needs `57` additional zero-false-split null replicates per
  case before the null confidence component can pass, with separate signal
  lower-bound validation still required.
- Ran the 73-replicate mixed candidate validation at root-stability threshold
  `0.15`. Point-transfer still passed, but confidence failed because null false
  splits appeared in `binary_2clusters`, `cat_clear_3cat_4c`, and
  `cat_mod_3cat_4c`; both binary and categorical family summaries had max null
  false-split rate `0.027397` and null upper confidence `0.094501`. Added
  `root_stability_threshold_sensitivity.csv` to the traversal panel. Post-run
  sensitivity on the same evidence identified threshold `0.24` as the first
  tested value passing both binary and categorical point/confidence checks:
  max null false split `0.0`, null upper confidence `0.049992`, binary signal
  lower confidence `0.762835`, and categorical signal lower confidence
  `0.758985`. This is a prospective-validation candidate, not a production
  constant, because the threshold was selected after inspecting the run.
- Ran the pre-alignment prospective 73-replicate validation with root-stability
  threshold `0.24` predeclared. The fixed coordinate BH +
  selected-topology penalty `50` + root-stability guard candidate passed point
  and confidence checks on the six-case binary/direct-categorical smoke suite.
  Binary max null false-split rate was `0.0`, null upper confidence
  `0.049992`, minimum signal mean ARI `0.835616`, and minimum signal lower
  confidence `0.762835`. Direct categorical max null false-split rate was
  `0.0`, null upper confidence `0.049992`, minimum signal mean ARI `0.782621`,
  and minimum signal lower confidence `0.758985`. The production contract still reports
  `diagnostic_only`, because candidate statuses are intentionally not
  production-ready without broader generalization or threshold derivation.
- Added a relational TooManyCells check and broader fixed-gate stress probes.
  TooManyCells is recorded only as a tree-first divisive spectral reference
  with Newman-Girvan modularity stopping, not as a direct KL-TE comparator. The
  full supported traversal surface contains 42 binary-template and 11 direct
  categorical-multinomial cases, but all-surface inline runs need checkpointed
  output before they are practical. Three-replicate targeted probes with the
  same fixed-gate constants found zero observed null false splits in targeted
  binary and direct categorical cases. Binary signal retained across seven
  targeted cases with minimum mean ARI `0.805955`; direct categorical signal
  failed because `cat_highcard_20cat_4c` had mean ARI `0.080649` and
  `cat_overlap_3cat_4c` had mean ARI `0.703263`. The next method problem is
  categorical power without reopening selected-root null inflation.
- Added checkpoint row output and selected-tree oracle ARI to the traversal
  panel. Checkpoints are written after every completed case-role-replicate
  unit, preserving partial evidence from broad validation runs. The oracle
  diagnostic shows that `cat_highcard_20cat_4c` is limited by the selected
  average-linkage tree itself: low-penalty `block_bh` reaches mean ARI
  `0.659781`, equal to the selected-tree oracle cut at the true cluster count,
  while edge alpha `0.01` does not improve the case. This moves the remaining
  categorical issue from sibling projection calibration to categorical tree
  construction or oracle-normalized validation.
- Added production-facing fixed-subspace sibling gate options:
  `sibling_gate_method="fixed_coordinate_bh"` and
  `sibling_gate_method="fixed_block_bh"`. These options are wired through
  `run_gate_annotation_pipeline`, `TreeDecomposition`, and `PosetTree.decompose`
  via forwarded kwargs. The default remains `projected_wald_inflation`. The new
  fixed gates compute covariance-whitened sibling contrasts and apply
  predeclared coordinate-wise or feature-block BH aggregation without using
  same-sample parent PCA projections or edge-derived sibling projection
  dimensions. Regression tests cover opt-in annotation, method metadata,
  categorical block aggregation, and cache invalidation across sibling-gate
  methods.
- Refactored the diagnostic BH fixed-gate p-value paths to delegate to the
  production `fixed_subspace_sibling_p_value` implementation. Added regression
  coverage proving that `fixed_coordinate_bh` does not call the parent-PCA
  sibling input resolver. This aligns the validation evidence with the
  production-facing method option and locks the main repair invariant into
  tests.
- Added `sibling_gate_alpha_penalty` to the gate annotation and decomposition
  path. The penalty is positive, defaults to `1.0`, is stored in gate config
  metadata, and divides `sibling_alpha` before sibling FDR. Tests cover
  effective alpha calculation, invalid penalties, and cache invalidation across
  penalty changes. An end-to-end `binary_2clusters` smoke with
  `fixed_coordinate_bh`, penalty `50`, edge alpha `0.001`, and sibling alpha
  `0.01` wrote `49` fixed-method sibling rows and `5` open sibling rows.
- Added the selected-root feature-subsample stability guard to the
  production-facing annotation and decomposition path. The guard is default
  off, records its threshold/subsampling settings in gate config metadata, and
  closes only an unstable open root by rewriting the root sibling decision to
  fail closed with `Root_Stability_*` diagnostics. Tests cover direct blocking,
  metadata capture, invalid inert guard configuration, and cache invalidation
  across guard changes. The runnable opt-in method now matches the diagnostic
  candidate: fixed-subspace sibling gate plus selected-topology alpha penalty
  plus selected-root stability guard, while production promotion remains gated
  by the admissibility contract.
- Added `fixed_global_chi_square` as the direct full fixed-subspace sibling
  gate. The statistic is `chi2.sf(z.T @ z, df=len(z))` on the existing
  covariance-whitened contrast, so it avoids parent PCA projection rows and
  edge-derived sibling dimensions while preserving the classical fixed-df Wald
  reference. Diagnostic panels expose the same option as
  `candidate_method="global_chi_square"`. A two-replicate `binary_2clusters`
  traversal smoke with penalty `50` and root-stability threshold `0.24` wrote
  all expected artifacts and produced point-transfer evidence, but production
  admissibility remained fail-closed because confidence was still
  null-uncertain at that sample size.
- Added named sibling-gate profiles `fixed_coordinate_guarded_v1` and
  `fixed_global_guarded_v1`. These profiles package the non-cross-fit fixed
  sibling gate, selected-topology penalty `50`, root-stability threshold
  `0.24`, `12` stability subsamples, feature fraction `0.8`, and deterministic
  seed `0` as auditable diagnostic candidates. The profile id is stored in
  gate annotation config metadata and included in cache reuse checks. Tests
  verify profile resolution, conflict detection, avoidance of parent-PCA
  sibling inputs, and recomputation when a cached adaptive bundle is reused
  under a fixed profile request.
- Wired fixed sibling-gate profiles through the shared KL benchmark runner.
  `_run_kl_method` and `_run_kl_on_distance` now forward profile/fixed-gate
  settings into gate annotation and decomposition and record them in
  `MethodRunResult.extra`. Added a runner regression for
  `fixed_global_guarded_v1`. The method-constants manifest now includes the
  fixed-profile constants `sibling_gate_profile`,
  `fixed_sibling_gate_alpha_penalty`, `root_stability_guard_threshold`,
  `root_stability_subsample_replicates`, and
  `root_stability_feature_fraction` as explicit validation targets; their
  skeleton evidence status remains `missing` until profile validation artifacts
  are attached.
- Added
  `benchmarks/diagnostics/calibration/fixed_sibling_gate_profile_validation.py`,
  `tests/validation/101_test_fixed_sibling_gate_profile_validation.py`,
  `raw/inbox/fixed-sibling-gate-profile-validation-20260613.md`, and
  [[fixed-sibling-gate-profile-validation-20260613]]. The panel validates named
  fixed sibling-gate profiles through the shared KL runner, records whether
  adaptive projected-Wald sibling rows were avoided, writes method-constant
  evidence fields, and emits conservative production-admissibility components.
  TooManyCells remains relational context only, not a direct comparator.
- Fixed the production-facing root-stability guard replay contract. The guard
  now records root replay distance metric and linkage method in gate config
  metadata and recomputes feature-subsample roots with the same Hamming/average
  contract used by the shared KL fixed-profile runner, instead of falling back
  to default Euclidean linkage replay. Focused tests cover metadata propagation,
  cache invalidation, runner extras, and profile-validation output columns.
- Aligned the data-independent sibling-gate traversal diagnostic with the same
  Hamming/average selected-tree replay contract. Selected trees, oracle cuts,
  root-selective p-values, and root feature-subsample stability now use
  explicit Hamming distance; row outputs and the manifest record the metric and
  linkage. Previous traversal numbers from before this correction should be
  treated as pre-alignment diagnostics unless rerun with these fields.
- Ran corrected Hamming/average traversal evidence for the fixed coordinate
  candidate. The 16-replicate six-case mixed smoke was a point-transfer
  candidate but confidence-uncertain. The 73-replicate support run remained a
  point-transfer candidate and signal confidence passed, but binary and direct
  categorical families each had one null false root (`1/73`, Wilson upper
  `0.073597`), so production admissibility stayed fail-closed. The traversal
  summary now reports observed-count-aware support sizing: with one false
  split, `110` total null replicates, or `37` additional zero-false null
  replicates, are needed to bring the Wilson upper bound below `0.05`. A
  corrected penalty-grid probe showed penalty `500` weakens categorical signal,
  so the evidence does not justify replacing the current penalty `50` profile
  constant.
- Added `root_selective_guard_sensitivity.csv` to the traversal panel. The
  diagnostic estimates a selected-root permutation guard that blocks opened
  roots whose selected-root p-value exceeds `sibling_alpha`. Targeted 99-draw
  checks on the corrected-Hamming false roots blocked the two null roots while
  retaining matched strong signal roots at the Monte Carlo floor; a nine-draw
  smoke was too coarse and blocked signal. This identifies selected-root
  permutation as the next method layer to validate, not a production default.
- Made selected-root permutation lazy in the traversal panel: all rows keep the
  observed root p-value, but expensive permutation draws are run only for
  opened or root-stability-blocked roots. Regression coverage verifies the lazy
  call pattern, and a 99-draw binary false-root smoke still writes the expected
  selected-root guard sensitivity artifact.
- Exposed the fixed coordinate-BH p-value as a direct helper so diagnostic
  selected-root permutation loops no longer rebuild a throwaway Bernoulli
  `FeatureSpace` for each coordinate-BH p-value. Focused validation confirms
  equality with the production fixed-subspace gate and statsmodels BH
  adjustment.
- Aligned the traversal diagnostic with production-facing fixed profiles by
  adding explicit `root_stability_seed` support, defaulting to `0`. The
  fixed-profile validation rows now record root-level sibling p-values,
  root-open status, root-stability mean/median/q10, and root-stability block
  status. An aligned replay at data seed `20309045` shows that
  `fixed_coordinate_guarded_v1` blocks the known binary null root but still
  opens the known direct-categorical null root; the 99-draw selected-root
  permutation diagnostic blocks that categorical false root while retaining the
  matched signal. Production summaries remain fail-closed because this is
  targeted evidence, not broad confidence evidence.

### 2026-06-14

- Exposed selected-root permutation as a production-facing, default-off runtime
  guard for fixed-subspace sibling gates. The guard preserves
  Bernoulli/categorical feature-block margins, reruns Hamming/average selected
  tree construction under permutation, records `Root_Selective_Permutation_*`
  audit columns, and closes only an open root whose selected-root p-value is
  above the guard alpha. It is intentionally rejected for
  `projected_wald_inflation`, because the same-sample adaptive PCA statistic is
  the invalid layer being avoided.
- Threaded the guard through `run_gate_annotation_pipeline`,
  `TreeDecomposition`, shared KL runner extras, gate annotation cache metadata,
  fixed-profile validation rows, summaries, method-constant evidence fields,
  and validation tests. Added method-constant targets for
  `root_selective_permutation_guard_replicates` and
  `root_selective_permutation_guard_alpha`.
- Ran a targeted runtime replay with `fixed_coordinate_guarded_v1`, data seed
  `20309045`, guard seed `20318458`, and `99` permutation draws. The known
  `cat_clear_3cat_4c` null root is now closed in the runtime profile path
  (`selected-root p = 0.17`, one final cluster), while the matched categorical
  signal is retained (`selected-root p = 0.01`, ARI `0.881909`). The known
  `binary_2clusters` null remains closed by root stability, with selected-root
  permutation also indicating a block (`p = 0.02`).
- Updated the raw notes and wiki synthesis to reflect the new boundary: the
  specific rooting/null-sibling artifact now has an executable non-cross-fit
  guard, but production promotion remains fail-closed until broad null
  confidence, signal confidence, guard replicate/alpha validation, and
  supported feature-family/tree-construction domains are established.
- Packaged the selected-root method layer as
  `fixed_coordinate_selective_root_v1`. The profile expands to fixed
  coordinate BH, selected-topology penalty `50`, root-stability threshold
  `0.24`, `12` stability subsamples, feature fraction `0.8`, selected-root
  permutation replicates `99`, seed `0`, and guard alpha `0.01`. Existing
  guarded profiles keep selected-root permutation disabled by default but still
  accept explicit guard settings for targeted replay. The shared KL runner now
  records resolved profile settings in `MethodRunResult.extra`, so validation
  artifacts report the actual method constants that ran.
- Updated fixed-profile method-constant evidence to use resolved profile
  settings. A profile-owned selected-root guard is now reported as enabled in
  evidence JSON even when the CLI guard override arguments are left at their
  defaults; the selective-root profile smoke records replicate grid `[99]` and
  alpha grid `[0.01]`.
- Rechecked the rooting/null-sibling boundary after the selected-root fix.
  Root-only guarding still leaves a `cat_clear_3cat_4c` pass-through
  descendant false split in the six-case two-replicate smoke. A broad
  `open_internal` selected-subtree guard closes nulls but weakens signal
  (`0.651330` binary mean signal ARI, `0.521072` categorical). Added the
  narrower `fixed_coordinate_selective_passthrough_v1` profile, which tests
  only descendant splits reached through an ordinary closed sibling ancestor
  and avoids compounding below explicit root guards. The v2 mixed smoke has
  zero observed null false splits, binary mean signal ARI `0.941101`, and
  categorical mean signal ARI `0.848463`; production remains fail-closed on
  smoke-scale confidence (`0.65762` null Wilson upper bound).
- Added `root_selective_permutation_guard_scope` as an explicit
  method-constant validation target and regenerated the method-constant
  manifest skeleton. Fixed-profile evidence now reports a `scope_grid`, so
  `root`, `open_internal`, and `passthrough_descendant` guard choices are
  auditable.
- Added Wilson support sizing to fixed-profile transfer summaries and
  manifest-style evidence. The two-replicate
  `fixed_coordinate_selective_passthrough_v1` mixed smoke now reports `73`
  required zero-false-split null replicates per case and `71` additional
  zero-false null replicates from the current support level. Fixed checkpoint
  resume to read CSVs with `keep_default_na=False`, preserving the literal
  `null` role in resumed rows so null confidence calculations are not dropped.
- Rechecked `fixed_coordinate_selective_passthrough_v1` on a six-case
  ten-replicate mixed run. Categorical nulls stayed closed, but
  `binary_many_clusters` null replicate `7` produced a pass-through descendant
  false split below an already closed unstable root. A hard closed-root
  pass-through barrier was rejected because it collapses two strong
  many-cluster signal rows from ARI `1.0` to `0.0`, and increasing the local
  selected-subtree permutation count to `999` still leaves the false row
  selected at p-value `0.002`. Diagnostic global selected-family replays move
  that false row to p-values around `0.05` to `0.06`, identifying the next
  non-cross-fit fix as an optimized global pass-through selected-family null,
  not a production constant change in this pass.
- Implemented the global selected-family pass-through null as
  `fixed_coordinate_global_passthrough_v1`. The new
  `global_sibling_min_passthrough_descendant` scope keeps selected-root
  permutation for roots and evaluates pass-through descendants against the
  minimum fixed-subspace sibling p-value over every binary parent in each
  fully reselected feature-block permutation null tree. A targeted
  `binary_many_clusters` replay closes null replicate `7` with selected p
  `0.05` and retains signal replicate `0` at ARI `1.0` with selected p
  `0.01`; the profile remains diagnostic because broad transfer and runtime
  optimization are still open.
- Added an exact Bernoulli fast path for the global selected-family
  fixed-coordinate BH statistic. The known false-row CLI replay now runs in
  about `16` seconds while preserving selected p `0.05`. Small transfer
  smokes are favorable but still confidence-limited:
  `/tmp/klte_global_passthrough_binary_10rep_20260614` has zero false splits
  across `30` binary null rows and signal mean ARI `0.945084`, while
  `/tmp/klte_global_passthrough_categorical_10rep_20260614` has zero false
  splits across `30` categorical null rows and signal mean ARI `0.832022`.
  Production remains fail-closed because Wilson confidence still needs
  additional zero-false null support; at this point categorical runtime still
  used the canonical covariance fallback, which was rechecked and optimized in
  the next pass.
- Rechecked the remaining fixed-profile elements and replaced the categorical
  fallback in the selected-family path with exact vectorized grouped
  multinomial whitening through `compute_whitened_wald_contrast`. The hard
  `binary_many_clusters` null seed replay remains closed with one cluster,
  selected-family p `0.05`, and scope
  `global_sibling_min_passthrough_descendant`. The rerun
  `/tmp/klte_global_passthrough_categorical_10rep_fast_20260614` preserves the
  prior direct-categorical smoke result, with zero false splits across `30`
  null rows and signal mean ARI `0.832022`, while wall time improves from about
  `309` seconds to about `147` seconds. Production remains fail-closed because
  support confidence still needs `63` additional zero-false null replicates per
  case; the remaining runtime issue is the repeated selected-family
  permutation loop, not categorical covariance object construction.
- Ran binary support-level validation for
  `fixed_coordinate_global_passthrough_v1` on `binary_2clusters`,
  `binary_many_clusters`, and `binary_unbalanced_low`. The first two cases
  have zero false splits through `142` null replicates each, but
  `binary_unbalanced_low` has three false splits, point false-split rate
  `0.021127`, and Wilson upper bound `0.060270`, so the confidence component
  remains fail-closed. All three false rows are pass-through descendants below
  closed roots and land on the `99`-draw selected-family p-value floor `0.01`.
  A `999`-draw replay gives p-values `0.017`, `0.005`, and `0.029`.
  Implemented the diagnostic successor
  `fixed_coordinate_global_passthrough_refined_v1`, which reruns global
  pass-through floor cases at `999` draws. Normal-runner replay closes two of
  the three observed false rows and keeps the genuinely stronger row with
  p-value `0.005`. The refined ten-replicate binary smoke has zero false
  splits across `30` null rows and signal mean ARI `0.945084`, but remains
  diagnostic until full support validation is run.
- Completed binary support-level validation for
  `fixed_coordinate_global_passthrough_refined_v1` on the same three binary
  transfer cases at `142` null/signal replicates per case. The refined profile
  has one false split across `426` null rows, the known
  `binary_unbalanced_low` replicate `96` with selected-family p-value `0.005`;
  the `binary_unbalanced_low` Wilson upper bound is `0.038809`, and the
  minimum signal mean ARI lower confidence across cases is `0.832434`. The
  production summary moves from `fail_closed_undefined` to `diagnostic_only`:
  confidence is now a candidate for the binary transfer set, while profile
  routing, transfer, and confidence components remain diagnostic-only pending
  broader method acceptance and direct-categorical support evidence.
- Ran the all-supported selected-null smoke and full 120-case single-seed
  performance pass for `fixed_coordinate_global_passthrough_refined_v1`; added
  `raw/inbox/refined-profile-all-benchmark-tests-20260614.md` and
  [[refined-profile-all-benchmark-tests-20260614]]. The selected-null smoke
  covered all 53 regenerable full-suite binary/direct-categorical cases with
  three null/signal replicates each. Direct categorical nulls stayed closed,
  but binary-template overlap nulls produced 11 false splits across 126 null
  rows, so all-supported production remains fail-closed. The full 120-case
  performance pass returned 107 ok rows and 13 continuous covariance-contract
  errors; ok rows had mean ARI `0.782567`, median ARI `0.955719`, and exact-K
  rate `0.439252`. The result keeps the refined profile diagnostic-only and
  identifies overlap-template null inflation, weak heavy-overlap/SBM/
  Dirichlet/deep-traversal signal, and continuous covariance support as the
  next blockers.
- Added `raw/inbox/root-selection-literature-20260614.md` and
  [[root-selection-literature-20260614]] after a literature check on root
  selection, hierarchical-clustering significance, TooManyCells, divisive
  high-dimensional split objectives, phylogenetic rooting, and trajectory-root
  orientation. The synthesis separates KL-TE's selected all-sample root split
  from biological or phylogenetic root orientation and records that the current
  method issue is selected-root/traversal null validity, not optimizing a free
  high-dimensional root point.
- Added `raw/inbox/selected-root-selected-family-traversal-literature-20260614.md`
  and [[selected-root-selected-family-traversal-literature-20260614]] after a
  focused literature check for the selected-root/selected-family traversal null
  law. The synthesis records that no exact KL-TE law was found, but the closest
  technical relatives are selective inference after hierarchical clustering,
  selective inference on multiple selected families, hierarchical FDR/FWER,
  TreeScan and scan-statistic max tests, and cluster-based permutation
  max-statistic methods. The current refined global pass-through diagnostic is
  best interpreted as a TreeScan/maxT-style selected-family permutation object
  for a data-selected hierarchy.
- Materialized a clean selected-root/pass-through null failure fixture under
  `raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/` and
  added [[selected-root-pass-through-null-fixture-20260614]]. The fixture uses
  iid Bernoulli null data with `400` samples, `80` features, and seed
  `20310054`; replaying `fixed_coordinate_global_passthrough_refined_v1`
  returns three clusters even though the root is closed by stability. The
  compact decision path records root `N798` closed with raw root sibling
  p-value `6.72e-6`, and descendant `N797` open with sibling p-value
  `5.21e-7`.
- Added
  `benchmarks/diagnostics/calibration/selected_family_traversal_panel.py`,
  `tests/validation/102_test_selected_family_traversal_panel.py`, and
  [[selected-family-traversal-panel-20260614]]. The panel compares baseline KL
  traversal with fixed-coordinate selected-root, local pass-through, and
  refined global selected-family profiles, writes selected-family guard rows,
  and emits multi-scale node, region, and sample outputs so stable boundaries,
  blocked selected families, pass-through zones, and leaf fragments are
  inspectable separately from the flat cluster labels.
- Added `scripts/analysis/multiscale_umap_overlay.py` for Julia-style
  multi-scale UMAP review. The helper joins `multiscale_gene_assignments.csv`
  to existing UMAP coordinates, colors stable regions first, and overlays
  pass-through or guard zones. The selected-family panel now writes the null
  role as `selected_null` and sample-level gene assignments with decomposition
  sample labels, so default CSV readers and UMAP joins do not lose the role or
  sample identifier.
- Added checkpoint/resume and per-row timeout support to
  `selected_family_traversal_panel.py`, then ran a one-replicate full-suite
  supported smoke for `fixed_coordinate_global_passthrough_refined_v1`. The run
  completed `101` of `106` supported binary/direct-categorical rows; five
  expensive high-dimensional categorical or overlap rows timed out at the
  `60` second per-row budget and were recorded in the manifest. Completed
  direct-categorical null rows stayed closed, but binary overlap-template nulls
  had four false splits and several binary/categorical signal rows were weak,
  so both production-admissibility summaries remain fail-closed.
- Added `scripts/analysis/run_selected_family_matrix.py` and ran the full
  Julia GO binary matrix through
  `fixed_coordinate_global_passthrough_refined_v1`, writing
  `raw/assets/benchmark-results/julia_selected_family_20260614/` and
  [[julia-selected-family-run-20260614]]. The selected-family profile returned
  `401` clusters, `306` singletons, `8` blocked selected-family/root guard
  rows, reference ARI `0.064157`, and reference NMI `0.618847`. The UMAP
  overlay shows stable regions more clearly after annotating the dominant
  selected-root guard zone `zone_N1404` covering `665` genes instead of
  outlining nearly every point.
- Added opt-in phylogenetic tree construction for KL runs:
  `kl_neighbor_joining` builds a neighbor-joining tree from the KL tree
  distance vector, `kl_iqtree3` calls an external IQ-TREE 3 binary and parses
  the resulting Newick tree, and both are rooted with minimum ancestor
  deviation before conversion to `PosetTree`. Added
  [[phylogenetic-tree-builders-20260614]] to keep this root-orientation
  implementation separate from the still-open selected-root/selected-family
  null-law problem.
- Ran the combined Julia GO matrix through baseline KL, MAD-rooted neighbor
  joining, and IQ-TREE 3 fast/MAD, then copied compact artifacts to
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/` and added
  [[julia-tree-estimator-run-20260614]]. Baseline KL produced `670` clusters
  with reference NMI `0.633958`; neighbor joining produced `494` clusters with
  reference NMI `0.616780`; IQ-TREE fast/MAD produced `566` clusters with
  reference NMI `0.629320`. The result shows tree estimator changes alter
  fragmentation but do not solve selected traversal calibration.
- Regenerated the Julia tree-estimator UMAP plots with high-contrast cluster
  color maps under
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/`.
  The clearer variants highlight only the largest non-singleton clusters with
  black outlines and encode cluster-size classes separately, because hundreds
  of tiny cluster IDs cannot be read reliably as one categorical colormap.
- Added `scripts/analysis/cluster_diagnostics_panel.py` and ran it over the
  Julia tree-estimator artifacts. The resulting
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/`
  bundle separates cluster-size fragmentation, reference recovery,
  reference-label fragmentation, method-cluster reference purity, UMAP
  compactness, and active-feature Jaccard coherence.
- Added full Julia UMAP exports under
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/`:
  a static all-gene reference-endotype view, a static all-gene method
  cluster-size view, and an interactive HTML view with per-gene cluster and
  reference metadata on hover.
- Ran a post-hoc alpha sensitivity audit for the Julia linkage and
  neighbor-joining tree-estimator runs, saving node-level default gate
  annotations and retresholded alpha-grid summaries under
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/alpha_audit/`.
  The audit shows edge decisions are nearly saturated, while sibling alpha
  controls a fragmentation versus UMAP-scatter tradeoff rather than producing
  clean UMAP-local clusters.
- Debugged the Julia UMAP/alpha analysis. The row-level interactive data has
  the expected `703 x 3` method rows with no duplicate gene-method entries, and
  stored assignment cluster sizes match recomputed sizes. Generated a
  standalone Plotly HTML because the first interactive export referenced the
  Plotly CDN, and added
  `raw/assets/benchmark-results/julia_tree_estimators_20260614/analysis_debug_report.md`
  to document the CDN issue and the post-hoc nature of the alpha audit.
- Added [[overlap-structural-sibling-panel-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_structural_sibling_panel.py` and
  `tests/validation/103_test_overlap_structural_sibling_panel.py`. The focused
  null/signal run over four binary overlap cases wrote `68` analytical rows
  under `raw/assets/benchmark-results/overlap_structural_sibling_20260614/`.
  All `38` selected-null rows were `weak_homogeneity_gain`, while accepted
  null splits retained the expected edge/sibling barycentric support alignment;
  the overlap failure is selected contrast without structural homogeneity gain.
- Tightened the overlap structural sibling panel so structure-versus-homogeneity
  support agreement is explicit. The regenerated outputs add
  `max_edge_homogeneity_jaccard_topk` and
  `subspace_consensus_jaccard_topk`; the same four-case run still classifies
  all selected-null analytical rows as `weak_homogeneity_gain`, not as
  structural-homogeneity subspace mismatches.
- Extended the same panel with heterogeneity-focused subspace support,
  heterogeneity pairwise gains, and `structural_change_mode`. The regenerated
  four-case output has no rows with strong same-subspace heterogeneity
  (`heterogeneity_gain_max >= 0.02`); selected-null rows remain weak/mixed
  structural change, while five signal rows are homogeneous same-subspace.
- Added [[overlap-structural-threshold-sensitivity-20260614]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_structural_threshold_sensitivity.py`
  and `tests/validation/104_test_overlap_structural_threshold_sensitivity.py`.
  The post-run sweep over `2304` case-level rows shows homogeneity gain around
  `0.02` as the focused overlap separator: zero null false accepts, all five
  truth-aligned signal accepts retained, and zero truth-misaligned accepts.
  Sibling p-value thresholds from `0.001` to `0.05` do not change the best
  rows, so alpha tightening alone is not the main traversal fix here.
- Extended the overlap threshold analyzer with case-replicate stability fields
  and ran a three-replicate four-case overlap follow-up under
  `raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/`.
  The replicate run weakens the one-seed threshold hypothesis: homogeneity
  threshold `0.02` still blocks null false accepts and truth-misaligned
  accepts, but retains only `12/17` truth-aligned accepts and `6/8`
  truth-aligned signal case-replicates. Threshold `0.01` retains `15/17`
  truth-aligned accepts but keeps two truth-misaligned accepts, while `0.005`
  leaves null false accepts. No tested threshold is stable enough for a method
  rule.
- Added [[overlap-structural-context-thresholds-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_structural_context_thresholds.py`
  and `tests/validation/105_test_overlap_structural_context_thresholds.py`.
  The context-binned run over the three-replicate overlap rows produced `864`
  sensitivity rows and `12` summary rows. Deep/internal contexts retained
  `7/7` truth-aligned signal accepts with zero null structural accepts at a
  permissive homogeneity threshold, while shallow and medium-parent contexts
  showed signal loss and large-parent contexts kept truth-misaligned signal.
  This points to context-conditioned traversal structure checks rather than a
  global cutoff.
- Added [[overlap-structural-continuous-rule-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_structural_continuous_rule.py`
  and `tests/validation/106_test_overlap_structural_continuous_rule.py`.
  The first continuous traversal-context grid evaluated `432` smooth threshold
  surfaces over depth, parent size, and barycentric balance; a finer follow-up
  evaluated `1200`. Neither produced a clean candidate. The best coarse rule
  blocked null and truth-misaligned accepts but retained only `7/8`
  truth-aligned signal case-replicates. Direct row inspection showed all null
  and truth-misaligned accepts in the weak-homogeneity region, while only
  `12/17` truth-aligned signal accepts were structurally same-subspace
  supported. The current method direction is therefore a continuous structural
  traversal rule with an explicit ambiguous multi-scale zone, not a forced
  global or additive accept cutoff.
- Added [[overlap-structural-decision-zones-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_structural_decision_zones.py`
  and `tests/validation/107_test_overlap_structural_decision_zones.py`.
  The three-way diagnostic zone run over the same three-replicate overlap rows
  writes stable, unstable weak-homogeneity, and nonaccepted rows. Accepted-row
  composition is cleanly separated: `stable_structural_accept` has `12`
  truth-aligned signal rows and zero null or truth-misaligned rows, while
  `unstable_weak_homogeneity_zone` has all `23` selected-null accepted rows,
  all `20` truth-misaligned signal accepted rows, and the remaining `5`
  truth-aligned accepted rows. This makes the next traversal method target an
  explicit unstable multi-scale zone plus selected-family null law, not alpha
  tuning.
- Added [[overlap-weak-zone-separability-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_weak_zone_separability.py` and
  `tests/validation/108_test_overlap_weak_zone_separability.py`. The panel
  scans scalar metrics inside `unstable_weak_homogeneity_zone`. Homogeneity
  gain nearly separates weak truth-aligned signal from selected null alone
  (`AUC = 0.991304`, zero-null retention `4/5`), but against selected null plus
  truth-misaligned signal the best zero-negative scalar threshold retains only
  `2/5` weak truth-aligned rows. The highest-AUC combined metric,
  context-homogeneity margin, retains `0/5` at zero leakage. This confirms
  that the weak-zone blocker is a selected-family mixture law, not a scalar
  threshold or alpha tuning issue.
- Added [[overlap-weak-family-thresholds-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_weak_family_thresholds.py` and
  `tests/validation/109_test_overlap_weak_family_thresholds.py`. The panel
  aggregates unstable weak-zone rows by `(case_id, data_role, replicate)`.
  Family-level selected p-values perfectly separate weak truth-aligned signal
  families from selected-null families in the focused run (`AUC = 1.0`,
  zero-null retention `5/5`), but fail against truth-misaligned signal
  families. Against selected-null plus truth-misaligned families, p-value
  metrics retain `0/5` positives at zero leakage, and the best combined
  family-level zero-negative rule retains only `2/5`. This confirms the open
  traversal object is a selected-family mixture law, not row or family alpha
  tightening.
- Added [[overlap-weak-truth-geometry-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_weak_truth_geometry.py` and
  `tests/validation/110_test_overlap_weak_truth_geometry.py`. The oracle-only
  panel classifies weak accepted signal rows by truth geometry. In the
  three-replicate overlap run, weak rows include `3` balanced truth recoveries,
  `2` partial truth recoveries, `8` one-sided pure fragments, `1` one-sided
  mixed remainder, `3` balanced wrong-granularity rows, and `8` diffuse truth
  mismatches. This explains why selected-family p-values are insufficient:
  pure-fragment splits can be statistically extreme while leaving a mixed
  sibling remainder, so the selected-family law needs a structural recovery
  target, not only an extremeness target.
- Added [[overlap-recovery-proxy-separability-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_recovery_proxy_separability.py`
  and `tests/validation/111_test_overlap_recovery_proxy_separability.py`.
  The panel evaluates non-oracle structural proxies against the oracle
  truth-geometry labels. Fragment-risk, size balance, edge-norm balance, and
  homogeneity symmetry distinguish truth recovery from one-sided fragment-like
  rows; `fragment_risk_proxy_score` reaches `AUC = 0.955556` and keeps `3/5`
  recovery rows at zero fragment leakage. The same proxies do not separate
  recovery from all non-recovery rows: the best full non-recovery metric keeps
  only `1/5` recovery rows at zero leakage. This supports a future fragment
  guard but not a complete selected-family law.
- Added [[overlap-fragment-risk-guard-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_fragment_risk_guard.py` and
  `tests/validation/112_test_overlap_fragment_risk_guard.py`. The guard scan
  evaluates non-oracle fragment-risk thresholds over all unstable weak-zone
  rows, including selected-null rows. The best focused candidate,
  `fragment_risk_proxy_score >= 1.252729`, retains `5/5` truth-recovery rows
  while blocking `8/9` fragment-like rows, `6/23` selected-null rows, and
  `1/11` diffuse/wrong rows. Size, barycentric, and edge-balance thresholds
  around `0.33`/`0.50` block `7/9` fragment-like rows with no recovery loss.
  This supports a diagnostic fragment-risk guard experiment, not production
  promotion.
- Updated [[overlap-structural-decision-zones-20260614]] after exposing the
  context-conditioned traversal rule as signed continuous margins in
  `benchmarks/diagnostics/calibration/overlap_structural_decision_zones.py`.
  The rule uses a smooth threshold over depth, parent size, and barycentric
  balance, then requires nonnegative homogeneity, subspace-consensus, and
  log-p margins for stable acceptance. Regenerated decision-zone outputs keep
  the same zone composition: `stable_structural_accept` has `12` truth-aligned
  signal rows and no null or truth-misaligned rows, while
  `unstable_weak_homogeneity_zone` contains all `23` selected-null accepted
  rows and all `20` truth-misaligned signal accepted rows. The median minimum
  continuous margin is `0.017777` in the stable zone and `-0.009088` in the
  weak zone.
- Added [[overlap-diagnostic-traversal-policy-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_diagnostic_traversal_policy.py`
  and `tests/validation/113_test_overlap_diagnostic_traversal_policy.py`.
  The composer applies the exact focused fragment-risk threshold
  `1.252728536810977` to the continuous decision-zone output without changing
  production behavior. In the focused overlap run it yields `12`
  `stable_region_accept` rows, `15` `weak_fragment_guard_blocked` rows, `33`
  `weak_unstable_multiscale_zone` rows, and `112` nonaccepted/leaf rows. The
  blocked weak rows include `8/9` fragment-like rows and no truth-recovery
  rows, but the remaining unstable zone still contains `17` selected-null rows,
  all `5` truth-recovery rows, `10` diffuse/wrong rows, and `11`
  truth-misaligned signal rows. This narrows the traversal fix to
  fragment-risk blocking plus an unresolved selected-family recovery law.
- Added [[overlap-residual-family-recovery-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_residual_family_recovery.py` and
  `tests/validation/114_test_overlap_residual_family_recovery.py`. The panel
  analyzes only rows left in `weak_unstable_multiscale_zone` after the
  continuous structural rule and fragment-risk guard. The residual family set
  contains `8` selected-null families, `5` truth-recovery families, and `3`
  non-recovery signal families. Residual family p-value extremeness separates
  recovery from selected null (`AUC = 1.0`, zero-null retention `5/5`) but not
  from non-recovery selected signal (`AUC = 0.666667`, zero-negative retention
  `0/5`), and still has zero-negative retention `0/5` against null plus
  non-recovery despite `AUC = 0.909091`. The best residual non-recovery
  structural metric, `residual_min_fragment_risk_proxy_score`, reaches
  `AUC = 0.866667` but retains only `3/5` recovery families at zero
  non-recovery leakage. This confirms the remaining traversal blocker is a
  selected-family structural recovery law, not p-value alpha tuning.
- Added [[overlap-residual-recovery-eligibility-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_residual_recovery_eligibility.py`
  and `tests/validation/115_test_overlap_residual_recovery_eligibility.py`.
  The eligibility layer separates residual selected-family null evidence from
  structural recovery evidence. In the focused overlap residual families,
  `residual_neg_log10_min_sibling_p_value > 8.454637` retains `5/5`
  truth-recovery families while selecting `0/8` selected-null families.
  `residual_min_fragment_risk_proxy_score > 0.747520` retains `3/5`
  truth-recovery families while selecting `0/3` non-recovery signal families,
  and the stricter all-negative threshold `> 1.076065` retains only `2/5`
  recovery families while selecting `0/11` negatives. This makes the current
  threshold hierarchy explicit: p-value evidence is a null filter; structural
  evidence is a recovery filter; neither alone is sufficient for production.
- Added [[overlap-threshold-hierarchy-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_threshold_hierarchy.py` and
  `tests/validation/116_test_overlap_threshold_hierarchy.py`. The hierarchy
  composes the focused overlap outputs into six ordered stages. Continuous
  structural stable acceptance retains `12/17` truth-aligned accepted rows and
  selects `0/43` selected-null or truth-misaligned accepted rows. The
  fragment-risk stage blocks `8/9` fragment-like weak rows and `0/5`
  truth-recovery rows. Residual selected-family null evidence retains `5/5`
  truth-recovery families with `0/8` selected-null families, while residual
  structural recovery evidence keeps only `3/5` or `2/5` truth-recovery
  families under non-recovery or all-negative control. The remaining blocker is
  the selected-family mixture where p-value evidence exists without sufficient
  structural recovery evidence.
- Added [[overlap-residual-threshold-transfer-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_residual_threshold_transfer.py`
  and `tests/validation/117_test_overlap_residual_threshold_transfer.py`. The
  transfer panel recomputes residual max-negative thresholds on leave-one-case
  and leave-one-replicate training splits. All residual threshold classes leak
  or lose recovery on held-out families. The residual p-value null-evidence
  threshold retains `5/5` held-out recovery families but selects `1/8`
  selected-null and `1/3` non-recovery families in both split kinds. The
  residual non-recovery structural threshold retains `3/5` recovery families
  but leaks `7/8` selected-null families in leave-one-case and `5/8` in
  leave-one-replicate, plus `1/3` non-recovery families. The strict
  all-negative structural threshold retains only `2/5` recovery families and
  still leaks selected-null families. This demotes focused cutpoints to
  explanatory diagnostics, not transferable calibration constants.
- Added [[overlap-threshold-stability-contract-20260614]] after implementing
  `benchmarks/diagnostics/calibration/overlap_threshold_stability_contract.py`
  and `tests/validation/118_test_overlap_threshold_stability_contract.py`.
  The contract classifies the six-stage threshold hierarchy after transfer
  testing: `1` stable-reporting stage, `1` diagnostic-only guard stage, `3`
  non-transferable focused cutpoints, and `4` stages requiring a selected-family
  law. The final status is `fail_closed_selected_family_law_required`, blocked
  by residual selected-family null evidence, residual structural evidence, the
  strict all-negative structural cutpoint, and the remaining p-value-only
  selected-family mixture. This records the current method state: report stable
  regions, keep fragment guard diagnostic, and do not promote residual
  thresholds without selected-family structural recovery theory.
- Added [[overlap-selected-family-law-requirements-20260614]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_selected_family_law_requirements.py`
  and `tests/validation/119_test_overlap_selected_family_law_requirements.py`.
  The diagnostic writes a conditioning envelope over residual family size,
  selected-family p-value evidence, homogeneity/context/subspace metrics,
  depth, parent size, barycentric balance, fragment risk, balanced recovery
  proxy, and size/edge balance. The four resulting requirements keep residual
  weak traversal fail-closed: selected-family null evidence is not
  transferable, the structural recovery target is missing, transfer validation
  is unsatisfied, and residual weak families must remain unstable multi-scale
  output until a selected-family structural recovery law is validated.
- Added [[overlap-conditional-bayesian-traversal-law-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_conditional_bayesian_traversal_law.py`
  and `tests/validation/120_test_overlap_conditional_bayesian_traversal_law.py`.
  The non-permutation diagnostic maps residual selected-family p-values to
  lower-bound Bayes-factor evidence, subtracts a selection-context penalty, and
  adds a structural neighborhood likelihood from homogeneity, continuous
  context, subspace, balance, recovery proxy, and fragment-risk terms. In the
  focused residual run, `0/8` selected-null families and `0/3` non-recovery
  families pass the strong-neighborhood coherent-candidate rule, while `1/5`
  truth-recovery families passes and `4/5` remain unstable. This confirms the
  next Bayesian law must model structural neighborhood likelihood, not only
  selected-family p-value evidence.
- Added [[overlap-bayesian-neighborhood-component-audit-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_bayesian_neighborhood_component_audit.py`
  and
  `tests/validation/121_test_overlap_bayesian_neighborhood_component_audit.py`.
  The audit shows selected-null residual families are blocked by context
  margin in `8/8` rows, non-recovery rows are blocked by context margin or
  subspace, and the blocked truth-recovery rows are limited by
  balanced-recovery, context-margin, or subspace terms. The next conditional
  Bayesian traversal law should preserve the context-margin selected-null
  blocker while replacing the coarse balanced-recovery and subspace proxies
  with an overlap-aware internal-node neighborhood likelihood.
- Added [[overlap-internal-node-bayesian-likelihood-probe-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_internal_node_bayesian_likelihood_probe.py`
  and
  `tests/validation/122_test_overlap_internal_node_bayesian_likelihood_probe.py`.
  The row-level overlap-aware likelihood probe selects `4/5` truth-recovery
  internal nodes and `0/28` selected-null, diffuse/wrong, or fragment-like
  rows in the focused residual panel. The result explains why family-level
  Bayesian aggregation was too coarse: selected families can contain one
  coherent internal node and one blocked neighboring node. The next traversal
  law should score internal nodes first and report mixed selected families as
  multi-scale structures, rather than forcing a whole-family promotion.
- Added [[overlap-internal-node-likelihood-sensitivity-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_sensitivity.py`
  and
  `tests/validation/123_test_overlap_internal_node_likelihood_sensitivity.py`.
  The `324`-point sensitivity grid reports `216` zero-negative rules and `24`
  zero-negative rules retaining at least `4/5` truth-recovery internal nodes.
  The default row-level likelihood rule is in that retaining band. Relaxing the
  context-margin floor to `-0.002` leaks negative rows for all `108/108` grid
  points, while floors `0.0` and `0.002` keep zero leakage. This records
  nonnegative context margin as the stable conditional gate; softer overlap
  tolerance should live in subspace, balance, and fragment-risk terms.
- Added [[overlap-internal-node-likelihood-transfer-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_transfer.py`
  and `tests/validation/124_test_overlap_internal_node_likelihood_transfer.py`.
  Leave-one-case transfer has `72` selected rule evaluations with zero leakage,
  while leave-one-replicate transfer has `24` leaking evaluations, all from
  `context_margin_floor = -0.002` rules. Restricting to nonnegative
  context-margin rules gives zero leakage for both split kinds, but replicate
  recovery retention remains incomplete. This keeps the row-level Bayesian
  likelihood diagnostic-only and sharpens the next target: improve
  overlap-internal subspace, balance, and fragment likelihood terms without
  relaxing nonnegative context margin.
- Added [[overlap-internal-node-transfer-gap-audit-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_internal_node_transfer_gap_audit.py`
  and
  `tests/validation/125_test_overlap_internal_node_transfer_gap_audit.py`.
  The audit shows the focused row-level panel recovers `4/5` truth-recovery
  internal nodes and selects `0/28` negative rows. The single missed truth row
  is `overlap_unbal_4c_small`, replicate `1`, node `N797`: it passes
  subspace, size-balance, edge-norm, and fragment-risk checks, but has
  negative local context margin. Since relaxed-context transfer rules produce
  `24` leakage evaluations and nonnegative-context rules produce `0`, the
  next repair must be a higher-order conditional/Bayesian law rather than a
  relaxed local context threshold.
- Added [[overlap-income-outcome-junction-law-20260615]] after implementing
  `benchmarks/diagnostics/calibration/overlap_income_outcome_junction_law.py`
  and `tests/validation/126_test_overlap_income_outcome_junction_law.py`.
  The diagnostic reframes the next law as directional incidence, not only
  node degree: incoming selected-parent context and outgoing child-sibling
  evidence are kept separate. In the focused residual panel, `32/33` rows have
  an observed incoming parent relation and `0/33` have nonnegative incoming
  context. Four truth-recovery rows are supported by outgoing local evidence,
  while the missed `overlap_unbal_4c_small` replicate `1` node `N797` is
  classified as `truth_recovery_income_outcome_context_transition_required`.
  This identifies the next mathematical object as a Bayesian transition law
  over \(I_u\), \(O_u\), and \(I_u \to O_u\) under the selected traversal
  event.
- Added [[overlap-branch-incidence-junction-panel-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_branch_incidence_junction_panel.py`
  and
  `tests/validation/127_test_overlap_branch_incidence_junction_panel.py`.
  This panel computes the actual incoming edge, incoming selected-family
  contrast, and outgoing child-sibling contrast from tree descendant means. In
  the focused overlap rerun it emits `148` branch-incidence rows and annotates
  `32` transfer-gap rows. All `5/5` annotated truth-recovery rows are
  coordinate branch-incidence mismatches, including the missed
  `overlap_unbal_4c_small` replicate `1` node `N797`; its
  incoming-family/outgoing top-k Jaccard is `0.0` and absolute cosine is
  `0.112750`. A follow-up metric-family extension added centered cosine,
  diagonal Fisher-weighted cosine, and Fisher-weighted top-coordinate overlap;
  the missed row's best metric-family score remains `0.112750`, classified as
  `metric_family_branch_mismatch`. The median truth-recovery metric-family
  score is `0.029662`, below the negative median `0.043478`. Therefore the
  next Bayesian traversal law should not enforce incoming/outgoing branch
  alignment as a hard recovery condition; it needs a separate emergent
  local-outcome mode.
- Added [[overlap-bayesian-incidence-mode-law-20260615]] after implementing
  `benchmarks/diagnostics/calibration/overlap_bayesian_incidence_mode_law.py`
  and `tests/validation/128_test_overlap_bayesian_incidence_mode_law.py`.
  The law has two diagnostic modes: continuation and emergent local outcome.
  Continuation has `0` candidates in the focused overlap panel because
  metric-family branch alignment is absent. Positive-context local outcome
  selects `4` rows, all truth-recovery with `0` negatives. Context-negative
  emergent mode is not identified: it contains `26` rows, with `1`
  truth-recovery and `25` negatives. The current admissible diagnostic
  boundary is therefore local outcome plus nonnegative context; the remaining
  task is to find an additional conditioning variable for the context-negative
  emergent row.
- Added [[overlap-context-negative-edge-conditioning-20260615]] after extending
  `benchmarks/diagnostics/calibration/overlap_branch_incidence_junction_panel.py`
  to export child-parent edge p-values and implementing
  `benchmarks/diagnostics/calibration/overlap_context_negative_edge_conditioning.py`
  plus `tests/validation/129_test_overlap_context_negative_edge_conditioning.py`.
  The focused overlap rerun shows edge tests exist but do not solve the
  context-negative emergent ambiguity: among `26` rows there is `1`
  truth-recovery row and `25` negatives, the best outgoing edge-strength metric
  has AUC `0.916667` but zero-negative retention `0`, and edge rejection flags
  are saturated for truth and negatives. The next conditioning variable must
  encode selected neighborhood/topology or a higher-order ancestor/incoming/
  outgoing/family relation, not raw edge significance alone.
- Added [[overlap-context-negative-topology-conditioning-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_context_negative_topology_conditioning.py`
  and
  `tests/validation/130_test_overlap_context_negative_topology_conditioning.py`.
  The focused scan joins incidence-mode rows, true branch-incidence topology,
  internal-node transfer-gap structural evidence, and income/outcome context
  inside the same `26` context-negative emergent rows. It finds `7` numeric
  zero-negative separators and no categorical separator. The strongest
  interpretable candidate is the higher-order topology relation
  `balance_product = incoming_branch_balance * outgoing_balance`: the single
  truth row has value `0.232891`, the largest negative is `0.222500`, and the
  max-negative margin is `0.010391`. Pure outgoing balance also separates but
  narrowly (`0.492891` truth versus `0.491304` max negative). The result moves
  the method target toward a smooth incoming/outgoing selected-neighborhood
  topology likelihood, but remains diagnostic-only because it is a
  single-positive focused cutpoint requiring transfer validation.
- Added [[overlap-context-negative-topology-transfer-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_context_negative_topology_transfer.py`
  and
  `tests/validation/131_test_overlap_context_negative_topology_transfer.py`.
  The transfer audit learns zero-negative rules on held-out case and replicate
  folds. `balance_product` keeps zero held-out negative leakage in folds with a
  training rule, but no fold simultaneously has a training separator and
  held-out truth because the focused context-negative emergent slice has only
  one truth-recovery row. Its status is
  `transfer_unvalidated_no_truth_holdout_support`; nearby topology metrics
  leak one held-out negative under replicate splits. The candidate therefore
  remains a Bayesian/conditional likelihood direction, not a promotable
  threshold.
- Added [[overlap-context-negative-bayesian-topology-law-20260615]] after
  implementing
  `benchmarks/diagnostics/calibration/overlap_context_negative_bayesian_topology_law.py`
  and
  `tests/validation/132_test_overlap_context_negative_bayesian_topology_law.py`.
  The diagnostic replaces the failed hard-threshold transfer direction with a
  continuous selected-neighborhood component law over incoming balance,
  outgoing balance, outgoing edge-norm balance, anti-fragment evidence,
  selected-family evidence, and soft context penalty. In the focused
  `context_negative_emergent` slice, the single truth row is top-ranked with
  posterior-style log odds `34.101999`; the strongest negative is `29.147130`,
  giving margin `4.954869`. Component summaries show outgoing balance and
  outgoing edge-norm balance carry the top rank, while raw edge significance is
  not used. The result remains diagnostic-only pending pooled support and
  weight uncertainty.
- Added [[overlap-context-negative-bayesian-topology-sensitivity-20260615]]
  after implementing
  `benchmarks/diagnostics/calibration/overlap_context_negative_bayesian_topology_sensitivity.py`
  and
  `tests/validation/133_test_overlap_context_negative_bayesian_topology_sensitivity.py`.
  The ablation audit evaluates `13` component profiles across context penalty
  weights `0`, `25`, `50`, `75`, and `100`. It finds `60/65` separating
  profile-weight combinations. Topology-only and outgoing-topology-only
  separate at all tested weights, while selected-family plus context separates
  in `0/5`. This supports the interpretation that the missing conditioning
  variable is structural selected-neighborhood topology, not selected-family
  evidence or context alone.
- Added [[topology-vector-benchmark-20260615]] after forwarding KL
  gate-profile parameters through `benchmarks/shared/runners/dispatch.py`,
  adding dispatch coverage in `tests/pipeline/51_test_dispatch_contract.py`,
  and running the regression-gate KL benchmark with default projected-Wald and
  `fixed_coordinate_global_passthrough_refined_v1` variants. The refined
  profile eliminates default KL skips (`17/17` ok versus `11/17`), improves
  skip-as-zero mean ARI from `0.388484` to `0.445742`, and improves exact-K
  count from `3/17` to `5/17`, but ok-only mean ARI is lower
  (`0.445742` versus `0.600384`) and several easy non-overlap rows under-split.
  The benchmark supports the topology vector as a diagnostic conditioning
  object, not as a broad production traversal default.
- Added [[overlap-conditional-topology-law-panel-20260615]] after implementing
  `benchmarks/diagnostics/calibration/overlap_conditional_topology_law_panel.py`,
  `tests/validation/134_test_overlap_conditional_topology_law_panel.py`, the
  diagnostic profile `fixed_coordinate_conditional_topology_diagnostic_v1`,
  and directed-incidence fields in selected-family multi-scale node decisions.
  The panel is non-cross-fit and non-permutation. On the focused `26`-row
  context-negative overlap slice, the single truth row remains rank `1`, with
  conditional log-odds `31.869341` versus strongest negative `27.174856`, but
  the status remains `support_insufficient_fail_closed` because the internal
  incidence support stratum has only one truth-recovery row. The result
  clarifies that the next mathematical object is a support-aware selected
  neighborhood conditional law, not raw edge strength, global homogeneity, or
  a tuned topology threshold. The benchmark-facing
  `kl_conditional_topology_diagnostic` method id also runs the 17-case
  regression gate with `17/17` ok rows, mean ARI `0.460293`, median ARI
  `0.480000`, and exact-K count `4/17`. A Julia binary matrix run writes a
  UMAP overlay and returns `410` final clusters from `703` samples, confirming
  that the profile is runnable but still fragmentation-heavy.
- Added [[cosine-band-coherence-comparator-20260615]] after porting the useful
  c2ef fixed cosine-band and coherence checks into
  `benchmarks/diagnostics/spectral/cosine_band_coherence_comparator.py` and
  `tests/validation/135_test_cosine_band_coherence_comparator.py`. The
  comparator keeps the historical fixed band labels, runs current
  gate/decomposition code on each band tree, reports feature-enrichment and
  within-cluster TF-IDF cosine coherence, and marks rows
  `diagnostic_only_not_production_calibration`.
- Updated [[overlap-conditional-topology-law-panel-20260615]] so the old
  neighborhood/log-scale idea is represented as the explicit
  `neighborhood_scale_log_component` with `neighborhood_scale_support_*`
  counts. Missing scale is neutral when absent from the input table, but sparse
  or row-missing scale evidence is fail-closed and reflected in the production
  summary.

## Evidence

- `raw/inbox/wiki-construction-brief.md` records the requested scaffold.
- `AGENTS.md` records the operating guide.

## Links

- [[project-overview]]
- [[wiki-construction]]
- [[schema]]
- [[maintenance]]
- [[wiki-search]]
