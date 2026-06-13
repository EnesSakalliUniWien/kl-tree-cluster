---
title: Wiki Log
type: control
status: reviewed
updated: 2026-06-12
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

## Evidence

- `raw/inbox/wiki-construction-brief.md` records the requested scaffold.
- `AGENTS.md` records the operating guide.

## Links

- [[project-overview]]
- [[wiki-construction]]
- [[schema]]
- [[maintenance]]
- [[wiki-search]]
