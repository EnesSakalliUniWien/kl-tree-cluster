---
title: Wiki Index
type: control
status: reviewed
updated: 2026-06-15
sources:
  - AGENTS.md
  - wiki/schema.md
  - wiki/maintenance.md
tags:
  - wiki
  - index
---

# Wiki Index

## Summary

This is the first read target for project questions. It maps the current wiki
pages to their main use and should be updated whenever a durable page is added,
renamed, or promoted.

## Details

### Control

- [[schema]] - frontmatter, body-section, naming, source, and wikilink contract.
- [[maintenance]] - routine checks, lint workflow, cleanup cadence, and `qmd`
  threshold.
- [[log]] - append-only chronology of scaffold, ingest, analysis,
  implementation, review, verification, and maintenance events.
- [[wiki-search]] - local search workflow using `rg`, `sed`, `nl`, `find`, and
  `make wiki-lint`.

### Project

- [[project-overview]] - concise map of the KL-TE repository, source surfaces,
  method purpose, and implementation entry points.

### Tools

- [[aws-selected-tail-equation-study]] - AWS Batch sharding and merge workflow
  for large selected-tail equation diagnostics, with row-level records and
  namespaced independent simulation ids.

### Source Summaries

- [[wiki-construction-brief]] - summary of the captured wiki construction
  request and its required layers, tooling, workflows, and open questions.
- [[github-wiki-structure-research]] - design-reference notes for Foam-style
  links, GitHub Docs frontmatter, markdownlint, and remark link validation.
- [[edge-selection-null-audit-20260601]] - edge-gate null audit showing that
  data-selected hierarchies can make nearly all child-parent edges significant
  under a global null, while fixed-tree permutations do not.
- [[selected-edge-type1-geometry-pilot-20260604]] - AWS pilot comparing
  fixed-tree and same-data selected-tree binary global-null edge behavior,
  with geometry covariates and strict sibling-calibration failure reporting.
- [[selected-edge-binary-categorical-pilot-20260604]] - AWS pilot extending
  selected-edge diagnostics to direct categorical multinomial nulls and
  exposing high-cardinality categorical fixed-tree frontier inflation.
- [[feature-split-selection-audit-20260601]] - cross-fit audit showing that
  feature-split hierarchy selection/testing restores null support in the null
  case and keeps signal examples interpretable.
- [[selected-hierarchy-null-audit-20260601]] - same-data selected-hierarchy
  null audit showing large selected-hierarchy correction factors without using
  cross-fitting as the method.
- [[selected-hierarchy-stratification-diagnostic-20260602]] - selected-null
  scale stratification by parent depth and parent-size bins, recorded as
  descriptive heterogeneity rather than calibration borrowing.
- [[selected-hierarchy-external-calibration-contract-20260602]] - production
  admissibility and scalar-vs-tail diagnostic for external selected-hierarchy
  calibration; the 500-replicate contract run remains production-undefined.
- [[selected-hierarchy-geometry-covariates-20260602]] - row-level tree,
  edge-selection, eigenvalue, and angular geometry covariates plus held-out
  candidate-equation transfer checks for selected sibling records under
  regenerated same-data null hierarchies.
- [[selected-ratio-tail-law-diagnostic-20260602]] - within-context
  selected-ratio tail-law diagnostic showing no production-admissible context
  in the broad 200-replicate run, followed by narrow admissible Gaussian,
  categorical, and binary projection-1 small-parent, high-edge-action contexts
  in focused runs.
- [[selected-tail-admissibility-domain-20260603]] - combined admissibility
  domain table showing which selected-tail contexts are production-admissible
  and which remain undefined under the external calibration support contract.
- [[phylogenetic-ml-topological-selected-tail-literature-20260603]] -
  literature scan connecting undefined selected-tail contexts to selective
  inference, phylogenetic covariance, distance-based inference, and
  topological graph variables.
- [[selected-tail-topology-refinement-20260603]] - row-level diagnostic showing
  that data-adaptive topology, merge-persistence, edge-path, and spectral
  refinements diagnose heterogeneity but do not define production-admissible
  selected-tail calibration contexts in the current 300-replicate
  Gaussian/categorical panel.
- [[selected-tail-equation-cloud-run-20260603]] - AWS Batch 1000-replicate
  per-case selected-tail equation run showing seven strict admissible base
  contexts while root/large high-edge contexts still fail tail precision.
- [[selected-tail-equation-cloud-run-20260604]] - rebuilt-image replication of
  the AWS selected-tail equation run, reproducing the same seven admissible
  contexts and no admissibility-status changes versus 2026-06-03.
- [[selected-geometry-mp-integral-literature-20260602]] - literature capture
  for the no-bootstrap selected-region geometry direction and the
  Stieltjes-transform integral route for deformed Marchenko--Pastur spectra.
- [[local-mp-identity-law-diagnostic-20260602]] - production-spectrum screen
  showing that the identity MP law is not a uniform description across
  Bernoulli, categorical, discretized Gaussian, and continuous contexts.
- [[hierarchy-gate-separation-20260603]] - latest strict full-suite separation
  of solved rows, hierarchy/metric failures, gate failures, calibration-support
  skips, covariance boundary, and oracle-matched-below-solved rows.
- [[root-selected-region-margins-20260603]] - average-linkage replay of
  observed root merge-selection inequalities, separating tie-heavy discrete
  hierarchy cells from smooth Euclidean first-order signed-distance cells, and
  showing edge-opening boundary/action, the edge/sibling barycentric
  relationship, and the fixed-projection edge-conditioned sibling tail.
- [[internal-vs-selected-hierarchy-inflation-20260603]] - diagnostic-only
  comparison showing that diffuse Gaussian root internal inflation agrees with
  selected-hierarchy scale, while null and high-cardinality categorical root
  contexts still lack internal production support.
- [[sibling-null-prior-interpolation-audit-20260604]] - diagnostic-only
  reconstruction of the old tree-neighborhood null-prior score, showing that
  it assigns positive weights to selected non-null records and therefore
  describes but does not solve strict empirical-null support failure.
- [[alpha-grid-full-20260604]] - AWS full-suite alpha grid showing that the
  current edge alpha `0.001` and sibling alpha `0.01` pair has the best mean
  ARI in the tested grid, while lower edge alpha gives more exact
  cluster-count hits.
- [[traversal-sibling-fdr-smoke-20260604]] - layered smoke diagnostic showing
  that traversal-aligned sibling BH is not automatically a global FDR guarantee
  across depths, and separating fixed-tree Wald, selected-tree Wald, and
  empirical-inflation support failures.
- [[calibration-contract-enhancement-request-20260604]] - review request
  arguing for explicit calibration decision statuses, fail-closed internal
  support, predeclared external selected-tail admissibility, and a manuscript
  correction to the production leaf-only spectral contract.
- [[selected-tail-law-q5-validation-20260604]] - diagnostic fit/validation of
  predeclared selected-tail law candidates using edge severity, parent size,
  projection dimension, feature family, and spectral geometry; edge-plus-
  spectral coordinates transfer best, while the full Q5 law is not production
  calibrated.
- [[recursive-method-followups-20260604]] - recursive follow-up implementing
  Q9/Q11 support-threshold reporting, Q10 weight-rule diagnostics, Q17
  projection-dimension rule grids, and Q14/Q15 targeted MP/k-min smoke outputs,
  while recording Q19--Q44 as larger validation/modeling tracks.
- [[barycentric-method-literature-request-20260604]] - barycentric method and
  literature follow-up adding explicit edge/sibling barycentric algebra,
  leverage notation, bibliography entries, and an expanded Q5 diagnostic where
  barycentric edge-plus-spectral predictors reduce residual-tail error.
- [[open-question-diagnostic-audit-20260604]] - all-open-question diagnostic
  audit assigning each of the 44 mathematical questions a current evidence
  status and next required diagnostic.
- [[open-question-full-diagnostic-contract-20260604]] - full diagnostic
  contract requiring every open mathematical question to have a concrete
  diagnostic family, scale, inputs, outputs, acceptance criteria, and method-
  claim blocker.
- [[selected-tail-promotion-gate-20260604]] - strict Q1/Q5/Q7/Q8 promotion
  gate over the 1000-replicate selected-tail run; no context is externally
  admissible because Q5 parent-size transfer fails and strict c-hat precision
  metadata is unavailable.
- [[selected-tail-parent-size-balance-stability-20260604]] - balance-aware
  parent-size stability diagnostic with c-hat precision metadata, finding
  three Gaussian Bernoulli projection-2 high-edge-action contexts that pass the
  row-level diagnostic gates.
- [[selected-tail-promotion-gate-debug-20260605]] - predicate-level debug of
  the zero-admissible selected-tail promotion gate result, showing six
  context-law candidates blocked by global Q5 and c-hat metadata, plus three
  narrower balance-conditioned diagnostic candidates.
- [[internal-calibration-q9-q10-q11-debug-20260605]] - root-cause debug of
  internal support thresholds and sibling null-weight validation: Q9/Q11 are
  implemented but opt-in, a threshold-panel entrypoint exists, and Q10 can now
  report labeled leakage when support labels are supplied; production-scale
  mixed null/signal validation is still missing.
- [[mixed-internal-calibration-sweeps-20260605]] - labeled method-proof,
  binary, and categorical Q9/Q10/Q11 sweeps showing near-zero Q10 leakage for
  the current product-BH rule but no global validation of internal support
  thresholds.
- [[mp-projection-dimension-behavior-sweeps-20260605]] - method-proof,
  binary, and categorical Q14/Q15/Q17 sweeps showing that raw MP count often
  chooses `k=0` and reduces false splits, while nonzero floors recover signal
  but keep selected-tree raw sibling p-values anti-conservative.
- [[traceable-benchmark-suite-request-20260604]] - benchmark-suite design
  request reframing new KL-TE cases as traceable mathematical probes with
  manifests, node traces, failure attribution, and metamorphic checks.
- [[full-benchmark-run-20260605]] - completed 120-case full benchmark including
  method-proof cases; k-means leads mean ARI over ok rows, KL remains strong on
  binary/overlapping ok rows, and method-proof cases expose calibration,
  covariance, and under-split stress modes.
- [[mnist-benchmark-run-20260605]] - canonical MNIST example over 20
  distance/linkage configurations; single linkage over-splits while the other
  configurations fail closed under strict sibling calibration support.
- [[mnist-continuous-pca50-run-20260605]] - continuous MNIST probe using PCA50
  and Euclidean linkage; complete/weighted linkage improve over binary
  single-linkage but remain weak digit clustering results.
- [[mnist-continuous-alpha-sweep-20260605]] - focused continuous MNIST PCA50
  alpha sweep showing strict Ward improves to ARI `0.514854`, while higher
  alphas over-split the Ward tree and complete/weighted remain weak.
- [[historical-kak-spectral-pipelines-20260605]] - git-history recovery of
  deleted root-level KAK/cosine spectral scripts, showing that they selected
  tree topology in spectral subspaces and then ran `TreeDecomposition` gates on
  the original feature matrix.
- [[adaptive-cosine-kak-benchmark-probe-20260605]] - current-compatible
  method-proof and full-suite adaptive cosine/KAK probe showing that early
  non-common spectral blocks usually carry recoverable tree signal, but support
  enforcement removes many ok rows and the full GO matrix has no admissible
  support-enforced KAK block; restored diagnostic pages expose radius, angle,
  invariant-axis, internal-tree merge geometry, and NMI-ordered Julia reference
  inspection for the full matrix; KAK/cosine lens alpha sweeps now compare
  average, complete, and Ward-Euclidean linkage, including the completed full
  AWS `450`-row lens/linkage/alpha sweep and regenerated semantic panel;
  covariance-axis sign/Procrustes stability now supports a stable
  covariance-axis diagnostic; raw KAK lens axes now map back to GO feature
  loadings before clustering; a binary feature-side `Q_B sqrt(Lambda_B)`
  subspace diagnostic now exposes that feature leaves are not yet supported by
  the current gate equations.
- [[barycentric-action-equation-diagnostic-20260606]] - diagnostic test of the
  candidate path-conditioned barycentric action equation, finding that compact
  barycentric edge/spectral variables remain best for selected-tail
  calibration while radius/angle variables explain KAK traversal fragmentation.
- [[full-benchmark-run-20260606]] - rerun of the canonical 120-case full
  benchmark with default methods, plots, relationship analysis, and failure
  diagnosis; results reproduce the prior leaderboard and strict KL
  calibration-support skip profile.
- [[clustering-root-audit-debug-20260606]] - corrected root-row interpretation
  for KL failure diagnosis and relationship audit factors; separates true
  root rejection from accepted-root post-root sibling traversal stalls.
- [[path-conditioned-barycentric-action-diagnostics-20260606]] - cached-output
  diagnostic and fresh full-benchmark join tracing the exact barycentric
  edge/sibling identity, KAK radius/angle traversal signal, action-budget
  candidate, high-action angular-shell guard and utility panels,
  calibration-support gap, and traversal-survival path.
- [[phase1-path-b-foundation-20260606]] - Phase 1 Path B full KL-only sweep
  over `k_min in {0,1,2,3}` and pass-through on/off, plus Q5 selected-tail
  geometry covariate gain; selects `k_min=1` with pass-through as the current
  penalized full-suite optimum.
- [[recursive-pvalue-geometry-20260606]] - full-suite recursive p-value and
  selected-subspace geometry diagnostic showing connected but asymmetric edge
  and sibling p-value fields, moderate subspace-rotation coupling, and
  mechanical chi-square consistency without selected-tree calibration.
- [[mixed-null-signal-geometry-validation-20260606]] - labeled full-suite
  null/signal sibling-context validation showing scikit-learn continuous
  edge/sibling/KAK surfaces give the strongest diagnostic held-out AUC, while
  remaining outside production selected-tail calibration.
- [[selected-sibling-lrt-diagnostic-20260612]] - whole-space fixed-diffusion
  Julia diagnostic adding a Bernoulli sibling deviance/LRT-like statistic beside
  projected-Wald evidence; the fragmented `k=30,t=3` regime amplifies
  projected-Wald selected ratios without increasing median deviance per changed
  feature.
- [[diagnostic-framework-github-scan-20260606]] - GitHub scan of analytical and
  machine learning frameworks for KL-TE diagnostics, selecting selective
  inference, simulation-based inference, conformal risk control, and
  calibration/observability tooling as the most relevant framework stack.
- [[data-independent-sibling-gate-panel-20260613]] - diagnostic same-data,
  non-cross-fit sibling-gate candidate using fixed coordinate-wise Wald
  p-values plus a selected-topology penalty; the `binary_2clusters` smoke
  controls the selected null near `0.01` while remaining diagnostic-only.
- [[data-independent-sibling-gate-traversal-panel-20260613]] - traversal-level
  diagnostic showing that fixed data-independent sibling gates recover strong
  binary signal and useful categorical signal; the selected-root permutation
  layer now has both traversal diagnostics and a default-off runtime guard, but
  production remains fail-closed pending broad confidence evidence.
- [[fixed-sibling-gate-profile-validation-20260613]] - shared-runner
  validation artifact proving that named fixed sibling-gate profiles avoid
  adaptive projected-Wald sibling rows, exposing the selected-root permutation
  guard in the runtime profile path, packaging
  `fixed_coordinate_selective_root_v1` and the narrower
  `fixed_coordinate_selective_passthrough_v1`, and recording the ten-replicate
  recheck that rejects a hard closed-root barrier, the support run showing
  `fixed_coordinate_global_passthrough_v1` still has boundary false splits,
  and `fixed_coordinate_global_passthrough_refined_v1` as the current
  selected-family pass-through diagnostic candidate after binary support
  validation moved its production summary from fail-closed to diagnostic-only.
- [[selected-family-traversal-panel-20260614]] - diagnostic runner comparing
  baseline traversal with fixed-coordinate selected-root and selected-family
  profiles, surfacing selected-family guard rows and multi-scale node, region,
  and sample outputs instead of only a flat clustering.
- [[refined-profile-all-benchmark-tests-20260614]] - all-supported
  selected-null smoke and full 120-case performance pass for
  `fixed_coordinate_global_passthrough_refined_v1`, showing overlap-template
  null inflation, several weak signal families, continuous covariance-contract
  errors, and strong but expensive high-dimensional categorical performance.

### Concepts

- [[llm-wiki-pattern]] - the docs-as-code memory pattern used by this project.
- [[kl-te-method]] - the main inferential pipeline: candidate hierarchy,
  subtree distributions, edge and sibling tests, and final traversal.
- [[projected-wald-statistic]] - projected quadratic statistic used in edge and
  sibling tests.
- [[top-down-traversal]] - decision extraction from precomputed edge and
  sibling annotations.

### Entities

- [[poset-tree]] - central directed tree structure used by the pipeline.
- [[tree-decomposition]] - decomposition class that turns test annotations into
  cluster assignments.

### Analyses

- [[null-edge-sibling-calibration-enhancement-plan]] - calibration roadmap
  separating selected edge-null law, sibling null support/external selected-tail
  law, and KAK/action traversal geometry before any production rule promotion.
- [[edge-null-calibration-panel-20260613]] - executable first-phase edge-null
  diagnostic contract separating fixed-tree null, selected-tree null, and
  selected-tree signal rows without changing production calibration.
- [[sibling-null-calibration-panel-20260613]] - sibling-null diagnostic
  contract separating strict-null, stopped-edge null, selected-nonnull, and
  external selected-tail contexts.
- [[traversal-guard-validation-panel-20260613]] - diagnostic guard-validation
  panel for action/angle traversal geometry, pure-fragment precision, and
  signal-context blocking.
- [[production-admissibility-contract-20260613]] - conservative component
  contract that turns separated diagnostic statuses into explicit
  production-admissible, diagnostic-only, or fail-closed decisions.
- [[selected-edge-sibling-null-equation-20260613]] - executable conditional
  empirical-null equation using barycentric balance, edge action, projection
  dimension, feature family, and edge-path-open status to estimate matched
  sibling tail probabilities only when support exists.
- [[statistic-distribution-shape-panel-20260613]] - diagnostic distribution
  panel comparing empirical statistic skew/tails against current chi-square df
  and covariance-inferred Satterthwaite df/scale references.
- [[covariance-laplacian-panel-20260613]] - graph-Laplacian covariance panel
  separating diagonal sibling contrast covariance from dense parent spectral
  covariance connectivity.
- [[selected-edge-sibling-postrun-analysis-20260613]] - post-run analyzer for
  enriched selected-edge sibling artifacts, producing distribution-shape and
  selected edge+sibling equation outputs without promoting a production rule.
- [[differential-statistic-validity-panel-20260613]] - diagnostic validity
  layer for projected-Wald sibling statistics, separating Fisher boundary,
  whitening, projection, and nonsmooth selection geometry before production
  promotion.
- [[regularized-wald-statistic-panel-20260613]] - diagnostic comparison of
  plug-in, Jeffreys-smoothed, Dirichlet-smoothed, and root-shrunk sibling
  projected-Wald statistics under fixed-tree-first null validation.
- [[null-law-decomposition-panel-20260613]] - fixed-topology null-law
  decomposition showing that same-sample adaptive projection breaks the
  sibling projected-Wald chi-square reference while independent fixed
  projections are near nominal.
- [[toomanycells-method-20260613]] - relational method note positioning
  TooManyCells as a tree-first divisive hierarchical spectral clustering
  approach with Newman-Girvan modularity stopping, not as a direct KL-TE
  comparator.
- [[overlap-structural-sibling-panel-20260614]] - overlap-focused structural
  sibling diagnostic showing that refined-profile binary overlap null failures
  have strong selected sibling contrasts but near-zero within-child homogeneity
  gain, consistent with selected barycentric contrast rather than structural
  cluster support.
- [[overlap-structural-threshold-sensitivity-20260614]] - post-run threshold
  sweeps over overlap structural rows showing that homogeneity gain, not
  sibling p-value tightening, is the main separator for focused binary overlap
  traversal failures, but the first three-replicate run finds no fully stable
  threshold in the tested grid.
- [[overlap-structural-context-thresholds-20260614]] - context-binned threshold
  diagnostic showing that deep/internal overlap traversal contexts can retain
  truth-aligned signal with permissive structural thresholds, while shallow,
  root, and large-parent contexts need stricter guards or multi-scale warnings.
- [[overlap-structural-continuous-rule-20260614]] - smooth traversal-context
  threshold diagnostic showing that depth, parent size, and barycentric balance
  can formalize the direction, but no tested continuous accept surface controls
  null and truth-misaligned rows while retaining all weak truth-aligned signal.
- [[overlap-structural-decision-zones-20260614]] - three-way structural
  traversal zone diagnostic exposing the context-conditioned rule as signed
  continuous margins; stable same-subspace accepts are clean truth-aligned
  evidence in the focused overlap run, while every null and truth-misaligned
  accepted split falls into the unstable weak-homogeneity zone.
- [[overlap-weak-zone-separability-20260614]] - unstable-zone separability
  diagnostic showing that scalar structural thresholds can rank weak signal but
  cannot recover all weak truth-aligned rows without selected-null or
  truth-misaligned leakage.
- [[overlap-weak-family-thresholds-20260614]] - family-wise weak-zone
  diagnostic showing that selected-family p-values separate weak signal from
  selected null, but fail against truth-misaligned signal families; family
  aggregation does not remove the selected-family mixture blocker.
- [[overlap-weak-truth-geometry-20260614]] - oracle truth-geometry diagnostic
  showing that many statistically extreme weak overlap families are one-sided
  pure fragments, diffuse mismatches, or wrong-granularity splits rather than
  balanced structural recovery.
- [[overlap-recovery-proxy-separability-20260614]] - non-oracle structural
  proxy diagnostic showing that size/edge/homogeneity symmetry can detect
  one-sided fragment risk, but does not separate truth recovery from all weak
  non-recovery modes.
- [[overlap-fragment-risk-guard-20260614]] - diagnostic guard-threshold scan
  showing that fragment-risk and symmetry thresholds can block most one-sided
  weak fragment rows while retaining truth-recovery rows in the focused overlap
  panel.
- [[overlap-diagnostic-traversal-policy-20260614]] - diagnostic composition of
  continuous structural zones and the fragment-risk guard, yielding stable
  region accepts, fragment-guarded weak rows, and an unresolved weak
  multi-scale zone.
- [[overlap-residual-family-recovery-20260614]] - post-fragment residual
  selected-family diagnostic showing that p-value extremeness separates
  recovery from selected null but not from non-recovery selected signal.
- [[overlap-residual-recovery-eligibility-20260614]] - residual eligibility
  diagnostic splitting selected-family null evidence from structural recovery
  evidence and showing why p-value-only promotion remains unsafe.
- [[overlap-threshold-hierarchy-20260614]] - ordered threshold synthesis
  showing which traversal thresholds act at stable-row, fragment-guard,
  residual-null, and residual-structural stages.
- [[overlap-residual-threshold-transfer-20260614]] - leave-one-case and
  leave-one-replicate transfer diagnostic showing that focused residual
  max-negative thresholds leak or lose recovery when learned off-split.
- [[overlap-threshold-stability-contract-20260614]] - diagnostic contract
  classifying the overlap thresholds as stable reporting, diagnostic-only,
  non-transferable, or selected-family-law required.
- [[overlap-selected-family-law-requirements-20260614]] - diagnostic envelope
  converting residual selected-family failures into explicit conditioning
  variables and validation obligations for the missing structural recovery law.
- [[overlap-conditional-bayesian-traversal-law-20260615]] - non-permutation
  posterior-style diagnostic showing that selected-family p-value Bayes
  evidence must be gated by strong structural neighborhood likelihood.
- [[overlap-bayesian-neighborhood-component-audit-20260615]] - component audit
  showing that selected-null families are blocked mainly by context margin,
  while blocked truth-recovery families expose balanced-recovery, context, and
  subspace likelihood gaps.
- [[overlap-internal-node-bayesian-likelihood-probe-20260615]] - row-level
  overlap-aware likelihood probe showing that internal-node candidates recover
  partial overlap signal inside otherwise mixed selected families.
- [[overlap-internal-node-likelihood-sensitivity-20260615]] - threshold
  sensitivity scan showing context margin must stay nonnegative, while
  subspace, balance, and fragment-risk tolerances form the softer overlap
  likelihood degrees of freedom.
- [[overlap-internal-node-likelihood-transfer-20260615]] - leave-one-case and
  leave-one-replicate transfer scan showing nonnegative context-margin rules
  keep zero leakage, while recovery retention remains incomplete.
- [[overlap-internal-node-transfer-gap-audit-20260615]] - conditional/Bayesian
  gap audit showing the single remaining row-level truth miss is
  context-negative but soft-structure supported, while relaxed context leaks.
- [[overlap-income-outcome-junction-law-20260615]] - income/outcome-aware
  junction diagnostic showing the next traversal law must model the transition
  from incoming selected-parent context to outgoing child-sibling evidence.
- [[overlap-branch-incidence-junction-panel-20260615]] - true branch-vector
  diagnostic showing overlap truth recovery is not explained by coordinate or
  metric-family incoming/outgoing branch alignment and needs an emergent
  local-outcome mode.
- [[overlap-bayesian-incidence-mode-law-20260615]] - two-mode diagnostic
  Bayesian law showing positive-context local outcome recovers `4/5` truth
  rows with zero negatives, while context-negative emergent rows remain
  unidentified and fail-closed.
- [[overlap-context-negative-edge-conditioning-20260615]] - edge-test
  conditioning scan showing child-parent edge evidence exists but does not
  separate the context-negative emergent truth row from negatives because edge
  strength overlaps and rejection flags saturate.
- [[overlap-context-negative-topology-conditioning-20260615]] - selected
  neighborhood/topology scan showing the higher-order
  incoming-balance/outgoing-balance product separates the single
  context-negative emergent truth row in the focused overlap slice, but only as
  a single-positive diagnostic candidate needing transfer validation.
- [[overlap-context-negative-topology-transfer-20260615]] - held-out
  case/replicate transfer audit showing the balance-product candidate is not
  yet validated because the focused slice has only one positive
  context-negative emergent row; neighboring topology metrics can leak held-out
  negatives under replicate splits.
- [[overlap-context-negative-bayesian-topology-law-20260615]] - continuous
  selected-neighborhood topology diagnostic showing the focused truth row is
  top-ranked by fixed Bayesian component scores over incoming/outgoing balance,
  outgoing edge-norm balance, anti-fragment evidence, selected-family evidence,
  and soft context penalty.
- [[overlap-context-negative-bayesian-topology-sensitivity-20260615]] -
  component-ablation audit showing topology-only and outgoing-topology-only
  profiles robustly separate the focused context-negative truth row, while
  selected-family plus context never separates.
- [[overlap-conditional-topology-law-panel-20260615]] - non-permutation,
  directed-incidence topology-law panel showing the focused truth row remains
  top-ranked, but the internal support stratum is still too thin for
  production promotion.
- [[topology-vector-benchmark-20260615]] - regression-gate KL benchmark and
  topology-vector summary showing the refined pass-through profile reduces
  skips and improves skip-as-zero ARI, but is not a broad production clustering
  default; the topology vector remains a diagnostic conditioning object.
- [[root-selection-literature-20260614]] - literature synthesis separating
  clustering root split inference from phylogenetic or trajectory root
  orientation, and arguing that KL-TE should validate selected-root stopping
  rather than optimize a free root parameter.
- [[selected-root-selected-family-traversal-literature-20260614]] - literature
  synthesis mapping KL-TE's selected-root and pass-through selected-family null
  problem to selective clustering inference, selective families, hierarchical
  FDR/FWER, TreeScan/scan statistics, and cluster-permutation max tests.
- [[selected-root-pass-through-null-fixture-20260614]] - compact iid Bernoulli
  null fixture showing a closed selected root with one pass-through descendant
  split under `fixed_coordinate_global_passthrough_refined_v1`.
- [[phylogenetic-tree-builders-20260614]] - implementation note for opt-in
  neighbor-joining and IQ-TREE 3 KL tree builders, both rooted by minimum
  ancestor deviation before conversion to `PosetTree`.
- [[julia-tree-estimator-run-20260614]] - full Julia GO matrix tree-estimator
  comparison showing baseline KL, MAD-rooted neighbor joining, and IQ-TREE 3
  fast/MAD remain highly fragmented and do not solve traversal calibration.
- [[julia-selected-family-run-20260614]] - full Julia GO matrix run of
  `fixed_coordinate_global_passthrough_refined_v1`, with multi-scale traversal
  outputs and a stable-region-first UMAP overlay; fragmentation drops to `401`
  clusters but a dominant selected-root guard zone covers most genes.

- [[wiki-construction]] - reusable argument for the three-layer wiki scaffold,
  local lint, and maintenance workflow.
- [[oracle-gate-path-diagnostic]] - mathematical recoverability and gate-path
  analysis separating tree failures, sibling-calibration under-splits, direct
  sibling false splits, and pass-through fragmentation.
- [[selected-hierarchy-selection-geometry]] - geometric explanation of why
  same-data hierarchy selection turns subtree barycenters, edge openings, and
  focal sibling contexts into selected high-contrast objects rather than
  ordinary fixed-tree null contrasts.
- [[selected-hierarchy-null-support-contract]] - diagnostic support contract
  for selected-hierarchy null studies, including no-fallback unsupported states,
  context matching, and Monte Carlo precision requirements.
- [[selected-hierarchy-geometric-law-map]] - map from selected-hierarchy
  geometry variables to their actual statistical laws and physical analogies.
- [[method-proof-web]] - proof-level map connecting fixed feature-space
  covariance, projected-Wald, MP, selected hierarchy, empirical-null support,
  and oracle recoverability while marking selected-inference proof gaps.
- [[root-selected-region-model]] - first simplified selected-region model for
  a root sibling context, written as hierarchy merge inequalities plus
  edge-opening inequalities in the null-whitened tangent geometry.
- [[local-marchenko-pastur-rule]] - audit of the local MP dimension rule,
  eigenvalue scale, finite-sample null behavior, leaf-only spectral rows, and
  calibration-support fallout.
- [[selected-pca-projected-wald-validation]] - locked local Gaussian validation
  showing that leaf-only selected PCA is calibrated in the tested settings while
  child-mean internal spectral rows break the fixed-subspace reference.
- [[dimensional-gaussian-representation-diagnostic]] - why selected continuous
  dimensional Gaussian cases improve consolidated signals but not diffuse
  high-noise signals.
- [[benchmark-pipeline-contract]] - active benchmark execution order and the
  strict case-generation, distance, method-dispatch, and report-orchestration
  contracts.
- [[spectral-backend-runtime-diagnostic]] - runtime evidence showing that
  null-whitened tangent matrix materialization, not SciPy eigendecomposition,
  dominates the current slow spectral workloads.
- [[manuscript-life-science-readiness]] - readiness check for turning the
  methods draft and biological feature matrices into a gap-marked or completed
  life-science application section.
- [[traceable-mathematical-benchmark-suite]] - contract for method-proof
  benchmark cases, trace artifacts, deterministic math-layer attribution, and
  non-promotion of unsupported calibration claims.

### Questions

- [[open-mathematical-questions]] - consolidated mathematical backlog for
  calibration, projection, feature-space covariance, FDR, traversal, hierarchy
  recoverability, and manuscript evidence gaps.
- [[generated-index]] - when to replace hand-maintained `wiki/index.md` with a
  generated index.
- [[markdown-validation]] - whether to add markdownlint or remark validation
  beyond the current dependency-free linter.

## Evidence

- `AGENTS.md` states that `wiki/index.md` is the first read target.
- `wiki/schema.md` defines required page shape.
- `wiki/maintenance.md` requires index updates during ingest and cleanup.

## Links

- [[project-overview]]
- [[wiki-construction]]
- [[wiki-search]]
- [[schema]]
- [[maintenance]]
- [[log]]
