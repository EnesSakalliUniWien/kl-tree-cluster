---
title: Wiki Index
type: control
status: reviewed
updated: 2026-06-04
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
