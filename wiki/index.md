---
title: Wiki Index
type: control
status: reviewed
updated: 2026-05-26
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

### Source Summaries

- [[wiki-construction-brief]] - summary of the captured wiki construction
  request and its required layers, tooling, workflows, and open questions.
- [[github-wiki-structure-research]] - design-reference notes for Foam-style
  links, GitHub Docs frontmatter, markdownlint, and remark link validation.

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
- [[local-marchenko-pastur-rule]] - audit of the local MP dimension rule,
  eigenvalue scale, finite-sample null behavior, internal spectral rows, and
  enhancement options.
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
