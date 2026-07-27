---
title: Method, Application, and Plot Seams 2026-07-27
type: analysis
status: reviewed
updated: 2026-07-27
sources:
  - README.md
  - applications/README.md
  - scripts/README.md
  - tree_break_selection/space_separation/README.md
  - tree_break_selection/plot/README.md
  - pyproject.toml
tags:
  - architecture
  - applications
  - plotting
  - space-separation
---

# Method, Application, and Plot Seams 2026-07-27

## Summary

Repository ownership is now explicit across three interfaces: reusable
space-separation methods in the importable package, dataset applications under
`applications/`, and reusable plotting engines under
`tree_break_selection/plot/`. Benchmarks consume those interfaces to validate
them and no longer own the adaptive-cosine method or the generic file-plotting
backend.

## Details

`tree_break_selection/space_separation/` is the method module. It owns adaptive
cosine weighting and spectral blocks, invariant/equivariant decomposition, and
fixed/adaptive diffusion coordinates and distances. Its interface accepts
in-memory matrices and has no dataset paths, labels, plots, or report layout.

`applications/` is the adapter layer. Endotype/GO, scRNA, and MNIST workflows
have separate directories and README entry-point maps. scRNA follow-up analyses
and dataset-specific figures are further separated under `analysis/` and
`plots/`. Applications may load datasets, interpret domain labels, orchestrate
the method, and compose application reports. The remaining `scripts/` directory
is limited to maintenance and external-request helpers. Manuscript-specific
toy and explanatory figure composition lives in `manuscript/tools/figures/`.

`tree_break_selection/plot/` is the reusable plot module. It owns cluster
colors, tree rendering, the noninteractive file backend, and common image
panels. Application-specific page composition stays with its application;
benchmark covers and result-record plots stay in `benchmarks/shared/plots/`.

These boundaries prevent the former reverse dependency where reusable
space-separation or plotting operations lived in benchmark diagnostics.
Applications that are themselves benchmark workflows may reuse public runner
and result contracts from `benchmarks/shared/`; they must not import private
underscore helpers or treat a diagnostic module as the owner of a method.

## Evidence

- `applications/README.md` lists all maintained application entry points.
- `tree_break_selection/space_separation/README.md` states the method contract
  and exclusions.
- `tree_break_selection/plot/README.md` separates reusable engines from
  benchmark and application composition.
- `scripts/README.md` defines when a script should graduate to a package module
  or application.
- `pyproject.toml` includes `applications` in first-party lint resolution.

## Links

- [[project-overview]]
- [[redundant-and-legacy-code-map-20260623]]
- [[repository-hygiene-and-completion-audit-20260727]]

## Open Questions

- Should manuscript-specific figure generators gain a common artifact manifest
  after the paper figure set stabilizes?
- Should application commands gain package entry points after their CLI names
  and output contracts stabilize?
