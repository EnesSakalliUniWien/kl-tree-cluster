---
title: Diagnostic Framework GitHub Scan 2026-06-06
type: source
status: reviewed
updated: 2026-06-06
sources:
  - raw/inbox/diagnostic-framework-github-scan-20260606.md
tags:
  - source
  - diagnostics
  - github
  - calibration
  - frameworks
---

# Diagnostic Framework GitHub Scan 2026-06-06

## Summary

This source summarizes a GitHub scan for analytical and machine learning
frameworks that could strengthen Tree-Break Selection selected-tail, sibling, edge, and
traversal diagnostics. The scan found no open GitHub issues or PRs in the
current project remote for the searched calibration terms, and it identified
external packages that are useful mainly as diagnostic patterns rather than
drop-in production calibration.

## Key Points

- The local GitHub remote is
  [EnesSakalliUniWien/tbs-tree-cluster](https://github.com/EnesSakalliUniWien/tbs-tree-cluster).
  Connector searches for open issues and PRs matching calibration,
  selected-tail, KAK, FDR, or sibling returned no results.
- [selective-inference/Python-software](https://github.com/selective-inference/Python-software)
  is the closest mathematical software reference. It supports post-selection
  inference ideas, but it is oriented toward regression selection rather than
  Tree-Break Selection selected trees, selected PCA frames, and sibling Wald statistics.
- Conformal risk-control tools such as
  [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk),
  [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE),
  and [deel-ai/puncc](https://github.com/deel-ai/puncc) are useful for
  held-out threshold validation and guard panels when exchangeability and
  monotone risk definitions are explicit.
- Calibration-specific tooling such as
  [EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework)
  can help report calibration error and recalibration diagnostics, but does not
  solve selected-tree null law derivation.
- Knockoff tooling such as
  [msesia/deepknockoffs](https://github.com/msesia/deepknockoffs) is useful as
  an FDR and null/signal symmetry design reference, not as a direct sibling
  p-value correction.
- [sbi-dev/sbi](https://github.com/sbi-dev/sbi) is the strongest machine
  learning framework candidate for this project because Tree-Break Selection already has
  simulators. It can learn selected-tail likelihood ratios or posterior
  summaries from generated null/signal panels, with strict held-out validation.
- [arviz-devs/arviz](https://github.com/arviz-devs/arviz) is useful if the
  selected-tail law or traversal hazard model becomes hierarchical Bayesian,
  because it provides model checking and diagnostic summaries.
- [pgmpy/pgmpy](https://github.com/pgmpy/pgmpy) can represent traversal as a
  probabilistic graphical dependency system, but it should be treated as a
  diagnostic representation rather than causal proof.
- Observability and interpretability tools such as
  [evidentlyai/evidently](https://github.com/evidentlyai/evidently),
  [whylabs/whylogs](https://github.com/whylabs/whylogs), and
  [shap/shap](https://github.com/shap/shap) are useful for panel drift,
  regression checks, and diagnostic-model explanation, not for production
  inferential calibration.
- The recommended enhancement stack is: selective-inference formalization for
  the equation, simulation-based inference for amortized diagnostic laws,
  conformal risk control for guard thresholds, and calibration/observability
  tooling for monitoring and interpretation.

## Evidence

- `raw/inbox/diagnostic-framework-github-scan-20260606.md` records the GitHub
  repository scan, local project remote, and framework interpretation.

## Links

- [[mixed-null-signal-geometry-validation-20260606]]
- [[selected-tail-law-q5-validation-20260604]]
- [[null-edge-sibling-calibration-enhancement-plan]]
- [[path-conditioned-barycentric-action-diagnostics-20260606]]
