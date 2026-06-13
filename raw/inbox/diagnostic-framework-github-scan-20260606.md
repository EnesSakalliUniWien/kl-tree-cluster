# Diagnostic Framework GitHub Scan 2026-06-06

This raw note records a lightweight GitHub scan for analytical and machine
learning frameworks that could help KL-TE calibration diagnostics after the
row-aligned KAK sibling-panel result.

Local GitHub context:

- Repository remote: https://github.com/EnesSakalliUniWien/kl-tree-cluster.git
- Current branch inspected locally: `dev`
- GitHub connector searches for open issues and PRs in
  `EnesSakalliUniWien/kl-tree-cluster` matching calibration, selected-tail,
  KAK, FDR, or sibling returned no results.

Relevant external repositories:

- Selective inference:
  https://github.com/selective-inference/Python-software
  Software for selective inference, mainly regression/post-selection tests.
  Relevant as mathematical reference for conditioning on selected objects, but
  not directly matched to selected trees or selected PCA sibling tests.

- Conformal risk control:
  https://github.com/aangelopoulos/conformal-risk
  Small codebase for conformal risk control with examples including
  hierarchical ImageNet. Relevant to threshold/guard calibration where the
  risk is monotone and a held-out calibration set is exchangeable.

- General conformal prediction and risk control:
  https://github.com/scikit-learn-contrib/MAPIE
  scikit-learn-compatible library for prediction intervals, prediction sets,
  and risk control.

- Conformal calibration:
  https://github.com/deel-ai/puncc
  Python library for conformal prediction and conformal anomaly detection.

- Calibration metrics and recalibration:
  https://github.com/EFS-OpenSource/calibration-framework
  The netcal framework for calibration-error metrics and post-hoc calibration,
  including regression uncertainty calibration.

- Knockoffs:
  https://github.com/msesia/deepknockoffs
  Approximate model-X knockoff sampling and diagnostics. Relevant as FDR
  design reference and for thinking about null/signal symmetry, but not a
  direct selected sibling p-value calibration.

- Simulation-based inference:
  https://github.com/sbi-dev/sbi
  Toolkit for likelihood-free posterior, likelihood, and ratio estimation from
  simulators. Relevant for amortizing selected-tail laws over simulated
  benchmark contexts.

- Bayesian diagnostics:
  https://github.com/arviz-devs/arviz
  Exploratory analysis, model checking, comparison, and diagnostics for
  Bayesian models. Relevant if KL-TE tail-law or hazard models become
  hierarchical Bayesian diagnostics.

- Probabilistic graphical models:
  https://github.com/pgmpy/pgmpy
  Python toolkit for graphical models, causal/probabilistic inference, model
  validation, parameter estimation, and simulations. Relevant as a diagnostic
  representation for traversal dependencies, not as causal proof.

- ML observability and drift:
  https://github.com/evidentlyai/evidently
  Reports, tests, and monitoring for data/model drift and quality. Useful for
  benchmark-panel drift and regression checks.

- Data profiling:
  https://github.com/whylabs/whylogs
  Lightweight data profiles for drift and quality monitoring. Useful for
  tracking covariate panel distribution shifts between benchmark runs.

- Explainability:
  https://github.com/shap/shap
  SHAP values for interpreting predictive models. Useful for diagnostic model
  interpretation, not for inferential calibration.

Project interpretation:

- The best direct mathematical framework is selective inference, but the
  available software is not shaped for tree-selected sibling Wald statistics.
- The best practical diagnostic workflow is simulation-based inference plus
  conformal/risk-control validation on held-out selected contexts.
- KAK geometry should enter first as a covariate in diagnostic tail-law and
  traversal-risk panels, not as a production p-value correction.
