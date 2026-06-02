---
title: Selected Hierarchy Null Support Contract
type: analysis
status: reviewed
updated: 2026-06-02
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py
  - benchmarks/diagnostics/calibration/selected_hierarchy_external_calibration_contract.py
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/external_calibration_contract.csv
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_parent_size.csv
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_depth.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_relationships.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_block_models.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/sources/selected-hierarchy-external-calibration-contract-20260602.md
  - wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md
  - wiki/sources/selected-hierarchy-geometry-covariates-20260602.md
  - wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md
  - wiki/analyses/selected-hierarchy-selection-geometry.md
  - wiki/questions/open-mathematical-questions.md
  - manuscript/sections/method/sibling_test.tex
tags:
  - analysis
  - calibration
  - selection
  - support
---

# Selected Hierarchy Null Support Contract

## Summary

The selected-hierarchy null support contract is a diagnostic contract, not a
production calibration fallback. It defines when a selected-hierarchy null
simulation has enough matched evidence to describe the same-data selection
phenomenon for a focal sibling context. If no matched selected-hierarchy null
records exist, the result is unsupported. If support exists but Monte Carlo
precision is weak, the result is descriptive and imprecise. Relaxing context
matching can explain where support is lost, but it must not be treated as a
rule for borrowing calibration evidence.

## Details

For a focal sibling context \(u\), the intended selected-hierarchy null object
is

\[
\mathcal L_0\!\left(
W_u
\mid
\mathcal S(X),\
E_u,\
M_u
\right),
\]

where \(\mathcal S(X)\) denotes same-data hierarchy selection, \(E_u\) denotes
the open child-parent edge path required for \(u\) to be selected, and \(M_u\)
denotes the matched focal context. The current diagnostic approximates this
law by regenerating null feature matrices, rebuilding the hierarchy inside
each replicate, rerunning edge tests, and collecting selected sibling
statistics.

The diagnostic support states are:

```text
matched_selected_hierarchy_records
  At least one regenerated selected-hierarchy replicate produced selected
  sibling records matching the declared context.

unsupported_no_matched_selected_hierarchy_records
  No regenerated selected-hierarchy replicate produced a selected sibling
  record matching the declared context. No c-hat, p-value, or blocking decision
  is defined.
```

The core no-fallback rule is:

\[
n_{\mathrm{match}}(u)=0
\quad\Longrightarrow\quad
\hat c_{\mathrm{sel}}(u)\ \text{undefined}.
\]

The current exact diagnostic context requires feature family, projection
dimension, parent-size band, and parent depth to match. The 500-replicate
context-relaxation ladder shows that exact depth matching is often the
sparsest variable for non-root binary and categorical targets. Therefore the
current mathematical status is:

```text
exact matching variables with current support:
  feature family
  projection dimension
  open edge path

candidate stratification variables, not validated borrowing variables:
  parent size
  parent depth
  target mode
```

This is not a final production contract. It is the current honest diagnostic
interpretation of the observed support geometry.

For Monte Carlo precision, the diagnostic records independent matching
simulation counts, relative simulation standard error for \(\hat c\), and
tail-resolution diagnostics. The 500-replicate descriptive study used the
following evidence-quality target:

```text
descriptive scale target:
  relative simulation SE(c-hat) near or below 5%

descriptive tail-resolution target:
  matching-simulation tail resolution near or below 0.01
```

This target is sufficient to describe whether \(c\) is near one or in the
tens. It is not sufficient to install a production empirical-tail calibration
at \(\alpha_{\mathrm{sib}}=0.01\). A production tail-calibration target would
need a stricter predeclared Monte Carlo error bound around the tail
probability.

The 500-replicate evidence gives:

```text
strict root context:
  gauss_null_large      c=49.9, relSE=2.2%, tail resolution=0.0021
  gauss_clear_medium    c=36.7, relSE=1.6%, tail resolution=0.0022
  binary_low_noise_4c   c=19.3, relSE=5.2%, tail resolution=0.0047
  cat_clear_3cat_4c     c=27.4, relSE=4.6%, tail resolution=0.0037

strict non-root context:
  gauss_null_large      c=66.2, relSE=2.2%, tail resolution=0.0049
  gauss_clear_medium    c=36.4, relSE=1.6%, tail resolution=0.0026
  binary_low_noise_4c   unsupported under exact depth matching
  cat_clear_3cat_4c     only five matching simulations under exact depth matching
```

The non-root relaxation ladder gives:

```text
projection and parent-size matching:
  binary matched simulations=25,  c=45.2
  categorical matched simulations=195, c=56.1

projection-only matching:
  binary matched simulations=206, c=37.5
  categorical matched simulations=304, c=51.2

feature-family-only matching:
  binary matched simulations=246, c=56.3
  categorical matched simulations=313, c=55.1
```

The conclusion is that same-data selected-hierarchy geometry produces
large-\(c\) selected null behavior across matched contexts. Exact non-root
depth matching can make support too sparse. Relaxed contexts help identify the
support bottleneck, but they do not define a calibration rule.

The 2026-06-02 stratification diagnostic supports the same interpretation.
Parent-size strata explain more of the visible heterogeneity than exact depth:
small selected parent nodes often have \(c\) in the `50`--`70` range, while
root-like selected nodes are lower but still far above one in the tested
contexts. Depth remains useful as a descriptive stratum, but exact depth
matching is too sparse to treat as a validated conditioning requirement.

The same stratification run now records the selected-ratio law
\[
R_u=\frac{W_u}{a_u\nu_u}
\]
inside each descriptive stratum. Reliable parent-size rows have q95 values
roughly in the `38`--`103` range, and the standard projected-Wald reference
rejects almost every selected-null record in most rows. This supports the
mathematical claim that the selected-hierarchy law is a different conditional
law, not a small perturbation of the fixed-context projected-Wald reference.
It still does not define a borrowing rule: these are law-shape diagnostics,
not admissible production calibration estimates.

The external calibration admissibility contract makes that boundary explicit.
For production external tail calibration at
\(\alpha_{\mathrm{sib}}=0.01\), the current diagnostic requires

```text
conditioning scope:
  same_data_selected_hierarchy_edge_path_open

exact stratum variables:
  case, feature family, n, p, sibling projection dimension, parent-size bin

tail-resolution target:
  1 / (m + 1) <= 0.2 * alpha_sib

resolved minimum at alpha_sib = 0.01:
  m >= 499 independent matching simulations
  matched selected records >= 499

scale-estimate precision:
  relative simulation SE(c-hat) <= 5%
```

The 500-replicate run does not make any row production-admissible: the largest
independent matching-simulation count is `470`, below the required `499`.
Therefore the production external estimator remains undefined everywhere in
this evidence set.

The same run checks scalar mean scaling. Among rows with at least 100 matching
simulations, scalar-\(c\) p-values have rejection rate `0.0` at
\(\alpha_{\mathrm{sib}}=0.01\), so scalar mean scaling is conservative here.
However, KS tests against Uniform(0,1) reject the scalar-\(c\) p-values in the
reliable rows. If an external selected-hierarchy model is pursued, the current
evidence points toward modeling the selected-ratio tail law rather than using a
single mean scale as calibrated p-value model.

The geometry covariate diagnostic adds candidate variables for the context
\(M_u\), but still does not make any variable production-admissible. In its
100-replicate representative run, edge-selection strength has the largest
recorded descriptive relationship with \(\log R_u\), while spectral and
angular summaries provide smaller additional structure. This means a future
external selected-hierarchy null study should test edge-selection severity as
an exact or modeled context variable before using parent size, depth, or
projection dimension alone. The current support contract remains descriptive:
these variables are candidate conditioning coordinates, not calibration
borrowing rules.

The selected-ratio tail-law diagnostic then tests a stricter context object:

```text
source family
feature family
parent-size bin
sibling projection dimension
edge-action bin
```

At `200` replicates, this table has `104` contexts and no
production-admissible rows. The most common failure is insufficient independent
matching simulations; sparse contexts also fail matched-record and held-out
tail standard-error requirements. Some small-parent, high-edge-action contexts
have descriptive held-out exceedance near `0.01`, but those rows still fail
the independent simulation threshold. This reinforces the no-fallback rule:
descriptive selected-tail behavior is evidence about the phenomenon, not an
external production calibration estimate.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py`
  implements selected records, support states, precision fields, and context
  matching for the diagnostic.
- `benchmarks/diagnostics/calibration/selected_hierarchy_external_calibration_contract.py`
  implements the external-calibration admissibility thresholds and scalar-vs-tail
  diagnostic.
- `raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/external_calibration_contract.csv`
  records the 500-replicate admissibility and scalar-vs-tail decisions.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_relationships.csv`
  records descriptive univariate relationships between selected-ratio scale
  and tree, edge-selection, spectral, and angular covariates.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_block_models.csv`
  records in-sample descriptive block-model summaries for the same covariate
  families.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
  records within-context selected-ratio tail support, held-out exceedance
  errors, and production-admissibility failures.
- `raw/assets/benchmark-results/selected_hierarchy_precision_20260601_summary.csv`
  records the 500-replicate root strict, non-root strict, and non-root
  relaxation-ladder summaries.
- `wiki/sources/selected-hierarchy-null-audit-20260601.md` summarizes the
  selected-hierarchy null audit and precision study.
- `wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md`
  summarizes the depth and parent-size stratification study, including
  selected-ratio quantiles and raw projected-Wald rejection rates.
- `wiki/analyses/selected-hierarchy-selection-geometry.md` explains the
  geometric selection mechanism that motivates this support contract.
- `manuscript/sections/method/sibling_test.tex` records the current internal
  empirical-null calibration support contract; the selected-hierarchy contract
  is a separate diagnostic object.

## Links

- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-null-audit-20260601]]
- [[selected-hierarchy-external-calibration-contract-20260602]]
- [[selected-hierarchy-stratification-diagnostic-20260602]]
- [[selected-hierarchy-geometry-covariates-20260602]]
- [[selected-ratio-tail-law-diagnostic-20260602]]
- [[open-mathematical-questions]]
- [[oracle-gate-path-diagnostic]]
- [[projected-wald-statistic]]

## Open Questions

- Which context variables should become exact conditioning variables for a
  production selected-hierarchy null, if such a model is ever added?
- Is edge-selection severity required in \(M_u\), and if so should it enter as
  a hard stratum, a continuous covariate, or part of a selected-ratio tail-law
  model?
- What Monte Carlo precision target would be required for production
  calibration at \(\alpha_{\mathrm{sib}}=0.01\)?
- Can parent size and depth be modeled continuously or stratified without
  becoming an unvalidated borrowing rule?
- How should a validated continuous selected-hierarchy null generator be
  defined before continuous cases enter this diagnostic?
