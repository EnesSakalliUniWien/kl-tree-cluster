---
title: Adaptive Alpha Gate Design 2026-06-28
type: analysis
status: reviewed
updated: 2026-06-28
sources:
  - raw/inbox/adaptive-alpha-method-literature-20260628.md
  - wiki/sources/alpha-structure-sweep-20260627.md
  - wiki/analyses/oracle-gate-path-diagnostic.md
  - wiki/concepts/projected-wald-statistic.md
  - wiki/analyses/traversal-neighborhood-method-comparison.md
  - wiki/analyses/edge-gate-distance-time-contract-20260623.md
  - wiki/questions/open-mathematical-questions.md
  - benchmarks/validation/alpha_structure_sweep.py
  - benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_structure_summary.csv
  - benchmarks/results/alpha_structure_sweep_hamming_nn_q4_q5_20260627/alpha_structure_summary.csv
tags:
  - alpha
  - adaptive
  - p-values
  - clustering
---

# Adaptive Alpha Gate Design 2026-06-28

## Summary

The next adaptive TBS alpha policy should be an adaptive-resolution policy, not
an adaptive-FDR claim. Keep edge alpha conservative by default and make sibling
alpha the first adaptive axis. The runtime rule should be no-perturbation: a
predeclared sibling-alpha ladder plus deterministic guards from the ordinary
edge/sibling test trace and calibration-support fields.

## Details

### Problem

The benchmark question is why `alpha=0.01` can be too strict for some visible
structures, especially the Hamming NN diffusion q4/q5 Gaussian-overlap cases.
The methodological question is broader: how can alpha adapt to local structure
without using a test's own p-value twice?

The answer is to separate threshold discovery from threshold use. A TBS node
may report how close its p-value is to each threshold, but the decision to relax
a threshold must come from a predeclared policy over the normal gate trace. It
should not require bootstrap, subsampling, diffusion-parameter perturbation, or
other perturbation loops at runtime.

### Literature Constraints

Multiple-testing methods support adaptivity only under explicit information
contracts. IHW and AdaPT use side information; online FDR and alpha-investing
use ordered decisions and alpha budgets; hierarchical FDR uses tree-structured
families. Selective-inference work on hierarchical clustering warns that tests
run after cluster selection do not inherit ordinary fixed-hypothesis
calibration.

For TBS, that means the first adaptive version should not claim formal adaptive
FDR control. It should be a transparent resolution rule, with every relaxed
decision labeled by the alpha level required and by the single-run test context
that justified the relaxed resolution.

### Recommended First Design

Use fixed `edge_alpha=0.001` as the default production safety gate, then define
a sibling ladder:

```text
core:      sibling_alpha <= 0.003
default:   sibling_alpha <= 0.010
adaptive:  sibling_alpha <= 0.030
weak:      sibling_alpha <= 0.100
closed:    not opened on the ladder
```

For every candidate sibling split, compute the smallest ladder value that opens
the corrected sibling p-value. This is `sibling_alpha_required`. Do not hide it:
write it to trace outputs, summaries, node tables, and benchmark diagnostics.

The adaptive runtime must not rerun the tree under perturbations. It uses only
fields that already belong to the normal test trace: child-parent edge p-values
and BH decisions, raw and corrected sibling p-values, projection dimension,
empirical-inflation status, calibration-support status, parent size, child
balance, traversal state, and pass-through state.

### Decision Rule

Accept `core` and `default` sibling splits when the existing edge and sibling
contracts pass.

Accept an `adaptive` sibling split, meaning `0.01 < sibling_alpha_required <=
0.03`, only when a predeclared no-perturbation context guard passes:

- The node is binary, visited, and edge-open.
- Both child-parent edge tests are open, or a predeclared strong-edge margin
  condition is met.
- The raw sibling test is already strong; the blocker is the corrected
  sibling/FDR or inflation layer rather than absence of sibling evidence.
- The calibration-support status is explicit and valid; missing support,
  zero calibration weight, or undefined selected-tail support fails closed.
- The node falls in a supported parent-size, projection-dimension, and
  child-balance bin.
- The rule is acting on the focal sibling split itself, not using descendant
  perturbation evidence to justify a pass-through.

Treat `weak` splits as report-only until selected-tree calibration or much
stronger validation exists. A `weak` split can be useful for diagnostics, plots,
or candidate biological interpretation, but it should not silently change the
production cluster assignment.

### Validation Plan

Rerun the alpha sweep as a method comparison with columns for
`sibling_alpha_required`, no-perturbation context fields, guard pass/fail,
final adaptive-resolution label, and accepted-vs-reported split status.

Use the existing q4/q5 Hamming NN diffusion cases as the first positive-control
artifact: baseline `edge=0.001`, `sibling=0.01` returns two clusters with ARI
`0.569784`, while the split opens at sibling `0.03` and recovers three clusters
with ARI `1.0` for q4 and `0.989983` for q5.

Use `phylo_large_64taxa` as the stress artifact: in the Cartesian adaptive
pydiffmap sweep, found clusters are invariant across edge alpha
`0.0003`, `0.001`, and `0.003`, but change with sibling alpha from `46` to
`61` to `70`. This is exactly the scenario where adaptive sibling resolution
should be explicit and audited.

Use the existing oracle-gate and projected-Wald notes as negative controls. They
show that one scalar alpha change cannot solve both open-edge/closed-sibling
under-splits and direct phylogenetic sibling false splits. The adaptive guard
therefore has to identify the test-behavior context, not merely raise alpha.

### Non-Goals

This design does not claim that sibling `0.03` is universally calibrated. It
does not make edge alpha adaptive. It does not use perturbation as a runtime
trigger. It records adaptive resolution decisions so that a future selected
hierarchy calibration can be layered on top without changing the result
contract.

## Evidence

- `benchmarks/results/alpha_structure_sweep_hamming_nn_q4_q5_20260627/alpha_structure_summary.csv`
  records the Hamming NN diffusion q4/q5 transition from two clusters at
  `edge=0.001`, `sibling=0.01` to three clusters at `edge=0.003`,
  `sibling=0.03`.
- `benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_structure_summary.csv`
  records that `phylo_large_64taxa` changes with sibling alpha and is stable
  across the compact edge-alpha grid.
- `wiki/sources/alpha-structure-sweep-20260627.md` records the sweep contract,
  p-value margin definition, partition-transition outputs, and diagnostic
  status.
- `wiki/analyses/oracle-gate-path-diagnostic.md` records the main gate-behavior
  split: under-splits often have open child-parent edges but closed corrected
  sibling gates, while over-splits can be direct sibling false splits or
  pass-through fragmentation.
- `wiki/concepts/projected-wald-statistic.md` records that same-sample adaptive
  sibling projection inflates selected-null tails and that fixed-coordinate
  sibling gates are the non-cross-fit repair direction in validated diagnostic
  paths.
- `wiki/analyses/traversal-neighborhood-method-comparison.md` records the
  traversal skeleton: a split requires binary, edge-open, and sibling-open;
  pass-through is a separate state when the local sibling gate is closed.
- `wiki/analyses/edge-gate-distance-time-contract-20260623.md` records that
  adaptive-diffusion branch lengths are topology/support diagnostics by default
  and do not automatically change edge variance.
- `raw/inbox/adaptive-alpha-method-literature-20260628.md` records the
  multiple-testing, stability, bootstrap-clustering, and selective-inference
  sources used to constrain the design.

## Links

- [[alpha-structure-sweep-20260627]]
- [[full-adaptive-pydiffmap-benchmark-run-20260627]]
- [[phylo-large-adaptive-pydiffmap-focus-audit-20260627]]
- [[categorical-adaptive-diffusion-focus-audit-20260626]]

## Open Questions

- Should the production adaptive cap be `0.03`, or should `0.05` be evaluated
  as an intermediate value before any `0.1` diagnostic setting?
- What exact strong-edge margin, parent-size, projection-dimension, and
  child-balance bins should define the no-perturbation adaptive context?
- Should pass-through contexts be excluded from adaptive alpha entirely, or can
  a focal split that opens at `0.03` convert a baseline pass-through into a
  direct split?
- Can sibling-alpha weighting be made formal with IHW/AdaPT-style covariates
  computed out of fold or from p-value-independent geometry?
- What selected-hierarchy null law is needed before this can be promoted from
  adaptive resolution to adaptive FDR control?
