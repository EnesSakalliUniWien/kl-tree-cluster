---
title: Barycentric Split Action Formula 2026-06-24
type: analysis
status: draft
updated: 2026-06-24
sources:
  - tree_break_selection/hierarchy_analysis/statistics/distributional_action.py
  - tests/statistics/47_test_distributional_action_contract.py
  - benchmarks/diagnostics/math_trace/barycentric_action.py
  - benchmarks/diagnostics/math_trace/path_conditioned_barycentric_action.py
  - wiki/analyses/scrna-distributional-action-audit-20260624.md
tags:
  - barycentric
  - clustering
  - topology
  - distribution
---

# Barycentric Split Action Formula 2026-06-24

## Summary

The correct distributional-action object for a tree decision is the internal
node split, not an isolated child edge. For an internal node \(p\) with children
\(c_i\), child masses \(m_i\), and child distributions \(\mu_i\), the parent is
the barycenter

\[
\mu_p = \frac{\sum_i m_i \mu_i}{\sum_i m_i}.
\]

The split action is the between-child part of the parent inertia:

\[
A(p) = \sum_i m_i \lVert \mu_i - \mu_p \rVert^2.
\]

For a binary split with children \(L,R\), total mass \(M=m_L+m_R\), and sibling
contrast \(\delta=\mu_L-\mu_R\),

\[
A(p) =
\frac{m_L m_R}{M}\lVert \mu_L-\mu_R \rVert^2.
\]

This is the concrete formula that ties mass, topology, and sibling structure
together.

## Details

Topology chooses which leaves belong to each child subtree. That determines
each child mass \(m_i\) and child barycenter \(\mu_i\). The parent distribution
is then not free: it is the mass-weighted barycenter of its children. Therefore
the child-parent edge vectors are constrained by the sibling vector.

For a binary split, let \(\beta=m_L/M\). Then

\[
\mu_p = \beta\mu_L + (1-\beta)\mu_R,
\]

\[
\mu_L-\mu_p = (1-\beta)(\mu_L-\mu_R),
\]

\[
\mu_R-\mu_p = -\beta(\mu_L-\mu_R).
\]

So the two child-parent edge directions are not independent evidence. They are
opposite mass-scaled views of the same sibling contrast. Their edge
contributions are

\[
A_{L|p} = m_L \lVert \mu_L-\mu_p \rVert^2,
\qquad
A_{R|p} = m_R \lVert \mu_R-\mu_p \rVert^2,
\]

and they sum to the parent split action:

\[
A_{L|p}+A_{R|p}=A(p).
\]

The shares are mass-tied:

\[
\frac{A_{L|p}}{A(p)}=\frac{m_R}{M},
\qquad
\frac{A_{R|p}}{A(p)}=\frac{m_L}{M}.
\]

This is why an edge-local filter can be wrong. In a \(1:99\) split with sibling
gap \(10\), the large child's edge contribution is only \(0.99\), but the split
action is \(99\). The small edge contribution does not mean the split is weak;
it means the large child is close to the parent barycenter because it dominates
the parent mass.

The topology-level identity is recursive. If leaves are points and each
internal node uses the same barycentric rule, then the total root inertia
decomposes into internal split actions:

\[
\sum_{\ell\in p}\lVert x_\ell-\mu_p\rVert^2 =
\sum_{\text{internal }u\subseteq p} A(u).
\]

For leaves \(0,2,10,14\) grouped as \((0,2)\) and \((10,14)\), the child split
actions are \(2\) and \(8\), the root split action is \(121\), and the root
total inertia is \(131\). This matches \(2+8+121\).

## Evidence

- `tests/statistics/47_test_distributional_action_contract.py` now verifies
  the binary sibling formula, the mass-tied child edge shares, the unbalanced
  split counterexample, and recursive topology-level inertia decomposition.
- `tree_break_selection/hierarchy_analysis/statistics/distributional_action.py`
  now exposes `split_distributional_action_summary`, which computes
  \(\sum_i m_i\lVert\mu_i-\mu_p\rVert^2\) from child masses and child means.
- `applications/scrna/pancreas_benchmark.py` copies
  `Distributional_Split_Action_*` annotation columns into tree-edge CSV outputs
  as diagnostics; non-`none` split-action filtering is rejected until calibrated
  against the projected-Wald edge/sibling gates.
- `benchmarks/diagnostics/math_trace/barycentric_action.py` already contained
  the same local barycentric contrast residual and parallel-axis split-action
  diagnostic; the statistics helper now carries that formula into tested
  reusable code.

## Links

- [[scrna-distributional-action-audit-20260624]]
- [[barycentric-action-equation-diagnostic-20260606]]
- [[path-conditioned-barycentric-action-diagnostics-20260606]]

## Open Questions

- If action is ever used as a gate, should it operate only on the parent split
  after the sibling contrast is defined?
- Which metric should define \(\lVert\cdot\rVert\): standardized PCA identity,
  projected-Wald covariance geometry, or another locally whitened metric?
