---
title: Specific Small Method Benchmark 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/context_negative_topology_conditioning
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/conditional_law_weight0_min2
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/conditional_law_weight1_min2
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/conditional_law_weight1_min1_diagnostic
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/focused_overlap_balance_product
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/small_overlap_selected_family
tags:
  - source
  - diagnostics
  - benchmark
  - overlap
  - topology
---

# Specific Small Method Benchmark 2026-06-15

## Summary

This run is the focused small benchmark for the current method question: can the
old topology-aware neighborhood bandwidth fix the context-negative overlap
failure when placed inside the support-aware Bayesian conditional topology law?
It reruns the `26`-row context-negative topology slice and compares conditional
law variants with topology-neighborhood bandwidth off, on, and on with relaxed
diagnostic support.

The benchmark also adds a focused multi-positive row-level fixture for
`balance_product`, then runs three small overlap clustering cases through the
selected-family traversal panel with the conditional-topology diagnostic profile
and the refined global pass-through comparator.

## Key Points

- The context-negative slice has `26` rows: `1` truth-recovery row, `17`
  `null_like` rows, `7` `diffuse_or_wrong` rows, and `1` `fragment_like` row.
- The explicit topology-neighborhood support contract is present: `17`
  `strict_null` support rows, `8` `selected_nonnull` exclusions, and `1`
  explicit `topology_signal_role = signal` row. Parent links are present for
  `25/26` rows.
- The same single-positive topology separators remain: outgoing balance,
  outgoing edge-norm balance, balance product, and outgoing balance times edge
  norm. The largest balance-product negative is `0.222500`; the truth value is
  `0.232891`, margin `0.010391`.
- With topology-neighborhood bandwidth disabled, the truth row is rank `1`,
  truth log odds are `31.869341`, maximum negative log odds are `27.174856`,
  and the margin is `4.694486`.
- With topology-neighborhood bandwidth enabled, the truth row is still rank
  `1`, truth log odds are `31.262811`, maximum negative log odds are
  `26.568325`, and the margin is still `4.694486`.
- The bandwidth component is active but not separating. The truth component is
  `-0.606531`, the negative median is `-0.471195`, and `13` negatives exceed
  the truth on this component alone.
- Relaxing support to `1` changes row status to
  `conditional_topology_candidate_diagnostic_only`, but the production summary
  remains `diagnostic_only_support_insufficient_fail_closed`. This is useful as
  a diagnostic but not a promotable calibration rule.
- The focused multi-positive row fixture has `3` truth-recovery rows and `4`
  hard negatives. `balance_product` is the only tested structural product that
  separates all truth rows from all negatives: truth minimum `0.226320`,
  negative maximum `0.210700`, margin `0.015620`. Outgoing balance alone does
  not separate because a hard negative has outgoing balance `0.491304`.
- The guarded internal recovery contract is now explicit:
  `recover_internal_split = root/null guards pass AND support sufficient AND
  balance_product/outgoing_edge evidence high`. In the focused fixture this
  recovers all `3` truth rows and blocks all `4` negatives: one by the
  root/null guard and three by insufficient balance product evidence.
- On the real `26`-row context-negative overlap slice, no row is promoted by
  the guarded recovery contract. The single truth row has coherent evidence
  (`balance_product = 0.232891`, outgoing edge-norm `0.971963`) but remains
  `support_insufficient_fail_closed`, so the rule correctly refuses to turn a
  one-positive diagnostic into a traversal calibration.
- The small selected-family clustering run covers
  `overlap_mod_4c_small`, `overlap_unbal_4c_small`, and `overlap_extreme_4c`
  with `2` signal and `2` selected-null replicates per method. The
  conditional-topology diagnostic profile is not a production traversal fix:
  it false-splits selected-null rows in `overlap_extreme_4c` and
  `overlap_mod_4c_small`, and fragments `overlap_unbal_4c_small` signal
  replicate `0` into `12` clusters. The refined global pass-through comparator
  is stronger on `overlap_extreme_4c` nulls and on the unbalanced signal row.

## Interpretation

The old topology-aware bandwidth layer fixes one engineering problem: it gives
the conditional law a cached, explicit, support-gated neighborhood component
without selected-nonnull leakage. It does not fix the statistical traversal
problem by itself. On the focused slice, the truth and the strongest negative
share the same node and parent context, so the bandwidth component moves both
by the same amount.

The signal is instead in coherent outgoing topology, especially when outgoing
evidence is conditioned on incoming support. In the original focused slice,
outgoing balance and outgoing edge-norm balance both rank the truth first with
zero negatives above the truth. In the multi-positive fixture, outgoing balance
alone fails while `balance_product` succeeds. The method should therefore treat
bandwidth as local context, not as the split decision. The next fix is a
selected-neighborhood Bayesian law whose dominant terms are outgoing balance,
outgoing edge-norm balance, and their income-conditioned product, with
bandwidth acting as a support/context regularizer.

The topology interpretation is directed. A candidate internal node has one
incoming edge from its parent side and two outgoing child edges; recovery is
plausible only when the incoming branch relation and outgoing child relation
are coherent in the same local structural context. `balance_product` is the
current diagnostic proxy for that income/outcome coherence, while outgoing
edge-norm balance checks that the child evidence is not merely a size-balanced
fragment. Root rows have no incoming edge, leaves have no outgoing sibling
test, and pass-through/null rows require guards before their local evidence can
be interpreted.

The small clustering run adds a guardrail: this posterior law should not replace
the refined selected-family/root guard globally. It should be applied only as an
internal ambiguous-node recovery rule after null/root protection remains in
force.

## Evidence

- The benchmark artifacts are stored under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/`.
- Focused verification passed:
  `pytest tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py tests/validation/calibration/overlap/134_test_overlap_conditional_topology_law_panel.py tests/validation/calibration/overlap/136_test_focused_overlap_balance_product_benchmark.py -q`.
- Lint passed for the touched benchmark files:
  `python -m ruff check benchmarks/diagnostics/calibration/overlap/overlap_context_negative_topology_conditioning.py benchmarks/diagnostics/calibration/overlap/overlap_conditional_topology_law_panel.py benchmarks/diagnostics/calibration/overlap/focused_overlap_balance_product_benchmark.py tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py tests/validation/calibration/overlap/134_test_overlap_conditional_topology_law_panel.py tests/validation/calibration/overlap/136_test_focused_overlap_balance_product_benchmark.py`.

## Links

- [[overlap-conditional-topology-law-panel-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[open-mathematical-questions]]
