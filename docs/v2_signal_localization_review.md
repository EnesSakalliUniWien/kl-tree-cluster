# Review of Split V2 (Signal Localization Decomposition)

**Date**: 2026-02-19

## Architecture Summary

The v2 pipeline extends v1's hard split/merge decisions with **signal localization** — when Gate 3 says "siblings ARE different", v2 drills down recursively to find *where* the divergence originates, enabling cross-boundary partial merges.

**Flow**: `decompose_tree_v2()` → `process_node_v2()` → `GateEvaluator.should_split_v2()` → `localize_divergence_signal()` → `extract_constrained_clusters()`

---

## Issues Identified

### 1. Fundamental Power Problem (structural, documented)
The copilot instructions already note this: v2 Mean ARI = 0.431 vs v1's 0.757 (74 cases). The root cause is sound — localization fragments one aggregate test (high pooled power) into many sub-pair tests, each with smaller sample sizes + BH penalty. This structurally destroys power, producing **false similarity edges** that trigger incorrect cross-boundary merges.

**Verdict**: This is inherent to the approach, not a bug. The power guard at gates.py (returning `(True, None)` when zero difference pairs survive BH) is a good mitigation but only catches the extreme case.

### 2. Recursion Strategy in `_recurse()` — Cartesian Explosion
At signal_localization.py, the recursion tests **all cross-boundary pairs** (`left_nodes × right_nodes`). If both subtrees are bushy (say k children each), this produces k² tests per level, and can cascade recursively. With branching factor 2 and depth d, worst-case is O(4^d) tests.

The `max_depth` parameter helps but defaults to `None` (unlimited). In `decompose_tree_v2()` at tree_decomposition.py, it passes `self.localization_max_depth` which defaults to `None` in the constructor. This means localization will recurse to the very bottom of the tree for every split point — likely the "computational cost issues" mentioned in the docs.

**Recommendation**: Default `localization_max_depth` to a sane value (e.g., 3-5) instead of `None`. Unbounded recursion on large trees is dangerous.

### 3. `_get_valid_children` Edge-Significance Filter — Asymmetric Stopping
At signal_localization.py, nodes whose edges are NOT significant (noise) stop recursion. But this creates an asymmetry: one side of a cross-boundary pair can drill deep while the other stays shallow. The resulting tests compare a deep sub-node (small n) against a shallow aggregate (large n), which systematically biases the test — the deep node will appear "similar" to anything because its small sample size yields low power.

**Recommendation**: Consider requiring both sides to be at comparable depths, or only drill into a side if it has ≥2 edge-significant children.

### 4. No FDR Coordination Across Split Points
Each call to `localize_divergence_signal()` (one per split point) runs its own independent BH correction (`signal_localization.py` L250). But `decompose_tree_v2()` then merges results from ALL split points into a single similarity/difference graph (`tree_decomposition.py` L430-L431). The effective FDR across the full graph is uncontrolled — if you have 10 split points each running BH at α=0.05, the global false similarity rate can be much higher.

**Recommendation**: Either (a) collect all raw p-values from all split points and run a single global BH, or (b) use a Bonferroni-style correction across split points.

### 5. `merge_similarity_graphs` Keeps Highest p-value — Correct but Subtle
At signal_localization.py, duplicate edges keep the higher p-value. This is correct for similarity (higher p = stronger evidence of no difference), but it means a single high-p result from one level can dominate over multiple low-p results from other levels, potentially masking real differences that were detected at a different localization granularity.

### 6. Union-Find in `extract_constrained_clusters` — Stale `cluster_members` Keys
At signal_localization.py, after `union(root_u, root_v)`, the code does:
```python
cluster_members[root_u].update(cluster_members[root_v])
del cluster_members[root_v]
```
But `find()` uses path compression, so after subsequent operations, `find(some_node)` might return a root that was previously `root_v` (now compressed to `root_u`). The issue is that the `root_u`/`root_v` from `find()` in future iterations might not match the keys in `cluster_members` if path compression runs lazily. In practice this works because the delete + update keeps `cluster_members` in sync with the representative, but it's fragile — a subtle bug if union-by-rank were added.

### 7. Leaf Deduplication Heuristic — Smallest-Cluster-Wins
At signal_localization.py, overlapping leaves go to the smallest cluster. This is a reasonable "most specific" heuristic, but it can produce very uneven cluster sizes and doesn't account for the statistical evidence. A leaf might have strong similarity to a large cluster but get assigned to a tiny one purely based on cluster size.

---

## Summary

| Area                                           | Severity   | Status                                  |
| ---------------------------------------------- | ---------- | --------------------------------------- |
| Power loss from fragmented testing             | High       | Structural limitation, documented       |
| Unbounded recursion depth                      | Medium     | Default `None`, should default to a cap |
| No global FDR across split points              | Medium     | Multiple independent BH corrections     |
| Cartesian explosion O(4^d) tests               | Medium     | No depth default mitigates this too     |
| Asymmetric depth from edge-significance filter | Low-Medium | Can bias power                          |
| Union-Find fragility                           | Low        | Works today but not robust to changes   |
| Smallest-cluster-wins deduplication            | Low        | Reasonable heuristic                    |

## Bottom Line

The v2 architecture is well-designed conceptually (drill down to find soft boundaries, constrained merge respecting Cannot-Link edges). The power guard is a good safety valve. The main practical problems are:

1. The fundamental power gap from fragmenting tests
2. Missing a default recursion depth cap
3. Uncoordinated FDR across split points

Issues 2 and 3 are fixable; issue 1 is why v2 stays opt-in experimental.