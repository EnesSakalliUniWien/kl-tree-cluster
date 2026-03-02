# `_should_split` (v1) vs `_should_split_v2` — Comparison

## Identical: Gates 1–3

Both methods apply the **exact same three-gate logic**:

```
Gate 1: Binary? ──NO──▶ MERGE
Gate 2: Either child edge-significant? ──NO──▶ MERGE
Gate 3: Sibling test skipped? ──YES──▶ MERGE
        Siblings different (BH)? ──NO──▶ MERGE
```

Up through this point, the two methods make **identical decisions**. The divergence happens only when all three gates pass (SPLIT).

## Where They Differ: What Happens After SPLIT

|                 | `_should_split` (v1)                                              | `_should_split_v2`                                                                                              |
| --------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Return type** | `bool`                                                            | `Tuple[bool, LocalizationResult \| None]`                                                                       |
| **On SPLIT**    | Returns `True` — hard binary split, push both children onto stack | Returns `(True, LocalizationResult)` — runs **signal localization** to discover WHERE the divergence originates |
| **On MERGE**    | Returns `False`                                                   | Returns `(False, None)`                                                                                         |

## Signal Localization (the v2 addition)

When v2 determines siblings ARE different, it calls `localize_divergence_signal()` which:

1. **Drills recursively** into the left and right subtrees
2. Tests **cross-boundary pairs** (subtrees of left vs. subtrees of right)
3. Identifies which specific sub-clusters are truly different vs. which could be merged across the split boundary
4. Returns a `LocalizationResult` containing `SimilarityEdge` objects (pairs that are NOT different)

## How the Decomposers Use Them

**`decompose_tree` (v1)** — standard top-down:
- Calls `_should_split` → if `True`, pushes children onto a DFS stack
- If `False`, collects all leaves under the node as one cluster
- Result: **hard partitions only**

**`decompose_tree_v2`** — signal-localized:
- Calls `_should_split_v2` → collects `split_points` and `localization_results`
- After traversal, builds a **combined similarity graph** from all `LocalizationResult` objects
- Also builds a **difference graph** (Cannot-Link constraints)
- Runs `extract_constrained_clusters()`: a greedy merge that respects Cannot-Link edges from significantly different pairs
- Result: **soft partitions** — sub-clusters from opposite sides of a split can be merged if localization found them similar

## Concrete Example

Consider a tree where node P splits into L and R, and L further splits into L1, L2, and R into R1, R2:

```
        P
       / \
      L    R
     / \  / \
    L1 L2 R1 R2
```

- **v1**: If Gate 3 says L ≠ R → hard split. L1, L2 stay together; R1, R2 stay together. No cross-boundary merging.
- **v2**: If Gate 3 says L ≠ R → localize. Tests L1↔R1, L1↔R2, L2↔R1, L2↔R2. If L2≈R1 (similar), adds a `SimilarityEdge`. Final clustering can merge L2 and R1 into one cluster while keeping L1 and R2 separate.

## Config Toggle

Controlled by `config.USE_SIGNAL_LOCALIZATION` (default `False`). When `True`, `PosetTree.decompose()` dispatches to `decompose_tree_v2()`.

---

## Known v2-Specific Bugs — Fixed (2026-02-17)

### Bug 1 (FIXED): `_should_split_v2` Localization Power Guard

When post-FDR correction causes ALL sub-pairs to become similarity edges (zero difference pairs), `_should_split_v2` now returns `(True, None)` — trusting the aggregate Gate 3 SPLIT decision but discarding the misleading localization result. This triggers a hard v1-style split. The aggregate test has higher power (one test, pooled signal) than the localization sub-tests (many tests, BH penalty), so its decision should be trusted even when localization can't pinpoint WHERE the difference is.

Previously returned `(True, loc_result)` with empty `difference_pairs`, causing false cross-boundary merges via uncontested similarity edges. Returning `(False, None)` was tried but caused K=1 collapse on most benchmark cases.

### Bug 2 (FIXED): `extract_constrained_clusters` Leaf Overlap

Two fixes applied:
1. **Ancestor merge_point pruning**: Before building `node_to_leaves`, prune `merge_points` that are ancestors of any graph node.
2. **Leaf deduplication pass**: After Union-Find extraction, assign overlapping leaves to the smallest (most specific) cluster ("smallest-cluster-wins" policy).

### Bug 3: No-Soft-Boundaries Fallback Fragility

When `combined_similarity.number_of_edges() == 0`, v2 falls back to `[self._get_all_leaves(node) for node in merge_points]`. This is correct only if every leaf is reachable via `merge_points` (guaranteed by the exhaustive traversal), but the assumption is fragile if traversal logic changes.

---

## v2 Benchmark Results (2026-02-17, 74 cases)

| Metric | v1 (default) | v2 (localization) |
|--------|--------------|-------------------|
| Mean ARI | 0.757 | 0.431 |
| Median ARI | 1.000 | 0.306 |
| Exact K | 59/95 (62%) | 21/74 (28%) |
| K=1 cases | 10 | 9 |

**Root cause of v2 regression**: Localization sub-tests lack statistical power. The aggregate Gate 3 test pools all signal into one test, while localization fragments it into many cross-boundary sub-pair tests with smaller sample sizes AND a BH multiple-testing penalty. This creates false **similarity edges** (Type II errors on genuinely different sub-pairs) that cause incorrect cross-boundary merges. Additionally, v2 has combinatorial cost issues — `overlap_mod_10c_highd` case timed out after >10 minutes due to exponential cross-boundary pair expansion.

**v2 strengths**: Dramatically improves phylogenetic cases where v1 over-splits:
- `phylo_divergent_8taxa`: v2 ARI=1.0 K=8 vs v1 K=72 (over-split)
- `phylo_large_32taxa`: v2 ARI=0.965 K=31
- `phylo_protein_4taxa`: v2 ARI=1.0 K=4

**Conclusion**: v2 remains an opt-in experimental feature (`USE_SIGNAL_LOCALIZATION=False` by default). It may be valuable for phylogenetic data types but harms standard binary/Gaussian clustering.

---

## Shared Limitation: Gate 2 Under-Sensitivity (2026-02-16)

Both v1 and v2 are **equally blocked** by the Gate 2 (edge significance) bottleneck on real sparse data. Since Gates 1–3 are identical, neither version reaches the point where they diverge if Gate 2 fails to detect signal.

**Empirical evidence** (626 × 456 GO-term matrix):
- Only 61/624 edges (9.8%) pass Gate 2 after BH correction
- K=1 for both v1 and v2 — the traversal stops at the root
- v2's signal localization **never activates** because Gate 3 is never reached at meaningful nodes

This means the v1-vs-v2 distinction is **irrelevant until Gate 2 is fixed**. The improvements needed (Felsenstein cap, BH leaf-edge exclusion, projection saturation fix — see [roadmap](roadmap_changes_and_modularization.md) Tier 0) affect both versions equally.

Once Gate 2 is calibrated, the v1-vs-v2 difference will matter again — v2's cross-boundary merging should produce better cluster assignments on data with non-binary cluster structure.
