# v1 (`_should_split` + `decompose_tree`) — Pitfalls & Mathematical Issues

## The good news: v1 is substantially cleaner

The v1 path is a straightforward three-gate DFS with no recursive sub-testing, so it avoids the entire class of selective-inference and power-explosion problems in v2. That said, there are still issues:

---

## Mathematically Questionable

### 1. Post-selection bias is acknowledged but not fully corrected

The cousin-weighted Wald addresses sibling inflation, but **Gate 2 (edge test) still runs uncorrected for post-selection**. On synthetic benchmarks it has ~15.9% Type I error at nominal α=5%. The tree-aware BH correction (`tree_bh`) adjusts for multiplicity but does **not** adjust for the fact that the tree was built from the same data. The edge test's nested variance $\theta(1-\theta)(1/n_c - 1/n_p)$ partially mitigates this (the child-is-subset-of-parent structure reduces inflation vs. a pure two-sample test), but it doesn't eliminate it.

This matters because Gate 2 is the **gatekeeper** for everything downstream — both the sibling test and the cousin-weighted calibration use Gate 2 results as ground truth. If Gate 2 has 15.9% Type I error, the "null-like" calibration set in `cousin_weighted_wald` contains ~16% contamination from false positives (pairs that should be null-like but one child was incorrectly flagged as edge-significant).

#### 1a. Empirical evidence: on real sparse data, Gate 2 is UNDER-powered, not over-powered

The 15.9% Type I inflation was measured on synthetic balanced benchmarks. On real-world sparse binary data (626 × 456 GO-term matrix, 97.5% zeros), Gate 2 exhibits the **opposite** problem — extreme under-sensitivity:

- **Only 61/624 edges (9.8%) pass Gate 2** after BH correction (at α=0.05)
- 203/624 edges have raw p < 0.05, but BH correction collapses this to 61
- The root node's children fail Gate 2: N1207 (2 leaves, p=0.14) and N1249 (624 leaves, p≈0 but irrelevant since Gate 2 requires *at least one child* to diverge — and N1207 doesn't)
- Gate 2 power is strongly size-dependent: edges with ≥200 leaves have 46/124 significant; edges with <50 leaves have only 15/455 significant

**Root causes of under-sensitivity:**

1. **Felsenstein variance inflation on outlier branches**: Root→N1207 has `BL/mean_BL = 4.29`, giving a Felsenstein multiplier of **5.29×**. This inflates variance by 5.29×, shrinking z-scores from |z|>17 (significant) to |z|<8 (still significant individually but diluted by projection). Outlier branches in average-linkage trees from sparse data routinely have BL/mean_BL > 3, causing massive variance inflation on exactly the edges that carry the most signal.

2. **Projection dimension saturation**: For n ≥ 100, `compute_projection_dimension(n, 456)` returns k=456 = d (the full feature dimension). The JL lemma with ε=0.3 gives k = O(log n), but log(100) × constant ≈ 500+ already exceeds d=456. So the "projection" is an identity transform for most of the tree — no dimensionality reduction occurs.  The test statistic T = Σz² ~ χ²(456) with 456 degrees of freedom. For a genuine but weak signal spread across many sparse features, this makes the test **conservative** (the signal-to-noise per feature is low, and summing 456 noisy z² values converges to null expectation).

3. **BH correction dilution**: Of 1250 edges tested, ~600 are leaf edges with p=1.0. These "dead" edges inflate the BH denominator, making it harder for real edges to pass correction. The effective threshold drops to α × rank/m where m includes all leaf edges.

### 2. Sequential gate logic is not joint-test-corrected

The three gates are applied **sequentially** with independent α levels:
- Gate 2: α = `SIGNIFICANCE_ALPHA` with tree-BH
- Gate 3: α = `SIBLING_ALPHA` with flat BH

But the overall false-split probability is $P(\text{split}) = P(\text{Gate 2 opens}) \times P(\text{Gate 3 opens} \mid \text{Gate 2 opened})$. These are **not** independent under the null — both use the same data, and both are inflated by post-selection. The effective per-node false-split rate is not $\alpha_2 \times \alpha_3$ because the conditioning on Gate 2 opening creates positive dependence (the same noise fluctuation that opened Gate 2 makes Gate 3 more likely to open).

No correction is applied for this joint testing structure. In practice, the cousin-weighted calibration partially compensates, but it only adjusts Gate 3 — Gate 2's inflation propagates unchecked.

### 3. Post-hoc merge uses different test semantics than the decomposition

The decomposition uses **pre-computed, BH-corrected** sibling divergence from the annotation pipeline. But post-hoc merge in [posthoc_merge.py](../kl_clustering_analysis/hierarchy_analysis/posthoc_merge.py) calls `test_divergence(lc, rc, node)` **fresh** — a raw `sibling_divergence_test()` call with no cousin-adjusted calibration. This means:

- The **decomposition** used a deflated $T_{\text{adj}} = T/\hat{c}$ to decide splits
- The **post-hoc merge** uses a raw inflated $T$ to decide merges

These are asymmetric: it's harder to merge than it was to split, because the merge test statistic is inflated while the split test was deflated. This bias systematically favors over-splitting.

---

## Structural Pitfalls

### 4. Post-hoc merge's LCA-blocking is overly conservative

In [posthoc_merge.py line 160](../kl_clustering_analysis/hierarchy_analysis/posthoc_merge.py#L155-L165):
```python
lca_has_reject: Dict[str, bool] = {}
for i, is_rejected in enumerate(reject):
    lca = pairs[i]["lca"]
    if is_rejected:
        lca_has_reject[lca] = True
```

If **any** pair under an LCA is significant, **all** merges under that LCA are blocked. Consider 4 clusters A, B, C, D under an LCA where A≠B (significant) but C≈D (p=0.9). Because A≠B, the merge of C and D is blocked — even though they're clearly similar. This is overly conservative and can prevent obvious merges.

### 5. Post-hoc merge descendant removal is too aggressive

In [posthoc_merge.py line 183](../kl_clustering_analysis/hierarchy_analysis/posthoc_merge.py#L180-L185):
```python
cluster_roots -= nx.descendants(tree, lca)
cluster_roots.discard(lca)
cluster_roots.add(lca)
```

When merging at an LCA, **all** descendant cluster roots are removed — not just the two being merged. If there are 3 clusters under the LCA and only 2 are merged, the third is also consumed. This is by design (the LCA subsumes everything), but it means a merge of 2 clusters can silently absorb a third unrelated cluster that happened to be under the same LCA.

### 6. Greedy merge ordering sensitivity

Like v2's constrained merge, the post-hoc merge processes pairs sorted by p-value (highest first = most similar). But the greedy logic checks `if lc not in cluster_roots or rc not in cluster_roots: continue` — once a cluster is consumed by a prior merge, later valid merges involving it are skipped. The final result depends on tie-breaking among similar p-values.

---

## Edge Significance Design Flaws (Empirical, 2026-02-16)

### 7. Felsenstein scaling is miscalibrated for data-dependent trees

The Felsenstein branch-length adjustment `Var *= 1 + BL/mean_BL` was designed for **fixed phylogenies** where branch lengths reflect evolutionary time. In data-dependent trees built by average linkage from binary data:

- **Outlier samples** produce long branches (BL/mean_BL > 4) that inflate variance by 5×+ and kill genuine signal
- **Large clusters** have short internal branches relative to outlier branches, so mean_BL is dominated by the many short internal edges
- The adjustment is asymmetric across siblings: root→small_child gets massive inflation while root→large_child gets moderate inflation

Empirically on real data (626×456 GO matrix): root→N1207 (2 outlier leaves) has Felsenstein multiplier **5.29×**, converting T/k from 6.22 (highly significant without Felsenstein) to 1.18 (p=0.14, not significant). The signal is real — those 2 samples have θ=1.0 on features where the population has θ=0.006 — but Felsenstein kills it.

### 8. Projection dimension saturates for d < O(log n / ε²)

The JL lemma gives k = O(log n / ε²). With ε=0.3 (default) and n≥100, k ≥ 456 = d for this dataset. The "random projection" becomes an identity transform — no dimensionality reduction occurs. This means:

- The test statistic T ~ χ²(456), requiring T > ~520 to reject at α=0.05
- Signal spread across many sparse features (each z_i slightly ≠ 0) may sum to T < critical value even when signal is present
- The test is designed for d >> k (many features projected to few dimensions), but on moderate-dimensional sparse data d ≈ k, removing the power advantage of projection

### 9. BH correction includes untestable edges

The multiple testing correction runs over ALL edges including leaf-edges that trivially have p=1.0. For a tree with 626 leaves, there are ~1250 edges but only ~624 are between internal nodes that could plausibly carry signal. The ~626 leaf-edges inflate the BH denominator m, reducing the effective significance threshold from α to α × rank / 1250. This is needlessly conservative — leaf edges are definitionally untestable and should be excluded from multiple testing.

---

## Summary Comparison

| Issue | v1 | v2 |
|---|---|---|
| Post-selection bias (edge test) | **Present** (15.9% Type I on synthetics) | Same |
| Edge test power on sparse data | **Under-powered** (9.8% detection rate on real data) | Same |
| Felsenstein on outlier branches | **Over-corrects** (5× variance inflation) | Same |
| Projection saturation (k=d) | **No dim reduction** when d < O(log n/ε²) | Same |
| Post-selection bias (sibling test) | **Partially corrected** via cousin-weighted | Same |
| Selective inference in sub-testing | **Not applicable** (no drill-down) | **Severe** |
| Test power | **Full** (one test per sibling pair) | **Diluted** ($4^d$ tests) |
| Post-hoc merge asymmetry | **Present** (raw T vs deflated T) | Same |
| Leaf coverage correctness | **Correct** (exact partition) | **Potentially broken** |
| Deterministic output | **Yes** (up to tie-breaking) | **No** (greedy + order-dependent) |

**Bottom line**: v1 is the more mathematically sound pipeline. On synthetic benchmarks its main weakness is over-splitting via uncorrected edge-test inflation. On real sparse data the opposite problem dominates: Gate 2 is **under-powered** due to Felsenstein overcorrection on outlier branches, projection dimension saturation, and BH dilution from untestable edges. The pipeline needs both a calibrated Gate 2 (not just Gate 3) and smarter handling of the Felsenstein/projection interaction on real-world data. v2 adds complexity without rigorous statistical underpinning for the recursive localization.
