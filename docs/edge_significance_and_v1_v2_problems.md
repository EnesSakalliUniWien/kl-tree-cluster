# Edge Significance & v1 vs v2 — Known Problems

## Edge Significance (Gate 2) — The Foundational Problem

### Post-Selection Inflation (Synthetic Benchmarks)

The edge test (child-parent divergence) has **~15.9% empirical Type I error** at nominal α=5% on synthetic benchmarks. This inflation is structural: the linkage algorithm builds the tree to optimally separate the data, so the child distribution is *expected* to diverge from the parent by construction, even under pure noise.

The nested variance formula partially mitigates this:

$$\text{Var}(\hat\theta_c - \hat\theta_p) = \theta(1-\theta) \cdot \left(\frac{1}{n_c} - \frac{1}{n_p}\right)$$

The $(1/n_c - 1/n_p)$ factor accounts for the child being a subset of the parent, reducing the effective degrees of freedom compared to a two-sample test. But it does **not** account for the tree being data-dependent — the variance formula assumes the partition is fixed a priori.

### Under-Sensitivity on Real Sparse Data (Empirical, 2026-02-16)

On real-world sparse binary data, Gate 2 exhibits the **opposite** problem — not inflation but extreme lack of power. Diagnostic results from a 626 × 456 GO-term binary matrix (97.5% zeros):

| Metric | Value |
|--------|-------|
| Edges passing Gate 2 (BH) | 61/624 (9.8%) |
| Raw p < 0.05 (before BH) | 203/624 (32.5%) |
| Mean T/k ratio (significant edges) | 2.08 (expect ~1 under null) |
| Root child N1207 p-value | 0.14 (FAILS Gate 2) |
| Root child N1249 p-value | ≈0 (passes but irrelevant without N1207) |

Three mechanisms cause this under-sensitivity:

#### 1. Felsenstein overcorrection on outlier branches

Average-linkage trees from sparse binary data produce **highly asymmetric root splits** — a few outlier samples with rare features are peeled off as a tiny cluster (n=2) with very long branch length (BL/mean_BL = 4.29). The Felsenstein multiplier `1 + BL/mean_BL = 5.29` inflates variance by 5.29×, converting a genuinely significant signal (T/k = 6.22 without Felsenstein) into a non-significant result (T/k = 1.18, p = 0.14).

The underlying signal is real: those 2 samples have θ = 1.0 on GO-terms where the population-wide θ = 0.006. But the Felsenstein formula assumes branch lengths reflect neutral evolutionary time, not data-dependent tree construction artifacts.

#### 2. Projection dimension saturation (k = d)

The JL lemma with ε = 0.3 gives k ≥ 456 (the full feature dimension) for any n ≥ 100. So for most of the tree, the random projection R is the identity — no dimensionality reduction occurs. The test statistic T ~ χ²(456) requires T > ~520 to reject at α = 0.05, which is hard to achieve when signal is spread across many sparse features.

Projection dimension by sample size (d = 456, ε = 0.3):

| n | k | k/d |
|---|---|-----|
| 10 | 255 | 0.56 |
| 50 | 434 | 0.95 |
| 100 | 456 | 1.00 |
| 200+ | 456 | 1.00 |

#### 3. BH correction includes untestable edges

The BH correction runs over ALL edges including ~626 leaf-edges that trivially have p = 1.0. This inflates the denominator m, reducing the effective threshold from α to α × rank/1250. Excluding leaf-edges from the correction set would reduce m by ~50% and recover sensitivity at small subtrees.

#### Edge p-value by subtree size

Power is strongly size-dependent — only large subtrees have detectable signal:

| Subtree size | # edges | # significant | Mean raw p |
|-------------|---------|---------------|------------|
| [2, 10) | 405 | 14 | 0.81 |
| [10, 50) | 50 | 1 | 0.84 |
| [50, 100) | 12 | 0 | 1.00 |
| [100, 200) | 33 | 0 | 0.48 |
| [200, 626) | 124 | 46 | 0.03 |

### Why This Matters for Everything Downstream

Gate 2 is the **gatekeeper** for the entire pipeline:

1. **Sibling test gating**: Gate 3 (sibling divergence) is never computed for parents where neither child passes Gate 2. A false positive in Gate 2 triggers an unnecessary sibling test.

2. **Cousin-weighted calibration contamination**: The weighted calibration in `cousin_weighted_wald` uses edge p-values as weights: $w_i = \min(p_{\text{edge},L}, p_{\text{edge},R})$. If Gate 2 has 15.9% Type I error, the "null-like" weight distribution is contaminated — pairs that should have high weights (true nulls) are assigned low weights because one child was falsely flagged as edge-significant.

3. **Joint gate inflation**: The overall false-split probability is:

$$P(\text{split}) = P(\text{Gate 2 opens}) \times P(\text{Gate 3 opens} \mid \text{Gate 2 opened})$$

These are **not independent under the null** — the same noise fluctuation that opens Gate 2 makes Gate 3 more likely to open. The effective per-node false-split rate is higher than $\alpha_2 \times \alpha_3$.

### No Current Correction

- Tree-BH corrects for **multiplicity** (many edges tested), not for **post-selection** (tree built from data).
- Felsenstein scaling adjusts for **branch length** (phylogenetic distance), not for **data-dependent tree construction**. On real sparse data it actively **hurts** by over-inflating variance on outlier branches.
- Neither the cousin-weighted Wald nor any other sibling test method corrects for Gate 2 inflation — they only calibrate Gate 3.
- The BH correction set includes leaf-edges (p=1.0 by definition), diluting the effective threshold for real edges.
- Projection dimension saturation (k = d) on moderate-dimensional data removes the dimensionality reduction that the test was designed around.

### The Dual Problem: Over-splitting on Synthetics, Under-splitting on Real Data

The edge test has **two opposite failure modes** depending on the data regime:
- **Balanced synthetic data** (well-separated clusters, θ near 0.5): Post-selection inflation dominates → 15.9% false positive rate → over-splitting
- **Sparse real-world data** (θ near 0, asymmetric trees, outlier branches): Felsenstein overcorrection + projection saturation + BH dilution dominate → 9.8% detection rate → under-splitting (K=1)

No single correction can fix both: reducing Felsenstein scaling would help real data but worsen synthetic over-splitting. A proper fix requires **calibrating Gate 2 itself** (analogous to what cousin-weighted Wald does for Gate 3), or switching to a permutation-based edge test that is intrinsically calibrated.

---

## v1 vs v2 — The Core Differences

### v1: Clean but Rigid

```
Gate 1 (binary?) → Gate 2 (edge signal?) → Gate 3 (sibling different?)
    │                    │                        │
    └─ MERGE             └─ MERGE                 ├─ YES → SPLIT (push children)
                                                  └─ NO  → MERGE (collect leaves)
```

**Strengths**:
- Deterministic output (up to p-value tie-breaking)
- Exact leaf partition (every leaf assigned to exactly one cluster)
- Single test per sibling pair — full statistical power
- No selective inference issues (no recursive sub-testing)

**Weaknesses**:
- Hard binary splits only — cannot merge sub-clusters across split boundaries
- Post-hoc merge partially compensates but uses uncalibrated raw tests (see Tier 1.1 in roadmap)
- If linkage puts similar sub-populations on opposite sides of a split, v1 cannot recover

### v2: Flexible but Statistically Unsound

```
Gates 1–3 identical to v1
    │
    └─ SPLIT → localize_divergence_signal()
                    │
                    ├─ Drill down recursively, testing cross-boundary pairs
                    ├─ Collect SimilarityEdges (non-significant pairs)
                    ├─ Collect DifferencePairs (significant pairs)
                    └─ extract_constrained_clusters() — greedy merge with Cannot-Link
```

**The 6 problems with v2**:

#### Problem 1: Selective Inference in Drilling

`_recurse()` uses raw `pval < alpha` to decide whether to drill deeper. This decision is itself a hypothesis test, but it uses **uncorrected** α. The set of tests that reach the final BH correction stage is data-dependent, violating BH's fixed-hypothesis-set assumption.

The references cited in the docstring (Meinshausen 2008, Yekutieli 2008, Goeman & Mansmann 2008) are hierarchical testing procedures that control FDR across the tree of tests — but **none of them are actually implemented**. The code uses flat BH on the leaf-level results, which is not the same thing.

#### Problem 2: BH on Dependent Tests

The leaf-level cross-boundary tests (e.g., L1↔R1, L1↔R2) share data: both compare against L1. Standard BH requires independence or PRDS. The tree-structured overlap violates both. BY correction (valid under arbitrary dependence) would be correct but more conservative.

#### Problem 3: $4^d$ Power Explosion

At each recursion level, the number of cross-pairs multiplies: $m_1 \times m_2$ tests where $m_i$ is the number of children on each side. After $d$ levels of binary tree: up to $4^d$ tests. BH correction over $4^d$ tests reduces the effective threshold to $\alpha \cdot i / 4^d$ for the $i$-th ranked p-value. At depth 3: 64 tests. Moderate signals that are clearly significant at the aggregate level can fail to survive correction at the leaf level.

**Consequence**: Localization systematically produces more `SimilarityEdge`s than warranted, causing the constrained merge to over-merge — the opposite of v1's over-splitting tendency.

#### Problem 4: Leaf Coverage Gaps

The v2 traversal pushes children onto `nodes_to_visit` after a split, and collects merge points. But `extract_constrained_clusters` initializes from `similarity_graph.nodes | difference_graph.nodes | merge_points`. If localization tested internal nodes (e.g., L↔R directly), these appear in the graph with leaf sets that **overlap** with what lower merge points contribute. The union-find can assign the same leaf to multiple clusters.

#### Problem 5: Wrong Conservatism in Similarity Merge

`merge_similarity_graphs()` keeps the **lower** p-value when the same pair appears from two split points. But similarity edges represent *failure to reject* — lower p-value means *weaker* evidence of similarity. The conservative choice should keep the **higher** p-value (strongest evidence of similarity).

#### Problem 6: Order-Dependent Greedy Merge

`extract_constrained_clusters()` processes similarity edges sorted by p-value (highest first), but the greedy merge is path-dependent. Merging A↔B first can transitively block C↔D via Cannot-Link propagation. The result is **not unique** and can be fragile to small changes in p-values.

---

## How v1 and v2 Interact with Edge Significance Issues

| Mechanism                              | v1 Impact                                                                                      | v2 Impact                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Gate 2 false positives (15.9%, synth)  | Unnecessary sibling tests computed → wasted computation, slight power dilution via BH          | Same + triggers localization drilling → amplified test explosion         |
| Gate 2 false negatives (90.2%, real)   | Traversal stops at root → K=1; 583/625 sibling tests SKIPPED because Gate 2 never opened them | Same — v2 localization never activates because Gate 2 blocks it          |
| Felsenstein overcorrection (outliers)  | Kills signal on outlier branches (5× variance inflation) → genuine clusters absorbed          | Same (Felsenstein runs before v2 localization)                           |
| Projection saturation (k=d)            | Test is chi²(456) — conservative; signal spread across sparse features doesn't aggregate      | Same                                                                     |
| BH dilution (leaf edges)               | Effective threshold halved by inclusion of ~n untestable leaf-edges with p=1.0                 | Same                                                                     |
| Calibration contamination              | Weighted calibration sees fewer true-null pairs → noisier ĉ                                   | Same (calibration runs before v2 localization)                           |
| Joint gate inflation                   | Over-splitting on synthetics, partially compensated by post-hoc merge                          | Over-splitting at Gate 3 → localization may under-correct via power loss |
| Cumulative false splits                | $\sim$15.9% × (Gate 3 rate) per node on synthetics                                            | Same base rate, but localization adds its own false similarities         |

---

## Summary: Which to Use

| Criterion                          | v1                                   | v2                                                                              |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| **Statistical validity**           | Sound (within post-selection limits) | Multiple violations                                                             |
| **Determinism**                    | Yes                                  | No (greedy order-dependent)                                                     |
| **Leaf partition correctness**     | Guaranteed                           | Potential duplicates                                                            |
| **Handles misplaced sub-clusters** | No (rigid)                           | Attempts to, but unreliably                                                     |
| **Implementation complexity**      | Simple (DFS + 3 gates)               | Complex (DFS + localization + similarity/difference graphs + constrained merge) |
| **Edge test on sparse data**       | **K=1** — blocked at root            | Same — blocked at root before v2 ever activates                                |
| **Recommended**                    | **Yes — default**                    | Not until selective inference and power issues are resolved                     |

v1 with calibrated post-hoc merge (Tier 1 fixes) is the recommended path. However, v1 also urgently needs **Gate 2 calibration** — without it, the pipeline produces K=1 on sparse real-world data because the edge test is under-powered after Felsenstein overcorrection and BH dilution. v2's signal localization concept is sound in principle but requires a hierarchical testing procedure (not flat BH) and careful leaf-coverage accounting before it can be trusted.
