# Enhancement Lab — Findings Summary

## Experiment Chronology

| Exp | Focus                                          | Key Finding                                   |
| --- | ---------------------------------------------- | --------------------------------------------- |
| 0   | Baseline                                       | 7 failure cases identified (ARI=0, K=1)       |
| 1–6 | Alpha, min samples, posthoc merge, passthrough | Minor ARI gains, not root-cause fixes         |
| 7   | Projection dimension (JL)                      | Confirmed k is too low when capped            |
| 8–9 | Sibling spectral dimension                     | Gate 3 uses JL dim, not spectral — mismatch   |
| 10  | Structural K diagnostic                        | Tree interleaving confirmed visually          |
| 11  | Symmetric power                                | Gate 2 + Gate 3 both underpowered at small n  |
| 12  | Deflation diagnostics                          | Deflation is not the bottleneck (see below)   |
| 13  | Power loss trace                               | Complete gate-by-gate attribution (see below) |

## Implementation (from exp7–9)

**Applied to codebase (all tests passing):**
- Gate 3 sibling test now uses `min(k_left, k_right)` spectral dims from Gate 2
- Removed `PROJECTION_MAX_DIMENSION` cap across 8 files
- ARI on failure cases improved from 0.059 → 0.220 (still low)

---

## Exp 12 — Deflation Is Not the Bottleneck

**Question:** Can better inflation estimation (deflation) rescue under-splitting?

**Method:** Tested 9 estimators (M0–M6 variants) across 7 failure cases. Simulated deflation: `T_adj = T / ĉ`, then re-ran BH correction to see if true splits become significant.

**Results:**
- **TP = 0** in ALL failure cases regardless of estimator
- Null p-values at k=2 are **overdispersed** (bimodal), not simply inflated
- k=2 dominates (861/2209 null pairs), mean T/k = 0.64 (below expected 1.0)
- All T/k-based estimators produce ĉ ≈ 1.0 because k=2 pulls the estimate down
- M6 (type-I ratio) detects inflation (ĉ = 1.4–6.3×) but only reduces FP, never creates TP

**Conclusion:** The problem is absence of power, not miscalibration.

---

## Exp 13 — Power Loss Trace (Definitive Diagnosis)

**Question:** Where exactly is statistical power lost for true cluster boundaries?

**Method:** For every internal node: (1) check if it's a "true split" (children have disjoint ground-truth label sets), (2) check which gate blocks it — Gate 2 (edge) or Gate 3 (sibling).

### Grand Summary (7 failure cases, 2973 nodes, 871 true boundaries)

| Blocked at          | Count | %     |
| ------------------- | ----- | ----- |
| Gate 2 (edge test)  | 363   | 41.7% |
| Gate 3 (sibling BH) | 508   | 58.3% |
| Passes all gates    | **0** | 0%    |

### Three Levels of Power Loss

#### Level 1: Tree Structure (Root Cause)

HAC with heavy inter-cluster overlap **interleaves** cluster members at the leaf level. True boundaries end up at tiny nodes (n_parent = 2–5) instead of clean high-n internal nodes.

| Case                        | True K | True-split nodes | Largest n_parent |
| --------------------------- | ------ | ---------------- | ---------------- |
| overlap_heavy_4c_small_feat | 4      | 169              | 16               |
| overlap_unbal_4c_small      | 4      | 62               | 5                |
| overlap_extreme_4c          | 4      | 279              | 7                |
| overlap_heavy_8c_large_feat | 8      | 329              | 12               |
| sbm_hard                    | 3      | 29               | n/a (all Gate 2) |
| overlap_hd_4c_1k            | 4      | 3                | 374 (but 1000-D) |

Contrast with a **working** case:
- `binary_perfect_4c` (K=4): only 3 true-split nodes at n_parent = 80, 40, 40

#### Level 2: Gate 2 (Edge Test)

At n = 1–3 per child, edge p-values = 1.0 (no signal). Median edge p_BH for blocked true splits = 1.0.

Special case: `overlap_hd_4c_1k` has n_parent up to 374, but Marchenko-Pastur finds spectral k = 0 in d=1000 (noise floor dominates signal).

#### Level 3: Gate 3 (Sibling Test)

All blocked pairs have k=2, median n_parent = 2, median raw sibling p = 0.55. At n=1 per side, there is zero power. Only 3.5% of raw p-values < 0.05 (≈ chance level).

---

## Strategic Implication

The current approach is **sound** for well-separated clusters where HAC produces clean tree structure. It is **fundamentally limited** for heavily overlapping clusters — not by calibration or alpha tuning, but by tree construction quality. Potential future directions:

1. **Better tree construction**: Alternative linkage methods, distance metrics, or ensemble trees
2. **Soft clustering**: Allow probabilistic membership instead of hard binary splits
3. **Pre-filtering overlapping features**: Dimensionality reduction before tree construction
4. **Hybrid approach**: Use KL decomposition for clear structure, fall back to another method for ambiguous regions
