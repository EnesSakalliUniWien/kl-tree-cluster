# Enhancement Lab — Experiment Map

**Last Updated**: 2026-03-20  
**Total Experiments**: 31+ (exp0–exp31)  
**Status**: Active investigation pipeline

---

## Experiment Taxonomy

```
enhancement_lab/
├── Core Power/Calibration Investigation (exp0–exp14)
│   ├── exp0_baseline.py              # Baseline: 7 failure cases identified
│   ├── exp1_min_samples.py           # Minimum sample size analysis
│   ├── exp2_passthrough_depth.py     # Pass-through traversal depth
│   ├── exp3_posthoc_merge.py         # Post-hoc merge strategies
│   ├── exp4_alpha_calibration.py     # Alpha threshold calibration
│   ├── exp5_combined.py              # Combined adjustments
│   ├── exp6_independent_alpha.py     # Independent alpha per gate
│   ├── exp7_projection_dimension.py  # JL projection dimension analysis
│   ├── exp8_sibling_spectral_diagnosis.py  # Gate 3 spectral diagnosis
│   ├── exp9_sibling_parent_spectral.py     # Parent vs sibling spectral dims
│   ├── exp10_structural_k_diagnostic.py    # Tree structure K diagnostic
│   ├── exp11_symmetric_power.py      # Symmetric power analysis
│   ├── exp12_deflation_diagnostics.py  # Inflation deflation study
│   ├── exp13_power_loss_trace.py     # Gate-by-gate power attribution
│   └── exp14_snn_distance.py         # SNN alternative distance metric
│
├── Spectral Dimension Investigation (exp15–exp19) ⭐ MAIN PIPELINE
│   ├── exp15_spectral_dim_regression.py    # Root cause confirmation
│   ├── exp16_spectral_dim_strategies.py    # 8 strategies compared
│   ├── exp17_literature_strategies.py      # 12 literature methods
│   ├── exp18_eigenvector_relationships.py  # Eigenspace mapping
│   └── exp19_spectral_equations.py         # 16 equations, 3-phase analysis
│
├── Gate 2/3 Combined Validation (exp20–exp25)
│   ├── exp20_extended_equations.py         # Extended spectral equations
│   ├── exp21_diagnose.py                   # Diagnostic tools
│   ├── exp21_full_suite_validation.py      # Full benchmark validation
│   ├── exp22_gate2_projection_dim.py       # Gate 2 projection dimension
│   ├── exp23_diagnose.py                   # Combined gate diagnosis
│   ├── exp23_combined_gate_validation.py   # Gate 2+3 combined tests
│   ├── exp24_signal_frac_validation.py     # Signal fraction analysis
│   ├── exp25_floored_lam12_hybrid.py       # Lam12 × JL floor hybrid
│   ├── exp25_jl_divisor_theory.py          # JL divisor theory
│   └── exp25b_principled_floor.py          # Principled floor derivation
│
├── Parent-PCA Projection (exp26–exp26b) 🆕
│   ├── exp26_parent_pca_projection.py      # Parent PCA for Gate 3
│   └── exp26b_satterthwaite_projection.py  # Satterthwaite approximation
│
├── Inflation Policy Discovery (exp27–exp31) 🆕
│   ├── exp27_neighborhood_padding.py        # Neighborhood-aware local context tests
│   ├── exp28_f_uncertainty_deflation.py     # ESS-based F-tail and robust c-hat tests
│   ├── exp29_low_ess_permutation_benchmark.py  # Low-ESS search + permutation benchmark
│   ├── exp30_global_local_shrinkage.py      # Partial-pooling global/local shrinkage
│   └── exp31_oracle_policy_equations.py     # Oracle rows, portability, Fourier summaries, spectra plots
│
└── Supporting Infrastructure
    ├── lab_helpers.py                      # Shared benchmark infrastructure
    ├── run_lab_baseline.py                 # Baseline runner
    ├── FINDINGS.md                         # Chronological findings summary
    ├── SPECTRAL_DIM_REPORT.md              # Spectral dimension deep-dive
    ├── _test_weighted_wald.py              # Weighted Wald test dev
    ├── _bisect_spectral_dims.py            # Spectral dim bisection
    ├── _tmp_k_distribution.py              # K distribution diagnostics
    ├── _tmp_verify_satt.py                 # Satterthwaite verification
    ├── analyze_calibration.py              # Calibration analysis
    ├── bisect_config_regression.py         # Configuration regression tests
    └── sync_lab_cases.py                   # Case synchronization
```

---

## Key Findings by Experiment

### Phase 1: Power/Calibration Foundation (exp0–exp14)

| Exp        | Question                         | Finding                                      | Status     |
| ---------- | -------------------------------- | -------------------------------------------- | ---------- |
| **exp0**   | What are baseline failure modes? | 7 cases with ARI=0, K=1                      | ✅ Complete |
| **exp1–6** | Can tuning fix under-splitting?  | Minor ARI gains, not root-cause              | ✅ Complete |
| **exp7**   | Is JL dimension too low?         | Confirmed: k capped too low                  | ✅ Complete |
| **exp8–9** | What dims does Gate 3 use?       | Uses JL, not spectral — mismatch             | ✅ Complete |
| **exp10**  | Is tree structure interleaved?   | Confirmed visually                           | ✅ Complete |
| **exp11**  | Is power symmetric?              | Gate 2 + Gate 3 both underpowered at small n | ✅ Complete |
| **exp12**  | Is deflation the bottleneck?     | **No** — TP=0 regardless of estimator        | ✅ Complete |
| **exp13**  | Where is power lost?             | **41.7% Gate 2, 58.3% Gate 3**, 0% pass      | ✅ Complete |
| **exp14**  | Does SNN distance help?          | Alternative metric explored                  | ✅ Complete |

**Phase 1 Conclusion**: Problem is **structural** (tree interleaving → small nodes → no power), not calibration.

---

### Phase 2: Spectral Dimension Solution (exp15–exp19) ⭐

> Update (2026-03-20): The `jl_floor_qrt` deployment recommendation below is stale. After repairing the enhancement-lab harness against the current `sibling_config` API, live reruns gave `exp16`: `min_child=0.998`, `jl_floor_qrt=0.994`, `none=0.979`; repaired `exp19`: `jl_floor_qrt=0.994`, below several alternatives at `0.998`.

| Exp       | Question                             | Finding                                                                | Status     |
| --------- | ------------------------------------ | ---------------------------------------------------------------------- | ---------- |
| **exp15** | Is min-child k the regression cause? | Historical claim only; not revalidated after harness repair            | ✅ Complete |
| **exp16** | What's the optimal k strategy?       | Repaired rerun: `min_child` 0.998, `jl_floor_qrt` 0.994, `none` 0.979  | ✅ Complete |
| **exp17** | Do literature methods help?          | Historical claim only; needs rerun                                     | ✅ Complete |
| **exp18** | What's the eigenspace structure?     | SPLIT: parent gains dims, children orthogonal                          | ✅ Complete |
| **exp19** | What's the best spectral equation?   | Repaired rerun: `jl_floor_qrt` = 0.994, below several 0.998 strategies | ✅ Complete |

**Winner**: no current winner claim should be treated as settled  
**Impact**: repaired reruns do not support the earlier `0.815 → 0.991` production narrative  
**Status**: not ready for production deployment

---

### Phase 3: Combined Gate Validation (exp20–exp25)

| Exp        | Question                     | Finding                             | Status     |
| ---------- | ---------------------------- | ----------------------------------- | ---------- |
| **exp20**  | Extended equations?          | Confirms jl_floor_qrt robustness    | ✅ Complete |
| **exp21**  | Full suite validation?       | Validates across 74 benchmark cases | ✅ Complete |
| **exp22**  | Gate 2 projection dim?       | Dimension sensitivity analysis      | ✅ Complete |
| **exp23**  | Combined gate performance?   | Gate 2+3 interaction mapping        | ✅ Complete |
| **exp24**  | Signal fraction validation?  | Signal-to-noise analysis            | ✅ Complete |
| **exp25**  | Lam12 × JL hybrid?           | Alternative floor strategy          | ✅ Complete |
| **exp25b** | Principled floor derivation? | Theoretical justification           | ✅ Complete |

**Phase 3 Conclusion**: the earlier `jl_floor_qrt` robustness claim is historical and should be revalidated before use.

---

### Phase 4: Parent-PCA Projection (exp26–exp26b) 🆕

| Exp        | Question                      | Finding                               | Status        |
| ---------- | ----------------------------- | ------------------------------------- | ------------- |
| **exp26**  | Should Gate 3 use parent PCA? | Tests parent PCA directions vs random | 🔄 In Progress |
| **exp26b** | Satterthwaite approximation?  | Alternative χ² approximation          | 🔄 In Progress |

**Hypothesis**: Parent PCA captures between-group variance better than random projections.

---

### Phase 5: Inflation Policy Discovery (exp27–exp31) 🆕

| Exp       | Question                                           | Finding                                                           | Status      |
| --------- | -------------------------------------------------- | ----------------------------------------------------------------- | ----------- |
| **exp27** | Does local neighborhood padding recover power?     | Additional local context benchmarked                              | ✅ Complete |
| **exp28** | Does uncertainty-aware F-tail help?                | F-tail stayed close to global χ²; median variants were unstable   | ✅ Complete |
| **exp29** | What happens in the lowest-ESS cases?              | No evidence that F-tail or permutation-median beats global χ²     | ✅ Complete |
| **exp30** | Can simple global/local shrinkage improve things?  | Null control preserved, but BH focal outcomes stayed flat         | ✅ Complete |
| **exp31** | Can we distill a portable oracle-policy rule?      | New row-level oracle dataset, family portability, Fourier summary | 🔄 In Progress |

**Phase 5 Conclusion**: ad hoc scalar shrinkage was too weak; the current direction is row-level oracle policy distillation with explicit portability checks across case families.

---

## Strategic Recommendations

### Immediate Deployment

1. **Do not deploy `jl_floor_qrt` to production yet**
   - The repaired 2026-03-20 reruns do not reproduce the earlier winner claim
   - Revalidate on the full-suite harness before changing Gate 3 production logic

2. **Remove `PROJECTION_MAX_DIMENSION` cap**
   - Already applied in exp7–9
   - Eliminates artificial power ceiling

### Medium-Term Investigation (In Progress)

3. **Parent-PCA projection directions** (exp26)
   - Replace random projections with parent PCA eigenvectors
   - Potential power gains for Gate 3

4. **Satterthwaite approximation** (exp26b)
   - Better χ² approximation for projected statistics
   - Improved calibration under H₀

### Long-Term Research (Fundamental Limits)

5. **Tree construction alternatives**
   - exp13 showed 58.3% of true boundaries blocked by Gate 3 at n=2–5
   - Root cause: HAC interleaves overlapping clusters
   - Potential directions:
     - Alternative linkage (Ward, complete)
     - Ensemble trees
     - Soft clustering

6. **Feature pre-filtering**
   - Remove overlapping/noisy features before tree construction
   - Dimensionality reduction (PCA, autoencoders)

---

## Benchmark Infrastructure

### Sentinel Cases (15 cases, used in exp15–exp25)

| Category                                  | Cases                                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Failure cases** (ARI=0 under min-child) | `binary_balanced_low_noise__2`, `gauss_clear_small`, `binary_low_noise_12c`, `binary_perfect_8c`, `binary_hard_4c` |
| **Regression guard** (must not regress)   | `binary_2clusters`, `binary_hard_8c`, `binary_low_noise_2c/4c/8c`, `gauss_clear_small/medium/large`                |
| **Partial success** (0.3 ≤ ARI < 0.8)     | `cat_highcard_10cat_4c`, `phylo_dna_4taxa_low_mut`, `overlap_unbal_4c_small`                                       |

### Metrics

- **Mean ARI**: Average Adjusted Rand Index across cases
- **Exact K**: Fraction of cases with correct cluster count
- **K=1 collapses**: Number of cases that under-split to single cluster
- **Wins**: Number of cases where strategy achieves best ARI

---

## Code Locations

| Component                   | File                         | Function                            |
| --------------------------- | ---------------------------- | ----------------------------------- |
| Gate 3 dimension derivation | `orchestrator.py`            | `_derive_sibling_spectral_dims()`   |
| Gate 3 PCA projections      | `orchestrator.py`            | `_derive_sibling_pca_projections()` |
| Edge test (Gate 2)          | `child_parent_divergence.py` | `annotate_edge_gate()`              |
| Sibling test (Gate 3)       | `sibling_divergence/`        | `annotate_sibling_divergence*()`    |
| Spectral decomposition      | `spectral.py`                | `compute_spectral_decomposition()`  |
| Marchenko-Pastur            | `marchenko_pastur.py`        | `estimate_k_marchenko_pastur()`     |

---

## Timeline

| Date       | Milestone                                                        |
| ---------- | ---------------------------------------------------------------- |
| 2026-02-14 | Post-selection bias identified, cousin-adjusted Wald deployed    |
| 2026-02-17 | Signal localization v2 benchmarked (ARI 0.431 vs v1 0.757)       |
| 2026-03-17 | Spectral dimension investigation complete (exp15–exp19)          |
| 2026-03-18 | Parent-PCA projection investigation started (exp26)              |
| 2026-03-20 | Oracle-policy row modeling and portability analysis started (exp31) |
| **Next**   | Revalidate Gate 3 k-selection on the repaired full-suite harness |

---

## References

- Marchenko, V.A. & Pastur, L.A. (1967). Distribution of eigenvalues for some sets of random matrices.
- Johnson, W.B. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space.
- Felsenstein, J. (1985). Phylogenies and the comparative method.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.

---

**Maintained by**: Enhancement Lab  
**Contact**: See `debug_scripts/enhancement_lab/lab_helpers.py`
