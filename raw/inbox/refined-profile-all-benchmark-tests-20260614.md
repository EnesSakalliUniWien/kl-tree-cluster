# Refined Profile All-Benchmark Tests 2026-06-14

## Commands

Selected-null all-supported smoke:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_global_passthrough_refined_all_supported_3rep_20260614 \
  --suite full \
  --case-names <53 binary/direct-categorical supported cases> \
  --profiles fixed_coordinate_global_passthrough_refined_v1 \
  --data-roles null,signal \
  --replicates 3 \
  --base-seed 20309045 \
  --resume-from-checkpoints
```

Full-suite single-seed performance pass:

```bash
nice -n 10 python - <<'PY'
# direct _run_kl_method loop over get_test_cases_by_suite("full")
# with sibling_gate_profile="fixed_coordinate_global_passthrough_refined_v1"
PY
```

## Artifacts

- `/tmp/klte_global_passthrough_refined_all_supported_3rep_20260614`
- `/tmp/klte_refined_full120_performance_20260614`

## Selected-Null All-Supported Smoke

The selected-null runner can regenerate nulls for 53 of the 120 full-suite
cases: 42 binary-template cases and 11 direct categorical multinomial cases.
It does not regenerate selected nulls for continuous, SBM, phylogenetic,
quantile, Dirichlet-multinomial, or method-proof generators.

The 3-replicate run wrote 318 rows. Result summary:

- Binary-template nulls: 11 false splits across 126 null rows.
- Direct categorical nulls: 0 false splits across 33 null rows.
- Binary-template signal: mean ARI 0.855829, median ARI 0.972200.
- Direct categorical signal: mean ARI 0.779888, median ARI 0.881909.
- Six null cases were inflated, all overlap-template binary cases:
  `overlap_unbal_4c_small`, `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_part_4c_small`, `overlap_extreme_4c`,
  and `overlap_part_8c_large`.
- Seven signal cases were weak, including `cat_highcard_20cat_4c`,
  `cat_mod_4cat_6c`, heavy-overlap binary cases, and
  `overlap_unbal_4c_small`.
- Production admissibility remains fail-closed for both binary-template and
  categorical-multinomial transfer summaries in this all-supported smoke.

The largest runtime bottleneck was high-dimensional direct categorical
selected-family p-value evaluation. A process sample showed CPU time inside
SciPy special-function chi-square tail evaluation, not an I/O wait or deadlock.

## Full 120-Case Performance Pass

The full benchmark performance pass used the refined profile directly through
`_run_kl_method` and recorded one row per full-suite case. It is not a
selected-null calibration law.

Result summary:

- 120 cases attempted.
- 107 cases returned `ok`.
- 13 continuous cases errored under the current continuous covariance contract.
  Most errors were `Continuous feature blocks require
  continuous_covariance_by_block`; high-dimensional continuous cases hit the
  dense covariance dimension/work-state guards.
- The resolved profile in ok rows was
  `fixed_coordinate_global_passthrough_refined_v1`, with
  `fixed_coordinate_bh`,
  `global_sibling_min_passthrough_descendant_refined`, and 99 base selected
  permutations.
- Across ok rows, mean ARI was 0.782567, median ARI was 0.955719, and exact-K
  rate was 0.439252.
- Strong families: blobs-quantile, Gaussian outliers, phylogenetic, ordinary
  binary, and many Gaussian/blob overlap cases by ARI.
- Weak families: SBM collapsed to one cluster; dimensional Gaussian median
  binaries mostly collapsed; categorical Dirichlet-multinomial was near zero;
  the planted deep-signal traversal case collapsed to one cluster.
- High-dimensional categorical performance was good but expensive:
  `cat_highd_3cat_500feat` returned ARI 0.888278 and K=14/4 in about 362 s;
  `cat_highd_4cat_1000feat` returned ARI 0.944379 and K=15/6 in about 1263 s.
- Heavy-overlap binary cases remain the main signal weakness:
  `overlap_heavy_8c_large_feat` and `overlap_extreme_4c` both returned one
  cluster and ARI 0.0.

## Interpretation

The refined global selected-family pass-through profile is a useful diagnostic
candidate, not a production calibration rule. It improves the binary support
boundary found in the earlier three-case support run, but all-benchmark testing
shows three remaining blockers:

1. Same-data selected-null control is still inflated in overlap-template binary
   cases.
2. Some signal classes are overblocked or collapsed, especially heavy overlap,
   SBM, dimensional Gaussian median binaries, categorical Dirichlet-multinomial,
   and the planted deep-signal traversal case.
3. Continuous feature families need the continuous covariance contract or a
   validated low-rank covariance implementation before this profile can be
   interpreted on continuous benchmarks.
