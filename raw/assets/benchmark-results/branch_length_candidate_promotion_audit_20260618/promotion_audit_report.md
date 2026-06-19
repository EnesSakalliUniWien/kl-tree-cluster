# Branch-Length Candidate Promotion Audit 2026-06-18

## Decision

Do not promote `kl_internal_filter_branch_length_v1` over current `kl` as the production default from this evidence. Keep it as the next candidate to test and as a guarded comparator.

## Method Summary

- `kl`: 93 ok / 28 skip / 0 fail, exact-K 66/121, mean ARI 0.819354, median ARI 1.000000.
- `kl_internal_filter_branch_length_v1`: 91 ok / 30 skip / 0 fail, exact-K 62/121, mean ARI 0.801364, median ARI 1.000000.
- `kl_legacy_c2ef9a69`: 121 ok / 0 skip / 0 fail, exact-K 78/121, mean ARI 0.726685, median ARI 0.994656.

## Matched Current Comparison

- Branch higher ARI than current on `6` cases.
- Branch lower ARI than current on `7` cases.
- Branch ties current on `66` cases.
- Branch is OK when current skips on `12` cases.
- Current is OK when branch skips on `14` cases.
- Both skip or are non-OK on `16` cases.

## Strongest Branch-Length Gains Versus Current

- `phylo_dna_8taxa_low_mut` (phylogenetic_dna): current ARI 0.638421, branch ARI 0.985649, delta 0.347228.
- `phylo_protein_8taxa` (phylogenetic_protein): current ARI 0.578652, branch ARI 0.869841, delta 0.291189.
- `cat_mod_4cat_6c` (categorical_moderate): current ARI 0.678058, branch ARI 0.819374, delta 0.141317.
- `binary_2clusters` (improved_binary_edge_cases): current ARI 0.960769, branch ARI 1.000000, delta 0.039231.
- `gauss_outlier_cluster_4c` (gaussian_outlier_contamination): current ARI 0.976327, branch ARI 1.000000, delta 0.023673.
- `gauss_noisy_many` (improved_gaussian): current ARI 0.990487, branch ARI 1.000000, delta 0.009513.

## Largest Branch-Length Losses Versus Current

- `gauss_clear_medium_continuous` (continuous_gaussian_examples): current ARI 1.000000, branch ARI 0.000000, delta -1.000000.
- `dim_consolidated_4c_24f_continuous` (continuous_dimensional_gaussian_examples): current ARI 1.000000, branch ARI 0.000000, delta -1.000000.
- `phylo_dna_4taxa_low_mut` (phylogenetic_dna): current ARI 0.959575, branch ARI 0.755372, delta -0.204203.
- `phylo_protein_4taxa` (phylogenetic_protein): current ARI 0.829355, branch ARI 0.762138, delta -0.067217.
- `phylo_dna_8taxa_med_mut` (phylogenetic_dna): current ARI 1.000000, branch ARI 0.949972, delta -0.050028.
- `gauss_single_outlier_4c_continuous` (continuous_gaussian_outlier_examples): current ARI 1.000000, branch ARI 0.991577, delta -0.008423.
- `binary_unbalanced_med` (improved_binary_unbalanced): current ARI 1.000000, branch ARI 0.993431, delta -0.006569.

## Promotion Blockers

- Current `kl` has higher full-suite completed-row mean ARI and more exact-K rows.
- Branch-length has more skips than current and introduces internal sparse-context inadmissibility skips that need a traversal/support audit before any production claim.
- The largest branch losses are continuous Gaussian/dimensional rows and some phylogenetic rows, so a default promotion would regress known solved cases.
- Branch-length sometimes completes rows that current skips, but those are not enough to offset the regressions and skip increase.

## Positive Evidence To Preserve

- Branch-length keeps fail-closed behavior on severe unsupported overlap such as `overlap_extreme_4c`, unlike legacy.
- Branch-length improves several rows where current under-recovers, especially `phylo_dna_8taxa_low_mut`, `phylo_protein_8taxa`, and `cat_mod_4cat_6c`.
- Binary section behavior is strong and should remain part of the candidate story.

## Next Evaluation Target

Run a fixed-candidate branch-length traversal/support audit rather than a routing policy: record the live traversal, the edge-reachable traversal that walks until edge tests close, the visited tuples, and branch lengths at those tuples. Use that evidence to identify the mathematical support problem before any production promotion.
