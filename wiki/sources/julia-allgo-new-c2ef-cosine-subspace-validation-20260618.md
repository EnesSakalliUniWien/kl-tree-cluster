---
title: Julia allGO New c2ef Cosine Subspace Validation 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/inbox/c2ef-cosine-subspace-method-notes-20260615.md
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/METADATA.md
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/config.py
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/reference_split_manifest.json
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/reference_tfidf_components_02_05_legacy_assignments.csv
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/validation/README.md
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/validation/cluster_biological_coherence_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/validation/perturbation_stability_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/validation/cluster_level_stability_summary.csv
tags:
  - source
  - julia
  - legacy
  - cosine
  - validation
---

# Julia allGO New c2ef Cosine Subspace Validation 2026-06-18

## Summary

The c2ef `validate_cosine_subspace_split.py` workflow was run against
`feature_matrix_julia_allGO_new.tsv` using the copied full legacy package from
commit `c2ef9a69e0888168950bdee4a41ae8ab9996e32f`. The reference split is the
historical TF-IDF gene-gene cosine components `2-5` tree, followed by legacy
`PosetTree.decompose` with `alpha_local = 0.001`, `sibling_alpha = 0.01`, and
legacy `PASSTHROUGH = True`.

The run produced a `602 x 6368` allGO-new matrix split into `19` clusters. The
largest reference cluster has `130` genes and only one cluster is a singleton.
The useful evidence is not external ARI. It is biological coherence,
within-cluster TF-IDF cosine, perturbation success/failure, and cluster-level
best-Jaccard stability.

## Key Points

- The legacy config used by the copied c2ef package matches the c2ef commit:
  `EDGE_ALPHA = 0.001`, `SIBLING_ALPHA = 0.01`,
  `TREE_DISTANCE_METRIC = hamming`, `TREE_LINKAGE_METHOD = average`, and
  `PASSTHROUGH = True`.
- The reference split has `19` clusters, largest cluster size `130`, and one
  singleton.
- Biological coherence passes for `11/19` clusters under the script rule:
  cluster size at least `3`, at least three GO terms with FDR `q < 0.05`,
  top-term `q < 0.05`, and top prevalence delta at least `0.25`.
- The coherent large clusters include DNA binding (`130` genes), endoplasmic
  reticulum membrane (`66`), glycosaminoglycan metabolic process (`52`), tumor
  necrosis factor-mediated signaling regulation (`44`), and ephrin receptor
  binding (`36`).
- Perturbation reruns succeed for `6/8` feature subsamples at fraction `0.8`,
  `4/8` feature subsamples at fraction `0.6`, and `8/8` gene subsamples at
  fraction `0.8`.
- All failed perturbations are feature-subsampling input failures where one or
  two genes lose all active terms after the feature drop. They are not
  decomposition exceptions.
- Cluster-level stability is heterogeneous. Reference clusters `2`, `1`, `0`,
  and `18` have mean best-Jaccard at least `0.82`, while clusters `12`, `10`,
  `14`, and `8` are weakly recovered under perturbation.
- A previously existing adaptive-diffusion run uses
  `feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv` (`703 x 14766`), not this
  allGO-new matrix. Its `kl_diffusion_adaptive` parameters were
  `k = 15`, diffusion time `3`, components `30`, pydiffmap hamming metric,
  bandwidth `-1/(d+2)`, and median epsilon. That run should not be treated as
  the adaptive-diffusion result for `feature_matrix_julia_allGO_new.tsv`.

## Evidence

- `reference_split_manifest.json` records the input matrix shape, legacy
  namespace, c2ef source commit, reference split, alpha settings, `19`
  clusters, largest cluster size `130`, and one singleton.
- `reference_tfidf_components_02_05_legacy_assignments.csv` stores the
  per-gene legacy c2ef reference assignments.
- `validation/README.md` summarizes the coherence and perturbation results.
- `cluster_biological_coherence_summary.csv` stores per-cluster enrichment,
  prevalence, coherence-rule, and within-cluster TF-IDF cosine metrics.
- `perturbation_stability_summary.csv` stores every perturbation run,
  including failure reasons.
- `cluster_level_stability_summary.csv` stores cluster-level best-Jaccard
  stability across successful perturbations.

## Links

- [[legacy-c2ef9a69-method-package-20260616]]
- [[cosine-band-coherence-comparator-20260615]]
- [[julia-tree-estimator-run-20260614]]
