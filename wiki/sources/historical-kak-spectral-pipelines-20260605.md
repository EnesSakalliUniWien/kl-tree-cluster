---
title: Historical KAK Spectral Pipelines 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - raw/inbox/historical-kak-spectral-pipelines-20260605.txt
tags:
  - source
  - scripts
  - spectral
  - kak
  - cosine
---

# Historical KAK Spectral Pipelines 2026-06-05

## Summary

Root-level KAK/cosine spectral scripts are not present in the current working
tree, but they are present in git history before commit `66c3def` deleted them
during repository simplification. Their shared contract was to select a tree
topology from a spectral subspace, then run the project `TreeDecomposition`
gates on the original feature matrix.

## Key Points

- `adaptive_cosine_spectral_blocks.py` built a gene-gene cosine operator,
  eigendecomposed it, selected adaptive spectral blocks from log-eigenvalue
  decay, built one average-linkage tree per block, and ran
  `tree.decompose(...)`.
- `kak_mp_tree_method_test.py` built gene-side Gram eigenvectors, used
  Marchenko--Pastur regimes to select signal/bulk/full/adaptive blocks, then
  built average-linkage trees in those KAK/MP coordinates before gate testing.
- `cosine_subspace_tree_sweep.py` was the earlier fixed-band cosine subspace
  sweep with bands such as `2-5`, `6-15`, and `36-80`.
- `kak_signal_adaptive_umap_tree_page.py` was visualization-only: it loaded KAK
  assignments and drew whole-UMAP, sub-UMAP, and tree pages.
- The named historical result folder is absent from the current checkout.
- Restoring the scripts would require replacing historical `alpha_local`
  keyword usage with the current `edge_alpha` API.

## Evidence

- `raw/inbox/historical-kak-spectral-pipelines-20260605.txt` records the git
  history paths, deletion commit, script behavior, and compatibility note.

## Links

- [[mp-projection-dimension-behavior-sweeps-20260605]]
- [[selected-pca-projected-wald-validation]]
- [[mnist-continuous-pca50-run-20260605]]
