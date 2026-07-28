# Space Separation

This package owns reusable methodological space separation. It has no
dataset-specific file paths, labels, plotting, or report writing.

- `decompose_invariant_equivariant_space(...)` separates one leading common
  axis from following orthogonal variation.
- `separate_adaptive_cosine_space(...)` weights a feature matrix, builds its
  sample cosine eigensystem, and segments the ordered spectrum into adaptive
  coordinate blocks.
- `hamming_knn_diffusion_geometry(...)`, `adaptive_diffusion_geometry(...)`,
  `block_diffusion_geometry(...)`, and `block_adaptive_diffusion_geometry(...)`
  return one explicit result containing coordinates, condensed distances, and
  construction evidence without importing benchmark runners.

Dataset adapters belong under `applications/`; benchmark evaluation and null
diagnostics belong under `benchmarks/`.
