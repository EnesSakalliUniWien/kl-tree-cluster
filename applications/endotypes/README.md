# Endotype and GO-Annotation Application

This application groups all user-facing feature-matrix, GO-annotation,
subspace, endotype/reference, and report commands.

Start with the orchestrator:

```bash
python applications/endotypes/pipelines/run_go_annotation_feature_matrix_pipeline.py --help
```

The canonical adaptive-cosine subspace application is
`pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py`. It consumes the
main `tree_break_selection.space_separation` interface; it does not import the
spectral method from a benchmark diagnostic.

Responsibilities are explicit:

- `pipelines/` owns executable workflows and orchestration.
- `analysis/` owns data-quality, correctness, and biological interpretation.
- `reports/` owns tabular and PDF artifact export.
- `plots/` owns endotype-specific visualization commands.
- `_shared.py` is the single internal seam for matrix naming, loading, reference
  endotype parsing, and symbol-to-Entrez resolution shared across those command
  categories.

Reference endotype tables belong in `data/reference/`. Canonical input feature
matrices belong in `data/feature_matrices/`.

The `plots/kak_signal_adaptive_umap_tree_page.py` command assembles KAK/cosine
geometry and tree pages from matrix-probe outputs. Application commands may
compose domain reports; reusable separation and plotting primitives belong in
`tree_break_selection/`.
