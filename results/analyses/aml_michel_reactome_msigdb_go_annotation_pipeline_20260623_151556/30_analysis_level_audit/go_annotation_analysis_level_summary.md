# GO Annotation Analysis Level Audit

Canonical target:
- `30_canonical_current_subspace_pipeline`: current TBS gates on adaptive-diffusion cosine subspace trees with connected rankings, manifests, per-subspace artifacts, and method PDFs.

## Levels Found

- `00_matrix_quality_only` (matrix quality only): 1
- `30_canonical_current_subspace_pipeline` (canonical current subspace pipeline): 1
- `90_unclassified_pdf_report` (unclassified PDF/report output): 1

## Inconsistent Levels

### `00_matrix_quality_only`
- Path: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/05_matrix_quality`
- Canonical: `False`
- Interpretation: Feature-matrix diagnostics before any tree or TBS gate is run.
- Inconsistency: Input-quality level only: no tree assignments, GO-IC ranking, or reader-facing tree pages.
- Signals: pdfs=1, assignments=0, axis_term_tables=0, methods=``, tree_geometries=``

### `30_canonical_current_subspace_pipeline`
- Path: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree`
- Canonical: `True`
- Interpretation: Current TBS gates on adaptive-diffusion cosine subspace trees with connected rankings, manifests, per-subspace artifacts, and method PDFs.
- Inconsistency: Canonical target: current method/version and one tree-geometry contract.
- Signals: pdfs=8, assignments=37, axis_term_tables=48, methods=`current`, tree_geometries=`adaptive_diffusion_cosine_subspace`

### `90_unclassified_pdf_report`
- Path: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/00_dataset_inventory`
- Canonical: `False`
- Interpretation: Report-like output that does not match a known allGO contract.
- Inconsistency: Unclassified output level: inspect manually before interpretation.
- Signals: pdfs=0, assignments=0, axis_term_tables=0, methods=``, tree_geometries=``
