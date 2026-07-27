---
title: GO Annotation Feature-Matrix Pipeline
type: tool
status: reviewed
updated: 2026-06-23
sources:
  - applications/endotypes/run_go_annotation_feature_matrix_pipeline.py
  - applications/endotypes/inventory_go_annotation_datasets.py
  - applications/endotypes/build_subspace_gene_annotation_pdf.py
  - applications/endotypes/export_subspace_cluster_rosters.py
  - applications/endotypes/audit_go_cluster_meaningfulness.py
  - scripts/rest_request.py
  - applications/endotypes/audit_go_annotation_analysis_levels.py
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - data/feature_matrices/feature_matrix_allGO_new_interactome.tsv
  - results/analyses/go_annotation_dataset_inventory_20260619/go_annotation_dataset_inventory.csv
  - results/analyses/go_annotation_dataset_inventory_20260619/go_annotation_directory_naming_summary.md
  - results/analyses/go_annotation_analysis_level_audit_20260619/go_annotation_analysis_level_inventory.csv
  - results/analyses/go_annotation_analysis_level_audit_20260619/go_annotation_analysis_level_summary.md
  - results/analyses/julia_allgo_new_go_annotation_pipeline_dryrun_20260619/pipeline_manifest.json
  - results/analyses/julia_allgo_new_go_annotation_pipeline_dryrun_20260619/PIPELINE.md
  - results/analyses/allgo_new_interactome_go_annotation_pipeline_dryrun_20260619/pipeline_manifest.json
  - results/analyses/allgo_new_interactome_go_annotation_pipeline_dryrun_20260619/PIPELINE.md
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/julia_allgo_new_systematic_subspace_gene_annotation_report.pdf
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/subspace_cluster_roster.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/subspace_cluster_status.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/meaningfulness_report.md
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/subspace_meaningfulness_summary.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/assignment_source_meaningfulness_summary.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/quickgo_top_terms_summary.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/allgo_new_interactome_systematic_subspace_gene_annotation_report.pdf
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/subspace_cluster_roster.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/subspace_cluster_status.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/meaningfulness_report.md
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/subspace_meaningfulness_summary.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/assignment_source_meaningfulness_summary.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/quickgo_top_terms_summary.csv
tags:
  - pipeline
  - allgo
  - go-annotation
  - audit
---

# GO Annotation Feature-Matrix Pipeline

## Summary

Use `applications/endotypes/run_go_annotation_feature_matrix_pipeline.py` as the
consistent entry point for GO annotation feature-matrix analyses. The active
datasets are `feature_matrix_julia_allGO_new.tsv` and
`feature_matrix_allGO_new_interactome.tsv`, both under
`data/feature_matrices/`. The canonical reader-facing method level is the
current adaptive-diffusion cosine-subspace tree pipeline; older quality-only,
candidate-generation, and mixed-method GO-IC PDF outputs should be treated as
different analysis levels rather than competing final reports.

## Usage

Plan or run the canonical pipeline with an explicit feature matrix:

```bash
MPLBACKEND=Agg python applications/endotypes/run_go_annotation_feature_matrix_pipeline.py \
  --input data/feature_matrices/feature_matrix_julia_allGO_new.tsv
```

Use `--dry-run` to write `PIPELINE.md`, per-stage command logs, and
`pipeline_manifest.json` without recomputing the expensive tree stages. The
wrapper is the source of truth for publishable GO annotation runs: by default
it now continues past the current subspace tree stage into radial-tree export,
systematic annotation PDF construction, cluster meaningfulness audit, input
copying, recurring-annotation summaries, and `RUN_SUMMARY.md`. Use
`--skip-subspace-package` only for low-level debugging runs where the
publishable package is intentionally not required. Use `--include-method-matrix`
only when the legacy/current by tree-geometry matrix is needed as an audit
stage; it is not the canonical reader PDF level.

The standard stages are:

- `00_dataset_inventory`: feature-matrix dataset inventory, duplicate check,
  and result-root naming audit.
- `05_matrix_quality`: input-matrix quality diagnostics and PDF.
- `10_current_adaptive_diffusion_subspace_tree`: canonical current-method
  adaptive-diffusion cosine-subspace trees, rankings, method PDF, manifests,
  axis term-loading files, and per-subspace artifacts.
- `20_method_tree_matrix_audit`: optional candidate-generation matrix across
  legacy/current and tree-geometry axes.
- `40_subspace_rosters_radial_trees`: organized per-subspace cluster rosters,
  copied source artifacts, full-space/subspace/tree-distance embeddings, and
  radial cluster trees.
- `45_subspace_annotation_pdf`: systematic two-page-per-subspace annotation
  PDF built from the organized radial-tree package.
- `50_cluster_meaningfulness_audit`: internal feature-enrichment and
  eigenband-coherence audit against size-preserving null partitions.
- `30_analysis_level_audit`: inventory that labels old and new output folders
  by analysis level.
- `90_package_for_github`: input-matrix copy, radial-output verification,
  recurring top-annotation tables, and `RUN_SUMMARY.md` for branch upload.

The 2026-06-19 audit found five relevant levels among the pre-existing allGO
outputs:

- `00_matrix_quality_only`: one input-quality PDF/report level.
- `10_candidate_tree_generation`: four candidate tree-generation folders.
- `20_mixed_method_go_ic_reader_report`: one mixed-method GO-IC PDF folder.
- `30_canonical_current_subspace_pipeline`: three canonical current pipeline
  folders.
- `90_unclassified_pdf_report`: two legacy or incomplete folders requiring
  manual inspection before interpretation.

The canonical level fixes the inconsistent axes to `method_version=current`
and `tree_geometry=adaptive_diffusion_cosine_subspace`. Mixed-method and
legacy/candidate outputs remain useful audits, but they should not be compared
as the same level as the current pipeline PDFs.

The 2026-06-19 dataset inventory recorded canonical matrices, Downloads
copies, and one root-level duplicate. All inspected matrices were binary and
had no missing values, zero rows, or zero columns. The
two active canonical datasets are:

- `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`: `602` genes by
  `6368` GO features.
- `data/feature_matrices/feature_matrix_allGO_new_interactome.tsv`: `339`
  genes by `5873` GO features.

The Downloads copies match these canonical files by SHA-256. The previously
observed root-level `feature_matrix_julia_allGO_new (1).tsv` copy was
byte-identical to the Julia matrix but non-canonical; it should not be used as
an input path.

Systematic subspace gene-annotation outputs are built by the wrapper through
`applications/endotypes/export_subspace_cluster_rosters.py` and
`applications/endotypes/build_subspace_gene_annotation_pdf.py`. These scripts remain
directly runnable for repair/debugging, but ordinary analysis runs should enter
through `applications/endotypes/run_go_annotation_feature_matrix_pipeline.py` so the
radial trees, PDFs, audit tables, input copy, and upload summary stay in one
contract. The roster exporter creates a
`subspaces/rank##_weighting_block_name/` directory for every
accepted subspace and a `subspaces/failed##_weighting_block_name/` directory
for every failed-gate subspace with saved linkage evidence. Accepted rows keep
`assignment_source=accepted_tbs`; failed-gate rows get an explicit
`assignment_source=diagnostic_linkage_cut` generated from the saved linkage
tree and should not be described as final accepted TBS output. Each directory
copies the source assignment/coherence/loading artifacts, writes complete
cluster rosters and gene-membership tables, renders darker radial trees colored
by cluster id, and renders full feature-space, subspace, and tree-distance
embeddings colored by the same subspace cluster assignments. The PDF builder
reads the canonical current subspace experiment, orders accepted subspaces by
`specificity_aware_rank`, appends failed-gate diagnostic subspaces, and writes
one cover page followed by two pages per subspace: first a plot page, then an
annotation page.

Each subspace plot page shows the radial tree, subspace embedding,
full feature-space embedding, tree-distance embedding, and combined axis
GO-term loading heatmap. The following annotation page lists top coherent
clusters, representative genes, local GO annotations, QuickGO term definitions,
and UniProt reviewed human protein labels where available.

The 2026-06-19 reports are:

- `julia_allgo_new_systematic_subspace_gene_annotation_report.pdf`: `31`
  pages, with `15` subspace plot/annotation pairs (`14` accepted TBS and `1`
  diagnostic linkage cut) and `120` top cluster summaries. The complete roster
  contains `2828` cluster rows and `9030` gene-subspace memberships. All
  subspace trees have `602` leaves and `602` unique genes.
- `allgo_new_interactome_systematic_subspace_gene_annotation_report.pdf`: `25`
  pages, with `12` subspace plot/annotation pairs (`7` accepted TBS and `5`
  diagnostic linkage cuts) and `96` top cluster summaries. The complete roster
  contains `920` cluster rows and `4068` gene-subspace memberships. All
  subspace trees have `339` leaves and `339` unique genes.

Both reports used `100` QuickGO term records and `80` UniProt gene/protein
lookups with no external lookup warnings; `79/80` representative genes were
resolved to UniProt protein metadata in each report. Non-protein or unresolved
symbols remain shown as local gene symbols.

The 2026-06-19 cluster meaningfulness audit uses
`applications/endotypes/audit_go_cluster_meaningfulness.py` as an internal
GO-coherence check. For each cluster it tests GO feature enrichment against the
full feature matrix with a one-sided hypergeometric test and BH correction, then
compares each eigenband to `30` random partitions preserving its cluster-size
distribution. This is not an external validation because the clusters are
formed from GO features, but it does test whether the discovered groups are
more coherent than size-matched random groupings.

For the Julia allGO-new matrix, accepted TBS subspaces have mean strong-cluster
fraction `0.776` and mean enriched-cluster fraction `0.940`; the diagnostic
linkage-cut eigenband has strong-cluster fraction `0.882` and enriched-cluster
fraction `1.000`. At the stricter eigenband level, only `2/14` accepted TBS
eigenbands are called `coherent`, `7/14` are above the null mean but not above
the null 95th percentile for enrichment, `4/14` are null-like or weak, and
`1/14` has too few tested clusters. The diagnostic failed-gate eigenband is
above the null mean but not coherent by the strict null-95 rule. Cluster-level
counts are `506` strong, `138` moderate, `8` weak or not enriched, `10`
untested, and `2166` too small for the audit threshold.

For the interactome matrix, accepted TBS subspaces have mean strong-cluster
fraction `0.787` and mean enriched-cluster fraction `1.000`; diagnostic
linkage-cut subspaces have mean strong-cluster fraction `0.713` and mean
enriched-cluster fraction `0.991`. At the eigenband level, `3/7` accepted TBS
eigenbands are coherent, `2/7` are above-null-mean but not strong, and `2/7`
are null-like or weak. The diagnostic linkage-cut eigenbands are not accepted
TBS output: `4/5` are above-null-mean but not strong and `1/5` is null-like or
weak. Cluster-level counts are `384` strong, `153` moderate, `1` weak or not
enriched, `1` untested, and `381` too small for the audit threshold.

QuickGO rechecks of the top audit terms found current ontology records for all
`21` interactome top terms. For the Julia top terms, `4/27` checked top terms
are obsolete (`GO:0031063`, `GO:0035521`, `GO:0048017`, `GO:0051091`), so those
specific labels should be treated cautiously even when their member genes are
internally enriched.

## Evidence

- `applications/endotypes/run_go_annotation_feature_matrix_pipeline.py` writes the
  canonical stage commands, manifest, and dry-run plan.
- `applications/endotypes/inventory_go_annotation_datasets.py` writes dataset,
  duplicate, result-root, and naming-convention inventories.
- `applications/endotypes/build_subspace_gene_annotation_pdf.py` writes systematic
  ranked two-page-per-subspace PDFs and accompanying cluster/gene
  interpretation CSVs.
- `applications/endotypes/export_subspace_cluster_rosters.py` writes the complete
  per-subspace directory tree, radial tree plots, full-space, subspace, and
  tree-distance cluster embedding plots, complete cluster rosters, diagnostic
  failed-gate linkage-cut rosters, and long gene-membership tables.
- `applications/endotypes/audit_go_cluster_meaningfulness.py` writes internal
  cluster meaningfulness calls, eigenband coherence calls, size-preserving null
  partition summaries, and top/weak cluster examples.
- `scripts/rest_request.py` is the compact REST helper used for QuickGO and
  UniProt requests following the life-science research plugin skill contract.
- `applications/endotypes/audit_go_annotation_analysis_levels.py` classifies existing
  result roots by file signals such as matrix-quality summaries, assignment
  files, GO-IC rankings, method-split PDFs, connected manifests, subspace
  directories, and axis term-loading tables.
- `results/analyses/go_annotation_dataset_inventory_20260619/` records the
  dataset and directory naming inventory for the active Downloads datasets and
  canonical feature-matrix directory.
- `results/analyses/go_annotation_analysis_level_audit_20260619/` records the
  11-path inventory over existing allGO result roots.
- `results/analyses/julia_allgo_new_go_annotation_pipeline_dryrun_20260619/`
  records a dry-run plan for the Julia allGO-new matrix.
- `results/analyses/allgo_new_interactome_go_annotation_pipeline_dryrun_20260619/`
  records a dry-run plan for the allGO-new interactome matrix.
- `systematic_subspace_gene_annotations/` under each current subspace
  experiment stores the PDF, `subspace_cluster_gene_annotation_summary.csv`,
  complete `subspace_cluster_roster.csv`,
  `subspace_gene_membership_long.csv`, `subspace_cluster_status.csv`,
  `full_space_embedding_coordinates.csv`, QuickGO and UniProt interpretation
  CSVs, `life_science_lookup_cache.json`, and one organized directory per
  subspace under `subspaces/`.

## Links

- [[current-adaptive-diffusion-subspace-tree-pipeline]]
- [[julia-allgo-new-feature-matrix-quality-20260618]]
- [[julia-allgo-new-method-version-tree-matrix-20260618]]
- [[julia-allgo-new-go-ic-tree-summary-plots-20260618]]
- [[julia-allgo-new-current-adaptive-diffusion-subspace-tree-20260618]]
