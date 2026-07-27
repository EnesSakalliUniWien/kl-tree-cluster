---
title: scRNA Plot Pipeline Audit 2026-06-24
type: analysis
status: draft
updated: 2026-06-24
sources:
  - applications/scrna/plot_pipeline.py
  - applications/scrna/pancreas_benchmark.py
  - applications/scrna/plots/pancreas_all_method_umap_clusters.py
  - applications/scrna/plots/pancreas_readable_umap_clusters.py
  - applications/scrna/plots/pancreas_cluster_radial_trees_ggtree.R
  - applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R
  - applications/scrna/plots/goncalves_progenitor_trees_ggtree.R
  - applications/scrna/analysis/analyze_goncalves_progenitors.py
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/plot_manifest.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/plot_manifest.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_relation_umap_grid_ggtree.png
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_relation_tree_grid_ggtree.png
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_relation_umap_tree_pages_ggtree.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_ggtree_outputs.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored_summary.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_cluster_radial_tree_combo_ggtree_outputs.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_tree_outputs.csv
  - wiki/sources/pancreas-scrna-clustering-benchmark-20260623.md
  - wiki/sources/goncalves-pancreas-progenitor-benchmark-prep-20260624.md
  - wiki/sources/goncalves-tbs-progenitor-analysis-20260624.md
tags:
  - scrna
  - pancreas
  - plotting
  - pipeline
---

# scRNA Plot Pipeline Audit 2026-06-24

## Summary

The rerun does not show a TBS cluster-to-tree assignment bug. The adult
pancreas and Goncalves fetal pancreas UMAP/tree highlighting audits report
`226/226` and `114/114` colored TBS clusters as exact clades, respectively.

The confusing part is the plot surface. Several generated files are duplicate
compatibility aliases or overlapping review views. This is now addressed by
`applications/scrna/plot_pipeline.py`, which defines one plot-generator order
per scRNA dataset and writes `plot_manifest.csv`/`.json` files marking
canonical plots, derivatives, compatibility aliases, and orphaned previous-run
artifacts. The fix is a pipeline/manifest contract, not a change to cluster
assignment.

## Details

### Current plot families

- `applications/scrna/pancreas_benchmark.py` is the core benchmark runner.
  It writes QC, classical UMAP overview, ARI, split/merge, quick method UMAP,
  branch-length, dendrogram, metric, assignment, and tree diagnostic outputs.
- `applications/scrna/plots/pancreas_all_method_umap_clusters.py` writes
  `all_methods_umap_clusters_all_colored.png` and `.pdf`, with all assigned
  clusters colored and only large cluster labels thresholded.
- `applications/scrna/plots/pancreas_readable_umap_clusters.py` writes the standalone
  TBS UMAP comparison and then copies the all-colored rendering to the older
  `tbs_readable_umap_clusters_ge50.*` names.
- `applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R` writes
  `tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.*` and copies it to
  `tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.*` and
  `tbs_umap_cluster_radial_tree_combo_ggtree.*`.
- `applications/scrna/analysis/analyze_goncalves_progenitors.py` writes the Goncalves
  progenitor signature UMAP, population/TBS heatmap, and exact meeting-node
  progenitor plot.
- `applications/scrna/plots/goncalves_progenitor_trees_ggtree.R` writes individual
  progenitor radial trees, a combined tree-only panel, a wide UMAP/tree panel,
  one PNG page per UMAP/tree view, relation UMAP/tree grids that include the
  TBS clustering result, and multi-page vector PDFs.

### Implemented plot contract

`applications/scrna/plot_pipeline.py` is the coherent entry point for plot
review. It has `--dataset pancreas` and `--dataset goncalves` modes, optional
`--run-generators`, `--print-commands`, and `--strict` validation. Without
rerunning generators, it scans the current output folder and writes a plot
manifest.

The adult pancreas manifest records `120` PNG/PDF plot artifacts: `9`
canonical plots, `51` derivatives, `6` aliases, and `54` orphaned previous-run
plots with no matching current `method_assignments.csv` column. This explains
why the folder looked incoherent: older TBS method names such as
`*_edge_dim30_*`, `*_native_nnls_*`, earlier adaptive-diffusion rows, and
superseded q50 action/split-action diagnostic rows still exist beside the
current six TBS methods.

The Goncalves manifest records `74` PNG/PDF plot artifacts: `21` canonical
plots, `41` derivatives, `4` aliases, and `8` orphaned previous-run q50
action/split-action diagnostic plots. These orphaned rows are not canonical
failures; they document superseded diagnostics that no longer have matching
current `method_assignments.csv` columns.

### Ambiguities found

- The three TBS UMAP-plus-tree combo PNGs are byte-identical within each run:
  the scaled, all-clusters, and legacy names are all copies of the same figure.
- In the adult pancreas run, `tbs_readable_umap_clusters_all_colored.png` and
  `tbs_readable_umap_clusters_ge50.png` are byte-identical. The `ge50` name is
  misleading because all clusters are colored; only text labels are restricted
  to clusters with at least `50` cells.
- `method_umap_clusters.png` and
  `all_methods_umap_clusters_all_colored.png` answer nearly the same review
  question. The all-method plot is the clearer canonical comparison because it
  records the all-clusters-colored policy and summary CSV.
- The Goncalves progenitor tree outputs are useful but should not be reviewed
  through a screenshot composite. The corrected relation outputs are generated
  directly from `method_assignments.csv`, progenitor score tables, and the
  adaptive-diffusion topology tree edges; the old stitched composite was
  removed from the pipeline.
- The run-level `manifest.json` records parameters and provenance, and some
  scripts write their own output CSVs, but the plot set previously lacked one
  manifest with `canonical`, `alias_of`, `stage`, and `intended_question`
  fields. The new `plot_manifest.csv` files fill that gap.

### Cluster interpretation

The exact-clade audits mean the plotted cluster colors are not being assigned
to the wrong tree leaves. The biological ambiguity instead comes from mixed
subtrees and processed fetal expression input. For Goncalves, the large
progenitor-rich `C22` group is not a pure progenitor cluster; it is better
treated as a broad mixed progenitor-rich neighborhood. Exact meeting nodes such
as `N2876` and `N2910` remain useful because they identify smaller
monophyletic proliferating-progenitor-rich joins.

### Remaining cleanup

1. Make `all_methods_umap_clusters_all_colored.*` the canonical all-method
   UMAP comparison. Keep `method_umap_clusters.png` only as a quick diagnostic
   or remove it from summary-level output lists.
2. Make `tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.*` the canonical
   TBS UMAP/tree combo. Keep the `all_clusters` and legacy names only as
   compatibility aliases recorded in the manifest.
3. Retire top-level `tbs_readable_umap_clusters_ge50.*` as a peer artifact.
   If compatibility requires the file, record it as an alias of
   `tbs_readable_umap_clusters_all_colored.*` and describe the label threshold
   explicitly.
4. For Goncalves progenitor review, make
   `goncalves_tbs_progenitor_umap_tree_pages_ggtree.pdf` the canonical review
   artifact. Treat individual radial trees, page PNGs, and the tree-only panel
   as derivatives.
5. For Goncalves cross-reference review, make
   `goncalves_tbs_relation_umap_tree_pages_ggtree.pdf`,
   `goncalves_tbs_relation_umap_grid_ggtree.*`, and
   `goncalves_tbs_relation_tree_grid_ggtree.*` the canonical relation
   artifacts.
6. Move adult orphaned previous-run plot files out of the top-level review
   folder, or keep them in place only as raw evidence that downstream readers
   filter through `plot_manifest.csv`.
7. Decide whether compatibility aliases should remain as physical image copies
   or become manifest-only rows after downstream references are updated.

## Evidence

- `applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R` explicitly writes the
  scaled combo, then copies it to the `all_clusters` and legacy combo names.
- `applications/scrna/plots/pancreas_readable_umap_clusters.py` explicitly copies the
  all-colored TBS UMAP rendering to the older `ge50` filenames.
- `applications/scrna/plot_pipeline.py` defines the dataset-specific plot
  generator order and writes the plot manifests.
- `applications/scrna/plots/goncalves_progenitor_trees_ggtree.R` now generates
  direct relation plots from the source CSVs instead of cropping an existing
  method-comparison PNG. The canonical relation outputs are
  `goncalves_tbs_relation_umap_grid_ggtree.*`,
  `goncalves_tbs_relation_tree_grid_ggtree.*`, and
  `goncalves_tbs_relation_umap_tree_pages_ggtree.pdf`.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/plot_manifest.csv`
  records `120` adult pancreas plot artifacts, including `54` orphaned
  previous-run files, `6` aliases, and `9` canonical plots.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/plot_manifest.csv`
  records `74` Goncalves plot artifacts, including `8` orphaned q50
  action/split-action diagnostic files, `4` aliases, and `21` canonical plots.
- Local SHA-256 checks found identical hashes for the three TBS combo PNGs in
  both the adult pancreas and Goncalves output folders, and identical hashes for
  the adult `tbs_readable_umap_clusters_all_colored.png` and
  `tbs_readable_umap_clusters_ge50.png`.
- `python3 -m py_compile applications/scrna/plot_pipeline.py
  applications/scrna/plots/pancreas_readable_umap_clusters.py
  applications/scrna/analysis/analyze_goncalves_progenitors.py` passed.
- `ruff check applications/scrna/plot_pipeline.py
  applications/scrna/plots/pancreas_readable_umap_clusters.py
  applications/scrna/analysis/analyze_goncalves_progenitors.py` passed.
- `python3 applications/scrna/plot_pipeline.py --dataset pancreas --strict` and
  `python3 applications/scrna/plot_pipeline.py --dataset goncalves --strict`
  passed and regenerated `plot_manifest.csv`/`.json` files for both output
  folders.
- Plot regeneration on 2026-06-24 succeeded by prepending
  `/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/bin` to
  `PATH` and running
  `/Users/berksakalli/miniconda3/bin/python
  applications/scrna/plot_pipeline.py --dataset pancreas --run-generators
  --strict` plus the corresponding Goncalves command. The rerender produced
  the expected `ggtree`/`ggplot2` deprecation warnings but no command failures.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_tree_highlighting_audit.csv`
  reports `226/226` colored clusters as exact clades.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_tree_highlighting_audit.csv`
  reports `114/114` colored clusters as exact clades.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_tree_outputs.csv`
  records the individual tree, panel, page, and multi-page PDF outputs plus
  dimensions.

## Links

- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]
- [[goncalves-tbs-progenitor-analysis-20260624]]
- [[benchmark-pipeline-contract]]
- [[redundant-and-legacy-code-map-20260623]]

## Open Questions

- Should compatibility aliases continue to be written as physical image copies,
  or should they move to manifest rows only?
- Should the adult pancreas and Goncalves runs share one scRNA plotting driver,
  or should Goncalves keep a small wrapper around the adult benchmark plot
  helpers because it adds progenitor-specific outputs?
