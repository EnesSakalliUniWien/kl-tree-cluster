---
title: Goncalves TBS Progenitor Analysis 2026-06-24
type: source
status: reviewed
updated: 2026-06-24
sources:
  - applications/scrna/analysis/analyze_goncalves_progenitors.py
  - applications/scrna/plots/goncalves_progenitor_trees_ggtree.R
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_population_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_cluster_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_inner_node_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_monophyletic_meeting_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_tree_outputs.csv
tags:
  - benchmark
  - pancreas
  - progenitor
  - scrna
---

# Goncalves TBS Progenitor Analysis 2026-06-24

## Summary

The Goncalves fetal pancreas progenitor analysis maps marker symbols to Ensembl
genes, scores fetal progenitor and context signatures on the processed/scaled
UCSC expression matrix, and compares population labels, adaptive-diffusion TBS
clusters, internal tree nodes, and exact monophyletic meeting nodes. Unlike the
adult pancreas subset, this dataset contains clear trunk, tip, proliferating,
and endocrine/endocrine-progenitor-like reference states.

## Key Points

- `37/39` requested markers mapped to raw AnnData genes. `GCG` and `PPY` were
  absent, so mature endocrine scoring uses `CHGA`, `INS`, and `SST`.
- Population-level signatures match the fetal labels: `tip` has the strongest
  tip-progenitor score, `trunk` has the strongest trunk-progenitor score among
  large epithelial progenitor labels, `proliferating` has the strongest
  proliferation signal, and `endocrine` has high endocrine-progenitor-core,
  endocrine-commitment, and mature-endocrine scores.
- Adaptive-diffusion TBS cluster `C6` is a pure tip-progenitor cluster
  (`31/31` tip; tip score `2.1351`).
- Cluster `C1` is a pure proliferating cluster (`19/19` proliferating;
  proliferation score `0.5741`), while `C2` and `C3` are small proliferating
  mixtures.
- Cluster `C10` is a compact mixed fetal progenitor cluster (`28` cells,
  `92.9%` trunk/tip/proliferating).
- Cluster `C22` is the large mixed progenitor-rich cluster (`683` cells,
  `83.9%` trunk/tip/proliferating) but includes mesenchyme, blood, neurons, and
  endocrine cells, so it is a broad fetal progenitor-rich neighborhood rather
  than a pure progenitor state.
- Cluster `C12` is the compact endocrine/endocrine-progenitor-like cluster
  (`19/19` endocrine; endocrine-progenitor-core score `2.0451`, endocrine
  commitment score `2.9916`, mature endocrine score `3.8332`).
- Within final clusters, many internal nodes are pure or nearly pure fetal
  progenitor groupings. Examples include `N2851` (`31` tip cells, tip score
  `2.1351`), `N2903` (`27` trunk and `27` proliferating cells), and `N2758`
  (`42` tip, `27` trunk, `1` proliferating).
- Exact monophyletic meeting node `N2876` joins `C1` and `C2` into a
  `27`-cell proliferating-progenitor-rich meeting (`96.3%` progenitor).
- Exact monophyletic meeting node `N2910` joins `C1`, `C2`, and `C3` into a
  `47`-cell proliferating arm (`87.2%` progenitor).
- Exact monophyletic meeting node `N2907` joins a small mesenchyme subtree
  (`C19-C21`) to the large progenitor-rich `C22`, so it is best interpreted as
  a progenitor/mesenchyme neighborhood join, not a pure progenitor ancestor.
- `applications/scrna/plots/goncalves_progenitor_trees_ggtree.R` renders the whole
  `1,465`-tip adaptive-diffusion TBS tree as three progenitor-specific radial
  trees: dominant fetal population, trunk/tip/proliferating fraction, and
  interpreted progenitor state. The plots label `N2851`, `N2903`, `N2758`,
  `N2876`, `N2910`, `N2907`, and `N2872`.
- The same plotting script is the consolidated Goncalves progenitor UMAP/tree
  plotting line: `Rscript applications/scrna/plots/goncalves_progenitor_trees_ggtree.R
  --pdf-width=30 --pdf-height=42 --page-width=30 --page-height=16
  --png-dpi=180`. It writes screen-readable PNG previews, a row-wise vector PDF,
  and a separate multi-page PDF with one UMAP/tree pair per page; UMAP cells are
  rendered with larger points and centered titles.

## Evidence

- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_progenitor_signature_umap.png`
  shows trunk, tip, proliferation, endocrine-progenitor-core, and endocrine
  commitment scores on the UMAP.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_population_tbs_cluster_signature_heatmap.png`
  compares reference populations and TBS clusters across marker signatures.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_monophyletic_meeting_progenitor_plot.png`
  summarizes exact monophyletic meeting nodes by progenitor population fraction.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_fraction_radial_tree_ggtree.png`
  colors the whole radial TBS tree by trunk/tip/proliferating descendant
  fraction.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_tree_panel_ggtree.png`
  combines the dominant-population, progenitor-fraction, and progenitor-state
  tree views.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_umap_tree_panel_wide_ggtree.pdf`
  is the `30` by `42` inch vector PDF with one UMAP/tree pair per row.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_progenitor_umap_tree_pages_ggtree.pdf`
  is the `30` by `16` inch-per-page vector PDF with the population,
  progenitor-fraction, and progenitor-state UMAP/tree views on separate pages.
- `ruff check applications/scrna/analysis/analyze_goncalves_progenitors.py` and
  `python -m py_compile applications/scrna/analysis/analyze_goncalves_progenitors.py` passed.
- `Rscript -e "parse('applications/scrna/plots/goncalves_progenitor_trees_ggtree.R')"`
  passed, and the plotting script regenerated
  `goncalves_tbs_progenitor_tree_outputs.csv`.

## Links

- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]
- [[pancreas-progenitor-dataset-selection-20260624]]
