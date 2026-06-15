# Julia Tree-Estimator Analysis Debug Report

Date: 2026-06-14

## Scope

This report checks the full UMAP, clustering diagnostics, and post-hoc alpha
audit generated under `raw/assets/benchmark-results/julia_tree_estimators_20260614/`.

## Checks

- The interactive long table has `2109` rows, exactly `703` genes for each of
  the three methods.
- There are no duplicate `(method, gene_symbol)` rows in the interactive long
  table.
- The UMAP coordinate table has `703` unique genes, and each assignment table
  has `703` unique genes.
- The stored `cluster_size` columns match recomputed per-cluster sizes for all
  three assignment files.
- The default post-hoc alpha rows reproduce the saved default cluster counts:
  `670` for baseline KL and `494` for neighbor joining plus MAD.

## Findings

1. The original interactive HTML,
   `umap_plots/julia_full_umap_interactive_methods.html`, references Plotly
   from `https://cdn.plot.ly/plotly-2.35.2.min.js`. This can fail in offline or
   external-script-restricted browser contexts.
2. A standalone interactive HTML was generated:
   `umap_plots/julia_full_umap_interactive_methods_standalone.html`. It embeds
   Plotly in the file and has no external `<script src=...>` dependency.
3. The alpha audit is a post-hoc sensitivity check, not a fresh calibrated
   alpha sweep. It retresholds saved child-parent and sibling q-value columns
   from the default run, so it does not recompute TreeBH ancestor testing,
   sibling-pair collection, empirical-null inflation, or selected statistics at
   each alpha.
4. Despite that limitation, the q-value distributions explain the observed
   behavior. For baseline KL, `1374/1392` child-parent edge q-values are
   already at most `1e-8`, and `1384/1392` are at most `0.001`. For neighbor
   joining, `1344/1392` edge q-values are at most `1e-8`, and `1382/1392` are
   at most `0.001`. Edge alpha is therefore nearly saturated in this audit.
5. Sibling q-values are the main driver. Baseline KL has `619/692` sibling
   q-values at most `1e-6` and `669/692` at most `0.01`. Neighbor joining has
   only `104/691` at most `1e-6`, `297/691` at most `0.01`, and `412/691` at
   most `0.1`.

## Interpretation

The scattered UMAP clusters are not caused by missing points, duplicate rows,
or incorrect cluster-size columns. They come from the combination of:

- a high-dimensional selected tree/gate geometry that is not equivalent to the
  two-dimensional UMAP neighborhood geometry,
- nearly saturated edge decisions,
- sibling alpha controlling a fragmentation-versus-breadth tradeoff, and
- passthrough traversal allowing descendant splits to survive below closed
  sibling ancestors.

Making sibling alpha stricter can merge many neighbor-joining fragments, but
the merged clusters become broader on the UMAP. Making sibling alpha looser
keeps clusters more local only by returning to many tiny or singleton clusters.

## Actionable Follow-Up

- Use the standalone HTML for local review when browser/network restrictions
  are possible.
- Treat `alpha_audit/posthoc_alpha_sensitivity_summary.csv` as a diagnostic
  sensitivity artifact, not as production calibration evidence.
- A production-grade alpha sweep must rerun the full gate pipeline for each
  alpha pair or add a dedicated runner that preserves all alpha-dependent
  TreeBH, sibling-collection, inflation, and traversal contracts.
