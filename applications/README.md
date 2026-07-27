# Applications

Applications adapt datasets, labels, reference annotations, and output formats
to reusable method/plotting interfaces in `tree_break_selection/` and public
benchmark orchestration contracts in `benchmarks/shared/`.

| Application | Primary entrypoint | Purpose |
| --- | --- | --- |
| Endotypes / GO annotation | `python applications/endotypes/run_go_annotation_feature_matrix_pipeline.py --help` | Feature-matrix quality, adaptive cosine subspaces, endotype/reference comparison, annotation, and report packaging |
| Endotype KAK geometry | `python applications/endotypes/kak_signal_adaptive_umap_tree_page.py --help` | KAK/cosine geometry and tree pages from matrix-probe outputs |
| Pancreas scRNA | `python applications/scrna/pancreas_benchmark.py --help` | Adult pancreas benchmark and TBS application |
| Goncalves fetal pancreas | `python applications/scrna/goncalves_benchmark.py --help` | Fetal pancreas/endocrine lineage application |
| scRNA space separation | `python applications/scrna/analyze_space_decomposition.py --help` | Invariant/equivariant diagnostic through the main method interface |
| scRNA plot suite | `python applications/scrna/plot_pipeline.py --help` | Dataset-specific plot orchestration and manifest |
| scRNA follow-up analyses | `applications/scrna/analysis/` | Branch-length/action audits and pancreas progenitor comparisons |
| scRNA report generators | `applications/scrna/plots/` | Dataset-specific Python/R report and figure composition |
| MNIST figures | `python applications/mnist/plot_interactive.py` | Interactive digit/tree inspection using saved benchmark outputs |

Benchmark cases, calibration panels, and method-comparison experiments remain
under `benchmarks/`; they validate the method rather than adapt one application.
Reusable separation belongs in `tree_break_selection/space_separation/` and
reusable plot construction belongs in `tree_break_selection/plot/`.

## Tree construction by application

| Application surface | Tree geometry and topology |
| --- | --- |
| Endotype feature-matrix runner | Direct configurable distance/linkage, fixed Hamming diffusion + average linkage, or adaptive diffusion + average linkage; the paper baseline is cosine + complete linkage |
| Endotype adaptive-cosine subspaces | Euclidean block distance or adaptive block-diffusion distance + average linkage |
| Adult and fetal-pancreas scRNA | Standardized-PCA Euclidean or adaptive-diffusion distance + average linkage; raw-linkage and NNLS rows share topology |
| MNIST benchmark and reports | Benchmark generators support binarized feature distance + configurable linkage; retained alpha-sweep reports reconstruct continuous PCA50 Euclidean + average-linkage trees recorded by those results |

The topology-builder, rooting, and branch-length boundaries are mapped in
`tree_break_selection/tree/README.md`; application plots and reports are not
additional tree estimators.
