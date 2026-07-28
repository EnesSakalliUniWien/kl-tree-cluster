---
title: Tree Construction Method Map
type: analysis
status: reviewed
updated: 2026-07-28
sources:
  - tree_break_selection/tree/README.md
  - tree_break_selection/tree/construction/build.py
  - tree_break_selection/tree/construction/defaults.py
  - tree_break_selection/tree/construction/hierarchical.py
  - tree_break_selection/tree/construction/phylogenetic.py
  - tree_break_selection/tree/optimized_branch_lengths.py
  - tree_break_selection/space_separation/diffusion.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/tree_consensus.py
  - tree_break_selection/hierarchy_analysis/bootstrap_consensus.py
  - applications/endotypes/pipelines/run_feature_matrix_with_umap.py
  - applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py
  - applications/scrna/pancreas_benchmark.py
  - benchmarks/experiments/mnist/run.py
tags:
  - architecture
  - tree
  - topology
  - rooting
  - applications
---

# Tree Construction Method Map

## Summary

The repository has three live registered topology builders: SciPy hierarchical
linkage, scikit-bio neighbor joining, and external IQ-TREE 3. Four main geometry
families feed them: direct or precomputed feature distances, fixed Hamming
nearest-neighbor diffusion, adaptive pydiffmap diffusion, and optional
graphtools kernel diffusion. Application-specific adaptive-cosine subspaces add
block-coordinate geometry but still terminate in average linkage.

Rooting and branch-length fitting are independent seams. Linkage is already
rooted at its final merge; neighbor joining and IQ-TREE are unrooted and are
oriented by minimum ancestor deviation (MAD). Raw linkage ultrametric lengths
and fixed-topology NNLS are alternative edge-length policies on an existing
topology, not additional tree builders. Gate profiles and traversal settings
are further downstream.

The canonical binary distance (`hamming`) and linkage method (`average`) are
immutable defaults owned by `tree/construction/defaults.py`. Alternative geometry and
topology choices are explicit method inputs; there is no mutable package-wide
runtime configuration.

The builders are ordered explicitly as `linkage`, `neighbor_joining`, and
`iqtree3` by `SUPPORTED_TREE_BUILDERS`. Method choice depends on the requested
builder name and validated rooting contract, never on filesystem, directory,
or import-discovery order.

## Details

### Methodological pipeline

The code implements this responsibility flow:

```text
feature representation
  -> geometry or condensed distance
  -> topology builder
  -> root/orientation adapter
  -> PosetTree representation
  -> optional fixed-topology branch-length fit
  -> edge/sibling gates and traversal
  -> clusters and plots
```

The highest-leverage interface is the condensed-distance seam. Direct feature
metrics and all diffusion engines can feed linkage; direct and graphtools
diffusion distances can also feed neighbor joining. IQ-TREE is the exception:
it consumes an encoded feature-state alignment and therefore does not consume
the condensed-distance seam.

### Registered topology routes

| Route | Geometry source | Topology implementation | Rooting | Registration and availability |
| --- | --- | --- | --- | --- |
| Direct linkage | Configured `pdist` metric or case-supplied condensed distance | SciPy `linkage` then `tree_from_linkage` | Final linkage merge (`linkage_root`) | `tbs` average is canonical; `tbs_complete` and `tbs_single` are explicit variants |
| Fixed Hamming diffusion linkage | Binary/one-hot Hamming kNN similarity, symmetric diffusion coordinates, Euclidean diffusion distance | Average linkage | `linkage_root` | Canonical `tbs_diffusion` |
| Adaptive pydiffmap linkage | Variable-bandwidth pydiffmap coordinates and Euclidean diffusion distance | Average linkage | `linkage_root` | `tbs_diffusion_adaptive`, with either linkage-ultrametric or fixed-topology NNLS lengths |
| Graphtools diffusion linkage | Optional graphtools kernel, fixed or fragmentation-guard adaptive K, then Euclidean diffusion distance | Average by default; adaptive-K grid also exposes complete, weighted, single, centroid, median, and Ward | `linkage_root` | Optional GPL methods |
| Distance neighbor joining | Configured direct or graphtools diffusion condensed distance | scikit-bio `nj` | MAD | `tbs_neighbor_joining` and the eighth adaptive-K grid route |
| IQ-TREE 3 | Per-feature categorical states encoded as an alignment | External IQ-TREE 3, default `JC2`, followed by Newick import | MAD | Opt-in `tbs_iqtree3`; external executable required |

Centroid and median linkage may return nonmonotone merge heights. When
fixed-topology NNLS is active, the runner preserves their merge topology with
placeholder edge lengths and replaces those lengths through NNLS before
branch-time use. It does not silently reinterpret nonmonotone linkage heights
as valid ultrametric time.

### Application routes

| Application | Representation and distance | Topology | Ownership note |
| --- | --- | --- | --- |
| Endotype/GO feature matrix | Configurable direct feature distance; fixed Hamming diffusion; adaptive pydiffmap diffusion | Configurable direct linkage, or average linkage for both diffusion routes | Dataset orchestration is in `applications/endotypes/`; reusable adaptive diffusion and adaptive-cosine geometry is in `tree_break_selection/space_separation/` |
| Paper endotype baseline | Cosine distance | Complete linkage followed by a fixed flat cut | Comparison baseline, not a TBS tree variant |
| Adaptive-cosine endotype blocks | Euclidean block coordinates or adaptive block-diffusion distance | Average linkage | Space separation is reusable; report/tree-page composition remains application-owned |
| Adult and fetal-pancreas scRNA | Standardized-PCA Euclidean distance or adaptive-diffusion distance | Average linkage | Topology-only, raw-linkage branch-time, and NNLS branch-time rows reuse the same topology for a given geometry |
| MNIST benchmark and reports | Benchmark generators support thresholded binary image features with configurable distance/linkage (Roger--Stanimoto plus average linkage by default); retained alpha-sweep reports use continuous PCA50 features with Euclidean distance | Configurable benchmark linkage; average linkage in the retained PCA50 report | `benchmarks/experiments/mnist/` owns evaluation; `applications/mnist/` reconstructs the geometry recorded by each retained result rather than imposing one universal MNIST tree |

### Construction package and removed dormant surfaces

`tree_break_selection/tree/construction/` now owns the complete construction
responsibility by category. `build.py` validates and dispatches the three live
methods and returns their topology, rooting, linkage, fallback, and IQ-TREE
evidence. `hierarchical.py` owns merge-array promotion and the topology-only
fallback. `phylogenetic.py` owns neighbor joining, IQ-TREE/Newick, and MAD
rooting. The unused sklearn-agglomerative and arbitrary edge-list adapters were
deleted, along with the shallow `PosetTree.from_*` facades. No compatibility
modules or aliases remain at the old paths.

### What is not a separate tree-construction method

- Fixed-topology NNLS changes only edge lengths; it does not change topology.
- Linkage-ultrametric branch lengths are merge-height-derived diagnostics, not
  a separate topology.
- Gate profiles, sibling multiple-testing rules, internal spectral filters,
  passthrough rules, and traversal settings consume a tree after construction.
- `benchmarks/shared/tree_consensus.py` ranks and selects among eight completed
  candidate clusterings. It does not merge them into a consensus topology.
- Bootstrap consensus repeatedly invokes direct distance plus linkage to
  estimate stability. It is a resampling analysis, not a fourth builder.
- BranchArchitect consumes exported Newick trees for RF-distance and optional
  interpolation diagnostics. It is not a production topology estimator here.
- Plotting engines either consume saved trees or reconstruct a known selected
  linkage for display; they do not define a new inference route.

### Redundancy and locality findings

The three registered topology algorithms each have one implementation. The
main runner now crosses one deep construction interface that owns validation,
topology-only fallback, and construction metadata. Direct `tree_from_linkage`
calls remain only where applications or analyses already own a prepared
linkage matrix; they share representation conversion without hiding their
distinct geometry preparation or output contracts.

The `tree_linkage_method` field remains present in neighbor-joining and
IQ-TREE method configurations. It does not control those topology builders,
although the runner can reuse the value for linkage-based gate replay defaults.
That dual meaning should remain explicit in future configuration cleanup.

## Evidence

- `benchmarks/shared/runners/tbs_runner.py` contains the single live dispatch
  over `linkage`, `neighbor_joining`, and `iqtree3`, plus the nonmonotone-linkage
  topology fallback and fixed-topology NNLS call.
- `benchmarks/shared/runners/method_registry.py` registers the canonical tree
  variants and the seven-linkage-plus-neighbor-joining adaptive-K grid.
- `tree_break_selection/tree/construction/phylogenetic.py` contains the only neighbor-joining,
  IQ-TREE/Newick, and MAD-rooting implementations.
- `tree_break_selection/space_separation/diffusion.py` owns reusable fixed
  Hamming-neighbor, adaptive, and block diffusion geometry; the benchmark
  runner owns graphtools-specific adaptive-neighbor policy and method dispatch.
- Exact repository search found the application, bootstrap, diagnostic, and
  plot-only direct-linkage call sites summarized above, and no live internal
  callers for the two dormant public adapters.
- Tests cover linkage representations, topology-only fallback, phylogenetic
  builders/rooting, diffusion runners, registry dispatch, consensus selection,
  application contracts, and plotting reconstruction.

## Links

- [[phylogenetic-tree-builders-20260614]]
- [[graphtools-adaptive-k-tree-inference-focus-benchmark-20260630]]
- [[scrna-branch-length-effect-audit-20260624]]
- [[benchmark-pipeline-contract]]
- [[edge-gate-distance-time-contract-20260623]]
- [[method-application-and-plot-seams-20260727]]
- [[redundant-and-legacy-code-map-20260623]]

## Open Questions

- Should the configuration split builder-specific fields from gate-replay
  linkage defaults so neighbor-joining and IQ-TREE runs do not appear to use
  `tree_linkage_method` for their primary topology?
- Should visualization-only tree reconstruction be replaced by persisted
  serialized `PosetTree`/Newick artifacts where exact inference replay is
  required?
