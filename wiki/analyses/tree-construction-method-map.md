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

### What happens after diffusion

The diffusion routes do not infer a tree directly. They return Euclidean
distances between diffusion coordinates, and the runner passes that condensed
distance vector into the ordinary construction interface:

```text
diffusion coordinates
  -> Euclidean condensed distance
  -> build_tree(...)
  -> SciPy linkage (average by default) or neighbor joining
  -> rooted PosetTree
  -> optional fixed-topology NNLS edge-length replacement
  -> bottom-up typed subtree distributions
  -> gate annotation
  -> depth-first top-down split/pass-through/boundary traversal
```

For linkage, every SciPy merge becomes one binary internal node. The last
merge is the root. Raw linkage edge lengths are parent height minus child
height, divided by the final merge height, so each positive-height
root-to-leaf path initially sums to one. These values are relative
ultrametric heights, not learned evolutionary or causal time.

Fixed-topology NNLS leaves every parent-child relationship unchanged. It fits
non-negative edge lengths so tree path distances approximate squared
standardized distances in the original feature matrix, or squared Euclidean
distances in an explicitly supplied aligned embedding. This changes the edge
semantics and no longer guarantees an ultrametric tree. The old linkage
lengths are retained as edge evidence.

Only after topology and edge lengths are fixed does `FeatureSpace` enter the
tree statistics. Leaves receive their raw typed coordinates; internal nodes
receive leaf-count-weighted subtree barycenters and family-aware covariance
state. The gate pipeline annotates child-parent and sibling evidence. The
iterative depth-first traversal starts at the root and either splits, passes
through, or declares the whole descendant leaf set a final cluster boundary.

This ordering exposes five important method contracts:

1. The topology builder does not see whether the data are Bernoulli,
   categorical, or continuous. Family-specific behavior must already be
   represented faithfully in the geometry delivered at the condensed-distance
   seam.
2. Average linkage is a hierarchical summary of diffusion distance, not a
   diffusion-native tree estimator. Its root is the last algorithmic merge,
   not a biological origin or trajectory direction.
3. Diffusion distance is not automatically additive tree distance.
   Neighbor joining plus MAD rooting is therefore a comparator whose
   additivity and rooting diagnostics must be checked, not a generally valid
   replacement for linkage.
4. TBS traversal requires exactly two children at a visited internal node.
   Linkage guarantees that shape, but tied or duplicate-heavy discrete data
   can receive arbitrary binary resolutions. Phylogenetic rooting can also
   return an existing degree-three node when the optimum lies at an edge
   endpoint; construction currently has no shared binary postcondition check,
   and traversal then stops at that non-binary boundary.
5. NNLS can create a dual-geometry method: diffusion chooses topology while
   original standardized features choose branch time. That is valid only when
   reported explicitly and when topology/branch-target coherence is audited.

The three feature families therefore require separate post-diffusion evidence,
even when they share the construction interface:

| Feature family | Post-diffusion tree interpretation | Required check |
| --- | --- | --- |
| Bernoulli | Binary-metric diffusion followed by a candidate hierarchy | Duplicate-state/tie burden, row-order and resampling topology stability |
| Categorical | Block/simplex diffusion followed by a candidate hierarchy | Category-block preservation and arbitrary binary resolution of equal states |
| Continuous | Manifold diffusion followed by average, weighted, complete, or Euclidean-coordinate Ward candidates | Topology stability; Ward only when the supplied distance is Euclidean |

The executed seven-case graphtools adaptive-K NNLS focus panel does not support
one global topology rule. Its label-free selector chose complete once, average
once, Ward once, neighbor joining once, and weighted three times. The selected
mean ARI was `0.885`, versus `0.765` for always-average and `0.832` for
always-weighted, but the panel is too small to promote the selector as a
production default. A later fail-closed diagnosis also found that radically
different rooted split sets could still all collapse to one final cluster
because the downstream gates closed. Topology difference and final-label
difference must therefore be audited separately.

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
main runner now crosses one deep construction interface that owns validation
and construction metadata. Non-monotone linkage heights fail closed instead of
building topology-only placeholder branch lengths. Direct `tree_from_linkage`
calls remain only where applications or analyses already own a prepared linkage
matrix; they share representation conversion without hiding their distinct
geometry preparation or output contracts.

The `tree_linkage_method` field remains present in neighbor-joining and
IQ-TREE method configurations. It does not control those topology builders,
although the runner can reuse the value for linkage-based gate replay defaults.
That dual meaning should remain explicit in future configuration cleanup.

The adaptive pydiffmap route also has a stricter backend contract than its
public neighbor resolver expresses. Pydiffmap's internal bandwidth KDE needs
seven retained positive neighbor distances per row: ordinary data fails below
effective K 9, while exact duplicates can yield insufficient sparse support or
zero local bandwidth at any larger K. Fixed-topology NNLS does not repair this
because the failure occurs before topology construction. See
[[adaptive-diffusion-nnls-method-library-audit]] for the executed boundary,
dual-geometry, solver, stability, and scaling audit.

### Enforced construction contracts

The 2026-07-28 construction correction moved the enforceable assumptions into
the deep construction and diffusion interfaces:

- `build_tree(...)` now validates the condensed-vector size and values, unique
  sample labels, a single directed root, exact leaf-label coverage, finite
  non-negative branch lengths, and exactly two children at every internal
  node. A malformed phylogenetic result can no longer become an ordinary
  traversal boundary.
- MAD rooting always represents an optimum on an edge with a new root. If the
  optimum is an endpoint, the new root receives one zero-length edge; the
  metric location is unchanged and the TBS tree remains binary.
- Samples are placed in natural label order before linkage, neighbor joining,
  or IQ-TREE input. Numbered strings use numeric-aware ordering; incomparable
  mixed label types use a stable type/representation fallback. Exact-tie
  resolution is therefore invariant to DataFrame row order without changing
  the established `S0`, `S1`, ..., `S10` order. This makes the arbitrary
  resolution reproducible; it does not turn tied discrete states into
  statistical evidence.
- `TreeBuildDiagnostics` records zero-distance pairs, repeated distance
  values, repeated merge heights, zero-length branches, and tree shape on
  every construction result. Tie-heavy runs are now auditable without
  rebuilding the tree.
- Diffusion methods return one `DiffusionGeometry` result containing
  coordinates, condensed distances, and backend evidence. The old
  distance-only interfaces were removed rather than retained as aliases.
- Diffusion linkage is required explicitly by the runner and registry. Average
  linkage remains the predeclared baseline, but is no longer a hidden function
  default.
- Fixed-topology NNLS now rejects an omitted branch geometry. The retained
  adaptive method explicitly supplies the original distributional feature
  matrix and reports that dual-geometry contract. An executed same-diffusion-
  coordinate NNLS alternative was rejected: on the same 14-case subset its
  mean ARI fell from `0.8722` to `0.3132`, exact-K fell from `10/13` to `2/13`,
  and runtime rose from about `11.6` to `29.6` seconds.
- The adaptive runner rejects Hamming geometry for an explicitly continuous
  `FeatureSpace`, rejects Euclidean geometry for explicitly Bernoulli or
  categorical inputs, and rejects mixed-family inference. Untyped binary
  inputs remain supported only after a direct binary-value check.

These corrections fix ambiguity, shape validity, and deterministic tie
handling. They do not promote a universal topology selector, repair
pydiffmap's duplicate-state failure, or establish family-specific topology
superiority.

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
- Tests cover linkage representations, fail-closed non-monotone heights, phylogenetic
  builders/rooting, diffusion runners, registry dispatch, consensus selection,
  application contracts, and plotting reconstruction.
- The focused construction/diffusion/dispatch tranche passed, the complete
  `tests/core` plus `tests/tree` tranche passed 122 tests, and the ordered
  repository suite passed all 1,260 tests.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/` records the
  seven-case topology-selector result; the follow-up
  `reports/tree_consensus_topology_difference_diagnosis_20260709/` separates
  topology disagreement from downstream fail-closed gate behavior.

## Links

- [[phylogenetic-tree-builders-20260614]]
- [[graphtools-adaptive-k-tree-inference-focus-benchmark-20260630]]
- [[scrna-branch-length-effect-audit-20260624]]
- [[benchmark-pipeline-contract]]
- [[repository-execution-and-benchmark-map]]
- [[adaptive-diffusion-nnls-method-library-audit]]
- [[edge-gate-distance-time-contract-20260623]]
- [[method-application-and-plot-seams-20260727]]
- [[redundant-and-legacy-code-map-20260623]]

## Open Questions

- Should builder-specific method interfaces be separated from gate-replay
  linkage inputs so neighbor-joining and IQ-TREE runs do not appear to use
  `tree_linkage_method` for their primary topology?
- Should visualization-only tree reconstruction be replaced by persisted
  serialized `PosetTree`/Newick artifacts where exact inference replay is
  required?
- Should adaptive pydiffmap remain canonical before its duplicate-state and
  positive-neighbor-support contract is replaced or made explicit?
- Which family-specific topology candidate sets survive a frozen full-grid
  benchmark with topology-stability diagnostics and no truth-label selection?
- What minimum bootstrap clade support is required before a tie-heavy topology
  may enter production gates?
