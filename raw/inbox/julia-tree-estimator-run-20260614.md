# Julia Tree Estimator Run 2026-06-14

Ran the combined Julia GOCC/GOBP/GOMF binary matrix (`703 x 14766`) through three KL tree-estimator paths:

- `kl`: Hamming average-linkage baseline.
- `kl_neighbor_joining`: Hamming neighbor-joining tree rooted by minimum ancestor deviation.
- `kl_iqtree3_fast`: IQ-TREE 3.1.2 using `JC2 --fast`, then minimum ancestor-deviation rooting and the KL gate/traversal layer.

Reference scoring used the cached Julia endotype labels from `kak_signal_adaptive_reference_labels.csv`, with `262` matched genes and `20` observed reference clusters.

Summary:

| method | status | clusters | elapsed_sec | reference_ari | reference_nmi |
| --- | --- | ---: | ---: | ---: | ---: |
| `kl` | ok | 670 | 161.311 | 0.001812 | 0.633958 |
| `kl_neighbor_joining` | ok | 494 | 178.330 | 0.021596 | 0.616780 |
| `kl_iqtree3_fast` | ok | 566 | 215.388 | 0.015612 | 0.629320 |

Interpretation:

- All three methods remain highly fragmented on the Julia matrix.
- Neighbor joining reduces fragmentation most and gives the best reference ARI, but lowers NMI versus baseline.
- IQ-TREE fast/MAD is intermediate: fewer clusters than baseline, more than NJ, reference NMI close to baseline, reference ARI below NJ.
- Tree construction changes root/topology behavior but does not solve the traversal-level selected-family/null-law problem.

Artifacts copied to `raw/assets/benchmark-results/julia_tree_estimators_20260614/`.
