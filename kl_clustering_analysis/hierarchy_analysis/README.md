# hierarchy_analysis/

Decomposition entrypoints, gate orchestration, and cluster-assignment helpers for the
clustering pipeline.

## Current Layout

| Path | Purpose |
| ---- | ------- |
| `tree_decomposition.py` | Public decomposition engine. Prepares or accepts explicit gate annotation bundles, evaluates gates, and emits cluster assignments. |
| `cluster_assignments.py` | Builds cluster-root and per-sample assignment tables from explicit final tree boundaries. |
| `bootstrap_consensus.py` | Bootstrap-based stability helpers layered on top of decomposition results. |
| `decomposition/gates/orchestrator.py` | Runs the edge-divergence gate then the sibling-divergence gate and returns a node-indexed annotation bundle. |
| `decomposition/gates/gate_evaluator.py` | Encapsulates the binary-structure prerequisite, edge-divergence gate, sibling-divergence gate, and pass-through traversal decision as one action. |
| `decomposition/gates/column_contracts.py` | Shared checks for the gate-column contract carried through the explicit annotation bundle. |

## Decomposition Flow

1. `TreeDecomposition` receives an annotation DataFrame or an explicit `GateAnnotationBundle`.
2. `run_gate_annotation_pipeline()` populates edge-divergence and sibling-divergence columns when they are missing.
3. `GateEvaluator.decision()` applies the runtime gates and returns a boundary, split, or pass-through action.
4. `TreeDecomposition.decompose_tree()` advances the worklist, optionally honoring pass-through mode for deeper descendant splits.
5. `cluster_assignments.py` converts the final boundary nodes into cluster-level and sample-level outputs.

## Notes

- The current runtime centers on the gate annotation pipeline under `decomposition/gates/`.
- Statistical test details live in `statistics/README.md`.
