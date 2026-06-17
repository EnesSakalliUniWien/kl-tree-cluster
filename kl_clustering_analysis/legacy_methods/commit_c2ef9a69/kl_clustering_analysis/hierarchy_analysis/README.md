# hierarchy_analysis/

Decomposition entrypoints, gate orchestration, and cluster-assignment helpers for the
clustering pipeline.

## Current Layout

| Path | Purpose |
| ---- | ------- |
| `tree_decomposition.py` | Public decomposition engine. Prepares annotations, caches node metadata, evaluates gates, and emits cluster assignments. |
| `cluster_assignments.py` | Builds cluster-root and per-sample assignment tables from traversal output. |
| `bootstrap_consensus.py` | Bootstrap-based stability helpers layered on top of decomposition results. |
| `annotations_df/README.md` | Schema reference for the node-indexed `annotations_df`, including column meanings and gate usage. |
| `decomposition/gates/orchestrator.py` | Runs Gate 2 then Gate 3 adapters and returns a legacy-compatible annotation bundle. |
| `decomposition/gates/gate_evaluator.py` | Encapsulates binary-structure, child-parent, sibling, and passthrough decisions at runtime. |
| `decomposition/gates/column_contracts.py` | Shared checks for the legacy gate-column contract carried through the pipeline metadata. |

## Decomposition Flow

1. `TreeDecomposition` resolves the projection-floor configuration and caches node metadata.
2. `run_gate_annotation_pipeline()` populates edge and sibling gate columns when they are missing.
3. `GateEvaluator.should_split()` applies the three runtime gates at each internal node.
4. `TreeDecomposition.decompose_tree()` advances the worklist, optionally honoring passthrough mode for deeper descendant splits.
5. `cluster_assignments.py` converts the final boundary nodes into cluster-level and sample-level outputs.

## Notes

- Older references to `_should_split_v2`, `signal_localization.py`, and `posthoc_merge.py` are obsolete.
- The current runtime centers on the gate adapters under `decomposition/gates/`.
- Statistical test details live in `statistics/README.md`.
