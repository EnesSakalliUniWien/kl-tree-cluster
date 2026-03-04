"""Gate-level orchestration wrappers for decomposition."""

from .edge_gate import annotate_edge_gate
from .orchestrator import run_gate_annotation_pipeline
from .pairwise_testing import build_branch_distance_cache, test_cluster_pair_divergence
from .posthoc_merge import apply_posthoc_merge
from .sibling_gate import annotate_sibling_gate
from .traversal import GateEvaluator, iterate_worklist, process_node

__all__ = [
    "GateEvaluator",
    "annotate_edge_gate",
    "annotate_sibling_gate",
    "apply_posthoc_merge",
    "build_branch_distance_cache",
    "iterate_worklist",
    "process_node",
    "run_gate_annotation_pipeline",
    "test_cluster_pair_divergence",
]
