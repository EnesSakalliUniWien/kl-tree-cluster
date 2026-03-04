"""Gate-level orchestration wrappers for decomposition."""

from .edge_gate import annotate_edge_gate
from .orchestrator import run_gate_annotation_pipeline
from .sibling_gate import annotate_sibling_gate

__all__ = [
    "annotate_edge_gate",
    "annotate_sibling_gate",
    "run_gate_annotation_pipeline",
]

