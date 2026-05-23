"""Child-parent divergence annotation modules for Gate 2."""

from .child_parent_divergence_annotation import annotate_child_parent_divergence
from .tree_testing import run_child_parent_tests_across_tree

__all__ = [
    "annotate_child_parent_divergence",
    "run_child_parent_tests_across_tree",
]
