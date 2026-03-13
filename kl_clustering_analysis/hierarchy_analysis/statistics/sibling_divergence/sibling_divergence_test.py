"""Backward-compatibility shim — re-exports from new locations.

Real implementations live in:
- ``standard_wald.py`` — orchestration (annotate, collect, run, apply)
- ``wald_kernel.py`` — statistical kernel (sibling_divergence_test)
- ``tree_traversal.py`` — tree helpers (_get_binary_children, etc.)
"""

from .standard_wald import (  # noqa: F401
    _collect_test_arguments,
    _run_tests,
    annotate_sibling_divergence,
)
from .tree_traversal import collect_significant_sibling_pairs  # noqa: F401
from .tree_traversal import get_binary_children as _get_binary_children  # noqa: F401
from .tree_traversal import get_sibling_data as _get_sibling_data  # noqa: F401
from .wald_kernel import sibling_divergence_test  # noqa: F401

__all__ = [
    "_collect_test_arguments",
    "_get_binary_children",
    "_get_sibling_data",
    "_run_tests",
    "annotate_sibling_divergence",
    "collect_significant_sibling_pairs",
    "sibling_divergence_test",
]
