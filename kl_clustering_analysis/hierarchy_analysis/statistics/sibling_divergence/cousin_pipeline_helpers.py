"""Backward-compatibility shim — re-exports from new locations.

Real implementations live in:
- ``tree_traversal.py`` — SiblingPairRecord, collect/deflate helpers
- ``bh_application.py`` — init, mark_non_binary, early_return, BH application
- ``wald_kernel.py`` — sibling_divergence_test kernel
"""

from .bh_application import early_return_if_no_records  # noqa: F401
from .bh_application import init_sibling_annotation_df  # noqa: F401
from .bh_application import mark_non_binary_as_skipped  # noqa: F401
from .tree_traversal import DeflatableSiblingRecord  # noqa: F401
from .tree_traversal import SiblingPairRecord  # noqa: F401
from .tree_traversal import collect_sibling_pair_records  # noqa: F401
from .tree_traversal import count_null_focal_pairs  # noqa: F401
from .tree_traversal import deflate_focal_pairs  # noqa: F401
from .tree_traversal import either_child_significant as _either_child_significant  # noqa: F401
from .wald_kernel import sibling_divergence_test  # noqa: F401

__all__ = [
    "DeflatableSiblingRecord",
    "SiblingPairRecord",
    "collect_sibling_pair_records",
    "count_null_focal_pairs",
    "deflate_focal_pairs",
    "early_return_if_no_records",
    "init_sibling_annotation_df",
    "mark_non_binary_as_skipped",
]
