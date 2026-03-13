"""Backward-compatibility shim — re-exports from new locations.

Real implementations live in:
- ``adjusted_wald.py`` — orchestration (annotate_sibling_divergence_adjusted)
- ``calibration.py`` — CalibrationModel, fit/predict inflation
- ``tree_traversal.py`` — collect_sibling_pair_records, etc.
"""

from .adjusted_wald import (  # noqa: F401
    _collect_all_pairs,
    annotate_sibling_divergence_adjusted,
)
from .calibration import CalibrationModel  # noqa: F401
from .calibration import fit_inflation_model as _fit_inflation_model  # noqa: F401
from .calibration import predict_inflation_factor  # noqa: F401

__all__ = [
    "CalibrationModel",
    "_collect_all_pairs",
    "_fit_inflation_model",
    "annotate_sibling_divergence_adjusted",
    "predict_inflation_factor",
]
