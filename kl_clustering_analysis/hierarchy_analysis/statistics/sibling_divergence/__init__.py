from .adjusted_wald_annotation import annotate_sibling_divergence_adjusted
from .inflation_correction import CalibrationModel

annotate_sibling_divergence = annotate_sibling_divergence_adjusted

__all__ = [
    "CalibrationModel",
    "annotate_sibling_divergence",
    "annotate_sibling_divergence_adjusted",
]
