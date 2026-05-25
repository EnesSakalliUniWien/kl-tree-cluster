"""External Gaussian null diagnostics for sibling empirical inflation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


CONTINUOUS_FIXED_SUBSPACE_SCOPE = "continuous_identity_gaussian_fixed_subspace"
NONCONTINUOUS_Z_PROXY_SCOPE = "standardized_z_gaussian_proxy_not_feature_null"


@dataclass(frozen=True)
class GaussianSiblingNullContext:
    """Observed sibling-test context for a fixed-subspace Gaussian null check."""

    case_id: str
    parent: object
    feature_family: str
    standardized_contrast_dimension: int
    projection_dimension: int
    degrees_of_freedom: float
    reference_scale: float
    observed_statistic: float
    observed_p_value: float
    sibling_alpha: float
    parent_sample_size: int
    left_sample_size: int
    right_sample_size: int

    @property
    def reference_expectation(self) -> float:
        return float(self.reference_scale * self.degrees_of_freedom)

    @property
    def validity_scope(self) -> str:
        if self.feature_family == "continuous":
            return CONTINUOUS_FIXED_SUBSPACE_SCOPE
        return NONCONTINUOUS_Z_PROXY_SCOPE


@dataclass(frozen=True)
class GaussianSiblingNullCalibration:
    """Monte-Carlo summary for the fixed-subspace Gaussian projected-Wald law."""

    context: GaussianSiblingNullContext
    n_replicates: int
    seed: int
    mean_statistic: float
    sd_statistic: float
    median_statistic: float
    q95_statistic: float
    q99_statistic: float
    q999_statistic: float
    mean_over_reference: float
    empirical_tail_p_value: float
    mean_scaled_chi2_p_value: float
    chi2_reference_p_value: float

    @property
    def external_blocks_at_alpha(self) -> bool:
        return bool(self.mean_scaled_chi2_p_value >= self.context.sibling_alpha)

    def as_row(self) -> dict[str, object]:
        """Return one flat CSV row."""
        context = self.context
        return {
            "case_id": context.case_id,
            "parent": context.parent,
            "feature_family": context.feature_family,
            "external_null_model": "fixed_subspace_isotropic_gaussian",
            "external_null_validity_scope": context.validity_scope,
            "standardized_contrast_dimension": context.standardized_contrast_dimension,
            "projection_dimension": context.projection_dimension,
            "degrees_of_freedom": context.degrees_of_freedom,
            "reference_scale": context.reference_scale,
            "reference_expectation": context.reference_expectation,
            "parent_sample_size": context.parent_sample_size,
            "left_sample_size": context.left_sample_size,
            "right_sample_size": context.right_sample_size,
            "observed_statistic": context.observed_statistic,
            "observed_p_value": context.observed_p_value,
            "sibling_alpha": context.sibling_alpha,
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "external_mean_statistic": self.mean_statistic,
            "external_sd_statistic": self.sd_statistic,
            "external_median_statistic": self.median_statistic,
            "external_q95_statistic": self.q95_statistic,
            "external_q99_statistic": self.q99_statistic,
            "external_q999_statistic": self.q999_statistic,
            "external_mean_over_reference": self.mean_over_reference,
            "external_empirical_tail_p_value": self.empirical_tail_p_value,
            "external_mean_scaled_chi2_p_value": self.mean_scaled_chi2_p_value,
            "external_chi2_reference_p_value": self.chi2_reference_p_value,
            "external_blocks_at_alpha": self.external_blocks_at_alpha,
        }


def validate_gaussian_sibling_null_context(
    context: GaussianSiblingNullContext,
) -> None:
    """Validate the fixed-subspace null diagnostic contract."""
    if not context.case_id:
        raise ValueError("Gaussian sibling null context requires a non-empty case_id.")
    if context.standardized_contrast_dimension <= 0:
        raise ValueError(
            "Gaussian sibling null context requires positive standardized_contrast_dimension."
        )
    if context.projection_dimension <= 0:
        raise ValueError(
            "Gaussian sibling null context requires positive projection_dimension."
        )
    if context.projection_dimension > context.standardized_contrast_dimension:
        raise ValueError(
            "projection_dimension cannot exceed standardized_contrast_dimension: "
            f"{context.projection_dimension} > {context.standardized_contrast_dimension}."
        )
    if not np.isfinite(context.degrees_of_freedom) or context.degrees_of_freedom <= 0.0:
        raise ValueError("Gaussian sibling null context requires positive degrees_of_freedom.")
    if not np.isclose(context.degrees_of_freedom, float(context.projection_dimension)):
        raise ValueError(
            "Fixed-subspace Gaussian null requires degrees_of_freedom to equal "
            "projection_dimension. "
            f"Got df={context.degrees_of_freedom}, k={context.projection_dimension}."
        )
    if not np.isfinite(context.reference_scale) or context.reference_scale <= 0.0:
        raise ValueError("Gaussian sibling null context requires positive reference_scale.")
    if not np.isfinite(context.observed_statistic) or context.observed_statistic < 0.0:
        raise ValueError(
            "Gaussian sibling null context requires a finite non-negative observed_statistic."
        )
    if (
        not np.isfinite(context.observed_p_value)
        or context.observed_p_value < 0.0
        or context.observed_p_value > 1.0
    ):
        raise ValueError("Gaussian sibling null context requires observed_p_value in [0, 1].")
    if not 0.0 < context.sibling_alpha < 1.0:
        raise ValueError("Gaussian sibling null context requires sibling_alpha in (0, 1).")
    if min(context.parent_sample_size, context.left_sample_size, context.right_sample_size) <= 0:
        raise ValueError("Gaussian sibling null context requires positive sample sizes.")


def simulate_fixed_subspace_gaussian_null(
    context: GaussianSiblingNullContext,
    *,
    n_replicates: int,
    seed: int,
) -> GaussianSiblingNullCalibration:
    """Simulate the projected-Wald law for a fixed orthonormal Gaussian subspace.

    The simulated object is the standardized projected contrast
    ``R z`` with ``z ~ N(0, I_d)`` and fixed orthonormal ``R`` of rank ``k``.
    Therefore the simulated statistic is exactly ``sum_i (R z)_i^2``.
    This isolates the projected-Wald reference law from tree-selection and
    feature-family effects.
    """
    validate_gaussian_sibling_null_context(context)
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    rng = np.random.default_rng(seed)
    projected_components = rng.standard_normal(
        size=(int(n_replicates), context.projection_dimension)
    )
    statistics = np.einsum(
        "ij,ij->i",
        projected_components,
        projected_components,
        optimize=True,
    )

    mean_statistic = float(np.mean(statistics))
    mean_over_reference = float(mean_statistic / context.reference_expectation)
    scaled_statistic = float(
        context.observed_statistic / (context.reference_scale * mean_over_reference)
    )
    return GaussianSiblingNullCalibration(
        context=context,
        n_replicates=int(n_replicates),
        seed=int(seed),
        mean_statistic=mean_statistic,
        sd_statistic=float(np.std(statistics, ddof=1)),
        median_statistic=float(np.quantile(statistics, 0.5)),
        q95_statistic=float(np.quantile(statistics, 0.95)),
        q99_statistic=float(np.quantile(statistics, 0.99)),
        q999_statistic=float(np.quantile(statistics, 0.999)),
        mean_over_reference=mean_over_reference,
        empirical_tail_p_value=float(
            (1 + np.count_nonzero(statistics >= context.observed_statistic))
            / (int(n_replicates) + 1)
        ),
        mean_scaled_chi2_p_value=float(
            chi2.sf(scaled_statistic, df=float(context.degrees_of_freedom))
        ),
        chi2_reference_p_value=float(
            chi2.sf(
                context.observed_statistic / context.reference_scale,
                df=float(context.degrees_of_freedom),
            )
        ),
    )


def append_external_gaussian_null_columns(
    target_rows: pd.DataFrame,
    calibrations: tuple[GaussianSiblingNullCalibration, ...],
) -> pd.DataFrame:
    """Join external null calibration summaries onto sibling target rows."""
    if target_rows.empty:
        raise ValueError("Cannot append external null columns to an empty target table.")
    calibration_rows = pd.DataFrame.from_records(
        [calibration.as_row() for calibration in calibrations]
    )
    if calibration_rows.empty:
        raise ValueError("At least one calibration row is required.")
    required = {"case_id", "parent"}
    missing_targets = required - set(target_rows.columns)
    missing_calibrations = required - set(calibration_rows.columns)
    if missing_targets:
        raise ValueError(f"Target table is missing columns: {sorted(missing_targets)}.")
    if missing_calibrations:
        raise ValueError(
            f"Calibration table is missing columns: {sorted(missing_calibrations)}."
        )
    merged = target_rows.merge(
        calibration_rows,
        on=["case_id", "parent"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_external"),
    )
    if len(merged) != len(target_rows):
        raise ValueError(
            "External Gaussian null calibration did not cover every target row: "
            f"{len(merged)} of {len(target_rows)} rows matched."
        )
    return merged


__all__ = [
    "CONTINUOUS_FIXED_SUBSPACE_SCOPE",
    "GaussianSiblingNullCalibration",
    "GaussianSiblingNullContext",
    "NONCONTINUOUS_Z_PROXY_SCOPE",
    "append_external_gaussian_null_columns",
    "simulate_fixed_subspace_gaussian_null",
    "validate_gaussian_sibling_null_context",
]
