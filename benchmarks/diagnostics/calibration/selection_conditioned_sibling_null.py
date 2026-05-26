"""Selection-conditioned sibling null diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE = (
    "continuous_fixed_projection_local_edge_selection"
)
NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE = (
    "standardized_z_proxy_fixed_projection_local_edge_selection"
)


@dataclass(frozen=True)
class SelectionConditionedSiblingNullContext:
    """Observed sibling-test context for a local selection-conditioned null."""

    case_id: str
    parent: object
    feature_family: str
    standardized_contrast_dimension: int
    edge_projection_dimension: int
    sibling_projection_dimension: int
    degrees_of_freedom: float
    reference_scale: float
    observed_statistic: float
    observed_p_value: float
    edge_alpha: float
    sibling_alpha: float
    parent_sample_size: int
    left_sample_size: int
    right_sample_size: int

    @property
    def reference_expectation(self) -> float:
        return float(self.reference_scale * self.degrees_of_freedom)

    @property
    def conditioning_scope(self) -> str:
        if self.feature_family == "continuous":
            return CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE
        return NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE


@dataclass(frozen=True)
class SelectionConditionedSiblingNullResult:
    """Monte-Carlo summary under local edge-gate selection."""

    context: SelectionConditionedSiblingNullContext
    seed: int
    n_candidates: int
    n_accepted: int
    edge_statistic_threshold: float
    conditional_mean_statistic: float
    conditional_sd_statistic: float
    conditional_median_statistic: float
    conditional_q95_statistic: float
    conditional_q99_statistic: float
    conditional_mean_over_reference: float
    empirical_tail_p_value: float
    mean_scaled_chi2_p_value: float

    @property
    def acceptance_rate(self) -> float:
        return float(self.n_accepted / self.n_candidates)

    @property
    def selection_blocks_at_alpha(self) -> bool:
        return bool(self.mean_scaled_chi2_p_value >= self.context.sibling_alpha)

    def as_row(self) -> dict[str, object]:
        """Return one flat CSV row."""
        context = self.context
        return {
            "case_id": context.case_id,
            "parent": context.parent,
            "feature_family": context.feature_family,
            "conditioning_scope": context.conditioning_scope,
            "standardized_contrast_dimension": context.standardized_contrast_dimension,
            "edge_projection_dimension": context.edge_projection_dimension,
            "sibling_projection_dimension": context.sibling_projection_dimension,
            "degrees_of_freedom": context.degrees_of_freedom,
            "reference_scale": context.reference_scale,
            "reference_expectation": context.reference_expectation,
            "observed_statistic": context.observed_statistic,
            "observed_p_value": context.observed_p_value,
            "edge_alpha": context.edge_alpha,
            "sibling_alpha": context.sibling_alpha,
            "parent_sample_size": context.parent_sample_size,
            "left_sample_size": context.left_sample_size,
            "right_sample_size": context.right_sample_size,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "n_accepted": self.n_accepted,
            "acceptance_rate": self.acceptance_rate,
            "edge_statistic_threshold": self.edge_statistic_threshold,
            "conditional_mean_statistic": self.conditional_mean_statistic,
            "conditional_sd_statistic": self.conditional_sd_statistic,
            "conditional_median_statistic": self.conditional_median_statistic,
            "conditional_q95_statistic": self.conditional_q95_statistic,
            "conditional_q99_statistic": self.conditional_q99_statistic,
            "selection_conditioned_c_hat": self.conditional_mean_over_reference,
            "selection_empirical_tail_p_value": self.empirical_tail_p_value,
            "selection_mean_scaled_chi2_p_value": self.mean_scaled_chi2_p_value,
            "selection_blocks_at_alpha": self.selection_blocks_at_alpha,
        }


def validate_selection_conditioned_context(
    context: SelectionConditionedSiblingNullContext,
) -> None:
    """Validate the local selection-conditioned diagnostic contract."""
    if not context.case_id:
        raise ValueError("Selection-conditioned context requires a non-empty case_id.")
    if context.standardized_contrast_dimension <= 0:
        raise ValueError(
            "Selection-conditioned context requires positive "
            "standardized_contrast_dimension."
        )
    if context.edge_projection_dimension <= 0:
        raise ValueError("edge_projection_dimension must be positive.")
    if context.sibling_projection_dimension <= 0:
        raise ValueError("sibling_projection_dimension must be positive.")
    if context.edge_projection_dimension > context.standardized_contrast_dimension:
        raise ValueError(
            "edge_projection_dimension cannot exceed standardized_contrast_dimension: "
            f"{context.edge_projection_dimension} > "
            f"{context.standardized_contrast_dimension}."
        )
    if context.sibling_projection_dimension > context.edge_projection_dimension:
        raise ValueError(
            "sibling_projection_dimension cannot exceed edge_projection_dimension "
            "for the fixed-parent-subspace diagnostic."
        )
    if (
        not np.isfinite(context.degrees_of_freedom)
        or context.degrees_of_freedom <= 0.0
    ):
        raise ValueError("degrees_of_freedom must be positive.")
    if not np.isclose(
        context.degrees_of_freedom,
        float(context.sibling_projection_dimension),
    ):
        raise ValueError(
            "Selection-conditioned diagnostic requires sibling degrees_of_freedom "
            "to equal sibling_projection_dimension."
        )
    if not np.isfinite(context.reference_scale) or context.reference_scale <= 0.0:
        raise ValueError("reference_scale must be positive.")
    if not np.isfinite(context.observed_statistic) or context.observed_statistic < 0.0:
        raise ValueError("observed_statistic must be finite and non-negative.")
    if (
        not np.isfinite(context.observed_p_value)
        or context.observed_p_value < 0.0
        or context.observed_p_value > 1.0
    ):
        raise ValueError("observed_p_value must lie in [0, 1].")
    if not 0.0 < context.edge_alpha < 1.0:
        raise ValueError("edge_alpha must lie in (0, 1).")
    if not 0.0 < context.sibling_alpha < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    if min(context.parent_sample_size, context.left_sample_size, context.right_sample_size) <= 0:
        raise ValueError("Selection-conditioned context requires positive sample sizes.")


def simulate_local_edge_selection_conditioned_null(
    context: SelectionConditionedSiblingNullContext,
    *,
    n_candidates: int,
    seed: int,
    chunk_size: int = 100_000,
) -> SelectionConditionedSiblingNullResult:
    """Simulate the sibling statistic conditional on a local edge gate opening.

    This level-1 diagnostic keeps the observed tree and parent PCA subspace
    fixed. It samples projected Gaussian coordinates under the null, accepts
    only draws whose parent child-edge statistic crosses ``edge_alpha``, and
    summarizes the sibling statistic in the leading sibling subspace.
    """
    validate_selection_conditioned_context(context)
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    rng = np.random.default_rng(seed)
    edge_k = int(context.edge_projection_dimension)
    sibling_k = int(context.sibling_projection_dimension)
    edge_threshold = float(chi2.isf(context.edge_alpha, df=float(edge_k)))

    accepted_statistics: list[np.ndarray] = []
    accepted_count = 0
    remaining = int(n_candidates)
    while remaining > 0:
        draw_count = min(int(chunk_size), remaining)
        projected_components = rng.standard_normal(size=(draw_count, edge_k))
        squared_components = projected_components * projected_components
        edge_statistics = np.sum(squared_components, axis=1)
        accepted_mask = edge_statistics >= edge_threshold
        if np.any(accepted_mask):
            sibling_statistics = np.sum(
                squared_components[accepted_mask, :sibling_k],
                axis=1,
            )
            accepted_statistics.append(sibling_statistics)
            accepted_count += int(sibling_statistics.shape[0])
        remaining -= draw_count

    if accepted_count == 0:
        raise ValueError(
            "Selection-conditioned null simulation accepted zero candidates; "
            "increase n_candidates or relax edge_alpha."
        )

    statistics = np.concatenate(accepted_statistics)
    conditional_mean = float(np.mean(statistics))
    conditional_mean_over_reference = float(
        conditional_mean / context.reference_expectation
    )
    scaled_observed_statistic = float(
        context.observed_statistic
        / (context.reference_scale * conditional_mean_over_reference)
    )
    return SelectionConditionedSiblingNullResult(
        context=context,
        seed=int(seed),
        n_candidates=int(n_candidates),
        n_accepted=int(accepted_count),
        edge_statistic_threshold=edge_threshold,
        conditional_mean_statistic=conditional_mean,
        conditional_sd_statistic=float(np.std(statistics, ddof=1)),
        conditional_median_statistic=float(np.quantile(statistics, 0.5)),
        conditional_q95_statistic=float(np.quantile(statistics, 0.95)),
        conditional_q99_statistic=float(np.quantile(statistics, 0.99)),
        conditional_mean_over_reference=conditional_mean_over_reference,
        empirical_tail_p_value=float(
            (1 + np.count_nonzero(statistics >= context.observed_statistic))
            / (accepted_count + 1)
        ),
        mean_scaled_chi2_p_value=float(
            chi2.sf(scaled_observed_statistic, df=float(context.degrees_of_freedom))
        ),
    )


__all__ = [
    "CONTINUOUS_LOCAL_EDGE_SELECTION_SCOPE",
    "NONCONTINUOUS_LOCAL_EDGE_Z_PROXY_SCOPE",
    "SelectionConditionedSiblingNullContext",
    "SelectionConditionedSiblingNullResult",
    "simulate_local_edge_selection_conditioned_null",
    "validate_selection_conditioned_context",
]
