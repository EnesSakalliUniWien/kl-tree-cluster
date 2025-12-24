"""Global divergence weighting for edge significance testing.

This module implements symmetric weighting based on the contribution fraction:
    f = KL_local / KL_global

The weight formula is:
    w = (f_median / f)^β

Effects:
- f > f_median → w < 1 (BONUS: easier to be significant)
- f = f_median → w = 1 (neutral)
- f < f_median → w > 1 (PENALTY: harder to be significant)

This reduces false positives in deep branches while preserving
sensitivity to genuine structure near the root.
"""

from __future__ import annotations

import numpy as np


def estimate_global_weight_strength(
    child_local_kl: np.ndarray,
    child_global_kl: np.ndarray,
    percentile: float = 50.0,
    min_beta: float = 0.1,
    max_beta: float = 1.0,
) -> float:
    """Estimate global weight strength (β) from data distribution.

    Analyzes the distribution of KL_local/KL_global ratios across edges
    and estimates β such that typical edges receive moderate weighting.

    Strategy:
    - Compute relative strength r = KL_local / KL_global for each edge
    - Find the percentile value r_p (e.g., median)
    - Set β so that edges at r_p get weight ≈ 1.5-2.0
    - This means: 1 + β·log(1 + 1/r_p) ≈ 1.5
    - Solve: β ≈ 0.5 / log(1 + 1/r_p)

    Parameters
    ----------
    child_local_kl
        Local KL(child||parent) values for all edges
    child_global_kl
        Global KL(child||root) values for all edges
    percentile
        Percentile to use for estimation (50 = median)
    min_beta
        Minimum allowed β value
    max_beta
        Maximum allowed β value

    Returns
    -------
    float
        Estimated β value in [min_beta, max_beta]

    Notes
    -----
    For the symmetric weighting scheme, β controls the strength of bonus/penalty:
    - weight = (median_fraction / edge_fraction)^β
    - β = 0.4 is moderate, β = 1.0 is strong

    max_beta is now estimated from data:
    - Based on the spread of fraction distribution (IQR)
    - Smaller spread → smaller max_beta (avoid over-weighting)
    - Larger spread → larger max_beta (can differentiate more)
    """
    # Filter valid edges (both KL values positive)
    valid = (
        (child_local_kl > 0)
        & (child_global_kl > 0)
        & np.isfinite(child_local_kl)
        & np.isfinite(child_global_kl)
    )
    if not valid.any():
        # No valid edges - return default
        return 0.4

    local_valid = child_local_kl[valid]
    global_valid = child_global_kl[valid]

    # Compute contribution fraction: KL_local / KL_global
    fraction = local_valid / global_valid
    n_edges = len(fraction)

    # Get the spread of fractions to calibrate β
    median_frac = float(np.percentile(fraction, 50))
    q10_frac = float(np.percentile(fraction, 10))  # Low contributors
    q25_frac = float(np.percentile(fraction, 25))
    q75_frac = float(np.percentile(fraction, 75))

    if q10_frac <= 0 or median_frac <= 0:
        return min_beta

    # Estimate max_beta from data characteristics:
    # - IQR ratio: how spread out is the distribution?
    # - Sample size: smaller datasets need gentler weighting
    iqr_ratio = q75_frac / max(q25_frac, 1e-6)

    # max_beta scales with log of IQR ratio and sample size
    # Small IQR (uniform fractions) → max_beta ≈ 0.3
    # Large IQR (varied fractions) → max_beta can be higher
    # Also scale down for small sample sizes
    sample_factor = min(
        1.0, np.log1p(n_edges) / np.log1p(100)
    )  # Saturates at ~100 edges
    spread_factor = min(
        1.0, np.log1p(iqr_ratio - 1) / np.log1p(3)
    )  # Saturates at IQR ratio ~4

    # max_beta ranges from 0.2 to 0.6 based on data
    estimated_max_beta = 0.2 + 0.4 * sample_factor * spread_factor

    # Override the fixed max_beta with data-driven estimate
    max_beta = max(min_beta, estimated_max_beta)

    # Target: At q10 (low contributor), weight should be ~1.5-2.0
    # weight = (median_frac / q10_frac)^β
    # 1.5 = (median_frac / q10_frac)^β
    # β = log(1.5) / log(median_frac / q10_frac)
    ratio = median_frac / q10_frac
    if ratio > 1:
        beta = np.log(1.5) / np.log(ratio)
    else:
        beta = 0.3  # Default if ratio <= 1

    # Clip to data-driven range
    beta = np.clip(beta, min_beta, max_beta)

    return float(beta)


def compute_neutral_point(
    child_local_kl: np.ndarray,
    child_global_kl: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute the data-driven neutral point for symmetric weighting.

    The neutral point is the median contribution fraction (KL_local/KL_global)
    across all valid edges. Edges with fraction equal to the neutral point
    receive weight = 1.0 (no adjustment).

    Parameters
    ----------
    child_local_kl
        Local KL(child||parent) values for all edges
    child_global_kl
        Global KL(child||root) values for all edges
    valid_mask
        Boolean mask indicating valid edges. If None, uses edges where
        both KL values are positive and finite.

    Returns
    -------
    float
        The neutral point (median contribution fraction), minimum 1e-6
    """
    if valid_mask is None:
        valid_mask = (
            (child_local_kl > 0)
            & (child_global_kl > 0)
            & np.isfinite(child_local_kl)
            & np.isfinite(child_global_kl)
        )

    # Further filter: global KL must be positive for fraction computation
    valid_for_neutral = (
        valid_mask & (child_global_kl > 0) & np.isfinite(child_global_kl)
    )

    if not valid_for_neutral.any():
        return 0.5  # Default neutral point

    fractions = child_local_kl[valid_for_neutral] / child_global_kl[valid_for_neutral]
    neutral_point = float(np.median(fractions))

    # Ensure it's positive
    return max(neutral_point, 1e-6)


def compute_global_weight(
    child_local_kl: float,
    child_global_kl: float,
    beta: float,
    method: str = "relative",
    neutral_point: float = 0.5,
) -> float:
    """Compute global divergence weight for an edge.

    This function implements symmetric weighting:
    - Bonus (weight < 1) for edges with strong local signal relative to global
    - Penalty (weight > 1) for edges with weak local signal relative to global

    Parameters
    ----------
    child_local_kl
        Local KL(child||parent) divergence
    child_global_kl
        Global KL(child||root) divergence
    beta
        Global weight strength parameter
    method
        Weighting method:
        - "fixed": weight = 1 + β·log(1 + KL_global) [penalty only]
        - "relative": symmetric bonus/penalty based on contribution fraction
          where fraction = KL_local / KL_global
    neutral_point
        The contribution fraction at which weight = 1.0 (neutral).
        - For symmetric distribution, use median(fraction) from data
        - Default 0.5 means "captures half the divergence → neutral"

    Returns
    -------
    float
        Weight multiplier:
        - < 1.0: bonus for global structures (easier to be significant)
        - = 1.0: neutral
        - > 1.0: penalty for noise (harder to be significant)

    Examples
    --------
    >>> compute_global_weight(0.02, 0.01, beta=0.4, neutral_point=0.01)
    0.76  # Bonus: fraction > neutral_point
    >>> compute_global_weight(0.005, 0.01, beta=0.4, neutral_point=0.01)
    1.32  # Penalty: fraction < neutral_point
    """
    if beta == 0 or child_global_kl <= 0:
        return 1.0

    if method == "fixed":
        # Simple depth penalty (no bonus)
        weight = 1.0 + beta * np.log(1.0 + child_global_kl)
        return max(1.0, float(weight))
    elif method == "relative":
        # Symmetric bonus/penalty based on contribution fraction
        # fraction = KL_local / KL_global = what fraction of total divergence is at this edge
        if child_local_kl <= 0:
            # No local signal - strong penalty
            return 2.0  # Maximum penalty

        fraction = child_local_kl / child_global_kl

        # fraction ranges from 0 to 1
        # - fraction > neutral_point: this edge captures more than median → BONUS
        # - fraction < neutral_point: this edge captures less than median → PENALTY
        # - fraction = neutral_point: weight = 1.0 (neutral)
        #
        # weight = (neutral_point / fraction)^β gives:
        #   If neutral_point = 0.01 (data-driven median):
        #     fraction=0.02 → weight = 0.5^β ≈ 0.76 (bonus) for β=0.4
        #     fraction=0.01 → weight = 1.0 (neutral)
        #     fraction=0.005 → weight = 2^β ≈ 1.32 (penalty) for β=0.4

        # Avoid division by zero and extreme values
        fraction = max(fraction, 1e-6)
        neutral_point = max(neutral_point, 1e-6)

        weight = (neutral_point / fraction) ** beta

        # Asymmetric clamping:
        # - Bonuses are more conservative (weight not too far below 1)
        # - Penalties can be stronger (weight up to 2)
        # This prevents over-splitting from aggressive bonuses
        min_weight = max(0.8, 1.0 - 0.3 * beta)  # At β=0.5: min_weight = 0.85
        max_weight = min(2.0, 1.0 + 1.5 * beta)  # At β=0.5: max_weight = 1.75
        weight = np.clip(weight, min_weight, max_weight)

        return float(weight)
    else:
        raise ValueError(f"Unknown global weight method: {method}")


def extract_global_weighting_config(
    nodes_dataframe: "pd.DataFrame",
    child_ids: list[str],
    child_local_kl: np.ndarray,
) -> tuple[np.ndarray | None, float, str]:
    """Extract global weighting configuration from dataframe and config.

    Reads global divergence weighting settings from config and extracts
    the necessary data from the nodes dataframe.

    Parameters
    ----------
    nodes_dataframe
        DataFrame with node statistics, must contain 'kl_divergence_global' column
    child_ids
        List of child node identifiers to extract data for
    child_local_kl
        Local KL values for computing data-driven beta

    Returns
    -------
    tuple[np.ndarray | None, float, str]
        (child_global_kl, global_weight_beta, global_weight_method)
        - child_global_kl: Global KL values or None if not available
        - global_weight_beta: The beta parameter for weighting
        - global_weight_method: The weighting method to use
    """
    import warnings
    import logging
    from kl_clustering_analysis import config

    if "kl_divergence_global" not in nodes_dataframe.columns:
        warnings.warn(
            "Global divergence weighting enabled but 'kl_divergence_global' "
            "column not found in nodes_dataframe. Proceeding without weighting.",
            UserWarning,
        )
        return None, 0.0, config.GLOBAL_WEIGHT_METHOD

    child_global_kl = (
        nodes_dataframe["kl_divergence_global"].reindex(child_ids).to_numpy()
    )

    # Determine beta and method based on config
    if config.GLOBAL_WEIGHT_METHOD == "data_driven":
        global_weight_beta = estimate_global_weight_strength(
            child_local_kl=child_local_kl,
            child_global_kl=child_global_kl,
            percentile=config.GLOBAL_WEIGHT_PERCENTILE,
        )
        # Data-driven uses relative weight formula with estimated beta
        global_weight_method = "relative"
        logger = logging.getLogger(__name__)
        logger.info(f"Estimated global_weight_beta from data: {global_weight_beta:.3f}")
    elif config.GLOBAL_WEIGHT_METHOD in ["fixed", "relative"]:
        global_weight_beta = config.GLOBAL_WEIGHT_STRENGTH
        global_weight_method = config.GLOBAL_WEIGHT_METHOD
    else:
        raise ValueError(f"Unknown GLOBAL_WEIGHT_METHOD: {config.GLOBAL_WEIGHT_METHOD}")

    return child_global_kl, global_weight_beta, global_weight_method


__all__ = [
    "estimate_global_weight_strength",
    "compute_neutral_point",
    "compute_global_weight",
    "extract_global_weighting_config",
]
