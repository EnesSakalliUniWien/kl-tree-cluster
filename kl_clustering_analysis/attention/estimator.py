"""Attention-weighted Bernoulli rate estimation with noise penalties."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kl_clustering_analysis.information_metrics.kl_divergence import (
    calculate_kl_divergence_vector,
)


def _kl_bernoulli(
    observed_probs: npt.NDArray[np.float64],
    reference_prob: float,
    eps: float = 1e-9,
) -> npt.NDArray[np.float64]:
    """Element-wise KL(Bern(observed) || Bern(reference)) using shared KL helper."""
    observed_probs = np.clip(observed_probs, eps, 1.0 - eps)
    reference_probs = np.clip(reference_prob, eps, 1.0 - eps)

    orig_shape = observed_probs.shape
    reference_probs = np.broadcast_to(reference_probs, orig_shape)

    # Build 2-class distributions per site and reuse the project-wide KL helper
    observed_distribution = np.stack(
        [observed_probs, 1.0 - observed_probs], axis=-1
    ).reshape(-1, 2)
    reference_distribution = np.stack(
        [
            reference_probs,
            1.0 - reference_probs,
        ],
        axis=-1,
    ).reshape(-1, 2)

    kl_flat = calculate_kl_divergence_vector(
        observed_distribution, reference_distribution
    )  # shape (n*2,)
    kl_per_site = kl_flat.reshape(-1, 2).sum(axis=1)
    return kl_per_site.reshape(orig_shape)


def _noise_scores(
    branch_site_probs: npt.NDArray[np.float64], eps: float = 1e-9
) -> npt.NDArray[np.float64]:
    """Per-site disagreement score across branches."""
    site_mean_probs = branch_site_probs.mean(axis=0)
    # KL of each branch to the site-wise mean, averaged over branches
    site_noise_scores = np.mean(
        _kl_bernoulli(branch_site_probs, site_mean_probs, eps=eps), axis=0
    )
    return site_noise_scores


def estimate_attention_rates(
    p: npt.NDArray[np.float64],
    *,
    tau: float = 0.1,
    gamma: float = 0.1,
    iters: int = 30,
    eps: float = 1e-9,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Estimate per-branch Bernoulli rates with attention weights over sites.

    Parameters
    ----------
    p : np.ndarray
        Observed frequencies with shape (B, J): branches x sites.
    tau : float, optional
        Temperature on per-branch fit (KL) term. Smaller -> sharper attention.
    gamma : float, optional
        Temperature on global noise term. Smaller -> stronger penalty for noisy sites.
    iters : int, optional
        Number of attention updates.
    eps : float, optional
        Numerical floor/ceiling for probabilities.

    Returns
    -------
    q_hat : np.ndarray
        Estimated rates per branch, shape (B,).
    attention : np.ndarray
        Attention weights per branch and site, shape (B, J).
    """
    branch_site_probs = np.asarray(p, dtype=np.float64)
    if branch_site_probs.ndim != 2:
        raise ValueError(
            f"Expected 2D array (branches x sites); got shape {branch_site_probs.shape}"
        )

    num_branches, num_sites = branch_site_probs.shape
    if num_branches == 0 or num_sites == 0:
        return (
            np.empty((num_branches,), dtype=np.float64),
            np.empty((num_branches, num_sites), dtype=np.float64),
        )

    branch_rate_estimates = branch_site_probs.mean(axis=1).copy()
    site_noise_scores = _noise_scores(branch_site_probs, eps=eps)  # (J,)
    attention_weights = np.full(
        (num_branches, num_sites), 1.0 / num_sites, dtype=np.float64
    )

    for _ in range(int(iters)):
        for branch_idx in range(num_branches):
            branch_site_kl = _kl_bernoulli(
                branch_site_probs[branch_idx], branch_rate_estimates[branch_idx], eps=eps
            )
            logits = -(branch_site_kl / tau + site_noise_scores / gamma)
            logits -= logits.max()  # stabilise
            branch_weights = np.exp(logits)
            branch_weights /= branch_weights.sum()
            attention_weights[branch_idx] = branch_weights
            branch_rate_estimates[branch_idx] = np.sum(
                branch_weights * branch_site_probs[branch_idx]
            )

    return branch_rate_estimates, attention_weights


__all__ = ["estimate_attention_rates"]
