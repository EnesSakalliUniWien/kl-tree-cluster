"""
Sequence evolution models for phylogenetic and temporal benchmarks.

Features:
- Jukes-Cantor substitution model (realistic sequence divergence).
- Evolution along branches of specified length.
- Divergence metrics (JS Divergence).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def jukes_cantor_transition_matrix(
    n_categories: int,
    branch_length: float,
) -> np.ndarray:
    """Compute Jukes-Cantor transition probability matrix.

    P(j|i, t) = (1/k) + (1 - 1/k) * exp(-k*mu*t)  if i == j
              = (1/k) * (1 - exp(-k*mu*t))        if i != j

    where k = n_categories, mu = 1 (normalized), t = branch_length
    """
    k = n_categories
    # Probability of staying in same state
    p_same = (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * branch_length / (k - 1))
    # Probability of changing to any other state
    p_diff = (1.0 / k) * (1 - np.exp(-k * branch_length / (k - 1)))

    # Build transition matrix
    P = np.full((k, k), p_diff)
    np.fill_diagonal(P, p_same)

    return P


def generate_ancestral_sequence(
    n_features: int,
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate a random ancestral sequence with uniform base frequencies."""
    return random_state.randint(0, n_categories, size=n_features)


def evolve_sequence(
    ancestor: np.ndarray,
    branch_length: float,
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Evolve a sequence along a branch using Jukes-Cantor model."""
    P = jukes_cantor_transition_matrix(n_categories, branch_length)
    evolved = np.zeros_like(ancestor)

    for i, state in enumerate(ancestor):
        evolved[i] = random_state.choice(n_categories, p=P[state])

    return evolved


def compute_expected_divergence(branch_length: float, n_categories: int) -> float:
    """Compute expected proportion of differing sites under Jukes-Cantor."""
    k = n_categories
    return ((k - 1) / k) * (1 - np.exp(-k * branch_length / (k - 1)))


def compute_js_divergence_per_feature(
    p: np.ndarray, q: np.ndarray, eps: float = 1e-10
) -> float:
    """Compute mean Jensen-Shannon divergence between two distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    js_per_feature = []
    for f in range(p.shape[0]):
        m = 0.5 * (p[f] + q[f])
        kl_pm = np.sum(p[f] * np.log(p[f] / m))
        kl_qm = np.sum(q[f] * np.log(q[f] / m))
        js_per_feature.append(0.5 * (kl_pm + kl_qm))

    return np.mean(js_per_feature)
