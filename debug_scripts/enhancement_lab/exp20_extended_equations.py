"""Lab exp20: Extended equation laboratory.

Massively expanded equation search beyond exp19's 16 formulas.
Tests ~50 equations organized by mathematical family:

Family 1  — Arithmetic: +, −, ×, ÷ of spectral features
Family 2  — Trigonometric: sin, cos, tan transformations of eigenvalue angles
Family 3  — Gap-based: eigenvalue gap ratios, consecutive gap products
Family 4  — Eigenvalue relationship: λ-ratios across parent/child, spectral norm
Family 5  — Conditional/Bayesian: posterior-weighted k using Gate 2 evidence
Family 6  — Hybrid blends: max/min/harmonic/geometric with JL floor

PHASE A — Feature collection (reuses exp19 infrastructure)
PHASE B — Equation evaluation across all nodes (correlation + separation)
PHASE C — Full benchmark of top equations as decomposition strategies
"""
from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import networkx as nx
import numpy as np
from lab_helpers import build_tree_and_data, compute_ari, temporary_experiment_overrides

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    effective_rank as compute_effective_rank,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    marchenko_pastur_signal_count,
)

CASES = [
    "binary_balanced_low_noise__2",
    "gauss_clear_small",
    "binary_low_noise_12c",
    "binary_perfect_8c",
    "binary_hard_4c",
    "gauss_noisy_3c",
    "gauss_overlap_4c_med",
    "gauss_moderate_3c",
    "gauss_moderate_5c",
    "binary_low_noise_4c",
    "binary_perfect_4c",
    "gauss_clear_large",
    "binary_multiscale_4c",
    "binary_many_features",
    "gauss_noisy_many",
]

_leaf_data_cache: dict = {}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _descendant_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    return sorted(d for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


def _node_data(tree, node, leaf_data):
    labels = [tree.nodes[lid].get("label", lid) for lid in _descendant_leaves(tree, node)]
    return leaf_data.loc[labels].values


def _eigendecompose(data):
    if data.shape[0] < 2:
        return None
    eig = eigendecompose_correlation_backend(data, compute_eigenvectors=False)
    if eig is None:
        return None
    return eig.eigenvalues, data.shape[0], eig.active_feature_count


def _mp_upper_bound(eigenvalues, n_samples, d_active):
    pos = eigenvalues[eigenvalues > 0]
    sigma2 = float(np.median(pos)) if len(pos) > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    gamma = d_active / n_samples
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _spectral_gap(eigenvalues, k_mp):
    if k_mp <= 0 or k_mp >= len(eigenvalues):
        return 1.0
    lam_k = eigenvalues[k_mp - 1]
    lam_k1 = eigenvalues[k_mp] if k_mp < len(eigenvalues) else 0.0
    return float(lam_k / max(lam_k1, 1e-12))


def _knee_elbow_k(eigenvalues, max_k=None):
    evals = eigenvalues[eigenvalues > 1e-12]
    if len(evals) < 3:
        return max(1, len(evals))
    if max_k is not None:
        evals = evals[:max_k]
    n = len(evals)
    coords = np.column_stack([np.arange(n), evals])
    line_vec = coords[-1] - coords[0]
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return 1
    line_unit = line_vec / line_len
    vecs = coords - coords[0]
    projs = vecs @ line_unit
    perps = np.sqrt(np.maximum(np.sum(vecs ** 2, axis=1) - projs ** 2, 0.0))
    return int(np.argmax(perps)) + 1


def _consecutive_gap_sum(eigenvalues, max_k=10):
    """Sum of consecutive eigenvalue ratios λ_i/λ_{i+1} for top eigenvalues."""
    total = 0.0
    for i in range(min(max_k, len(eigenvalues) - 1)):
        if eigenvalues[i + 1] > 1e-12:
            total += eigenvalues[i] / eigenvalues[i + 1]
        else:
            total += eigenvalues[i] / 1e-12 if eigenvalues[i] > 0 else 0
    return total


def _max_consecutive_gap_index(eigenvalues, max_k=20):
    """Index (1-based) of the largest consecutive gap λ_i/λ_{i+1}."""
    best_gap = 0.0
    best_idx = 1
    for i in range(min(max_k, len(eigenvalues) - 1)):
        denom = max(eigenvalues[i + 1], 1e-12)
        gap = eigenvalues[i] / denom
        if gap > best_gap:
            best_gap = gap
            best_idx = i + 1
    return best_idx


# ── Feature collection ──────────────────────────────────────────────────────

def collect_features(case_name):
    """Collect extended spectral features at every binary parent."""
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.SIBLING_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    annotations_df = tree.annotations_df

    rows = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        n_L = len(_descendant_leaves(tree, left))
        n_R = len(_descendant_leaves(tree, right))
        if n_L < 2 and n_R < 2:
            continue

        try:
            data_L = _node_data(tree, left, data_df)
            data_R = _node_data(tree, right, data_df)
            data_P = np.vstack([data_L, data_R])
        except (KeyError, IndexError):
            continue

        eig_L = _eigendecompose(data_L)
        eig_R = _eigendecompose(data_R)
        eig_P = _eigendecompose(data_P)

        if eig_P is None:
            continue

        ev_L, ns_L, da_L = eig_L if eig_L is not None else (np.array([0.0]), 1, 1)
        ev_R, ns_R, da_R = eig_R if eig_R is not None else (np.array([0.0]), 1, 1)
        ev_P, ns_P, da_P = eig_P

        k_L = marchenko_pastur_signal_count(ev_L, ns_L, da_L) if eig_L else 0
        k_R = marchenko_pastur_signal_count(ev_R, ns_R, da_R) if eig_R else 0
        k_P = marchenko_pastur_signal_count(ev_P, ns_P, da_P)

        er_L = compute_effective_rank(ev_L) if eig_L else 1.0
        er_R = compute_effective_rank(ev_R) if eig_R else 1.0
        er_P = compute_effective_rank(ev_P)

        n_features = data_df.shape[1]
        jl_k = compute_jl_dim(ns_P, n_features)
        jl_qrt = max(1, jl_k // 4)

        mp_bound_P = _mp_upper_bound(ev_P, ns_P, da_P)
        gap_P = _spectral_gap(ev_P, k_P)
        gap_L = _spectral_gap(ev_L, k_L) if eig_L and k_L > 0 else 1.0
        gap_R = _spectral_gap(ev_R, k_R) if eig_R and k_R > 0 else 1.0

        # Variance-explained thresholds
        cumvar = np.cumsum(ev_P) / max(np.sum(ev_P), 1e-12)
        var90_P = next((i + 1 for i, cv in enumerate(cumvar) if cv >= 0.90), len(ev_P))
        var95_P = next((i + 1 for i, cv in enumerate(cumvar) if cv >= 0.95), len(ev_P))

        knee_P = _knee_elbow_k(ev_P)

        # Top eigenvalues
        lam1_P = float(ev_P[0]) if len(ev_P) > 0 else 0.0
        lam2_P = float(ev_P[1]) if len(ev_P) > 1 else 0.0
        lam3_P = float(ev_P[2]) if len(ev_P) > 2 else 0.0
        lam1_L = float(ev_L[0]) if eig_L and len(ev_L) > 0 else 0.0
        lam1_R = float(ev_R[0]) if eig_R and len(ev_R) > 0 else 0.0
        lam2_L = float(ev_L[1]) if eig_L and len(ev_L) > 1 else 0.0
        lam2_R = float(ev_R[1]) if eig_R and len(ev_R) > 1 else 0.0

        # Consecutive gap features
        max_gap_idx_P = _max_consecutive_gap_index(ev_P)
        gap_sum_P = _consecutive_gap_sum(ev_P, max_k=min(20, len(ev_P)))

        # Spectral norm ratio: how much stronger is the biggest eigenvalue relative to bulk?
        trace_P = float(np.sum(ev_P))
        spectral_ratio_P = lam1_P / max(trace_P, 1e-12)

        # Edge p-values (Gate 2 evidence for Bayesian weighting)
        edge_p_L = 1.0
        edge_p_R = 1.0
        if left in annotations_df.index:
            edge_p_L = float(annotations_df.loc[left].get("Child_Parent_Divergence_P_Value_BH", 1.0))
        if right in annotations_df.index:
            edge_p_R = float(annotations_df.loc[right].get("Child_Parent_Divergence_P_Value_BH", 1.0))

        # Gate 3 decision
        sibling_diff = False
        if parent in annotations_df.index:
            sibling_diff = bool(annotations_df.loc[parent].get("Sibling_BH_Different", False))

        rows.append({
            "parent": parent,
            "n_L": n_L, "n_R": n_R, "n_P": n_L + n_R,
            "d_active": da_P, "n_features": n_features,
            # MP signal counts
            "k_L": k_L, "k_R": k_R, "k_P": k_P,
            "k_min": min(k_L, k_R) if eig_L and eig_R else 0,
            "k_max": max(k_L, k_R),
            "k_sum": k_L + k_R,
            # Effective rank
            "er_L": er_L, "er_R": er_R, "er_P": er_P,
            # JL reference
            "jl_k": jl_k, "jl_qrt": jl_qrt,
            # Spectral gaps
            "gap_P": gap_P, "gap_L": gap_L, "gap_R": gap_R,
            "max_gap_idx_P": max_gap_idx_P, "gap_sum_P": gap_sum_P,
            # Variance explained
            "var90_P": var90_P, "var95_P": var95_P,
            # Knee/elbow
            "knee_P": knee_P,
            # Eigenvalues (top 3 parent, top 2 per child)
            "lam1_P": lam1_P, "lam2_P": lam2_P, "lam3_P": lam3_P,
            "lam1_L": lam1_L, "lam1_R": lam1_R,
            "lam2_L": lam2_L, "lam2_R": lam2_R,
            # Derived spectral features
            "mp_bound_P": mp_bound_P,
            "trace_P": trace_P,
            "spectral_ratio_P": spectral_ratio_P,
            # Gate 2 evidence (for Bayesian weighting)
            "edge_p_L": edge_p_L, "edge_p_R": edge_p_R,
            "min_edge_p": min(edge_p_L, edge_p_R),
            # Gate 3
            "SPLIT": sibling_diff,
        })

    return rows, tc, decomp


# ═════════════════════════════════════════════════════════════════════════════
# EQUATIONS — organized by mathematical family
# ═════════════════════════════════════════════════════════════════════════════

# Helper: safe operations
def _safe_div(a, b, default=0.0):
    return a / b if b > 1e-12 else default

def _clamp_k(x):
    """Clamp to integer ≥ 2."""
    return max(2, int(round(x)))


# ── Family 0: Baselines ────────────────────────────────────────────────────

def eq_jl_qrt(r):
    """JL/4 — our champion baseline."""
    return max(2, r["jl_qrt"])

def eq_min_child(r):
    """min(k_L, k_R) — the regression."""
    return max(2, r["k_min"])


# ── Family 1: Arithmetic combinations ──────────────────────────────────────

def eq_k_plus_er(r):
    """k_P + er_P: signal + rank."""
    return _clamp_k(r["k_P"] + r["er_P"])

def eq_k_minus_gap(r):
    """k_P − 1/gap_P: penalize weak gaps."""
    return _clamp_k(r["k_P"] - _safe_div(1.0, r["gap_P"], 0.0))

def eq_k_times_gap(r):
    """k_P × min(gap_P, 3): reward strong gaps, cap to avoid explosion."""
    return _clamp_k(r["k_P"] * min(r["gap_P"], 3.0))

def eq_er_times_spectral_ratio(r):
    """er_P × (λ₁/trace): rank scaled by dominance."""
    return _clamp_k(r["er_P"] * r["spectral_ratio_P"])

def eq_k_div_gap_sum(r):
    """k_P × n_P / gap_sum: normalize by total gap structure."""
    return _clamp_k(r["k_P"] * r["n_P"] / max(r["gap_sum_P"], 1.0))

def eq_sum_child_er(r):
    """er_L + er_R: additive child ranks."""
    return _clamp_k(r["er_L"] + r["er_R"])

def eq_diff_parent_child_er(r):
    """er_P − (er_L + er_R)/2: excess rank in parent."""
    delta = r["er_P"] - (r["er_L"] + r["er_R"]) / 2
    return _clamp_k(max(delta, 1))

def eq_product_child_k(r):
    """√(k_L × k_R): geometric mean of child signal counts."""
    return _clamp_k(np.sqrt(max(r["k_L"], 1) * max(r["k_R"], 1)))

def eq_k_parent_minus_k_max(r):
    """k_P − max(k_L, k_R) + JL/4: novel dimensions in parent + floor."""
    novel = r["k_P"] - r["k_max"]
    return _clamp_k(novel + r["jl_qrt"])

def eq_er_div_k(r):
    """er_P / max(k_P, 1) × JL/4: effective-rank-to-mp ratio as multiplier."""
    ratio = r["er_P"] / max(r["k_P"], 1)
    return _clamp_k(ratio * r["jl_qrt"])


# ── Family 2: Trigonometric ────────────────────────────────────────────────
# Treat spectral features as angles (normalized to [0, π/2]) or use
# trigonometric functions to create nonlinear mappings.

def eq_sin_lam_ratio(r):
    """sin(π/2 × λ₁_P/max_lam) × JL/4: smoothly maps dominance to k."""
    max_lam = max(r["lam1_P"], 1.0)
    ratio = min(r["lam1_P"] / max_lam, 1.0)
    return _clamp_k(np.sin(np.pi / 2 * ratio) * r["jl_qrt"])

def eq_cos_n_ratio(r):
    """cos(π/2 × min(n_L,n_R)/n_P) × JL: balance-sensitive.
    Balanced children → cos(~π/4) ≈ 0.71 → 71% of JL.
    Unbalanced → cos(~0) ≈ 1.0 → full JL (more power needed)."""
    balance = min(r["n_L"], r["n_R"]) / max(r["n_P"], 1)
    return _clamp_k(np.cos(np.pi / 2 * balance) * r["jl_k"])

def eq_sin_er_ratio(r):
    """sin(π × er_P/(er_L+er_R+1)) × JL/4: phase angle of rank ratio."""
    ratio = r["er_P"] / max(r["er_L"] + r["er_R"], 0.1)
    return _clamp_k(np.sin(np.pi * min(ratio, 1.0)) * r["jl_qrt"])

def eq_atan_gap(r):
    """2/π × atan(gap_P) × JL/4: saturating gap→k via arctangent."""
    return _clamp_k((2.0 / np.pi) * np.arctan(r["gap_P"]) * r["jl_qrt"])

def eq_cos_spectral_angle(r):
    """cos(π × k_P/(k_P + JL/4)) × JL: phase between spectral and JL."""
    ratio = r["k_P"] / max(r["k_P"] + r["jl_qrt"], 1)
    return _clamp_k(np.cos(np.pi * ratio) * r["jl_k"])

def eq_sin_eigenvalue_phase(r):
    """sin(atan2(λ₂, λ₁)) × k_P + JL/4: eigenvalue angle → k boost."""
    angle = np.arctan2(r["lam2_P"], max(r["lam1_P"], 1e-12))
    return _clamp_k(np.sin(angle) * max(r["k_P"], 1) + r["jl_qrt"])


# ── Family 3: Gap-based ───────────────────────────────────────────────────

def eq_max_gap_idx(r):
    """Index of largest consecutive gap in parent eigenvalues."""
    return _clamp_k(r["max_gap_idx_P"])

def eq_max_gap_idx_jl(r):
    """max(max_gap_idx_P, JL/4): gap-index with JL floor."""
    return max(2, r["max_gap_idx_P"], r["jl_qrt"])

def eq_gap_ratio_children(r):
    """(gap_P / (gap_L × gap_R)^0.5) × k_P: relative gap strength."""
    child_gap = np.sqrt(max(r["gap_L"], 0.01) * max(r["gap_R"], 0.01))
    return _clamp_k(r["gap_P"] / max(child_gap, 0.01) * max(r["k_P"], 1))

def eq_gap_sum_normalized(r):
    """gap_sum / n_eigenvalues × JL/4: average gap quality × floor."""
    avg_gap = r["gap_sum_P"] / max(r["k_P"], 1)
    return _clamp_k(min(avg_gap, 5.0) * r["jl_qrt"])

def eq_gap_times_er(r):
    """min(gap_P, 3) × er_P: gap quality × rank breadth."""
    return _clamp_k(min(r["gap_P"], 3.0) * r["er_P"])

def eq_inv_gap_as_penalty(r):
    """JL/4 × (1 − 1/gap_P): penalize dimension if gap is small."""
    penalty = max(0.0, 1.0 - _safe_div(1.0, r["gap_P"], 1.0))
    return _clamp_k(r["jl_qrt"] * max(penalty, 0.3))


# ── Family 4: Eigenvalue relationships ─────────────────────────────────────

def eq_lam1_div_lam2(r):
    """λ₁/λ₂ as proxy for signal dimensionality."""
    ratio = _safe_div(r["lam1_P"], r["lam2_P"], 1.0)
    return _clamp_k(ratio)

def eq_lam1_plus_lam2_div_trace(r):
    """(λ₁ + λ₂)/trace × JL: top-2 dominance fraction → scale JL."""
    frac = (r["lam1_P"] + r["lam2_P"]) / max(r["trace_P"], 1e-12)
    return _clamp_k(frac * r["jl_k"])

def eq_lam_parent_div_child_max(r):
    """λ₁_P / max(λ₁_L, λ₁_R): parent dominance vs children."""
    child_max = max(r["lam1_L"], r["lam1_R"], 1e-12)
    return _clamp_k(r["lam1_P"] / child_max)

def eq_spectral_norm_ratio(r):
    """(λ₁_P / mp_bound × JL/4): how far above noise × floor."""
    snr = _safe_div(r["lam1_P"], r["mp_bound_P"], 1.0)
    return _clamp_k(snr * r["jl_qrt"])

def eq_log_lam1(r):
    """log(1 + λ₁_P) × 2: logarithmic eigenvalue → k."""
    return _clamp_k(np.log1p(r["lam1_P"]) * 2)

def eq_lam_spread(r):
    """(λ₁ − λ₃) / λ₂ : how spread the top eigenvalues are."""
    spread = (r["lam1_P"] - r["lam3_P"]) / max(r["lam2_P"], 1e-12)
    return _clamp_k(spread)

def eq_child_lam_sum_over_parent(r):
    """(λ₁_L + λ₁_R) / λ₁_P × JL/4: child contributes relative to parent."""
    ratio = (r["lam1_L"] + r["lam1_R"]) / max(r["lam1_P"], 1e-12)
    return _clamp_k(ratio * r["jl_qrt"])

def eq_lam2_chain(r):
    """(λ₂_L + λ₂_R) / λ₂_P: second-eigenvalue relationship."""
    ratio = (r["lam2_L"] + r["lam2_R"]) / max(r["lam2_P"], 1e-12)
    return _clamp_k(ratio * max(r["k_P"], 2))


# ── Family 5: Conditional / Bayesian ───────────────────────────────────────
# Use Gate 2 p-values as prior evidence to weight k.

def eq_bayesian_posterior_k(r):
    """Bayesian posterior: prior=JL/4, evidence=edge p-values.
    Low p-values (strong signal) → trust spectral k; high p → fall back to JL/4.
    posterior_k = (1−w) × spectral_k + w × JL/4 where w = min_edge_p."""
    w = min(r["min_edge_p"], 1.0)
    spectral_k = max(r["k_P"], r["knee_P"])
    return _clamp_k((1 - w) * spectral_k + w * r["jl_qrt"])

def eq_bayesian_er_jl(r):
    """Bayesian blend: prior=JL/4, likelihood=er_P.
    Blend weight from edge p-value strength: strong evidence → trust er, weak → JL."""
    w = min(r["min_edge_p"], 1.0)
    return _clamp_k((1 - w) * r["er_P"] + w * r["jl_qrt"])

def eq_conditional_gap_boost(r):
    """If gap > 2 (strong separation): use k_P.
    If gap == 1 (no separation): use JL/4.
    Interpolate between."""
    strength = min((r["gap_P"] - 1.0) / 2.0, 1.0)
    strength = max(strength, 0.0)
    return _clamp_k(strength * max(r["k_P"], r["knee_P"]) + (1 - strength) * r["jl_qrt"])

def eq_conditional_n_threshold(r):
    """If n_P > 30: use spectral (enough data). If n_P <= 10: use JL/4 (too little).
    Smooth transition between."""
    t = min(max((r["n_P"] - 10) / 20, 0.0), 1.0)
    spectral_k = max(r["k_P"], int(round(r["er_P"])))
    return _clamp_k(t * spectral_k + (1 - t) * r["jl_qrt"])

def eq_posterior_lam_evidence(r):
    """Posterior k from eigenvalue evidence:
    evidence = λ₁_P / mp_bound (how much signal above noise).
    k = JL/4 × sigmoid(evidence − 1): above noise → boost, below → JL/4."""
    evidence = _safe_div(r["lam1_P"], r["mp_bound_P"], 0.0) - 1.0
    sigmoid = 1.0 / (1.0 + np.exp(-2 * evidence))
    return _clamp_k(r["jl_qrt"] + sigmoid * max(r["k_P"], 0))

def eq_conditional_var_gate(r):
    """Gate on variance explained: if var90 < 5 dims → compact signal, trust it.
    If var90 > 20 → diffuse signal, fall back to JL/4."""
    if r["var90_P"] <= 5:
        return _clamp_k(r["var90_P"])
    elif r["var90_P"] >= 20:
        return max(2, r["jl_qrt"])
    t = (r["var90_P"] - 5) / 15
    return _clamp_k((1 - t) * r["var90_P"] + t * r["jl_qrt"])

def eq_bayesian_gap_prior(r):
    """Bayesian with gap as prior confidence.
    High gap → strong prior on k_P; low gap → fall back to JL/4.
    prior_strength = tanh(gap − 1) ∈ [0, 1]."""
    prior_strength = np.tanh(max(r["gap_P"] - 1.0, 0.0))
    return _clamp_k(prior_strength * max(r["k_P"], 2) + (1 - prior_strength) * r["jl_qrt"])


# ── Family 6: Hybrid blends with JL floor ──────────────────────────────────

def eq_max_gap_idx_er_jl(r):
    """max(max_gap_idx, er_P, JL/4): triple-max hybrid."""
    return max(2, r["max_gap_idx_P"], int(round(r["er_P"])), r["jl_qrt"])

def eq_harmonic_er_knee(r):
    """Harmonic mean of er_P and knee_P, floored by JL/4."""
    er = max(r["er_P"], 1)
    kn = max(r["knee_P"], 1)
    h = 2 * er * kn / (er + kn)
    return max(2, int(round(h)), r["jl_qrt"])

def eq_weighted_avg_jl(r):
    """Weighted avg: 0.5×JL/4 + 0.3×er_P + 0.2×k_P."""
    return _clamp_k(0.5 * r["jl_qrt"] + 0.3 * r["er_P"] + 0.2 * max(r["k_P"], 2))

def eq_min_var90_jl(r):
    """min(var90_P, JL/4): cap at JL/4 rather than floor."""
    return max(2, min(r["var90_P"], r["jl_qrt"]))

def eq_adaptive_blend(r):
    """Adaptive: if k_P close to JL/4 (within 2×), use geometric mean.
    Otherwise use softer of the two."""
    if r["k_P"] <= 0:
        return max(2, r["jl_qrt"])
    ratio = r["jl_qrt"] / max(r["k_P"], 1)
    if 0.5 <= ratio <= 2.0:
        return _clamp_k(np.sqrt(r["jl_qrt"] * max(r["k_P"], 1)))
    return max(2, r["jl_qrt"])

def eq_cubic_root_product(r):
    """∛(JL/4 × er_P × knee_P): triple geometric mean."""
    vals = [max(r["jl_qrt"], 1), max(r["er_P"], 1), max(r["knee_P"], 1)]
    return _clamp_k(np.cbrt(vals[0] * vals[1] * vals[2]))


# ═════════════════════════════════════════════════════════════════════════════
# Equation registry
# ═════════════════════════════════════════════════════════════════════════════

EQUATIONS = {
    # Family 0: Baselines
    "jl_qrt": eq_jl_qrt,
    "min_child": eq_min_child,
    # Family 1: Arithmetic
    "k_plus_er": eq_k_plus_er,
    "k_minus_gap": eq_k_minus_gap,
    "k_times_gap": eq_k_times_gap,
    "er_x_spec_ratio": eq_er_times_spectral_ratio,
    "k_div_gapsum": eq_k_div_gap_sum,
    "sum_child_er": eq_sum_child_er,
    "diff_par_child_er": eq_diff_parent_child_er,
    "geom_child_k": eq_product_child_k,
    "kP_minus_kmax_jl": eq_k_parent_minus_k_max,
    "er_div_k_x_jl": eq_er_div_k,
    # Family 2: Trigonometric
    "sin_lam_ratio": eq_sin_lam_ratio,
    "cos_n_ratio": eq_cos_n_ratio,
    "sin_er_ratio": eq_sin_er_ratio,
    "atan_gap": eq_atan_gap,
    "cos_spec_angle": eq_cos_spectral_angle,
    "sin_eig_phase": eq_sin_eigenvalue_phase,
    # Family 3: Gap-based
    "max_gap_idx": eq_max_gap_idx,
    "max_gap_idx_jl": eq_max_gap_idx_jl,
    "gap_ratio_children": eq_gap_ratio_children,
    "gap_sum_norm": eq_gap_sum_normalized,
    "gap_x_er": eq_gap_times_er,
    "inv_gap_penalty": eq_inv_gap_as_penalty,
    # Family 4: Eigenvalue relationships
    "lam1_div_lam2": eq_lam1_div_lam2,
    "lam12_frac_jl": eq_lam1_plus_lam2_div_trace,
    "lam_P_div_child": eq_lam_parent_div_child_max,
    "spec_norm_x_jl": eq_spectral_norm_ratio,
    "log_lam1": eq_log_lam1,
    "lam_spread": eq_lam_spread,
    "child_lam_over_P": eq_child_lam_sum_over_parent,
    "lam2_chain": eq_lam2_chain,
    # Family 5: Conditional / Bayesian
    "bayes_post_k": eq_bayesian_posterior_k,
    "bayes_er_jl": eq_bayesian_er_jl,
    "cond_gap_boost": eq_conditional_gap_boost,
    "cond_n_thresh": eq_conditional_n_threshold,
    "post_lam_evid": eq_posterior_lam_evidence,
    "cond_var_gate": eq_conditional_var_gate,
    "bayes_gap_prior": eq_bayesian_gap_prior,
    # Family 6: Hybrid blends
    "max_gapidx_er_jl": eq_max_gap_idx_er_jl,
    "harm_er_knee": eq_harmonic_er_knee,
    "weighted_avg_jl": eq_weighted_avg_jl,
    "min_var90_jl": eq_min_var90_jl,
    "adaptive_blend": eq_adaptive_blend,
    "cbrt_product": eq_cubic_root_product,
}

N_EQ = len(EQUATIONS)


# ── Strategy factories ──────────────────────────────────────────────────────

def _make_equation_strategy(eq_fn):
    """Produce a _derive function from a row-level equation."""
    def _derive(tree, annotated_df):
        leaf_data = _leaf_data_cache.get("leaf_data")
        if leaf_data is None:
            return None

        sibling_dims = {}
        for parent in tree.nodes:
            children = list(tree.successors(parent))
            if len(children) != 2:
                continue
            left, right = children

            try:
                data_L = _node_data(tree, left, leaf_data)
                data_R = _node_data(tree, right, leaf_data)
            except (KeyError, IndexError):
                continue

            data_P = np.vstack([data_L, data_R])
            eig_P = _eigendecompose(data_P)
            if eig_P is None:
                continue

            ev_P, ns_P, da_P = eig_P
            eig_L = _eigendecompose(data_L)
            eig_R = _eigendecompose(data_R)
            ev_L, ns_L, da_L = eig_L if eig_L is not None else (np.array([0.0]), 1, 1)
            ev_R, ns_R, da_R = eig_R if eig_R is not None else (np.array([0.0]), 1, 1)

            n_features = leaf_data.shape[1]
            n_L = len(_descendant_leaves(tree, left))
            n_R = len(_descendant_leaves(tree, right))
            n_P = n_L + n_R

            k_P = marchenko_pastur_signal_count(ev_P, ns_P, da_P)
            k_L = marchenko_pastur_signal_count(ev_L, ns_L, da_L) if eig_L else 0
            k_R = marchenko_pastur_signal_count(ev_R, ns_R, da_R) if eig_R else 0
            er_P = compute_effective_rank(ev_P)
            er_L = compute_effective_rank(ev_L) if eig_L else 1.0
            er_R = compute_effective_rank(ev_R) if eig_R else 1.0

            cumvar = np.cumsum(ev_P) / max(np.sum(ev_P), 1e-12)
            var90 = next((i + 1 for i, cv in enumerate(cumvar) if cv >= 0.90), len(ev_P))
            var95 = next((i + 1 for i, cv in enumerate(cumvar) if cv >= 0.95), len(ev_P))

            # Edge p-values from annotated_df
            edge_p_L = 1.0
            edge_p_R = 1.0
            if left in annotated_df.index:
                edge_p_L = float(annotated_df.loc[left].get(
                    "Child_Parent_Divergence_P_Value_BH", 1.0))
            if right in annotated_df.index:
                edge_p_R = float(annotated_df.loc[right].get(
                    "Child_Parent_Divergence_P_Value_BH", 1.0))

            jl_k_val = compute_jl_dim(n_P, n_features)
            trace_P = float(np.sum(ev_P))

            row = {
                "n_L": n_L, "n_R": n_R, "n_P": n_P,
                "d_active": da_P, "n_features": n_features,
                "k_L": k_L, "k_R": k_R, "k_P": k_P,
                "k_min": min(k_L, k_R) if eig_L and eig_R else 0,
                "k_max": max(k_L, k_R),
                "k_sum": k_L + k_R,
                "er_L": er_L, "er_R": er_R, "er_P": er_P,
                "jl_k": jl_k_val, "jl_qrt": max(1, jl_k_val // 4),
                "gap_P": _spectral_gap(ev_P, k_P),
                "gap_L": _spectral_gap(ev_L, k_L) if eig_L and k_L > 0 else 1.0,
                "gap_R": _spectral_gap(ev_R, k_R) if eig_R and k_R > 0 else 1.0,
                "max_gap_idx_P": _max_consecutive_gap_index(ev_P),
                "gap_sum_P": _consecutive_gap_sum(ev_P, max_k=min(20, len(ev_P))),
                "var90_P": var90, "var95_P": var95,
                "knee_P": _knee_elbow_k(ev_P),
                "lam1_P": float(ev_P[0]) if len(ev_P) > 0 else 0.0,
                "lam2_P": float(ev_P[1]) if len(ev_P) > 1 else 0.0,
                "lam3_P": float(ev_P[2]) if len(ev_P) > 2 else 0.0,
                "lam1_L": float(ev_L[0]) if eig_L and len(ev_L) > 0 else 0.0,
                "lam1_R": float(ev_R[0]) if eig_R and len(ev_R) > 0 else 0.0,
                "lam2_L": float(ev_L[1]) if eig_L and len(ev_L) > 1 else 0.0,
                "lam2_R": float(ev_R[1]) if eig_R and len(ev_R) > 1 else 0.0,
                "mp_bound_P": _mp_upper_bound(ev_P, ns_P, da_P),
                "trace_P": trace_P,
                "spectral_ratio_P": float(ev_P[0]) / max(trace_P, 1e-12) if len(ev_P) > 0 else 0,
                "edge_p_L": edge_p_L, "edge_p_R": edge_p_R,
                "min_edge_p": min(edge_p_L, edge_p_R),
            }

            k = eq_fn(row)
            sibling_dims[parent] = k

        return sibling_dims if sibling_dims else None

    return _derive


def _derive_none(tree, annotated_df):
    return None


def _derive_jl_floor_qrt(tree, annotated_df):
    edge_spectral_dims = annotated_df.attrs.get("_spectral_dims")
    if not edge_spectral_dims:
        return None
    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        k_left = edge_spectral_dims.get(left, 0)
        k_right = edge_spectral_dims.get(right, 0)
        positive = [k for k in (k_left, k_right) if k > 0]
        if not positive:
            continue
        spectral_k = min(positive)
        n_left = tree.nodes[left].get("leaf_count", 1)
        n_right = tree.nodes[right].get("leaf_count", 1)
        n_parent = n_left + n_right
        dist = tree.nodes[left].get("distribution")
        n_features = len(dist) if dist is not None else 100
        jl_k = compute_jl_dim(n_parent, n_features)
        floor_k = max(1, jl_k // 4)
        sibling_dims[parent] = max(spectral_k, floor_k)
    return sibling_dims if sibling_dims else None


def run_case_strategy(case_name, strategy_fn):
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)
    with temporary_experiment_overrides(
        leaf_data_cache=_leaf_data_cache,
        leaf_data=data_df,
        sibling_dims=strategy_fn,
    ):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )
    ari = compute_ari(decomp, data_df, true_labels) if true_labels is not None else float("nan")
    return {
        "true_k": tc.get("n_clusters", "?"),
        "found_k": decomp["num_clusters"],
        "ari": round(ari, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print("        SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed)")
    print(f"        Equations: {N_EQ}")
    print()

    # ═══ PHASE A: Feature collection ═══
    print("═══ PHASE A: FEATURE COLLECTION ═══\n")
    all_rows = []
    for case_name in CASES:
        rows, tc, _ = collect_features(case_name)
        true_k = tc.get("n_clusters", "?")
        splits = [r for r in rows if r["SPLIT"]]
        merges = [r for r in rows if not r["SPLIT"]]
        print(f"  {case_name:<35} K={true_k}  nodes={len(rows):>3}  "
              f"SPLIT={len(splits):>2}  MERGE={len(merges):>3}")
        for r in rows:
            r["case"] = case_name
            r["true_k"] = true_k
        all_rows.extend(rows)

    splits_all = [r for r in all_rows if r["SPLIT"]]
    merges_all = [r for r in all_rows if not r["SPLIT"]]
    print(f"\n  TOTAL: {len(all_rows)} nodes ({len(splits_all)} SPLIT, {len(merges_all)} MERGE)")

    # ═══ PHASE B: Equation evaluation ═══
    print("\n═══ PHASE B: EQUATION EVALUATION ═══")
    print(f"\n  {N_EQ} equations across 6 families\n")

    families = {
        "Baselines": ["jl_qrt", "min_child"],
        "Arithmetic": ["k_plus_er", "k_minus_gap", "k_times_gap", "er_x_spec_ratio",
                        "k_div_gapsum", "sum_child_er", "diff_par_child_er",
                        "geom_child_k", "kP_minus_kmax_jl", "er_div_k_x_jl"],
        "Trigonometric": ["sin_lam_ratio", "cos_n_ratio", "sin_er_ratio",
                          "atan_gap", "cos_spec_angle", "sin_eig_phase"],
        "Gap-based": ["max_gap_idx", "max_gap_idx_jl", "gap_ratio_children",
                      "gap_sum_norm", "gap_x_er", "inv_gap_penalty"],
        "Eigenvalue": ["lam1_div_lam2", "lam12_frac_jl", "lam_P_div_child",
                       "spec_norm_x_jl", "log_lam1", "lam_spread",
                       "child_lam_over_P", "lam2_chain"],
        "Bayesian": ["bayes_post_k", "bayes_er_jl", "cond_gap_boost",
                     "cond_n_thresh", "post_lam_evid", "cond_var_gate",
                     "bayes_gap_prior"],
        "Hybrid": ["max_gapidx_er_jl", "harm_er_knee", "weighted_avg_jl",
                   "min_var90_jl", "adaptive_blend", "cbrt_product"],
    }

    print(f"  {'Equation':<22} {'Family':<14} │ "
          f"{'Mean k':>7} {'Med k':>7} │ "
          f"{'k(SPLIT)':>8} {'k(MERGE)':>8} │ "
          f"{'Corr JL/4':>10} │ "
          f"{'Sep':>6}")
    print("  " + "─" * 110)

    phase_b_results = {}
    for fam_name, eq_names in families.items():
        for eq_name in eq_names:
            eq_fn = EQUATIONS[eq_name]
            ks = [eq_fn(r) for r in all_rows]
            jl_qrts = [r["jl_qrt"] for r in all_rows]
            ks_split = [eq_fn(r) for r in splits_all]
            ks_merge = [eq_fn(r) for r in merges_all]

            corr = float(np.corrcoef(ks, jl_qrts)[0, 1]) \
                if np.std(ks) > 0 and np.std(jl_qrts) > 0 else 0.0

            # Separation: how well does the equation's k separate SPLIT from MERGE?
            s_mean = np.mean(ks_split) if ks_split else 0
            m_mean = np.mean(ks_merge) if ks_merge else 0
            s_std = np.std(ks_split) if ks_split else 1
            m_std = np.std(ks_merge) if ks_merge else 1
            sep = abs(s_mean - m_mean) / (s_std + m_std + 1e-6)

            phase_b_results[eq_name] = {
                "corr": corr, "sep": sep,
                "mean_k": np.mean(ks), "med_k": np.median(ks),
                "mean_k_split": s_mean, "mean_k_merge": m_mean,
            }

            print(f"  {eq_name:<22} {fam_name:<14} │ "
                  f"{np.mean(ks):>7.1f} {np.median(ks):>7.1f} │ "
                  f"{s_mean:>8.1f} {m_mean:>8.1f} │ "
                  f"{corr:>10.3f} │ "
                  f"{sep:>6.3f}")

    # ═══ PHASE C: Benchmark ═══
    print("\n═══ PHASE C: BENCHMARK ═══")
    print(f"\n  Running {N_EQ} equations + 2 baselines across {len(CASES)} cases\n")

    # Build strategy map: baselines + all equations
    benchmark_strategies = {
        "none": _derive_none,
        "jl_floor_qrt": _derive_jl_floor_qrt,
    }
    for eq_name, eq_fn in EQUATIONS.items():
        if eq_name in ("jl_qrt", "min_child"):
            continue  # Covered by baselines
        benchmark_strategies[eq_name] = _make_equation_strategy(eq_fn)

    strat_names = list(benchmark_strategies.keys())

    # Run all strategies on all cases
    all_results: dict[str, dict] = {}
    for case_name in CASES:
        print(f"  Running {case_name}...", end=" ", flush=True)
        for sname in strat_names:
            try:
                r = run_case_strategy(case_name, benchmark_strategies[sname])
                all_results.setdefault(case_name, {})[sname] = r
            except Exception:
                all_results.setdefault(case_name, {})[sname] = {
                    "true_k": "?", "found_k": -1, "ari": 0.0,
                }
        print("done")

    # Summary by family
    print()
    print("═══ RESULTS BY FAMILY ═══\n")

    print(f"  {'Strategy':<22} {'Family':<14} │ {'Mean ARI':>9} │ {'Wins':>5} │ {'Perfect':>7}")
    print("  " + "─" * 72)

    ari_by_strat = {}
    for sname in strat_names:
        aris = [all_results[c][sname]["ari"] for c in CASES
                if c in all_results and sname in all_results[c]]
        ari_by_strat[sname] = np.mean(aris) if aris else 0.0

    # Print baselines first
    for sname in ["none", "jl_floor_qrt"]:
        aris = [all_results[c][sname]["ari"] for c in CASES
                if c in all_results and sname in all_results[c]]
        wins = sum(1 for c in CASES if c in all_results and sname in all_results[c]
                   and abs(all_results[c][sname]["ari"] -
                           max(all_results[c][s]["ari"] for s in all_results[c])) < 0.001)
        perfect = sum(1 for a in aris if a >= 0.999)
        print(f"  {sname:<22} {'baseline':<14} │ {np.mean(aris):>9.3f} │ {wins:>5} │ {perfect:>7}")

    print("  " + "─" * 72)

    # Print by family
    for fam_name, eq_names in families.items():
        strategy_eq_names = [n for n in eq_names if n not in ("jl_qrt", "min_child")]
        for eq_name in strategy_eq_names:
            aris = [all_results[c][eq_name]["ari"] for c in CASES
                    if c in all_results and eq_name in all_results[c]]
            if not aris:
                continue
            mean_ari = np.mean(aris)
            wins = sum(1 for c in CASES if c in all_results and eq_name in all_results[c]
                       and abs(all_results[c][eq_name]["ari"] -
                               max(all_results[c][s]["ari"] for s in all_results[c])) < 0.001)
            perfect = sum(1 for a in aris if a >= 0.999)
            print(f"  {eq_name:<22} {fam_name:<14} │ {mean_ari:>9.3f} │ {wins:>5} │ {perfect:>7}")

    # Final ranking
    print()
    print("═══ FINAL RANKING (top 20) ═══\n")
    ranked = sorted(ari_by_strat.items(), key=lambda x: -x[1])
    for rank, (sname, mean_ari) in enumerate(ranked[:20], 1):
        wins = sum(1 for c in CASES if c in all_results and sname in all_results[c]
                   and abs(all_results[c][sname]["ari"] -
                           max(all_results[c][s]["ari"] for s in all_results[c])) < 0.001)
        aris = [all_results[c][sname]["ari"] for c in CASES
                if c in all_results and sname in all_results[c]]
        perfect = sum(1 for a in aris if a >= 0.999)
        # Identify family
        fam = "baseline"
        for fn, eqs in families.items():
            if sname in eqs:
                fam = fn
                break
        print(f"  {rank:>2}. {sname:<22} [{fam:<14}]  ARI={mean_ari:.3f}  wins={wins}  perfect={perfect}")

    # Per-case detail for top-5
    print()
    print("═══ PER-CASE DETAIL (top 5 strategies) ═══\n")
    top5 = [s for s, _ in ranked[:5]]
    header = f"  {'Case':<30} {'TK':>3}"
    for s in top5:
        header += f" │ {s[:12]:>12}"
    print(header)
    print("  " + "─" * (34 + 15 * len(top5)))

    for case_name in CASES:
        tk = all_results[case_name][top5[0]]["true_k"]
        line = f"  {case_name:<30} {tk:>3}"
        for sname in top5:
            r = all_results[case_name].get(sname, {"found_k": "?", "ari": 0.0})
            line += f" │ K={r['found_k']:>2} {r['ari']:>5.3f}"
        print(line)

    # Family summary
    print()
    print("═══ FAMILY SUMMARY ═══\n")
    for fam_name, eq_names in families.items():
        fam_aris = [(n, ari_by_strat.get(n, 0)) for n in eq_names
                    if n in ari_by_strat]
        if not fam_aris:
            continue
        best_name, best_ari = max(fam_aris, key=lambda x: x[1])
        avg_ari = np.mean([a for _, a in fam_aris])
        print(f"  {fam_name:<14}: best={best_name:<22} ARI={best_ari:.3f}  "
              f"family_avg={avg_ari:.3f}  n_equations={len(fam_aris)}")
