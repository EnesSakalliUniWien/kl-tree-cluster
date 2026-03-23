"""Lab exp19: Spectral equation laboratory.

Deep analysis of eigenvalue/eigenvector structure at every binary parent,
then creates and benchmarks multiple equation-based k derivation strategies.

PHASE A — Raw spectral map:
    For every binary parent, extract full eigenvalue spectra of L, R, P.
    Compute: MP bounds, spectral gaps, variance-explained curves, effective
    rank, condition numbers, eigenvalue ratios.

PHASE B — Candidate equations:
    Derive k from spectral quantities via ~15 different formulas.
    Correlate each formula's k with the "ideal" k (JL/4 — our best so far)
    and with Gate 3 SPLIT/MERGE outcomes.

PHASE C — Benchmark:
    Run decomposition with each equation-based strategy, measure ARI.
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
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_spectral_dims as current_derive_sibling_spectral_dims,
)

orig_derive = current_derive_sibling_spectral_dims

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

_leaf_data_cache = {}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _descendant_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    return sorted(d for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


def _node_data(tree, node, leaf_data):
    labels = [tree.nodes[lid].get("label", lid) for lid in _descendant_leaves(tree, node)]
    return leaf_data.loc[labels].values


def _eigendecompose(data):
    """Return (eigenvalues_sorted_desc, n_samples, d_active) or None."""
    if data.shape[0] < 2:
        return None
    eig = eigendecompose_correlation_backend(data, need_eigh=False)
    if eig is None:
        return None
    return eig.eigenvalues, data.shape[0], eig.d_active


def _mp_upper_bound(eigenvalues, n_samples, d_active):
    """Compute MP upper edge λ_+ = σ²(1 + √(d/n))²."""
    pos = eigenvalues[eigenvalues > 0]
    sigma2 = float(np.median(pos)) if len(pos) > 0 else 0.0
    if sigma2 <= 0:
        return 0.0
    gamma = d_active / n_samples
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _spectral_gap(eigenvalues, k_mp):
    """Ratio of k-th to (k+1)-th eigenvalue (gap between signal & noise)."""
    if k_mp <= 0 or k_mp >= len(eigenvalues):
        return 1.0
    lam_k = eigenvalues[k_mp - 1]
    lam_k1 = eigenvalues[k_mp] if k_mp < len(eigenvalues) else 0.0
    return float(lam_k / max(lam_k1, 1e-12))


def _variance_explained(eigenvalues, k):
    """Fraction of total variance explained by top-k eigenvalues."""
    total = float(np.sum(eigenvalues))
    if total <= 0 or k <= 0:
        return 0.0
    return float(np.sum(eigenvalues[:k])) / total


def _knee_elbow_k(eigenvalues, max_k=None):
    """Find elbow/knee point in eigenvalue curve via max-distance method."""
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


# ── PHASE A: Spectral map ──────────────────────────────────────────────────

def collect_spectral_map(case_name):
    """For every binary parent, collect rich spectral features."""
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)

    # Run decomposition to get Gate 3 decisions
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.SIBLING_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    stats_df = tree.stats_df

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

        mp_bound_P = _mp_upper_bound(ev_P, ns_P, da_P)
        gap_P = _spectral_gap(ev_P, k_P)
        gap_L = _spectral_gap(ev_L, k_L) if eig_L and k_L > 0 else 1.0
        gap_R = _spectral_gap(ev_R, k_R) if eig_R and k_R > 0 else 1.0

        var90_P = 0
        cumvar = np.cumsum(ev_P) / max(np.sum(ev_P), 1e-12)
        for i, cv in enumerate(cumvar):
            if cv >= 0.90:
                var90_P = i + 1
                break
        if var90_P == 0:
            var90_P = len(ev_P)

        var95_P = 0
        for i, cv in enumerate(cumvar):
            if cv >= 0.95:
                var95_P = i + 1
                break
        if var95_P == 0:
            var95_P = len(ev_P)

        knee_P = _knee_elbow_k(ev_P)
        knee_L = _knee_elbow_k(ev_L) if eig_L else 0
        knee_R = _knee_elbow_k(ev_R) if eig_R else 0

        # Top eigenvalue ratios between parent and children
        lam1_L = float(ev_L[0]) if eig_L and len(ev_L) > 0 else 0.0
        lam1_R = float(ev_R[0]) if eig_R and len(ev_R) > 0 else 0.0
        lam1_P = float(ev_P[0]) if len(ev_P) > 0 else 0.0
        lam2_P = float(ev_P[1]) if len(ev_P) > 1 else 0.0

        # Gate 3 decision
        sibling_diff = False
        if parent in stats_df.index:
            sibling_diff = bool(stats_df.loc[parent].get("Sibling_BH_Different", False))

        rows.append({
            "parent": parent, "n_L": n_L, "n_R": n_R, "n_P": n_L + n_R,
            "d_active": da_P, "n_features": n_features,
            # MP signal counts
            "k_L": k_L, "k_R": k_R, "k_P": k_P,
            # Effective rank
            "er_L": er_L, "er_R": er_R, "er_P": er_P,
            # JL reference
            "jl_k": jl_k, "jl_qrt": max(1, jl_k // 4),
            # Spectral gaps
            "gap_P": gap_P, "gap_L": gap_L, "gap_R": gap_R,
            # Variance explained thresholds
            "var90_P": var90_P, "var95_P": var95_P,
            # Knee/elbow
            "knee_P": knee_P, "knee_L": knee_L, "knee_R": knee_R,
            # Top eigenvalue structure
            "lam1_P": lam1_P, "lam2_P": lam2_P,
            "lam1_L": lam1_L, "lam1_R": lam1_R,
            "mp_bound_P": mp_bound_P,
            # Gate 3
            "SPLIT": sibling_diff,
        })

    return rows, tc, decomp


# ── PHASE B: Candidate equations ───────────────────────────────────────────
# Each equation maps spectral features → k (integer ≥ 2).

def _eq_jl_qrt(r):
    """Baseline: JL/4."""
    return max(2, r["jl_qrt"])


def _eq_k_parent(r):
    """Parent MP signal count."""
    return max(2, r["k_P"])


def _eq_er_parent(r):
    """Effective rank of parent."""
    return max(2, int(round(r["er_P"])))


def _eq_knee_parent(r):
    """Elbow of parent eigenvalue curve."""
    return max(2, r["knee_P"])


def _eq_var90(r):
    """Dims for 90% variance explained in parent."""
    return max(2, r["var90_P"])


def _eq_var95(r):
    """Dims for 95% variance explained in parent."""
    return max(2, r["var95_P"])


def _eq_gap_weighted(r):
    """k_P × spectral_gap_P — scale signal count by gap quality."""
    return max(2, int(round(r["k_P"] * min(r["gap_P"], 5.0))))


def _eq_sqrt_n(r):
    """√n_parent — sample-size scaling."""
    return max(2, int(round(np.sqrt(r["n_P"]))))


def _eq_log_n_d(r):
    """log(n) × log(d) / 2 — dimensional scaling."""
    return max(2, int(round(np.log(max(r["n_P"], 2)) * np.log(max(r["d_active"], 2)) / 2)))


def _eq_er_children_sum(r):
    """Sum of children's effective ranks."""
    return max(2, int(round(r["er_L"] + r["er_R"])))


def _eq_max_knee_jl_blend(r):
    """max(knee_P, JL/4) — elbow with JL floor."""
    return max(2, r["knee_P"], r["jl_qrt"])


def _eq_er_parent_jl_floor(r):
    """max(er_P, JL/4) — effective rank with JL floor."""
    return max(2, int(round(r["er_P"])), r["jl_qrt"])


def _eq_var90_jl_floor(r):
    """max(var90_P, JL/4) — 90% variance dims with JL floor."""
    return max(2, r["var90_P"], r["jl_qrt"])


def _eq_lam_ratio(r):
    """k based on how much λ₁_P exceeds MP bound: max(2, ceil(λ₁_P / mp_bound))."""
    if r["mp_bound_P"] > 0:
        return max(2, int(np.ceil(r["lam1_P"] / r["mp_bound_P"])))
    return 2


def _eq_harmonic_jl_er(r):
    """Harmonic mean of JL and er_P — balances both."""
    jl = r["jl_k"]
    er = max(1, r["er_P"])
    return max(2, int(round(2 * jl * er / (jl + er))))


def _eq_geom_jl_knee(r):
    """Geometric mean of JL and knee_P."""
    return max(2, int(round(np.sqrt(max(1, r["jl_k"]) * max(1, r["knee_P"])))))


EQUATIONS = {
    "jl_qrt":          _eq_jl_qrt,
    "k_parent":        _eq_k_parent,
    "er_parent":       _eq_er_parent,
    "knee_parent":     _eq_knee_parent,
    "var90":           _eq_var90,
    "var95":           _eq_var95,
    "gap_weighted":    _eq_gap_weighted,
    "sqrt_n":          _eq_sqrt_n,
    "log_n_d":         _eq_log_n_d,
    "er_child_sum":    _eq_er_children_sum,
    "knee_jl":         _eq_max_knee_jl_blend,
    "er_jl":           _eq_er_parent_jl_floor,
    "var90_jl":        _eq_var90_jl_floor,
    "lam_ratio":       _eq_lam_ratio,
    "harm_jl_er":      _eq_harmonic_jl_er,
    "geom_jl_knee":    _eq_geom_jl_knee,
}


# ── PHASE C: Benchmark strategies ──────────────────────────────────────────
# Convert the best equations into actual decomposition strategies.

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

            row = {
                "n_L": n_L, "n_R": n_R, "n_P": n_L + n_R,
                "d_active": da_P, "n_features": n_features,
                "k_L": marchenko_pastur_signal_count(ev_L, ns_L, da_L) if eig_L else 0,
                "k_R": marchenko_pastur_signal_count(ev_R, ns_R, da_R) if eig_R else 0,
                "k_P": marchenko_pastur_signal_count(ev_P, ns_P, da_P),
                "er_L": compute_effective_rank(ev_L) if eig_L else 1.0,
                "er_R": compute_effective_rank(ev_R) if eig_R else 1.0,
                "er_P": compute_effective_rank(ev_P),
                "jl_k": compute_jl_dim(ns_P, n_features),
                "jl_qrt": max(1, compute_jl_dim(ns_P, n_features) // 4),
                "gap_P": _spectral_gap(ev_P, marchenko_pastur_signal_count(ev_P, ns_P, da_P)),
                "var90_P": 0, "var95_P": 0,
                "knee_P": _knee_elbow_k(ev_P),
                "knee_L": _knee_elbow_k(ev_L) if eig_L else 0,
                "knee_R": _knee_elbow_k(ev_R) if eig_R else 0,
                "lam1_P": float(ev_P[0]) if len(ev_P) > 0 else 0.0,
                "lam2_P": float(ev_P[1]) if len(ev_P) > 1 else 0.0,
                "lam1_L": float(ev_L[0]) if eig_L and len(ev_L) > 0 else 0.0,
                "lam1_R": float(ev_R[0]) if eig_R and len(ev_R) > 0 else 0.0,
                "mp_bound_P": _mp_upper_bound(ev_P, ns_P, da_P),
            }
            # Var90/95
            cumvar = np.cumsum(ev_P) / max(np.sum(ev_P), 1e-12)
            for i, cv in enumerate(cumvar):
                if cv >= 0.90 and row["var90_P"] == 0:
                    row["var90_P"] = i + 1
                if cv >= 0.95 and row["var95_P"] == 0:
                    row["var95_P"] = i + 1
            if row["var90_P"] == 0:
                row["var90_P"] = len(ev_P)
            if row["var95_P"] == 0:
                row["var95_P"] = len(ev_P)

            k = eq_fn(row)
            sibling_dims[parent] = k

        return sibling_dims if sibling_dims else None

    return _derive


def _derive_none(tree, annotated_df):
    return None


def _derive_jl_floor_qrt(tree, annotated_df):
    """Reproduce jl_floor_qrt from exp17."""
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


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
    print(f"        SPECTRAL_METHOD={config.SPECTRAL_METHOD}")
    print()

    # ═════════════════════════════════════════════════════════════════════
    # PHASE A: Spectral map — collect features at every binary parent
    # ═════════════════════════════════════════════════════════════════════
    print("═══ PHASE A: SPECTRAL MAP ═══")
    print()

    all_rows = []
    for case_name in CASES:
        rows, tc, _ = collect_spectral_map(case_name)
        true_k = tc.get("n_clusters", "?")
        splits = [r for r in rows if r["SPLIT"]]
        merges = [r for r in rows if not r["SPLIT"]]

        # Summary per case
        print(f"  {case_name:<35} K={true_k}  nodes={len(rows):>3}  "
              f"SPLIT={len(splits):>2}  MERGE={len(merges):>3}")

        for r in rows:
            r["case"] = case_name
            r["true_k"] = true_k
        all_rows.extend(rows)

    splits_all = [r for r in all_rows if r["SPLIT"]]
    merges_all = [r for r in all_rows if not r["SPLIT"]]
    print(f"\n  TOTAL: {len(all_rows)} nodes ({len(splits_all)} SPLIT, {len(merges_all)} MERGE)")

    # Key spectral features: SPLIT vs MERGE distributions
    print()
    print("═══ SPECTRAL FEATURE DISTRIBUTIONS (SPLIT vs MERGE) ═══")
    print()

    features_to_show = [
        ("k_P", "Parent MP signal count"),
        ("k_L", "Left child MP signal count"),
        ("k_R", "Right child MP signal count"),
        ("er_P", "Parent effective rank"),
        ("er_L", "Left effective rank"),
        ("er_R", "Right effective rank"),
        ("gap_P", "Parent spectral gap (λ_k/λ_{k+1})"),
        ("knee_P", "Parent elbow k"),
        ("var90_P", "Parent dims for 90% var"),
        ("var95_P", "Parent dims for 95% var"),
        ("lam1_P", "Parent λ₁"),
        ("lam2_P", "Parent λ₂"),
        ("mp_bound_P", "Parent MP upper bound"),
        ("jl_qrt", "JL/4"),
        ("n_P", "n_parent"),
        ("d_active", "d_active"),
    ]

    print(f"  {'Feature':<35} │ {'SPLIT':>30} │ {'MERGE':>30} │ {'Separation':>10}")
    print(f"  {'':─<35}─┼─{'':─>30}─┼─{'':─>30}─┼─{'':─>10}")

    for fname, label in features_to_show:
        s_vals = [r[fname] for r in splits_all]
        m_vals = [r[fname] for r in merges_all]
        if not s_vals or not m_vals:
            continue
        s_med, s_mean = np.median(s_vals), np.mean(s_vals)
        m_med, m_mean = np.median(m_vals), np.mean(m_vals)
        sep = abs(s_mean - m_mean) / (np.std(s_vals) + np.std(m_vals) + 1e-6)
        print(f"  {label:<35} │ med={s_med:>7.1f} mean={s_mean:>7.1f} │ "
              f"med={m_med:>7.1f} mean={m_mean:>7.1f} │ {sep:>8.3f}")

    # Derived ratios
    print()
    print("═══ DERIVED SPECTRAL RATIOS (SPLIT vs MERGE) ═══")
    print()

    derived_features = {
        "k_P / k_sum":       lambda r: r["k_P"] / max(r["k_L"] + r["k_R"], 1),
        "k_P / k_max":       lambda r: r["k_P"] / max(max(r["k_L"], r["k_R"]), 1),
        "er_P / er_sum":     lambda r: r["er_P"] / max(r["er_L"] + r["er_R"], 0.1),
        "er_P / er_max":     lambda r: r["er_P"] / max(max(r["er_L"], r["er_R"]), 0.1),
        "knee_P / k_P":      lambda r: r["knee_P"] / max(r["k_P"], 1),
        "var90 / k_P":       lambda r: r["var90_P"] / max(r["k_P"], 1),
        "λ₁_P / mp_bound":   lambda r: r["lam1_P"] / max(r["mp_bound_P"], 1e-6),
        "λ₁_P / λ₂_P":       lambda r: r["lam1_P"] / max(r["lam2_P"], 1e-6),
        "gap_P":              lambda r: r["gap_P"],
        "n_P / d_active":    lambda r: r["n_P"] / max(r["d_active"], 1),
        "√n_P":               lambda r: np.sqrt(r["n_P"]),
        "log(n)·log(d)/2":   lambda r: np.log(max(r["n_P"], 2)) * np.log(max(r["d_active"], 2)) / 2,
    }

    print(f"  {'Ratio':<25} │ {'SPLIT':>25} │ {'MERGE':>25} │ {'Sep':>6}")
    print(f"  {'':─<25}─┼─{'':─>25}─┼─{'':─>25}─┼─{'':─>6}")

    for rname, rfn in derived_features.items():
        s_vals = [rfn(r) for r in splits_all]
        m_vals = [rfn(r) for r in merges_all]
        s_med, s_mean = np.median(s_vals), np.mean(s_vals)
        m_med, m_mean = np.median(m_vals), np.mean(m_vals)
        sep = abs(s_mean - m_mean) / (np.std(s_vals) + np.std(m_vals) + 1e-6)
        print(f"  {rname:<25} │ med={s_med:>6.2f} mean={s_mean:>6.2f} │ "
              f"med={m_med:>6.2f} mean={m_mean:>6.2f} │ {sep:>5.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE B: Equation evaluation — compute k per equation, compare
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("═══ PHASE B: EQUATION EVALUATION ═══")
    print()
    print("For each equation, compute k at every node and compare to:")
    print("  1. JL/4 (our best strategy)")
    print("  2. Gate 3 SPLIT/MERGE correlation")
    print()

    print(f"  {'Equation':<17} │ {'Mean k':>7} {'Med k':>7} │ "
          f"{'Mean k(S)':>9} {'Mean k(M)':>9} │ "
          f"{'Corr(k,JL/4)':>12} │ "
          f"{'k>2 SPLIT%':>10} {'k>2 MERGE%':>11}")
    print(f"  {'':─<17}─┼─{'':─>7}─{'':─>7}─┼─{'':─>9}─{'':─>9}─┼─{'':─>12}─┼─{'':─>10}─{'':─>11}")

    for eq_name, eq_fn in EQUATIONS.items():
        ks = [eq_fn(r) for r in all_rows]
        jl_qrts = [r["jl_qrt"] for r in all_rows]

        ks_split = [eq_fn(r) for r in splits_all]
        ks_merge = [eq_fn(r) for r in merges_all]

        # Correlation with JL/4
        if np.std(ks) > 0 and np.std(jl_qrts) > 0:
            corr = float(np.corrcoef(ks, jl_qrts)[0, 1])
        else:
            corr = 0.0

        # % of nodes where k > 2, split by SPLIT/MERGE
        pct_split = 100 * np.mean([k > 2 for k in ks_split]) if ks_split else 0
        pct_merge = 100 * np.mean([k > 2 for k in ks_merge]) if ks_merge else 0

        print(f"  {eq_name:<17} │ {np.mean(ks):>7.1f} {np.median(ks):>7.1f} │ "
              f"{np.mean(ks_split):>9.1f} {np.mean(ks_merge):>9.1f} │ "
              f"{corr:>12.3f} │ "
              f"{pct_split:>9.1f}% {pct_merge:>10.1f}%")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE C: Benchmark — run each equation as a strategy
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("═══ PHASE C: BENCHMARK ═══")
    print()

    # Include baselines + all equations
    benchmark_strategies = {
        "none": _derive_none,
        "jl_floor_qrt": _derive_jl_floor_qrt,
    }
    for eq_name, eq_fn in EQUATIONS.items():
        if eq_name == "jl_qrt":
            continue  # Already covered by jl_floor_qrt baseline
        benchmark_strategies[eq_name] = _make_equation_strategy(eq_fn)

    strat_names = list(benchmark_strategies.keys())

    # Header
    header = f"{'Case':<30} {'TK':>3}"
    for s in strat_names:
        header += f" │ {'K':>3} {'ARI':>5}"
    print(header)
    sub = f"{'':<30} {'':>3}"
    for s in strat_names:
        sub += f" │ {s[:9]:>9}"
    print(sub)
    print("─" * len(header))

    all_results = {}
    for name in CASES:
        row = f"{name:<30}"
        true_k = None
        for sname in strat_names:
            try:
                r = run_case_strategy(name, benchmark_strategies[sname])
                if true_k is None:
                    true_k = r["true_k"]
                    row = f"{name:<30} {true_k:>3}"
                all_results.setdefault(name, {})[sname] = r
                row += f" │ {r['found_k']:>3} {r['ari']:>5.3f}"
            except Exception:
                row += " │  ERR   "
                import traceback
                traceback.print_exc()
        print(row)

    # Summary
    print()
    print(f"{'MEAN ARI':<34}")
    for sname in strat_names:
        aris = [all_results[c][sname]["ari"] for c in CASES
                if c in all_results and sname in all_results[c]]
        mean_ari = np.mean(aris) if aris else float("nan")
        print(f"  {sname:<17}: {mean_ari:.3f}")

    # Rank table
    print()
    print("Strategy ranking by mean ARI:")
    ari_by_strat = {}
    for sname in strat_names:
        aris = [all_results[c][sname]["ari"] for c in CASES
                if c in all_results and sname in all_results[c]]
        ari_by_strat[sname] = np.mean(aris) if aris else 0.0

    ranked = sorted(ari_by_strat.items(), key=lambda x: -x[1])
    for rank, (sname, mean_ari) in enumerate(ranked, 1):
        wins = sum(1 for c in CASES if c in all_results and sname in all_results[c]
                   and abs(all_results[c][sname]["ari"] -
                           max(all_results[c][s]["ari"] for s in all_results[c])) < 0.001)
        print(f"  {rank:>2}. {sname:<17}  ARI={mean_ari:.3f}  wins={wins}/{len(CASES)}")
