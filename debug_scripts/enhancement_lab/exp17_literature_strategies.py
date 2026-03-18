"""Lab: literature-informed sibling spectral-dim derivation strategies.

Builds on exp16 findings (jl_floor_qrt=0.991 best).  Adds strategies
motivated by the high-dimensional two-sample testing literature:

Existing winners (from exp16):
  1. none           — pure JL fallback (mean ARI 0.979)
  2. jl_floor_qrt   — max(spectral_min, JL/4) (mean ARI 0.991)
  3. sum_child_2x   — 2×(k_L + k_R) (mean ARI 0.981)

New literature-informed strategies:
  A. pooled_mp      — Marchenko-Pastur on POOLED sibling data (between-group
                       signal appears as extra eigenvalues in the joint space)
  B. effective_rank — continuous effective rank via Shannon entropy on pooled
                       data (Vershynin 2018; avoids hard MP cutoff)
  C. ncp_power_opt  — pick k that maximises theoretical χ² power at the
                       observed test statistic (noncentrality parameter idea
                       from Steiger 1985, chi-square power literature)
  D. diff_signal    — count "significant" components in the sibling mean
                       difference vector δ = μ_L − μ_R (Fisher discriminant
                       spirit: rank of between-group scatter ≤ 1 for 2 groups,
                       but we extend via MP on the standardised residual)
  E. jl_floor_8th   — max(spectral_min, JL/8) — finer grid between qrt and min
  F. jl_spectral_geo — geometric mean of JL and spectral min: √(JL × spec_min)
"""
from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data, compute_ari, temporary_experiment_overrides
from scipy.stats import chi2, ncx2

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_jl_dim,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import (
    effective_rank as compute_effective_rank,
    marchenko_pastur_signal_count,
)

orig_derive = orchestrator._derive_sibling_spectral_dims


# ── Sentinel cases ──────────────────────────────────────────────────────────

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


# ── Helper to get leaf data rows for a node ─────────────────────────────────

def _descendant_leaf_indices(tree, node):
    """Return sorted list of leaf indices under *node*."""
    import networkx as nx
    leaves = []
    for desc in nx.descendants(tree, node):
        if tree.out_degree(desc) == 0:
            leaves.append(desc)
    if tree.out_degree(node) == 0:
        leaves.append(node)
    return sorted(leaves)


def _node_leaf_data(tree, node, leaf_data):
    """Return (n_leaves, d) array of leaf data under *node*."""
    leaf_ids = _descendant_leaf_indices(tree, node)
    labels = [tree.nodes[lid].get("label", lid) for lid in leaf_ids]
    return leaf_data.loc[labels].values


# ── Strategy: none (JL) ────────────────────────────────────────────────────

def _derive_none(tree, annotated_df):
    return None


# ── Strategy: min_child (HEAD) ──────────────────────────────────────────────

def _derive_min_child(tree, annotated_df):
    return orig_derive(tree, annotated_df)


# ── Strategy: jl_floor_qrt (exp16 winner) ──────────────────────────────────

def _make_jl_floor_strategy(fraction):
    """max(spectral_min, fraction * JL_dim)."""
    def _derive(tree, annotated_df):
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
            floor_k = max(1, int(jl_k * fraction))
            sibling_dims[parent] = max(spectral_k, floor_k)
        return sibling_dims if sibling_dims else None
    return _derive


# ── Strategy: sum_child_2x ──────────────────────────────────────────────────

def _derive_sum_child_2x(tree, annotated_df):
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
        total = k_left + k_right
        if total > 0:
            sibling_dims[parent] = 2 * total
    return sibling_dims if sibling_dims else None


# ── Strategy A: pooled_mp ───────────────────────────────────────────────────
# Run Marchenko-Pastur on the POOLED data from both siblings.
# Rationale: Between-group signal appears as extra eigenvalues in the joint
# space that aren't visible in either child alone.

_pooled_mp_leaf_data_cache = {}

def _derive_pooled_mp(tree, annotated_df):
    """Marchenko-Pastur on pooled sibling data."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        # Pool leaf data from both siblings
        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        pooled = np.vstack([data_left, data_right])
        if pooled.shape[0] < 2:
            continue

        result = eigendecompose_correlation_backend(pooled, need_eigh=False)
        if result is None:
            continue

        k = marchenko_pastur_signal_count(
            result.eigenvalues,
            n_samples=pooled.shape[0],
            n_features=result.d_active,
        )
        if k > 0:
            sibling_dims[parent] = k

    return sibling_dims if sibling_dims else None


# ── Strategy B: effective_rank ──────────────────────────────────────────────
# Continuous effective dimension via Shannon entropy of pooled eigenspectrum.
# Avoids hard MP cutoff — more robust for borderline cases.

def _derive_effective_rank(tree, annotated_df):
    """Effective rank (Shannon entropy) on pooled sibling data."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        pooled = np.vstack([data_left, data_right])
        if pooled.shape[0] < 2:
            continue

        result = eigendecompose_correlation_backend(pooled, need_eigh=False)
        if result is None:
            continue

        eff_rank = compute_effective_rank(result.eigenvalues)
        k = max(1, int(round(eff_rank)))
        sibling_dims[parent] = k

    return sibling_dims if sibling_dims else None


# ── Strategy C: ncp_power_opt ───────────────────────────────────────────────
# For each candidate k, estimate the non-centrality parameter δ(k) and pick
# the k maximizing power = P(χ²(k, δ) > χ²_k,α).
#
# We use the edge spectral dim range to bound the search (2..max_k) and
# approximate δ(k) ≈ k × (T_obs / k_obs) where T_obs is the raw Wald
# statistic at some reference k.  Since we don't have T_obs directly at
# derivation time, we use a simplified heuristic: the pooled MP signal
# count gives the "true" signal dimension s; then for projection k,
# δ(k) ≈ s × SNR × min(k, s) / k  (signal captured diminishes with
# noise dimensions).  We search k ∈ {s, 2s, ..., JL_dim}.

def _derive_ncp_power(tree, annotated_df):
    """Power-optimal k via simplified noncentrality heuristic."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    alpha = config.SIBLING_ALPHA
    sibling_dims = {}

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        pooled = np.vstack([data_left, data_right])
        if pooled.shape[0] < 2:
            continue

        n_parent = pooled.shape[0]
        n_features = pooled.shape[1]

        # Estimate effective signal dimension from pooled data
        result = eigendecompose_correlation_backend(pooled, need_eigh=False)
        if result is None:
            continue
        s = marchenko_pastur_signal_count(
            result.eigenvalues, n_samples=n_parent, n_features=result.d_active
        )

        # Estimate SNR from the mean-difference norm (crude)
        mean_l = data_left.mean(axis=0)
        mean_r = data_right.mean(axis=0)
        diff = mean_l - mean_r
        n1, n2 = len(data_left), len(data_right)
        # Pooled variance per feature (diagonal)
        var_pooled = np.var(pooled, axis=0, ddof=1)
        var_pooled = np.maximum(var_pooled, 1e-12)
        # Approximate noncentrality for full-dimensional test
        ncp_full = float(np.sum(diff ** 2 / var_pooled)) * (n1 * n2) / (n1 + n2)

        # Now search over candidate k for best power
        jl_k = compute_jl_dim(n_parent, n_features)
        best_k, best_power = s, 0.0

        for k in range(max(s, 2), jl_k + 1):
            # Expected NCP when projecting to k dims:
            # If signal occupies s dimensions, random k-dim projection
            # captures roughly min(k, s)/max(k, s) fraction of signal
            if k <= s:
                ncp_k = ncp_full * k / s
            else:
                ncp_k = ncp_full * s / k  # noise dilution

            crit = chi2.ppf(1 - alpha, df=k)
            # Power = P(χ²_nc(k, ncp_k) > crit)
            power = 1.0 - ncx2.cdf(crit, df=k, nc=max(ncp_k, 0))
            if power > best_power:
                best_power = power
                best_k = k

        sibling_dims[parent] = max(best_k, 2)

    return sibling_dims if sibling_dims else None


# ── Strategy D: diff_signal ─────────────────────────────────────────────────
# Count significant components in the sibling mean-difference vector.
# Standardise δ = (μ_L − μ_R) / σ_pooled and count entries above a
# noise threshold (like a 1D Marchenko-Pastur on |δ_j / σ_j|²).

def _derive_diff_signal(tree, annotated_df):
    """Signal dimension from sibling mean difference vector."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        if len(data_left) < 2 or len(data_right) < 2:
            continue

        mean_l = data_left.mean(axis=0)
        mean_r = data_right.mean(axis=0)
        diff = mean_l - mean_r

        # Pooled standard deviation per feature
        n1, n2 = len(data_left), len(data_right)
        var1 = np.var(data_left, axis=0, ddof=1)
        var2 = np.var(data_right, axis=0, ddof=1)
        var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sd_pooled = np.sqrt(np.maximum(var_pooled, 1e-12))

        # Standardised absolute difference (Cohen's d per feature)
        z = np.abs(diff / sd_pooled)

        # Count features with |d_j| > threshold (Bonferroni-ish)
        # Use a soft threshold: median + 2 * MAD (robust to sparsity)
        med_z = np.median(z)
        mad_z = np.median(np.abs(z - med_z))
        threshold = med_z + 2.0 * max(mad_z, 0.01)
        k = int(np.sum(z > threshold))
        k = max(k, 2)  # at least 2 dimensions
        sibling_dims[parent] = k

    return sibling_dims if sibling_dims else None


# ── Strategy E: jl_floor_8th ───────────────────────────────────────────────

_derive_jl_floor_8th = _make_jl_floor_strategy(1.0 / 8)

# ── Strategy F: jl_spectral_geo ─────────────────────────────────────────────
# Geometric mean of JL dim and spectral min-child — balances both signals.

def _derive_jl_spectral_geo(tree, annotated_df):
    """Geometric mean of JL dim and spectral min-child."""
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
        # Geometric mean
        geo_k = max(2, int(round(np.sqrt(spectral_k * jl_k))))
        sibling_dims[parent] = geo_k
    return sibling_dims if sibling_dims else None


# ── Strategy G: k_parent ────────────────────────────────────────────────────
# Use the PARENT's Marchenko-Pastur signal count as the sibling test dim.
# exp18 showed: at SPLIT nodes, k_P >> k_L + k_R because pooling two
# different populations creates inter-cluster eigenvalues. At MERGE nodes,
# k_P ≈ max(k_L, k_R). So k_P directly measures the "true" dimensionality
# of the between-sibling signal.

def _derive_k_parent(tree, annotated_df):
    """Parent's Marchenko-Pastur signal count as sibling test dimension."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        pooled = np.vstack([data_left, data_right])
        if pooled.shape[0] < 2:
            continue

        result = eigendecompose_correlation_backend(pooled, need_eigh=False)
        if result is None:
            continue

        k = marchenko_pastur_signal_count(
            result.eigenvalues,
            n_samples=pooled.shape[0],
            n_features=result.d_active,
        )
        # Floor at 2 to avoid degenerate chi^2(1) tests
        sibling_dims[parent] = max(k, 2)

    return sibling_dims if sibling_dims else None


# ── Strategy H: k_parent_jl_floor ───────────────────────────────────────────
# Hybrid: max(k_parent, JL/4). Uses parent's signal count but keeps the
# JL floor to guard against low-power edge cases where k_parent is small.

def _derive_k_parent_jl_floor(tree, annotated_df):
    """max(k_parent_MP, JL_dim/4)."""
    leaf_data = _pooled_mp_leaf_data_cache.get("leaf_data")
    if leaf_data is None:
        return None

    sibling_dims = {}
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        try:
            data_left = _node_leaf_data(tree, left, leaf_data)
            data_right = _node_leaf_data(tree, right, leaf_data)
        except (KeyError, IndexError):
            continue

        pooled = np.vstack([data_left, data_right])
        if pooled.shape[0] < 2:
            continue

        n_features = pooled.shape[1]
        n_parent = pooled.shape[0]

        result = eigendecompose_correlation_backend(pooled, need_eigh=False)
        if result is None:
            continue

        k_mp = marchenko_pastur_signal_count(
            result.eigenvalues,
            n_samples=n_parent,
            n_features=result.d_active,
        )

        jl_k = compute_jl_dim(n_parent, n_features)
        floor_k = max(1, jl_k // 4)
        sibling_dims[parent] = max(k_mp, floor_k)

    return sibling_dims if sibling_dims else None


# ── Strategy registry ──────────────────────────────────────────────────────

STRATEGIES = {
    # exp16 baselines
    "none":           _derive_none,
    "min_child":      _derive_min_child,
    "jl_floor_qrt":   _make_jl_floor_strategy(0.25),
    "sum_child_2x":   _derive_sum_child_2x,
    # Literature-informed
    "pooled_mp":      _derive_pooled_mp,
    "effective_rank": _derive_effective_rank,
    "ncp_power_opt":  _derive_ncp_power,
    "diff_signal":    _derive_diff_signal,
    "jl_floor_8th":   _derive_jl_floor_8th,
    "jl_spectral_geo": _derive_jl_spectral_geo,
    # exp18-informed: parent's signal dimension
    "k_parent":       _derive_k_parent,
    "k_parent_jl":    _derive_k_parent_jl_floor,
}


# ── Runner ─────────────────────────────────────────────────────────────────

def run_case_strategy(case_name: str, strategy_fn) -> dict:
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)
    with temporary_experiment_overrides(
        leaf_data_cache=_pooled_mp_leaf_data_cache,
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

    strat_names = list(STRATEGIES.keys())

    # Header
    header = f"{'Case':<30} {'TK':>3}"
    for s in strat_names:
        header += f" | {'K':>3} {'ARI':>5}"
    print(header)
    header2 = f"{'':<30} {'':>3}"
    for s in strat_names:
        header2 += f" | {s[:9]:>9}"
    print(header2)
    print("-" * len(header))

    all_results = {}
    for name in CASES:
        row = f"{name:<30}"
        true_k = None
        for sname in strat_names:
            try:
                r = run_case_strategy(name, STRATEGIES[sname])
                if true_k is None:
                    true_k = r["true_k"]
                    row = f"{name:<30} {true_k:>3}"
                all_results.setdefault(name, {})[sname] = r
                row += f" | {r['found_k']:>3} {r['ari']:>5.3f}"
            except Exception:
                row += " |  ERR   "
                import traceback
                traceback.print_exc()
        print(row)

    # Summary: mean ARI per strategy
    print()
    summary = f"{'Mean ARI':<30} {'':>3}"
    for sname in strat_names:
        aris = [
            all_results[c][sname]["ari"]
            for c in CASES
            if c in all_results and sname in all_results[c]
        ]
        mean_ari = np.mean(aris) if aris else float("nan")
        summary += f" | {'':>3} {mean_ari:>5.3f}"
    print(summary)

    # Wins count
    print()
    print("Strategy win counts (tied wins counted for each):")
    wins = {s: 0 for s in strat_names}
    for name in CASES:
        if name not in all_results:
            continue
        results = all_results[name]
        best_ari = max(results[s]["ari"] for s in results)
        for s in results:
            if abs(results[s]["ari"] - best_ari) < 0.001:
                wins[s] += 1
    for s in strat_names:
        print(f"  {s:<18}: {wins[s]:>2} / {len(CASES)}")

    # Best per case
    print()
    print("Best strategy per case:")
    for name in CASES:
        if name not in all_results:
            continue
        results = all_results[name]
        best_s = max(results, key=lambda s: results[s]["ari"])
        best_ari = results[best_s]["ari"]
        print(f"  {name:<30} → {best_s:<18} (ARI={best_ari:.3f})")
