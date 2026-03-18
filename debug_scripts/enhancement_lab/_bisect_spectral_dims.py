"""Quick A/B: auto-derived sibling spectral dims (HEAD) vs None (pre-9b474c4).

Tests whether _derive_sibling_spectral_dims in the orchestrator causes regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from lab_helpers import build_tree_and_data, compute_ari, temporary_experiment_overrides  # noqa: E402


from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import orchestrator  # noqa: E402

# Mix of guard, intermediate, and failure cases
SENTINEL = [
    "binary_hard_4c",
    "gauss_clear_small",
    "gauss_moderate_3c",
    "gauss_moderate_5c",
    "binary_low_noise_12c",
    "binary_perfect_8c",
    "binary_balanced_low_noise__2",
    "gauss_clear_large",
    "gauss_noisy_3c",
    "binary_low_noise_4c",
    "binary_perfect_4c",
    "gauss_overlap_4c_med",
]

orig_fn = orchestrator._derive_sibling_spectral_dims


def run_case(case_name: str, disable_auto_derive: bool) -> dict:
    tree, data_df, true_labels, tc = build_tree_and_data(case_name)
    strategy_fn = (lambda t, d: None) if disable_auto_derive else orig_fn
    with temporary_experiment_overrides(sibling_dims=strategy_fn):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )
    ari = compute_ari(decomp, data_df, true_labels) if true_labels is not None else float("nan")
    return {
        "true_k": tc.get("n_clusters"),
        "found_k": decomp["num_clusters"],
        "ari": round(ari, 3),
    }


print(f"Config: SIBLING_ALPHA={config.SIBLING_ALPHA}, METHOD={config.SIBLING_TEST_METHOD}")
print(f"{'Case':<35} {'TK':>3} {'K_a':>4} {'ARI_a':>7} {'K_n':>4} {'ARI_n':>7} {'delta':>7}")
print("-" * 75)

sum_auto = sum_none = n = 0
for name in SENTINEL:
    try:
        r_auto = run_case(name, disable_auto_derive=False)
        r_none = run_case(name, disable_auto_derive=True)
    except Exception as e:
        print(f"{name:<35} ERROR: {e}")
        continue
    delta = round(r_none["ari"] - r_auto["ari"], 3)
    sum_auto += r_auto["ari"]
    sum_none += r_none["ari"]
    n += 1
    print(
        f"{name:<35} {r_auto['true_k']:>3} "
        f"{r_auto['found_k']:>4} {r_auto['ari']:>7.3f} "
        f"{r_none['found_k']:>4} {r_none['ari']:>7.3f} "
        f"{delta:>+7.3f}"
    )

if n:
    print(
        f"\nMean ARI  auto={sum_auto/n:.3f}  none={sum_none/n:.3f}  delta={sum_none/n - sum_auto/n:+.3f}"
    )
