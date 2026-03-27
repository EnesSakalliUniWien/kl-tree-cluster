#!/usr/bin/env python3
"""
Sweep config parameter combinations on overlap benchmark cases.

Tests combinations of:
  - SIBLING_WHITENING: per_component / satterthwaite
  - INCLUDE_INTERNAL_IN_SPECTRAL: True / False
  - FELSENSTEIN_SCALING: True / False
  - EDGE_ALPHA / SIBLING_ALPHA: 0.05 / 0.01 / 0.001
"""

import itertools
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault("KL_TE_N_JOBS", "1")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

TARGET_CASES = {
    "gauss_overlap_3c_small",
    "overlap_heavy_4c_small_feat",
    "overlap_mod_4c_small",
}

# --- Parameter grid ---
PARAM_GRID = {
    "SIBLING_WHITENING": ["per_component", "satterthwaite"],
    "INCLUDE_INTERNAL_IN_SPECTRAL": [False, True],
    "FELSENSTEIN_SCALING": [False, True],
    "alpha": [0.05, 0.01, 0.001],
}


def set_config(combo: dict) -> str:
    """Apply a config combination and return a short label."""
    config.SIBLING_WHITENING = combo["SIBLING_WHITENING"]
    config.INCLUDE_INTERNAL_IN_SPECTRAL = combo["INCLUDE_INTERNAL_IN_SPECTRAL"]
    config.FELSENSTEIN_SCALING = combo["FELSENSTEIN_SCALING"]
    config.EDGE_ALPHA = combo["alpha"]
    config.SIBLING_ALPHA = combo["alpha"]

    parts = []
    parts.append(f"whiten={combo['SIBLING_WHITENING'][:3]}")
    parts.append(f"internal={'Y' if combo['INCLUDE_INTERNAL_IN_SPECTRAL'] else 'N'}")
    parts.append(f"fels={'Y' if combo['FELSENSTEIN_SCALING'] else 'N'}")
    parts.append(f"α={combo['alpha']}")
    return " | ".join(parts)


def run_case(tc: dict) -> tuple:
    """Run decomposition and return (found_k, ari)."""
    data, labels, _orig, _meta = generate_case_data(tc)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.dtypes.apply(lambda d: d.kind == "f").any():
        data = (data > np.median(data.values, axis=0)).astype(int)

    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

    try:
        results = tree.decompose(
            leaf_data=data,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
        )
    except Exception as e:
        return None, None, str(e)

    found_k = results["num_clusters"]
    # Compute ARI
    if labels is not None:
        assignments = results["cluster_assignments"]
        pred = np.full(len(labels), -1)
        for cid, info in assignments.items():
            for leaf in info["leaves"]:
                idx = data.index.get_loc(leaf)
                pred[idx] = cid
        ari = adjusted_rand_score(labels, pred)
    else:
        ari = float("nan")

    return found_k, ari, None


def main():
    all_cases = get_default_test_cases()
    cases = [c for c in all_cases if c["name"] in TARGET_CASES]
    # Sort for consistent output
    cases.sort(key=lambda c: c["name"])

    print(f"Sweeping config combinations on {len(cases)} overlap cases\n")

    # Build all combos
    keys = [
        "SIBLING_WHITENING",
        "INCLUDE_INTERNAL_IN_SPECTRAL",
        "FELSENSTEIN_SCALING",
        "alpha",
    ]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[PARAM_GRID[k] for k in keys])]

    # Header
    case_names = [c["name"] for c in cases]
    true_ks = [c.get("n_clusters", "?") for c in cases]
    header_cases = "  ".join(f"{'K':>3s} {'ARI':>6s}" for _ in cases)
    print(f"{'Config':<52s}  " + "  ".join(f"{n[:20]:>11s}" for n in case_names))
    print(f"{'':52s}  " + "  ".join(f"{'(K=' + str(k) + ')':>11s}" for k in true_ks))
    print("-" * (54 + 13 * len(cases)))

    rows = []
    for combo in combos:
        label = set_config(combo)
        if label is None:
            continue

        row = {"config": label}
        parts = []
        for tc in cases:
            found_k, ari, err = run_case(tc)
            if err:
                parts.append(f"{'ERR':>3s} {'':>6s}")
                row[tc["name"] + "_K"] = "ERR"
                row[tc["name"] + "_ARI"] = None
            else:
                ari_str = f"{ari:.3f}" if not np.isnan(ari) else "  N/A"
                parts.append(f"{found_k:>3d} {ari_str:>6s}")
                row[tc["name"] + "_K"] = found_k
                row[tc["name"] + "_ARI"] = ari
        rows.append(row)
        print(f"{label:<52s}  {'  '.join(parts)}")

    # Summary: find best combo per case
    print(f"\n{'=' * 70}")
    print("BEST CONFIGS PER CASE (by ARI)")
    print("=" * 70)
    for tc in cases:
        name = tc["name"]
        true_k = tc.get("n_clusters", "?")
        best = max(rows, key=lambda r: r.get(name + "_ARI", -1) or -1)
        print(
            f"  {name} (true K={true_k}): K={best[name + '_K']}, "
            f"ARI={best[name + '_ARI']:.3f}  [{best['config']}]"
        )

    # Also find best overall (mean ARI)
    print("\nBEST OVERALL (mean ARI across 3 cases)")
    for r in rows:
        aris = [r.get(tc["name"] + "_ARI", None) for tc in cases]
        aris = [a for a in aris if a is not None and not np.isnan(a)]
        r["mean_ari"] = np.mean(aris) if aris else -1

    rows.sort(key=lambda r: r["mean_ari"], reverse=True)
    for r in rows[:10]:
        print(f"  mean_ARI={r['mean_ari']:.3f}  [{r['config']}]")


if __name__ == "__main__":
    main()
