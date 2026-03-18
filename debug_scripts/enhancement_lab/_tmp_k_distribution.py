"""Diagnose: what are the actual sibling spectral k values across benchmarks?

Runs benchmark cases, builds trees through Gate 2, and reports
the distribution of min-child spectral k per parent node.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data

from benchmarks.shared.cases import get_default_test_cases
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.edge_gate import (
    annotate_edge_gate,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    _derive_sibling_spectral_dims,
)

all_cases = get_default_test_cases()

all_k_values = []
case_summaries = []

for i, tc in enumerate(all_cases):
    name = tc["name"]
    try:
        tree, data_df, y_true, tc_info = build_tree_and_data(name)
        annotations_df = tree.stats_df.copy()

        edge_bundle = annotate_edge_gate(
            tree,
            annotations_df,
            significance_level_alpha=config.EDGE_ALPHA,
            leaf_data=data_df,
            spectral_method="marchenko_pastur",
            minimum_projection_dimension=None,
            fdr_method="tree_bh",
        )

        sibling_dims = _derive_sibling_spectral_dims(tree, edge_bundle.annotated_df)
        if sibling_dims:
            ks = list(sibling_dims.values())
            all_k_values.extend(ks)
            case_summaries.append(
                {
                    "name": name,
                    "n_parents": len(ks),
                    "k_values": Counter(ks),
                    "min_k": min(ks),
                    "max_k": max(ks),
                    "mean_k": float(np.mean(ks)),
                    "pct_k1": sum(1 for k in ks if k == 1) / len(ks) * 100,
                }
            )
        else:
            case_summaries.append(
                {
                    "name": name,
                    "n_parents": 0,
                    "k_values": {},
                    "min_k": 0,
                    "max_k": 0,
                    "mean_k": 0.0,
                    "pct_k1": 100.0,
                }
            )
        ks_str = str(dict(Counter(ks))) if sibling_dims else "None"
        print(f"  [{i+1:3d}] {name:<40} {ks_str}")
    except Exception as e:
        print(f"  [{i+1:3d}] SKIP {name}: {e}")

# Aggregate
print("\n" + "=" * 70)
print("SIBLING SPECTRAL k DISTRIBUTION ACROSS BENCHMARK CASES")
print("=" * 70)

global_counter = Counter(all_k_values)
total = len(all_k_values)
print(f"\nTotal parent nodes across all cases: {total}")
print("\nGlobal k distribution:")
for k in sorted(global_counter.keys()):
    count = global_counter[k]
    print(f"  k={k:3d}: {count:5d} ({count/total*100:5.1f}%)")

pct_k1 = global_counter.get(1, 0) / total * 100 if total else 0
pct_k_le2 = sum(global_counter.get(k, 0) for k in [1, 2]) / total * 100 if total else 0
pct_k_le3 = sum(global_counter.get(k, 0) for k in [1, 2, 3]) / total * 100 if total else 0
print(f"\n  k=1 : {pct_k1:.1f}%")
print(f"  k<=2: {pct_k_le2:.1f}%")
print(f"  k<=3: {pct_k_le3:.1f}%")

print(f"\n{'Case':<40} {'Parents':>7} {'%k=1':>6} {'min':>4} {'max':>4} {'mean':>6}")
print("-" * 70)
for s in case_summaries:
    print(
        f"{s['name']:<40} {s['n_parents']:>7} {s['pct_k1']:>5.1f}% {s['min_k']:>4} {s['max_k']:>4} {s['mean_k']:>6.1f}"
    )
