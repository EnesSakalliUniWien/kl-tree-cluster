"""Diagnostic for exp21 silent eigendecompose failure.

Identifies why _make_strategy's _eigendecompose returns None for every node.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

import numpy as np
from lab_helpers import build_tree_and_data

from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)


def _descendant_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        if tree.out_degree(n) == 0:
            leaves.append(n)
        else:
            stack.extend(tree.successors(n))
    return leaves


def _node_data(tree, node, leaf_data):
    leaves = _descendant_leaves(tree, node)
    labels = [tree.nodes[lf].get("label", lf) for lf in leaves]
    return leaf_data.loc[labels].values


if __name__ == "__main__":
    case_name = "gauss_clear_small"
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    print(f"Case: {case_name}, shape: {data_df.shape}")
    print()

    n_binary = 0
    n_eigendecompose_ok = 0
    n_eigendecompose_fail = 0
    first_error = None

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        n_binary += 1
        left, right = children

        try:
            data_L = _node_data(tree, left, data_df)
            data_R = _node_data(tree, right, data_df)
        except (KeyError, IndexError) as e:
            print(f"  {parent}: _node_data failed: {e}")
            continue

        data_P = np.vstack([data_L, data_R])

        # ── Test 1: exp21 style (need_eigh=True, result.n_samples) ──
        try:
            result = eigendecompose_correlation_backend(data_P, need_eigh=True)
            _ = result.eigenvalues, result.n_samples, result.d_active
            n_eigendecompose_ok += 1
        except Exception as e:
            n_eigendecompose_fail += 1
            if first_error is None:
                first_error = (parent, type(e).__name__, str(e))

    print(f"Binary parents: {n_binary}")
    print(f"  Eigendecompose OK:   {n_eigendecompose_ok}")
    print(f"  Eigendecompose FAIL: {n_eigendecompose_fail}")
    if first_error:
        print(f"\n  First error at {first_error[0]}:")
        print(f"    {first_error[1]}: {first_error[2]}")

    # ── Test 2: exp20 style (need_eigh=False, data.shape[0]) ──
    print("\n── Exp20 style (need_eigh=False, data.shape[0]): ──")
    ok2 = fail2 = 0
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        try:
            data_P = np.vstack(
                [
                    _node_data(tree, left, data_df),
                    _node_data(tree, right, data_df),
                ]
            )
            eig = eigendecompose_correlation_backend(data_P, need_eigh=False)
            if eig is None:
                fail2 += 1
            else:
                _ = eig.eigenvalues, data_P.shape[0], eig.d_active
                ok2 += 1
        except Exception:
            fail2 += 1
    print(f"  OK: {ok2}, FAIL: {fail2}")
