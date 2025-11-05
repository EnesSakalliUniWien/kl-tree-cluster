"""
Lightweight helpers to diagnose significance gates that control splitting.

Usage
-----
from misc.significance_diagnostics import (
    summarize_significance,
    explain_root_decision,
)

summarize_significance(results_df)
explain_root_decision(tree, results_df, alpha_local=0.05)

These print concise counts and indicate which gate(s) prevent splitting
at the root (local child-vs-parent KL or sibling CMI independence).
"""

from __future__ import annotations

from typing import Optional
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from hierarchy_analysis.local_kl_utils import get_local_kl_series, get_local_kl_value


def _fmt_ratio(x: int, n: int) -> str:
    if n <= 0:
        return "0/0 (0.0%)"
    pct = 100.0 * (x / n)
    return f"{x}/{n} ({pct:.1f}%)"


def summarize_significance(results_df: pd.DataFrame) -> None:
    """Print key significance counts used by the decomposer.

    - Global BH/FDR (Are_Features_Dependent)
    - Local BH (Local_BH_Significant)
    - Sibling CMI independence (Sibling_BH_Independent)
    """
    df = results_df.copy()
    n_rows = len(df)
    g = int(df.get("Are_Features_Dependent", pd.Series(False, index=df.index)).sum())
    l = int(df.get("Local_BH_Significant", pd.Series(False, index=df.index)).sum())
    s = int(df.get("Sibling_BH_Independent", pd.Series(False, index=df.index)).sum())

    print("Significance summary (counts | percent):")
    print(f"  Global BH dependent (Are_Features_Dependent):  {_fmt_ratio(g, n_rows)}")
    print(f"  Local BH significant (child vs parent):       {_fmt_ratio(l, n_rows)}")
    print(f"  Sibling BH independent (parents):             {_fmt_ratio(s, n_rows)}")


def explain_root_decision(
    tree: nx.DiGraph,
    results_df: pd.DataFrame,
    *,
    alpha_local: float = 0.05,
    n_features: Optional[int] = None,
) -> None:
    """
    Explain whether the root splits and why, based on gate conditions:
      1) Both children must diverge from parent (local KL p < alpha_local),
         or be marked as Local_BH_Significant.
      2) Parent's children must be independent (Sibling_BH_Independent is True).
    """
    # Find root and its two children
    roots = [n for n, d in tree.in_degree() if d == 0]
    if len(roots) != 1:
        print(f"Expected exactly 1 root, found {len(roots)}: {roots}")
        return
    root = roots[0]
    children = list(tree.successors(root))
    if len(children) != 2:
        print(f"Root is not binary (|children|={len(children)}); no split.")
        return
    c1, c2 = children

    # Pull BH local flags if present
    local_bh = results_df.get("Local_BH_Significant", pd.Series(False, index=[]))
    has_bh_local = "Local_BH_Significant" in results_df.columns

    # Compute raw p-values if BH not present or unknown nodes
    def _pval(child: str, parent: str) -> float:
        if child not in results_df.index:
            return math.nan
        kl = get_local_kl_value(tree, child, parent)
        n = results_df.loc[child].get("leaf_count", math.nan)
        # infer features from a distribution if not provided
        F = n_features
        if F is None:
            dist = results_df.loc[child].get("distribution", None)
            if dist is not None:
                try:
                    F = int(np.asarray(dist).size)
                except Exception:
                    F = None
        if not (np.isfinite(kl) and np.isfinite(n) and n > 0 and F and F > 0):
            return math.nan
        stat = 2.0 * float(n) * float(kl)
        return float(chi2.sf(stat, df=int(F)))

    if has_bh_local:
        c1_div = bool(local_bh.get(c1, False))
        c2_div = bool(local_bh.get(c2, False))
        gate_local = c1_div and c2_div
        print(
            f"Local gate (BH): child1={c1_div}, child2={c2_div} -> both={gate_local}"
        )
    else:
        p1 = _pval(c1, root)
        p2 = _pval(c2, root)
        gate_local = (p1 < alpha_local) and (p2 < alpha_local)
        print(
            f"Local gate (raw): p(child1|parent)={p1:.3g}, p(child2|parent)={p2:.3g} "
            f"-> both={gate_local} at alpha={alpha_local}"
        )

    # Sibling independence gate
    sib_indep = bool(results_df.get("Sibling_BH_Independent", pd.Series()).get(root, False))
    print(f"Sibling independence (BH) at root: {sib_indep}")

    will_split = gate_local and sib_indep
    print(f"Root split decision: {'SPLIT' if will_split else 'MERGE'}")
