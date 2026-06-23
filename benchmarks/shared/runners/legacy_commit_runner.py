"""Runners for importable legacy Tree-Break Selection method snapshots."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from tree_break_selection.legacy_methods.commit_c2ef9a69 import COMMIT
from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection import (
    config as legacy_config,
)
from tree_break_selection.legacy_methods.commit_c2ef9a69.tree_break_selection.tree.poset_tree import (
    PosetTree as LegacyPosetTree,
)

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import _labels_and_report_from_decomposition
from benchmarks.shared.util.time import elapsed_since


def _run_legacy_c2ef9a69_tbs_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray | None,
    sibling_significance_level: float,
    tree_linkage_method: str = legacy_config.TREE_LINKAGE_METHOD,
    *,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    edge_alpha: float = legacy_config.EDGE_ALPHA,
    passthrough: bool = legacy_config.PASSTHROUGH,
    **unused_modern_kwargs: Any,
) -> MethodRunResult:
    """Run the full Tree-Break Selection method snapshot from commit ``c2ef9a69``.

    The historical method only supports SciPy linkage trees over a condensed
    distance vector and Bernoulli-style leaf matrices. Newer options such as
    typed feature spaces, phylogenetic builders, spectral transport, and
    root-stability guards are intentionally ignored here; using them would make
    this a hybrid rather than the copied old method.
    """
    if tree_builder != "linkage":
        raise ValueError(
            "Legacy c2ef9a69 TBS runner only supports tree_builder='linkage'."
        )
    if tree_rooting != "linkage_root":
        raise ValueError(
            "Legacy c2ef9a69 TBS runner only supports tree_rooting='linkage_root'."
        )
    if distance_condensed is None:
        raise ValueError("Legacy c2ef9a69 TBS runner requires distance_condensed.")

    stage_timings: dict[str, float] = {}
    tree_build_start_sec = perf_counter()
    linkage_matrix = linkage(distance_condensed, method=tree_linkage_method)
    tree = LegacyPosetTree.from_linkage(
        linkage_matrix,
        leaf_names=data_df.index.tolist(),
    )
    stage_timings["tree_build_sec"] = elapsed_since(tree_build_start_sec)

    decomposition_start_sec = perf_counter()
    decomposition = tree.decompose(
        leaf_data=data_df,
        alpha_local=float(edge_alpha),
        sibling_alpha=float(sibling_significance_level),
        passthrough=bool(passthrough),
    )
    stage_timings["legacy_decomposition_sec"] = elapsed_since(decomposition_start_sec)

    labels, report_df = _labels_and_report_from_decomposition(
        decomposition,
        data_df.index.tolist(),
    )
    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomposition["num_clusters"]),
        report_df=report_df,
        status="ok",
        skip_reason=None,
        extra={
            "legacy_commit": COMMIT,
            "tree": tree,
            "decomposition": decomposition,
            "annotations": tree.annotations_df,
            "linkage_matrix": linkage_matrix,
            "tree_builder": tree_builder,
            "tree_rooting": tree_rooting,
            "stage_timings": stage_timings,
            "passthrough": bool(passthrough),
            "ignored_modern_kwargs": tuple(sorted(unused_modern_kwargs)),
        },
    )


__all__ = ["_run_legacy_c2ef9a69_tbs_method"]
