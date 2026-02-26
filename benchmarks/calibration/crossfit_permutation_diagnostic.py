"""Decomposition-respecting cross-fit + permutation diagnostic.

This script does NOT change production inference. It evaluates whether
selection-aware inference reduces null inflation by:

1) Selecting topology/candidates on feature split A and testing on B.
2) Swapping (B -> A).
3) Computing hypothesis-level permutation p-values on inference split with
   fixed topology and decomposition-consistent candidate sets.
4) Combining fold p-values with Bonferroni-min: p_comb = min(1, 2*min(p1,p2)).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.util.time import format_timestamp_utc
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_projected_test,  # diagnostic-only use of internal routine
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    derive_projection_seed,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


@dataclass(frozen=True)
class EdgeHypothesis:
    parent: str
    child: str


@dataclass(frozen=True)
class SiblingHypothesis:
    parent: str
    left: str
    right: str


def _default_null_scenarios() -> list[dict[str, Any]]:
    return [
        {"scenario": "null_small", "n_samples": 64, "n_features": 32, "p_one": 0.5},
        {"scenario": "null_medium", "n_samples": 128, "n_features": 64, "p_one": 0.5},
        {"scenario": "null_large", "n_samples": 192, "n_features": 96, "p_one": 0.5},
    ]


def _make_null_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    p_one: float,
) -> pd.DataFrame:
    arr = (rng.random((n_samples, n_features)) < p_one).astype(np.float64)
    rows = [f"S{i}" for i in range(n_samples)]
    cols = [f"F{j}" for j in range(n_features)]
    return pd.DataFrame(arr, index=rows, columns=cols)


def _feature_split_indices(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_features)
    return idx[::2], idx[1::2]


def _select_candidates_from_decomposition(
    tree: PosetTree,
    stats_df: pd.DataFrame,
) -> tuple[list[EdgeHypothesis], list[SiblingHypothesis]]:
    """Freeze decomposition-consistent candidate set from selection fold.

    Candidate parent rule follows current production sibling setup:
    - Parent must be binary.
    - At least one child has significant child-parent divergence.
    For each candidate parent:
    - include both child-parent edges as candidate edge hypotheses.
    - include one sibling hypothesis (left,right).
    """
    edge_h: list[EdgeHypothesis] = []
    sib_h: list[SiblingHypothesis] = []

    if "Child_Parent_Divergence_Significant" not in stats_df.columns:
        return edge_h, sib_h

    sig = stats_df["Child_Parent_Divergence_Significant"].to_dict()
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        if not (bool(sig.get(left, False)) or bool(sig.get(right, False))):
            continue
        edge_h.append(EdgeHypothesis(parent=parent, child=left))
        edge_h.append(EdgeHypothesis(parent=parent, child=right))
        if left <= right:
            sib_h.append(SiblingHypothesis(parent=parent, left=left, right=right))
        else:
            sib_h.append(SiblingHypothesis(parent=parent, left=right, right=left))

    return edge_h, sib_h


def _mean_over_leaves(df: pd.DataFrame, leaves: tuple[str, ...]) -> np.ndarray:
    if len(leaves) == 0:
        raise ValueError("Cannot compute distribution over empty leaf set.")
    return df.loc[list(leaves)].to_numpy(dtype=np.float64).mean(axis=0)


def _edge_stat(
    *,
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    parent: str,
    child: str,
    branch_length: float | None,
    mean_branch_length: float | None,
) -> float:
    seed = derive_projection_seed(
        config.PROJECTION_RANDOM_SEED,
        f"edge:{parent}->{child}",
    )
    stat, _, _, invalid = _compute_projected_test(
        child_dist=np.asarray(child_dist, dtype=np.float64),
        parent_dist=np.asarray(parent_dist, dtype=np.float64),
        n_child=int(n_child),
        n_parent=int(n_parent),
        seed=int(seed),
        branch_length=branch_length,
        mean_branch_length=mean_branch_length,
    )
    if invalid or not np.isfinite(stat):
        return np.nan
    return float(stat)


def _sibling_stat(
    *,
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: int,
    n_right: int,
    parent: str,
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
) -> float:
    stat, _, p, = sibling_divergence_test(
        left_dist=np.asarray(left_dist, dtype=np.float64),
        right_dist=np.asarray(right_dist, dtype=np.float64),
        n_left=float(n_left),
        n_right=float(n_right),
        branch_length_left=branch_length_left,
        branch_length_right=branch_length_right,
        mean_branch_length=mean_branch_length,
        test_id=f"sibling:{parent}",
    )
    if not np.isfinite(stat) or not np.isfinite(p):
        return np.nan
    return float(stat)


def _perm_p_value(stats_perm: list[float], stat_obs: float) -> float:
    if not np.isfinite(stat_obs):
        return 1.0
    arr = np.asarray(stats_perm, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    b = arr.size
    if b == 0:
        return 1.0
    return float((1 + np.sum(arr >= stat_obs)) / (b + 1))


def _evaluate_fold(
    *,
    linkage_matrix: np.ndarray,
    leaf_names: list[str],
    select_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    alpha: float,
    n_perms: int,
    base_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return fold-level permutation p-values for edge/sibling hypotheses."""
    # Selection: build tree + run current decomposition flow
    tree_sel = PosetTree.from_linkage(linkage_matrix, leaf_names=leaf_names)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No eligible parent nodes for sibling tests",
            category=UserWarning,
        )
        tree_sel.decompose(
            leaf_data=select_df,
            alpha_local=float(alpha),
            sibling_alpha=float(alpha),
        )
    stats_sel = tree_sel.stats_df if tree_sel.stats_df is not None else pd.DataFrame(index=tree_sel.nodes())
    edge_h, sib_h = _select_candidates_from_decomposition(tree_sel, stats_sel)

    # Inference: same topology, inference features only
    tree_inf = PosetTree.from_linkage(linkage_matrix, leaf_names=leaf_names)
    tree_inf.populate_node_divergences(infer_df)
    desc = tree_inf.compute_descendant_sets(use_labels=True)
    mean_bl = compute_mean_branch_length(tree_inf)

    out_edge: dict[str, float] = {}
    out_sib: dict[str, float] = {}

    # Edge hypotheses
    for hyp in edge_h:
        parent, child = hyp.parent, hyp.child
        if parent not in desc or child not in desc:
            continue
        parent_leaves = tuple(sorted(desc[parent]))
        child_leaves = tuple(sorted(desc[child]))
        if len(parent_leaves) == 0 or len(child_leaves) == 0:
            continue
        if len(child_leaves) >= len(parent_leaves):
            continue

        parent_set = set(parent_leaves)
        child_set = set(child_leaves)
        if not child_set.issubset(parent_set):
            continue

        parent_dist = _mean_over_leaves(infer_df, parent_leaves)
        child_dist_obs = _mean_over_leaves(infer_df, child_leaves)
        n_parent = len(parent_leaves)
        n_child = len(child_leaves)
        branch_length = sanitize_positive_branch_length(
            tree_inf.edges[parent, child].get("branch_length")
            if tree_inf.has_edge(parent, child)
            else None
        )

        stat_obs = _edge_stat(
            child_dist=child_dist_obs,
            parent_dist=parent_dist,
            n_child=n_child,
            n_parent=n_parent,
            parent=parent,
            child=child,
            branch_length=branch_length,
            mean_branch_length=mean_bl,
        )

        seed = derive_projection_seed(
            int(base_seed),
            f"perm:edge:{parent}->{child}:d={infer_df.shape[1]}",
        )
        rng = np.random.default_rng(int(seed))

        perms: list[float] = []
        parent_arr = np.asarray(parent_leaves, dtype=object)
        for _ in range(int(n_perms)):
            perm = rng.permutation(parent_arr)
            child_star = tuple(perm[:n_child].tolist())
            child_dist_star = _mean_over_leaves(infer_df, child_star)
            stat_b = _edge_stat(
                child_dist=child_dist_star,
                parent_dist=parent_dist,
                n_child=n_child,
                n_parent=n_parent,
                parent=parent,
                child=child,
                branch_length=branch_length,
                mean_branch_length=mean_bl,
            )
            perms.append(stat_b)

        p_perm = _perm_p_value(perms, stat_obs)
        out_edge[f"edge:{parent}->{child}"] = p_perm

    # Sibling hypotheses
    for hyp in sib_h:
        parent, left, right = hyp.parent, hyp.left, hyp.right
        if parent not in desc or left not in desc or right not in desc:
            continue
        left_leaves = tuple(sorted(desc[left]))
        right_leaves = tuple(sorted(desc[right]))
        if len(left_leaves) == 0 or len(right_leaves) == 0:
            continue
        union = tuple(sorted(set(left_leaves) | set(right_leaves)))
        n_left = len(left_leaves)
        n_right = len(right_leaves)
        if n_left + n_right == 0:
            continue

        left_dist_obs = _mean_over_leaves(infer_df, left_leaves)
        right_dist_obs = _mean_over_leaves(infer_df, right_leaves)
        bl_left = sanitize_positive_branch_length(
            tree_inf.edges[parent, left].get("branch_length")
            if tree_inf.has_edge(parent, left)
            else None
        )
        bl_right = sanitize_positive_branch_length(
            tree_inf.edges[parent, right].get("branch_length")
            if tree_inf.has_edge(parent, right)
            else None
        )

        stat_obs = _sibling_stat(
            left_dist=left_dist_obs,
            right_dist=right_dist_obs,
            n_left=n_left,
            n_right=n_right,
            parent=parent,
            branch_length_left=bl_left,
            branch_length_right=bl_right,
            mean_branch_length=mean_bl,
        )

        seed = derive_projection_seed(
            int(base_seed),
            f"perm:sibling:{parent}:{left}|{right}:d={infer_df.shape[1]}",
        )
        rng = np.random.default_rng(int(seed))
        union_arr = np.asarray(union, dtype=object)

        perms: list[float] = []
        for _ in range(int(n_perms)):
            perm = rng.permutation(union_arr)
            left_star = tuple(perm[:n_left].tolist())
            right_star = tuple(perm[n_left : n_left + n_right].tolist())
            left_dist_star = _mean_over_leaves(infer_df, left_star)
            right_dist_star = _mean_over_leaves(infer_df, right_star)
            stat_b = _sibling_stat(
                left_dist=left_dist_star,
                right_dist=right_dist_star,
                n_left=n_left,
                n_right=n_right,
                parent=parent,
                branch_length_left=bl_left,
                branch_length_right=bl_right,
                mean_branch_length=mean_bl,
            )
            perms.append(stat_b)

        p_perm = _perm_p_value(perms, stat_obs)
        out_sib[f"sibling:{parent}|{left}|{right}"] = p_perm

    return out_edge, out_sib


def _combine_two_fold_pvals(
    pvals_1: dict[str, float],
    pvals_2: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = set(pvals_1) | set(pvals_2)
    for k in keys:
        p1 = float(pvals_1.get(k, 1.0))
        p2 = float(pvals_2.get(k, 1.0))
        out[k] = float(min(1.0, 2.0 * min(p1, p2)))
    return out


def run_crossfit_permutation_diagnostic(
    output_dir: Path,
    *,
    alpha: float = 0.05,
    n_reps: int = 10,
    n_perms: int = 100,
    seed: int = 20260213,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    scenarios = _default_null_scenarios()

    rep_rows: list[dict[str, Any]] = []
    hyp_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        sc_name = str(scenario["scenario"])
        n_samples = int(scenario["n_samples"])
        n_features = int(scenario["n_features"])
        p_one = float(scenario["p_one"])

        for rep in range(int(n_reps)):
            data_df = _make_null_data(rng, n_samples, n_features, p_one)
            idx_a, idx_b = _feature_split_indices(n_features)
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            features_a = [data_df.columns[i] for i in idx_a]
            features_b = [data_df.columns[i] for i in idx_b]

            data_a = data_df.loc[:, features_a]
            data_b = data_df.loc[:, features_b]

            # Selection topology for A->B
            dist_a = pdist(data_a.to_numpy(dtype=np.float64), metric=config.TREE_DISTANCE_METRIC)
            linkage_a = linkage(dist_a, method=config.TREE_LINKAGE_METHOD)
            leaf_names = data_df.index.tolist()

            p_edge_ab, p_sib_ab = _evaluate_fold(
                linkage_matrix=linkage_a,
                leaf_names=leaf_names,
                select_df=data_a,
                infer_df=data_b,
                alpha=float(alpha),
                n_perms=int(n_perms),
                base_seed=derive_projection_seed(int(seed), f"{sc_name}:rep{rep}:AtoB"),
            )

            # Selection topology for B->A
            dist_b = pdist(data_b.to_numpy(dtype=np.float64), metric=config.TREE_DISTANCE_METRIC)
            linkage_b = linkage(dist_b, method=config.TREE_LINKAGE_METHOD)
            p_edge_ba, p_sib_ba = _evaluate_fold(
                linkage_matrix=linkage_b,
                leaf_names=leaf_names,
                select_df=data_b,
                infer_df=data_a,
                alpha=float(alpha),
                n_perms=int(n_perms),
                base_seed=derive_projection_seed(int(seed), f"{sc_name}:rep{rep}:BtoA"),
            )

            p_edge_comb = _combine_two_fold_pvals(p_edge_ab, p_edge_ba)
            p_sib_comb = _combine_two_fold_pvals(p_sib_ab, p_sib_ba)

            edge_vals = np.array(list(p_edge_comb.values()), dtype=np.float64)
            sib_vals = np.array(list(p_sib_comb.values()), dtype=np.float64)
            edge_rej = int(np.sum(edge_vals <= alpha)) if edge_vals.size > 0 else 0
            sib_rej = int(np.sum(sib_vals <= alpha)) if sib_vals.size > 0 else 0

            rep_rows.append(
                {
                    "scenario": sc_name,
                    "replicate": int(rep),
                    "alpha": float(alpha),
                    "edge_hypotheses": int(edge_vals.size),
                    "edge_rejects": int(edge_rej),
                    "edge_type1": float(edge_rej / edge_vals.size) if edge_vals.size > 0 else np.nan,
                    "sibling_hypotheses": int(sib_vals.size),
                    "sibling_rejects": int(sib_rej),
                    "sibling_type1": float(sib_rej / sib_vals.size) if sib_vals.size > 0 else np.nan,
                }
            )

            for h, p in p_edge_comb.items():
                hyp_rows.append(
                    {
                        "scenario": sc_name,
                        "replicate": int(rep),
                        "kind": "edge",
                        "hypothesis_id": h,
                        "p_comb": float(p),
                        "reject": int(p <= alpha),
                    }
                )
            for h, p in p_sib_comb.items():
                hyp_rows.append(
                    {
                        "scenario": sc_name,
                        "replicate": int(rep),
                        "kind": "sibling",
                        "hypothesis_id": h,
                        "p_comb": float(p),
                        "reject": int(p <= alpha),
                    }
                )

    rep_df = pd.DataFrame(rep_rows)
    hyp_df = pd.DataFrame(hyp_rows)

    if rep_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "scenario",
                "alpha",
                "edge_hypotheses",
                "edge_rejects",
                "edge_type1",
                "sibling_hypotheses",
                "sibling_rejects",
                "sibling_type1",
            ]
        )
    else:
        summary_df = (
            rep_df.groupby("scenario", as_index=False)
            .agg(
                alpha=("alpha", "first"),
                edge_hypotheses=("edge_hypotheses", "sum"),
                edge_rejects=("edge_rejects", "sum"),
                sibling_hypotheses=("sibling_hypotheses", "sum"),
                sibling_rejects=("sibling_rejects", "sum"),
            )
            .sort_values("scenario")
        )
        summary_df["edge_type1"] = summary_df["edge_rejects"] / summary_df["edge_hypotheses"].replace(0, np.nan)
        summary_df["sibling_type1"] = summary_df["sibling_rejects"] / summary_df["sibling_hypotheses"].replace(0, np.nan)

    rep_csv = output_dir / "crossfit_permutation_replicate_level.csv"
    hyp_csv = output_dir / "crossfit_permutation_hypothesis_level.csv"
    summary_csv = output_dir / "crossfit_permutation_summary.csv"
    rep_df.to_csv(rep_csv, index=False)
    hyp_df.to_csv(hyp_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    return {
        "replicate_csv": str(rep_csv),
        "hypothesis_csv": str(hyp_csv),
        "summary_csv": str(summary_csv),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run decomposition-respecting cross-fit permutation diagnostic."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-reps", type=int, default=8)
    parser.add_argument("--n-perms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260213)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/run_<timestamp>/calibration/crossfit_permutation_diagnostic"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "results"
            / f"run_{format_timestamp_utc()}"
            / "calibration"
            / "crossfit_permutation_diagnostic"
        )
    outputs = run_crossfit_permutation_diagnostic(
        out_dir,
        alpha=float(args.alpha),
        n_reps=int(args.n_reps),
        n_perms=int(args.n_perms),
        seed=int(args.seed),
    )
    print("Cross-fit permutation diagnostic complete:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
