"""Pipeline-level false-positive rate under the null hypothesis.

Answers the question: when all samples come from a *single* iid Bernoulli
distribution (true K=1), how often does the full pipeline produce K > 1?

This is the critical calibration gap identified in PUBLICATION_READINESS_REVIEW.md
(M-007).  The existing null calibration in ``run.py`` reports *test-level* rejection
rates (what fraction of individual edge/sibling tests reject).  Those are necessary
but insufficient — a 38% sibling rejection rate might or might not translate into a
pipeline-level false split depending on how the three gates interact.

Design
------
For each null scenario × sibling method × replicate:

1. Generate iid Bernoulli(p) binary matrix (true K = 1).
2. Build linkage tree from the data (in-sample — the whole point).
3. Run ``tree.decompose()`` using the configured sibling method.
4. Record ``K_found`` from the decomposition result.
5. A false positive = K_found > 1.

After all replicates, compute:
- **Pipeline FPR** = P(K_found > 1)  — the "bottom line" metric
- **Mean K** and **Max K** under null
- 95% binomial confidence interval on FPR
- Test-level edge/sibling Type-I rates (for diagnostic context)

Usage
-----
Standalone::

    python -m benchmarks.calibration.null_pipeline_fpr
    python -m benchmarks.calibration.null_pipeline_fpr --n-reps 100 --seed 42
    python -m benchmarks.calibration.null_pipeline_fpr --methods wald,cousin_tree_guided

Programmatic::

    from benchmarks.calibration.null_pipeline_fpr import run_null_pipeline_fpr
    summary_df, replicate_df = run_null_pipeline_fpr(n_reps=50)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Null data generation
# ---------------------------------------------------------------------------


def _make_null_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    p_one: float,
) -> pd.DataFrame:
    """Generate iid Bernoulli(p) binary matrix — true K = 1."""
    matrix = (rng.random((n_samples, n_features)) < p_one).astype(int)
    rows = [f"S{i}" for i in range(n_samples)]
    cols = [f"F{j}" for j in range(n_features)]
    return pd.DataFrame(matrix, index=rows, columns=cols)


# ---------------------------------------------------------------------------
# Default scenarios
# ---------------------------------------------------------------------------


def _default_scenarios() -> list[dict[str, Any]]:
    """Null scenarios spanning small / medium / large sample sizes."""
    return [
        {"scenario": "null_small", "n_samples": 64, "n_features": 32, "p_one": 0.5},
        {"scenario": "null_medium", "n_samples": 128, "n_features": 64, "p_one": 0.5},
        {"scenario": "null_large", "n_samples": 192, "n_features": 96, "p_one": 0.5},
    ]


ALL_SIBLING_METHODS = [
    "wald",
    "cousin_ftest",
    "cousin_adjusted_wald",
    "cousin_tree_guided",
]


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def _binomial_ci_95(successes: int, trials: int) -> tuple[float, float]:
    """Normal-approximation 95% CI for a Bernoulli proportion."""
    if trials <= 0:
        return float("nan"), float("nan")
    p_hat = successes / trials
    se = np.sqrt(max(p_hat * (1.0 - p_hat) / trials, 0.0))
    z = 1.96
    return (max(0.0, p_hat - z * se), min(1.0, p_hat + z * se))


# ---------------------------------------------------------------------------
# Single replicate
# ---------------------------------------------------------------------------


def _run_one_replicate(
    rng: np.random.Generator,
    scenario: dict[str, Any],
    alpha: float,
    method: str,
) -> dict[str, Any]:
    """Run one null replicate: generate data → build tree → decompose → record K."""
    data_df = _make_null_data(
        rng,
        n_samples=int(scenario["n_samples"]),
        n_features=int(scenario["n_features"]),
        p_one=float(scenario["p_one"]),
    )

    # Temporarily set the sibling method for this replicate.
    original_method = config.SIBLING_TEST_METHOD
    try:
        config.SIBLING_TEST_METHOD = method

        distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        linkage_matrix = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())

        result = tree.decompose(
            leaf_data=data_df,
            alpha_local=float(alpha),
            sibling_alpha=float(alpha),
        )
    finally:
        config.SIBLING_TEST_METHOD = original_method

    k_found = int(result.get("num_clusters", 1))

    # Collect test-level diagnostics from stats_df.
    stats_df = tree.stats_df if tree.stats_df is not None else pd.DataFrame()

    edge_tests = 0
    edge_rejects = 0
    sibling_tests = 0
    sibling_rejects = 0

    if not stats_df.empty:
        if "Child_Parent_Divergence_P_Value_BH" in stats_df.columns:
            mask = stats_df["Child_Parent_Divergence_P_Value_BH"].notna()
            edge_tests = int(mask.sum())
            if edge_tests > 0:
                edge_rejects = int(
                    stats_df.loc[mask, "Child_Parent_Divergence_Significant"].astype(bool).sum()
                )
        if "Sibling_Divergence_P_Value_Corrected" in stats_df.columns:
            mask = stats_df["Sibling_Divergence_P_Value_Corrected"].notna()
            sibling_tests = int(mask.sum())
            if sibling_tests > 0:
                sibling_rejects = int(stats_df.loc[mask, "Sibling_BH_Different"].astype(bool).sum())

    return {
        "scenario": scenario["scenario"],
        "n_samples": int(scenario["n_samples"]),
        "n_features": int(scenario["n_features"]),
        "sibling_method": method,
        "alpha": float(alpha),
        "k_found": k_found,
        "false_positive": int(k_found > 1),
        "edge_tests": edge_tests,
        "edge_rejects": edge_rejects,
        "sibling_tests": sibling_tests,
        "sibling_rejects": sibling_rejects,
    }


# ---------------------------------------------------------------------------
# Summarize across replicates
# ---------------------------------------------------------------------------


def _summarize(replicate_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level results into per-(scenario, method) summary."""
    grouped = replicate_df.groupby(["scenario", "sibling_method"], as_index=False).agg(
        n_samples=("n_samples", "first"),
        n_features=("n_features", "first"),
        alpha=("alpha", "first"),
        n_reps=("false_positive", "count"),
        false_positives=("false_positive", "sum"),
        mean_k=("k_found", "mean"),
        max_k=("k_found", "max"),
        edge_type1=(
            "edge_rejects",
            lambda s: s.sum() / max(replicate_df.loc[s.index, "edge_tests"].sum(), 1),
        ),
        sibling_type1=(
            "sibling_rejects",
            lambda s: s.sum() / max(replicate_df.loc[s.index, "sibling_tests"].sum(), 1),
        ),
    )

    grouped["pipeline_fpr"] = grouped["false_positives"] / grouped["n_reps"]
    cis = [
        _binomial_ci_95(int(r), int(t))
        for r, t in zip(grouped["false_positives"], grouped["n_reps"])
    ]
    grouped["fpr_ci_low"] = [c[0] for c in cis]
    grouped["fpr_ci_high"] = [c[1] for c in cis]

    # Reorder columns for readability.
    col_order = [
        "scenario",
        "sibling_method",
        "n_samples",
        "n_features",
        "alpha",
        "n_reps",
        "false_positives",
        "pipeline_fpr",
        "fpr_ci_low",
        "fpr_ci_high",
        "mean_k",
        "max_k",
        "edge_type1",
        "sibling_type1",
    ]
    return grouped[[c for c in col_order if c in grouped.columns]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_null_pipeline_fpr(
    *,
    scenarios: list[dict[str, Any]] | None = None,
    methods: list[str] | None = None,
    alpha: float | None = None,
    n_reps: int = 30,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the null-pipeline FPR benchmark.

    Parameters
    ----------
    scenarios : list[dict], optional
        Null data scenarios.  Defaults to small/medium/large.
    methods : list[str], optional
        Sibling test methods to evaluate.  Defaults to all four.
    alpha : float, optional
        Significance level.  Defaults to ``config.SIBLING_ALPHA``.
    n_reps : int
        Number of replicates per scenario × method.
    seed : int
        Base RNG seed.
    output_dir : Path, optional
        If provided, write CSV artifacts here.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (scenario, method) with pipeline FPR and diagnostics.
    replicate_df : pd.DataFrame
        One row per replicate with K_found and test-level counts.
    """
    if scenarios is None:
        scenarios = _default_scenarios()
    if methods is None:
        methods = list(ALL_SIBLING_METHODS)
    alpha_eff = float(config.SIBLING_ALPHA if alpha is None else alpha)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(methods) * n_reps
    done = 0

    for scenario in scenarios:
        for method in methods:
            for rep in range(n_reps):
                row = _run_one_replicate(rng, scenario, alpha_eff, method)
                row["replicate"] = rep
                rows.append(row)
                done += 1
                if done % 20 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {scenario['scenario']} / {method} "
                        f"rep={rep} → K={row['k_found']}",
                        flush=True,
                    )

    replicate_df = pd.DataFrame(rows)
    summary_df = _summarize(replicate_df)

    # Print results table.
    print("\n" + "=" * 80)
    print("NULL-PIPELINE FALSE-POSITIVE RATE SUMMARY")
    print(f"alpha = {alpha_eff},  n_reps = {n_reps},  seed = {seed}")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        status = "PASS" if row["pipeline_fpr"] <= alpha_eff else "FAIL"
        print(
            f"  [{status}] {row['scenario']:15s} | {row['sibling_method']:25s} | "
            f"FPR={row['pipeline_fpr']:.3f} [{row['fpr_ci_low']:.3f}, {row['fpr_ci_high']:.3f}] | "
            f"mean_K={row['mean_k']:.2f} max_K={int(row['max_k'])} | "
            f"edge_T1={row['edge_type1']:.3f} sib_T1={row['sibling_type1']:.3f}"
        )
    print("=" * 80)

    # Persist if output_dir given.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        replicate_df.to_csv(output_dir / "null_pipeline_replicate.csv", index=False)
        summary_df.to_csv(output_dir / "null_pipeline_fpr_summary.csv", index=False)
        print(f"\nArtifacts written to {output_dir}")

    return summary_df, replicate_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Null-pipeline FPR benchmark: generate single-cluster data, "
            "build tree, decompose, measure false-split rate across sibling methods."
        ),
    )
    parser.add_argument("--n-reps", type=int, default=30, help="Replicates per scenario×method")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=None, help="Significance level")
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated sibling methods (default: all four)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parents[1]
        / "results"
        / f"null_fpr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}"
    )
    run_null_pipeline_fpr(
        methods=methods,
        alpha=args.alpha,
        n_reps=args.n_reps,
        seed=args.seed,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
