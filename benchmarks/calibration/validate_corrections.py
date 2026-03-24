"""Validate correction procedures under the null hypothesis.

Generates pure null data (iid Bernoulli(0.5)), builds trees, and runs
the full pipeline many times to measure:

  1. Edge test (Gate 2) Type I error under flat BH vs TreeBH
  2. Sibling test (Gate 3) Type I error: raw Wald vs adjusted Wald
  3. T/k distribution vs χ²(k) reference (calibration quality)
  4. p-value uniformity (KS test against Uniform(0,1))
  5. End-to-end: how often K > 1 is declared under pure null

Usage
-----
    python benchmarks/calibration/validate_corrections.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import kstest

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

N_TRIALS = 100
N_SAMPLES = 100
N_FEATURES = 50
ALPHA = 0.01

RESULTS_DIR = Path("benchmarks/results")
OUT_PDF = RESULTS_DIR / "correction_validation.pdf"


# ---------------------------------------------------------------------------
# Null data generator
# ---------------------------------------------------------------------------


def _generate_null(n: int, p: int, seed: int) -> pd.DataFrame:
    """Pure null: all rows iid Bernoulli(0.5)."""
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    return pd.DataFrame(
        X,
        index=[f"S{i}" for i in range(n)],
        columns=[f"F{j}" for j in range(p)],
    )


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------


def _run_null_trial(seed: int, method: str, alpha: float) -> dict:
    """Run full pipeline on null data and extract diagnostics."""
    data = _generate_null(N_SAMPLES, N_FEATURES, seed)

    orig = (config.SIBLING_TEST_METHOD, config.SIBLING_ALPHA, config.EDGE_ALPHA)
    try:
        config.SIBLING_TEST_METHOD = method
        config.SIBLING_ALPHA = alpha
        config.EDGE_ALPHA = alpha

        dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = tree.decompose(
                leaf_data=data,
                alpha_local=alpha,
                sibling_alpha=alpha,
            )

        k_found = decomp.get("num_clusters", 0)
        annotations = tree.annotations_df

        # Edge test diagnostics
        edge_tested = annotations[annotations["Child_Parent_Divergence_P_Value"].notna()]
        edge_p_raw = edge_tested["Child_Parent_Divergence_P_Value"].dropna().values
        edge_p_bh = edge_tested["Child_Parent_Divergence_P_Value_BH"].dropna().values
        edge_reject = edge_tested["Child_Parent_Divergence_Significant"].sum()
        edge_total = len(edge_tested)

        # Sibling test diagnostics
        sib_tested = annotations[
            (annotations["Sibling_Divergence_Skipped"] == False)  # noqa: E712
            & annotations["Sibling_Divergence_P_Value"].notna()
        ]
        sib_p_raw = sib_tested["Sibling_Divergence_P_Value"].dropna().values
        sib_p_bh = sib_tested["Sibling_Divergence_P_Value_Corrected"].dropna().values
        sib_T = sib_tested["Sibling_Test_Statistic"].dropna().values
        sib_k = sib_tested["Sibling_Degrees_of_Freedom"].dropna().values
        sib_reject = sib_tested["Sibling_BH_Different"].sum() if len(sib_tested) > 0 else 0
        sib_total = len(sib_tested)

        # Calibration audit
        audit = annotations.attrs.get("sibling_divergence_audit", {})
        c_hat = audit.get("global_inflation_factor", None)
        diag = audit.get("diagnostics", {})
        r2 = diag.get("r_squared", None)

        return {
            "seed": seed,
            "method": method,
            "k_found": k_found,
            "edge_total": edge_total,
            "edge_reject": edge_reject,
            "edge_p_raw": edge_p_raw,
            "edge_p_bh": edge_p_bh,
            "sib_total": sib_total,
            "sib_reject": sib_reject,
            "sib_p_raw": sib_p_raw,
            "sib_p_bh": sib_p_bh,
            "sib_T": sib_T,
            "sib_k": sib_k,
            "c_hat": c_hat,
            "r2": r2,
        }
    finally:
        config.SIBLING_TEST_METHOD, config.SIBLING_ALPHA, config.EDGE_ALPHA = orig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    methods = ["wald", "cousin_adjusted_wald"]
    all_results: list[dict] = []

    for method in methods:
        print(f"\n{'='*70}")
        print(
            f"Method: {method}, α={ALPHA}, {N_TRIALS} null trials "
            f"(n={N_SAMPLES}, p={N_FEATURES})"
        )
        print(f"{'='*70}")

        for i in range(N_TRIALS):
            if (i + 1) % 20 == 0:
                print(f"  Trial {i+1}/{N_TRIALS}")
            result = _run_null_trial(seed=1000 + i, method=method, alpha=ALPHA)
            all_results.append(result)

    # =====================================================================
    # Aggregate results
    # =====================================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for method in methods:
        trials = [r for r in all_results if r["method"] == method]

        # 1. End-to-end: K > 1 rate (false discovery of structure)
        k_vals = [r["k_found"] for r in trials]
        k_gt1 = sum(1 for k in k_vals if k > 1)
        mean_k = np.mean(k_vals)

        # 2. Edge test Type I
        edge_rejects = [r["edge_reject"] for r in trials]
        edge_totals = [r["edge_total"] for r in trials]
        edge_fdr = np.mean([r / t if t > 0 else 0 for r, t in zip(edge_rejects, edge_totals)])

        # 3. Sibling test Type I
        sib_rejects = [r["sib_reject"] for r in trials]
        sib_totals = [r["sib_total"] for r in trials]
        sib_fdr = np.mean([r / t if t > 0 else 0 for r, t in zip(sib_rejects, sib_totals)])
        sib_any_reject = sum(1 for r in sib_rejects if r > 0)

        # 4. Collect all sibling p-values for uniformity test
        all_sib_p_raw = np.concatenate([r["sib_p_raw"] for r in trials if len(r["sib_p_raw"]) > 0])
        all_sib_p_bh = np.concatenate([r["sib_p_bh"] for r in trials if len(r["sib_p_bh"]) > 0])

        # 5. Edge p-value uniformity
        all_edge_p_raw = np.concatenate(
            [r["edge_p_raw"] for r in trials if len(r["edge_p_raw"]) > 0]
        )

        # 6. T/k ratio for sibling test
        all_T = np.concatenate([r["sib_T"] for r in trials if len(r["sib_T"]) > 0])
        all_k = np.concatenate([r["sib_k"] for r in trials if len(r["sib_k"]) > 0])
        valid = (all_k > 0) & np.isfinite(all_T) & np.isfinite(all_k)
        T_over_k = all_T[valid] / all_k[valid]

        # 7. Calbration model
        c_hats = [r["c_hat"] for r in trials if r["c_hat"] is not None]
        r2s = [r["r2"] for r in trials if r["r2"] is not None]

        # KS test for uniformity of raw edge p-values
        ks_edge = kstest(all_edge_p_raw, "uniform") if len(all_edge_p_raw) > 10 else None

        # KS for sibling raw p
        ks_sib = kstest(all_sib_p_raw, "uniform") if len(all_sib_p_raw) > 10 else None

        print(f"\n--- {method} ---")
        print(
            f"  End-to-end K > 1 rate: {k_gt1}/{N_TRIALS} ({100*k_gt1/N_TRIALS:.1f}%)"
            f"  [should be ≤ {100*ALPHA:.0f}%]"
        )
        print(f"  Mean K found: {mean_k:.2f}  [should be ≈ 1.0]")
        print(f"  Edge FDR (per-trial mean): {edge_fdr:.4f}  [should be ≤ {ALPHA}]")
        print(f"  Sibling FDR (per-trial mean): {sib_fdr:.4f}  [should be ≤ {ALPHA}]")
        print(f"  Trials with any sibling rejection: {sib_any_reject}/{N_TRIALS}")
        print(f"  Edge p-values collected: {len(all_edge_p_raw)}")
        print(f"  Sibling p-values collected: {len(all_sib_p_raw)}")

        if ks_edge:
            print(
                f"  Edge raw p KS test: D={ks_edge.statistic:.4f}, "
                f"p={ks_edge.pvalue:.4f} {'✓ uniform' if ks_edge.pvalue > 0.05 else '✗ NOT uniform'}"
            )
        if ks_sib:
            print(
                f"  Sibling raw p KS test: D={ks_sib.statistic:.4f}, "
                f"p={ks_sib.pvalue:.4f} {'✓ uniform' if ks_sib.pvalue > 0.05 else '✗ NOT uniform'}"
            )

        if T_over_k.size > 0:
            print(
                f"  Sibling T/k: mean={T_over_k.mean():.3f}, "
                f"median={np.median(T_over_k):.3f}  [should be ≈ 1.0 under χ²(k)]"
            )

        if c_hats:
            print(
                f"  Calibration ĉ: mean={np.mean(c_hats):.3f}, " f"median={np.median(c_hats):.3f}"
            )
        if r2s:
            print(f"  Calibration R²: mean={np.mean(r2s):.3f}")

    # =====================================================================
    # Plot
    # =====================================================================

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Correction Validation Under Null (n={N_SAMPLES}, p={N_FEATURES}, "
        f"α={ALPHA}, {N_TRIALS} trials)",
        fontsize=13,
        fontweight="bold",
    )

    colors = {"wald": "red", "cousin_adjusted_wald": "steelblue"}

    # -- Panel A: Edge raw p-value QQ plot --
    ax = axes[0, 0]
    for method in methods:
        trials = [r for r in all_results if r["method"] == method]
        p_raw = np.concatenate([r["edge_p_raw"] for r in trials if len(r["edge_p_raw"]) > 0])
        if len(p_raw) > 0:
            p_sorted = np.sort(p_raw)
            expected = np.linspace(0, 1, len(p_sorted) + 2)[1:-1]
            ax.plot(
                expected,
                p_sorted,
                ".",
                color=colors[method],
                alpha=0.3,
                ms=2,
                label=method.replace("_", " "),
            )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Expected (Uniform)")
    ax.set_ylabel("Observed (raw p)")
    ax.set_title("A. Edge test raw p-values (Gate 2)")
    ax.legend(fontsize=7)

    # -- Panel B: Sibling raw p-value QQ plot --
    ax = axes[0, 1]
    for method in methods:
        trials = [r for r in all_results if r["method"] == method]
        p_raw = np.concatenate([r["sib_p_raw"] for r in trials if len(r["sib_p_raw"]) > 0])
        if len(p_raw) > 0:
            p_sorted = np.sort(p_raw)
            expected = np.linspace(0, 1, len(p_sorted) + 2)[1:-1]
            ax.plot(
                expected,
                p_sorted,
                ".",
                color=colors[method],
                alpha=0.3,
                ms=2,
                label=method.replace("_", " "),
            )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Expected (Uniform)")
    ax.set_ylabel("Observed (raw p)")
    ax.set_title("B. Sibling test raw p-values (Gate 3)")
    ax.legend(fontsize=7)

    # -- Panel C: T/k histogram vs χ²(1) density --
    ax = axes[0, 2]
    for method in methods:
        trials = [r for r in all_results if r["method"] == method]
        all_T = np.concatenate([r["sib_T"] for r in trials if len(r["sib_T"]) > 0])
        all_k = np.concatenate([r["sib_k"] for r in trials if len(r["sib_k"]) > 0])
        valid = (all_k > 0) & np.isfinite(all_T) & np.isfinite(all_k)
        ratio = all_T[valid] / all_k[valid]
        if len(ratio) > 0:
            bins = np.linspace(0, 4, 50)
            ax.hist(
                ratio,
                bins=bins,
                density=True,
                alpha=0.4,
                color=colors[method],
                label=f"{method.replace('_', ' ')} (mean={ratio.mean():.2f})",
            )
    # Reference: E[χ²(k)/k] = 1 under null
    x = np.linspace(0.01, 4, 200)
    ax.axvline(1.0, color="black", ls="--", lw=1, label="E[T/k]=1 under null")
    ax.set_xlabel("T/k")
    ax.set_ylabel("Density")
    ax.set_title("C. Sibling T/k distribution")
    ax.legend(fontsize=7)

    # -- Panel D: K found histogram --
    ax = axes[1, 0]
    for method in methods:
        trials = [r for r in all_results if r["method"] == method]
        k_vals = [r["k_found"] for r in trials]
        bins = np.arange(0.5, max(k_vals) + 1.5, 1)
        ax.hist(
            k_vals,
            bins=bins,
            alpha=0.5,
            color=colors[method],
            label=f"{method.replace('_', ' ')} (mean={np.mean(k_vals):.2f})",
        )
    ax.axvline(1.0, color="green", ls="--", lw=2, label="True K=1")
    ax.set_xlabel("K found")
    ax.set_ylabel("Count")
    ax.set_title("D. End-to-end K under null")
    ax.legend(fontsize=7)

    # -- Panel E: Per-trial sibling rejection count --
    ax = axes[1, 1]
    for method in methods:
        trials = [r for r in all_results if r["method"] == method]
        rejects = [r["sib_reject"] for r in trials]
        ax.hist(
            rejects,
            bins=(
                np.arange(-0.5, max(rejects) + 1.5, 1)
                if max(rejects) > 0
                else np.arange(-0.5, 3.5, 1)
            ),
            alpha=0.5,
            color=colors[method],
            label=f"{method.replace('_', ' ')}",
        )
    ax.set_xlabel("# sibling rejections per trial")
    ax.set_ylabel("Count")
    ax.set_title("E. Sibling BH rejections per trial")
    ax.legend(fontsize=7)

    # -- Panel F: Calibration ĉ distribution (adjusted Wald only) --
    ax = axes[1, 2]
    adj_trials = [r for r in all_results if r["method"] == "cousin_adjusted_wald"]
    c_hats = [r["c_hat"] for r in adj_trials if r["c_hat"] is not None]
    if c_hats:
        ax.hist(c_hats, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(1.0, color="red", ls="--", lw=1.5, label="No inflation (ĉ=1)")
        ax.axvline(
            np.mean(c_hats), color="orange", ls="-", lw=1.5, label=f"Mean ĉ = {np.mean(c_hats):.3f}"
        )
        ax.set_xlabel("ĉ (median T/k of null-like pairs)")
        ax.set_ylabel("Count")
        ax.set_title("F. Calibration ĉ under null")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("F. Calibration ĉ under null")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    main()
