"""Plot the structural relationship between raw Wald and adjusted Wald.

Reads the node-level and grid-search CSVs produced by
``method_relationship.py`` and ``grid_search.py``, generating a
multi-panel figure.

Usage
-----
    python benchmarks/calibration/plot_method_relationship.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("benchmarks/results")
NODE_CSV = RESULTS / "method_relationship_nodes.csv"
GRID_CSV = RESULTS / "grid_search_calibration_bh.csv"
OUT_PDF = RESULTS / "method_relationship.pdf"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(NODE_CSV)
    grid = pd.read_csv(GRID_CSV)
    return nodes, grid


# ── colours / markers ────────────────────────────────────────────────────────

CASE_COLORS = {
    "easy_2c": "#1f77b4",
    "easy_3c": "#2ca02c",
    "easy_4c": "#ff7f0e",
    "noisy_3c": "#d62728",
    "large_5c": "#9467bd",
}


def _plot(nodes: pd.DataFrame, grid: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Raw Wald vs Adjusted Wald: Structural Relationship",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel A: T_raw vs T_adj (deflation identity) ─────────────────────
    ax = axes[0, 0]
    for case, color in CASE_COLORS.items():
        sub = nodes[nodes["case"] == case]
        ax.scatter(sub["T_wald"], sub["T_adj"], c=color, label=case, s=40, zorder=3)
    lo, hi = 0, nodes["T_wald"].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x (no deflation)")
    ax.set_xlabel("T_raw (raw Wald)")
    ax.set_ylabel("T_adj (adjusted Wald)")
    ax.set_title("A. Deflation: T_adj = T_raw / ĉ_i")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel B: p_raw vs p_deflated with decision boundaries ────────────
    ax = axes[0, 1]
    agree = nodes[nodes["split_wald"] == nodes["split_adj"]]
    flip = nodes[(nodes["split_wald"] == True) & (nodes["split_adj"] == False)]  # noqa: E712
    ax.scatter(
        agree["p_raw"],
        agree["p_deflated"],
        c="steelblue",
        s=30,
        label=f"Agree ({len(agree)})",
        alpha=0.7,
        zorder=3,
    )
    ax.scatter(
        flip["p_raw"],
        flip["p_deflated"],
        c="red",
        marker="x",
        s=60,
        lw=2,
        label=f"Wald-only SPLIT ({len(flip)})",
        zorder=4,
    )
    ax.axhline(0.01, color="orange", ls=":", lw=1, label="α = 0.01")
    ax.axvline(0.01, color="orange", ls=":", lw=1)
    ax.set_xlabel("p_raw")
    ax.set_ylabel("p_deflated (after ĉ_i)")
    ax.set_title("B. p-value shift: raw → deflated")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.legend(fontsize=7)

    # ── Panel C: Per-node ĉ_i distribution ───────────────────────────────
    ax = axes[0, 2]
    c_focal = nodes[nodes["c_hat_i"] > 1.001]["c_hat_i"]
    c_null = nodes[abs(nodes["c_hat_i"] - 1.0) < 0.001]["c_hat_i"]
    bins = np.linspace(0.9, 3.0, 25)
    ax.hist(
        c_null, bins=bins, color="gray", alpha=0.6, label=f"ĉ_i = 1.0 (null-like, n={len(c_null)})"
    )
    ax.hist(
        c_focal, bins=bins, color="steelblue", alpha=0.7, label=f"ĉ_i > 1 (focal, n={len(c_focal)})"
    )
    if len(flip) > 0:
        ax.hist(
            flip["c_hat_i"],
            bins=bins,
            color="red",
            alpha=0.5,
            label=f"ĉ_i at over-splits (n={len(flip)})",
        )
    ax.axvline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("ĉ_i (per-node inflation factor)")
    ax.set_ylabel("Count")
    ax.set_title("C. Distribution of ĉ_i")
    ax.legend(fontsize=7)

    # ── Panel D: Grid search — ARI vs sibling_alpha ──────────────────────
    ax = axes[1, 0]
    for method, style in [
        ("wald", ("red", "--", "o")),
        ("cousin_adjusted_wald", ("steelblue", "-", "s")),
    ]:
        sub = grid[grid["sibling_method"] == method]
        pivot = sub.pivot_table(values="ari", index="sibling_alpha", aggfunc="mean")
        ax.plot(
            pivot.index,
            pivot.values,
            color=style[0],
            ls=style[1],
            marker=style[2],
            ms=5,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("Sibling α")
    ax.set_ylabel("Mean ARI")
    ax.set_title("D. Grid search: ARI vs sibling α")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)

    # ── Panel E: Grid search — Exact-K rate vs sibling_alpha ─────────────
    ax = axes[1, 1]
    for method, style in [
        ("wald", ("red", "--", "o")),
        ("cousin_adjusted_wald", ("steelblue", "-", "s")),
    ]:
        sub = grid[grid["sibling_method"] == method]
        pivot = sub.pivot_table(values="exact_k", index="sibling_alpha", aggfunc="mean")
        ax.plot(
            pivot.index,
            pivot.values,
            color=style[0],
            ls=style[1],
            marker=style[2],
            ms=5,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("Sibling α")
    ax.set_ylabel("Exact-K rate")
    ax.set_title("E. Grid search: Exact-K rate vs sibling α")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # ── Panel F: Mean K found vs sibling_alpha ───────────────────────────
    ax = axes[1, 2]
    for method, style in [
        ("wald", ("red", "--", "o")),
        ("cousin_adjusted_wald", ("steelblue", "-", "s")),
    ]:
        sub = grid[grid["sibling_method"] == method]
        pivot = sub.pivot_table(values="k_found", index="sibling_alpha", aggfunc="mean")
        ax.plot(
            pivot.index,
            pivot.values,
            color=style[0],
            ls=style[1],
            marker=style[2],
            ms=5,
            label=method.replace("_", " "),
        )
    # True K reference band
    true_ks = grid["true_k"].unique()
    ax.axhspan(true_ks.min(), true_ks.max(), color="green", alpha=0.08, label="True K range")
    ax.set_xlabel("Sibling α")
    ax.set_ylabel("Mean K found")
    ax.set_title("F. Grid search: mean K vs sibling α")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    nodes, grid = _load()
    fig = _plot(nodes, grid)
    fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    main()
