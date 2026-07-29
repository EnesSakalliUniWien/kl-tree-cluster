"""Shared CLI arguments for endotype tree-analysis pipeline entry points."""

from __future__ import annotations

import argparse


def add_tree_analysis_arguments(
    parser: argparse.ArgumentParser,
    *,
    edge_alpha_default: float | None,
    sibling_alpha_default: float | None,
) -> None:
    """Add the common subspace/tree-analysis knobs to a parser."""

    parser.add_argument("--edge-alpha", type=float, default=edge_alpha_default)
    parser.add_argument("--sibling-alpha", type=float, default=sibling_alpha_default)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--diffusion-k-neighbors", type=int, default=15)
    parser.add_argument("--diffusion-time", type=int, default=3)
    parser.add_argument("--diffusion-components", type=int, default=30)
    parser.add_argument("--adaptive-bandwidth-type", default="-1/(d+2)")
    parser.add_argument("--adaptive-epsilon", default="median")
    parser.add_argument("--adaptive-metric", default="euclidean")
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
    )
    parser.add_argument("--block-names", nargs="*", default=None)
