#!/usr/bin/env python3
"""Render selected-family regions on existing UMAP coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tree_break_selection.plot import load_overlay_data, render_multiscale_umap_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-assignments", type=Path, required=True)
    parser.add_argument("--umap-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-id", default=None)
    parser.add_argument("--data-role", default=None)
    parser.add_argument("--replicate", type=int, default=None)
    parser.add_argument("--top-regions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_data = load_overlay_data(
        gene_assignments_path=args.gene_assignments,
        umap_coordinates_path=args.umap_coordinates,
        method_id=args.method_id,
        data_role=args.data_role,
        replicate=args.replicate,
    )
    table_path = args.output_dir / "multiscale_umap_overlay_data.csv"
    plot_path = args.output_dir / "multiscale_umap_overlay.png"
    overlay_data.to_csv(table_path, index=False)
    render_multiscale_umap_overlay(
        overlay_data,
        plot_path,
        top_regions=int(args.top_regions),
    )
    manifest = {
        "gene_assignments": str(args.gene_assignments),
        "umap_coordinates": str(args.umap_coordinates),
        "method_id": args.method_id,
        "data_role": args.data_role,
        "replicate": args.replicate,
        "top_regions": int(args.top_regions),
        "outputs": {"overlay_data": str(table_path), "plot": str(plot_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
