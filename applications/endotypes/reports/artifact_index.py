"""Shared artifact-index loading for endotype report generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_artifact_index(experiment_dir: Path) -> pd.DataFrame:
    """Load and normalize an endotype experiment artifact index."""
    path = experiment_dir / "artifact_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    rank_col = (
        "specificity_aware_rank" if "specificity_aware_rank" in frame.columns else "display_rank"
    )
    frame["_rank"] = pd.to_numeric(frame[rank_col], errors="coerce")
    if "run_id" not in frame.columns:
        if "method_run_id" in frame.columns:
            frame["run_id"] = frame["method_run_id"].astype(str)
        else:
            frame["run_id"] = (
                "current__adaptive_diffusion_cosine_subspace__"
                + frame["weighting"].astype(str)
                + "__"
                + frame["block_name"].astype(str)
            )
    return frame.sort_values(["_rank", "weighting", "block_name"], na_position="last").reset_index(
        drop=True
    )
