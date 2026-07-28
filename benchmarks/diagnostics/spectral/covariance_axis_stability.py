"""Stability diagnostic for covariance/PCA axes.

This checks whether a candidate common axis is stable as a covariance object,
not as the cosine/KAK operator. The diagnostic keeps rows aligned, resamples
feature columns, recomputes top PCA sample axes of the centered matrix, then
aligns each replicate to the full-data reference by sign and Procrustes.

It is diagnostic evidence only; it does not promote an invariant-axis claim.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from tree_break_selection.space_separation import weight_feature_matrix

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_matrix_probe import load_matrix

SCHEMA_VERSION = "covariance_axis_stability/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test covariance/PCA axis stability under feature resampling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
    )
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument(
        "--resample-mode",
        choices=["subsample", "bootstrap"],
        default="subsample",
    )
    parser.add_argument("--random-state", type=int, default=1729)
    parser.add_argument("--stability-threshold", type=float, default=0.9)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return Path("benchmarks/results/diagnostics") / f"covariance_axis_stability_{stamp}"


def centered_values(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D data matrix.")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError("Centered matrix contains non-finite values.")
    if np.allclose(centered, 0.0):
        raise ValueError("Centered matrix is degenerate.")
    return centered


def top_sample_covariance_axes(
    values: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    centered = centered_values(values)
    max_components = min(centered.shape) - 1
    if max_components < 1:
        raise ValueError("At least two rows and columns are required for PCA axes.")
    k = min(int(n_components), max_components)
    u, singular_values, vt = randomized_svd(
        centered,
        n_components=k,
        n_iter=7,
        random_state=random_state,
    )
    total_ss = float(np.sum(centered * centered))
    explained_ratio = (
        (singular_values * singular_values) / total_ss
        if total_ss > 0
        else np.full_like(singular_values, np.nan, dtype=float)
    )
    return {
        "sample_axes": u,
        "sample_scores": u * singular_values[np.newaxis, :],
        "singular_values": singular_values,
        "feature_axes": vt,
        "explained_ratio": explained_ratio,
    }


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(left, right) / denom)


def sign_aligned_axis_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    raw_cosine = vector_cosine(reference, candidate)
    sign = -1.0 if math.isfinite(raw_cosine) and raw_cosine < 0 else 1.0
    aligned = candidate * sign
    return {
        "axis1_signed_cosine_before_alignment": raw_cosine,
        "axis1_alignment_sign": sign,
        "axis1_cosine_after_alignment": vector_cosine(reference, aligned),
        "axis1_abs_cosine": abs(raw_cosine) if math.isfinite(raw_cosine) else math.nan,
    }


def procrustes_subspace_metrics(
    reference_axes: np.ndarray, candidate_axes: np.ndarray
) -> dict[str, float]:
    k = min(reference_axes.shape[1], candidate_axes.shape[1])
    reference = np.asarray(reference_axes[:, :k], dtype=float)
    candidate = np.asarray(candidate_axes[:, :k], dtype=float)
    u, singular_values, vt = np.linalg.svd(candidate.T @ reference, full_matrices=False)
    rotation = u @ vt
    aligned = candidate @ rotation
    residual = float(np.linalg.norm(aligned - reference, ord="fro") / math.sqrt(k))
    return {
        "subspace_components": int(k),
        "subspace_min_canonical_corr": float(np.min(singular_values)),
        "subspace_mean_canonical_corr": float(np.mean(singular_values)),
        "subspace_max_canonical_corr": float(np.max(singular_values)),
        "procrustes_residual": residual,
    }


def resampled_feature_indices(
    *,
    n_features: int,
    feature_fraction: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0.0 < feature_fraction <= 1.0:
        raise ValueError("feature_fraction must be in (0, 1].")
    n_selected = max(2, int(round(n_features * feature_fraction)))
    if mode == "subsample":
        n_selected = min(n_selected, n_features)
        return np.sort(rng.choice(n_features, size=n_selected, replace=False))
    if mode == "bootstrap":
        return rng.choice(n_features, size=n_selected, replace=True)
    raise ValueError(f"Unknown resample mode: {mode!r}")


def run_weighting_stability(
    data: pd.DataFrame,
    *,
    weighting: str,
    n_components: int,
    replicates: int,
    feature_fraction: float,
    resample_mode: str,
    random_state: int,
    stability_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    values = weight_feature_matrix(data, weighting)
    reference = top_sample_covariance_axes(
        values,
        n_components=n_components,
        random_state=random_state,
    )
    reference_axis = reference["sample_scores"][:, 0]
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, object]] = []
    for replicate in range(int(replicates)):
        feature_idx = resampled_feature_indices(
            n_features=values.shape[1],
            feature_fraction=feature_fraction,
            mode=resample_mode,
            rng=rng,
        )
        replicate_axes = top_sample_covariance_axes(
            values[:, feature_idx],
            n_components=n_components,
            random_state=random_state + replicate + 1,
        )
        axis_metrics = sign_aligned_axis_metrics(
            reference_axis,
            replicate_axes["sample_scores"][:, 0],
        )
        subspace_metrics = procrustes_subspace_metrics(
            reference["sample_axes"],
            replicate_axes["sample_axes"],
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "axis_source": "centered_dual_covariance_sample_pca",
                "weighting": weighting,
                "replicate": int(replicate),
                "n_samples": int(values.shape[0]),
                "n_features": int(values.shape[1]),
                "n_resampled_features": int(len(feature_idx)),
                "feature_fraction": float(feature_fraction),
                "resample_mode": resample_mode,
                "reference_axis1_explained_ratio": float(reference["explained_ratio"][0]),
                "replicate_axis1_explained_ratio": float(replicate_axes["explained_ratio"][0]),
                **axis_metrics,
                **subspace_metrics,
            }
        )

    replicate_table = pd.DataFrame.from_records(rows)
    axis_abs = replicate_table["axis1_abs_cosine"].to_numpy(dtype=float)
    subspace_mean = replicate_table["subspace_mean_canonical_corr"].to_numpy(dtype=float)
    residual = replicate_table["procrustes_residual"].to_numpy(dtype=float)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "axis_source": "centered_dual_covariance_sample_pca",
        "weighting": weighting,
        "n_samples": int(values.shape[0]),
        "n_features": int(values.shape[1]),
        "n_components": int(reference["sample_axes"].shape[1]),
        "replicates": int(replicates),
        "feature_fraction": float(feature_fraction),
        "resample_mode": resample_mode,
        "stability_threshold": float(stability_threshold),
        "axis1_abs_cosine_q10": float(np.quantile(axis_abs, 0.10)),
        "axis1_abs_cosine_q50": float(np.quantile(axis_abs, 0.50)),
        "axis1_abs_cosine_q90": float(np.quantile(axis_abs, 0.90)),
        "axis1_stable_fraction": float(np.mean(axis_abs >= stability_threshold)),
        "subspace_mean_canonical_corr_q50": float(np.quantile(subspace_mean, 0.50)),
        "procrustes_residual_q50": float(np.quantile(residual, 0.50)),
        "reference_axis1_explained_ratio": float(reference["explained_ratio"][0]),
    }
    return replicate_table, summary


def write_plots(replicates: pd.DataFrame, output_dir: Path) -> None:
    if replicates.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for weighting, group in replicates.groupby("weighting"):
        axes[0].hist(
            group["axis1_abs_cosine"],
            bins=20,
            alpha=0.55,
            label=str(weighting),
        )
        axes[1].hist(
            group["procrustes_residual"],
            bins=20,
            alpha=0.55,
            label=str(weighting),
        )
    axes[0].set_title("Sign-aligned axis-1 stability")
    axes[0].set_xlabel("abs cosine to full-data covariance axis")
    axes[0].set_ylabel("replicates")
    axes[1].set_title("Top-subspace Procrustes residual")
    axes[1].set_xlabel("Frobenius residual / sqrt(k)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_axis_stability_histograms.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Covariance Axis Stability",
        "",
        "This diagnostic tests the centered dual covariance/PCA sample axis, not the cosine/KAK common axis.",
        "Rows remain aligned; feature columns are resampled, covariance sample axes are recomputed, then sign and Procrustes aligned to the full-data reference.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- High `axis1_abs_cosine_q50` means the first covariance axis is stable under feature resampling.",
        "- High `subspace_mean_canonical_corr_q50` and low `procrustes_residual_q50` mean the selected top-k covariance subspace is stable after rotation.",
        "- This is still diagnostic evidence. A formal invariant-axis claim requires a specified symmetry group and selected-subspace calibration law.",
        "",
    ]
    (output_dir / "covariance_axis_stability_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(args.input)

    replicate_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for offset, weighting in enumerate(args.weightings):
        table, summary = run_weighting_stability(
            data,
            weighting=weighting,
            n_components=args.n_components,
            replicates=args.replicates,
            feature_fraction=args.feature_fraction,
            resample_mode=args.resample_mode,
            random_state=args.random_state + 1000 * offset,
            stability_threshold=args.stability_threshold,
        )
        replicate_tables.append(table)
        summary_rows.append(
            {
                **summary,
                "input_path": str(args.input),
            }
        )

    replicates = pd.concat(replicate_tables, ignore_index=True)
    summary = pd.DataFrame.from_records(summary_rows)
    replicates.to_csv(output_dir / "covariance_axis_stability_replicates.csv", index=False)
    summary.to_csv(output_dir / "covariance_axis_stability_summary.csv", index=False)
    write_plots(replicates, output_dir)
    write_report(summary, output_dir)
    print(f"Wrote covariance axis stability diagnostic: {output_dir}")


if __name__ == "__main__":
    main()
