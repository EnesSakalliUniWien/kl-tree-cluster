"""Adaptive cosine spectral-space separation.

This module owns weighting, cosine-operator eigendecomposition, adaptive
log-spectrum segmentation, and per-block coordinate extraction. Applications
and diagnostics can therefore use the method without importing benchmark code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectralBlock:
    """One contiguous one-based block of ordered cosine spectral modes."""

    block_id: int
    block_name: str
    block_start: int
    block_end: int
    block_type: str


@dataclass(frozen=True)
class AdaptiveCosineSpace:
    """Adaptive cosine eigensystem and its selected spectral blocks."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    blocks: tuple[SpectralBlock, ...]
    diagnostics: dict[str, object]
    weighting: str

    def coordinates(self, block: SpectralBlock) -> np.ndarray:
        """Return sample coordinates for one selected block."""

        return coordinates_for_block(self.eigenvalues, self.eigenvectors, block)


def weight_feature_matrix(data: pd.DataFrame | np.ndarray, weighting: str) -> np.ndarray:
    """Return raw or TF-IDF-weighted feature values for spectral separation."""

    values = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if weighting == "binary":
        return values
    if weighting == "tfidf":
        if np.any(values < 0):
            raise ValueError("tfidf weighting requires nonnegative feature values.")
        from sklearn.feature_extraction.text import TfidfTransformer

        return TfidfTransformer(norm=None, use_idf=True, smooth_idf=True).fit_transform(
            values
        ).toarray()
    raise ValueError(f"Unknown weighting: {weighting!r}")


def cosine_eigendecomposition(
    values: np.ndarray,
    max_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose the sample cosine operator in descending order."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if max_rank < 1:
        raise ValueError("max_rank must be positive.")
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms <= 1e-12):
        raise ValueError("Rows with zero norm cannot enter the cosine operator.")

    row_normalized = matrix / row_norms[:, None]
    operator = row_normalized @ row_normalized.T
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = eigenvalues > 1e-10
    return eigenvalues[keep][:max_rank], eigenvectors[:, keep][:, :max_rank]


def _interval_linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    design = np.vstack([x, np.ones_like(x)]).T
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficient
    return float(np.sum(residual**2))


def _should_isolate_common_mode(log_eigenvalues: np.ndarray) -> bool:
    if len(log_eigenvalues) < 8:
        return False
    gaps = log_eigenvalues[:-1] - log_eigenvalues[1:]
    if len(gaps) < 4:
        return False
    first_gap = gaps[0]
    tail = gaps[1:]
    return bool(first_gap > np.median(tail) + 2.0 * np.std(tail))


def adaptive_spectral_blocks(
    eigenvalues: np.ndarray,
    *,
    min_segment_length: int,
    max_segments: int,
) -> tuple[list[SpectralBlock], dict[str, object]]:
    """Segment a descending positive eigenspectrum into adaptive decay regimes."""

    if min_segment_length < 1:
        raise ValueError("min_segment_length must be positive.")
    if max_segments < 1:
        raise ValueError("max_segments must be positive.")
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    rank = len(eigenvalues)
    if rank == 0:
        return [], {"selected_segments": 0, "bic": math.nan, "segmentation_sse": math.nan}
    if rank <= min_segment_length:
        return [
            SpectralBlock(0, f"adaptive_modes_01_{rank:02d}", 1, rank, "all_available")
        ], {"selected_segments": 1, "bic": math.nan, "segmentation_sse": 0.0}

    log_eigenvalues = np.log(np.maximum(eigenvalues, 1e-300))
    offset = 0
    blocks: list[SpectralBlock] = []
    if _should_isolate_common_mode(log_eigenvalues):
        blocks.append(SpectralBlock(0, "adaptive_common_mode_01", 1, 1, "common_mode"))
        offset = 1

    y = log_eigenvalues[offset:]
    n_values = len(y)
    x = np.arange(offset + 1, rank + 1, dtype=float)
    if n_values < min_segment_length:
        if n_values:
            blocks.append(
                SpectralBlock(
                    len(blocks),
                    f"adaptive_modes_{offset + 1:02d}_{rank:02d}",
                    offset + 1,
                    rank,
                    "tail",
                )
            )
        return blocks, {
            "selected_segments": len(blocks),
            "bic": math.nan,
            "segmentation_sse": 0.0,
            "common_mode_isolated": bool(offset == 1),
        }

    max_block_count = min(max_segments, max(1, n_values // min_segment_length))
    sse = np.full((n_values, n_values), np.inf)
    for start in range(n_values):
        for end in range(start + min_segment_length - 1, n_values):
            sse[start, end] = _interval_linear_sse(x[start : end + 1], y[start : end + 1])

    dynamic = np.full((max_block_count + 1, n_values), np.inf)
    previous = np.full((max_block_count + 1, n_values), -1, dtype=int)
    for end in range(min_segment_length - 1, n_values):
        dynamic[1, end] = sse[0, end]
    for block_count in range(2, max_block_count + 1):
        first_valid_end = block_count * min_segment_length - 1
        for end in range(first_valid_end, n_values):
            for cut in range(
                (block_count - 1) * min_segment_length - 1,
                end - min_segment_length + 1,
            ):
                value = dynamic[block_count - 1, cut] + sse[cut + 1, end]
                if value < dynamic[block_count, end]:
                    dynamic[block_count, end] = value
                    previous[block_count, end] = cut

    selected: tuple[float, int, float] | None = None
    for block_count in range(1, max_block_count + 1):
        total_sse = float(dynamic[block_count, n_values - 1])
        if not np.isfinite(total_sse):
            continue
        total_sse = max(total_sse, 1e-12)
        parameter_count = 3 * block_count
        bic = n_values * math.log(total_sse / n_values) + parameter_count * math.log(n_values)
        if selected is None or bic < selected[0]:
            selected = (bic, block_count, total_sse)
    if selected is None:
        raise RuntimeError("Could not select adaptive spectral segmentation.")

    _, block_count, total_sse = selected
    segments: list[tuple[int, int]] = []
    end = n_values - 1
    while block_count >= 1:
        cut = previous[block_count, end]
        start = 0 if block_count == 1 else cut + 1
        segments.append((start + offset + 1, end + offset + 1))
        end = cut
        block_count -= 1

    for start, end in reversed(segments):
        blocks.append(
            SpectralBlock(
                len(blocks),
                f"adaptive_modes_{start:02d}_{end:02d}",
                int(start),
                int(end),
                "adaptive_decay_regime",
            )
        )

    return blocks, {
        "selected_segments": len(segments),
        "bic": float(selected[0]),
        "segmentation_sse": float(total_sse),
        "common_mode_isolated": bool(offset == 1),
        "min_segment_length": int(min_segment_length),
        "max_segments": int(max_segments),
    }


def coordinates_for_block(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    block: SpectralBlock,
) -> np.ndarray:
    """Return sample coordinates for a one-based spectral block."""

    start = block.block_start - 1
    end = block.block_end
    if start < 0 or end < block.block_start or end > len(eigenvalues):
        raise ValueError(f"Spectral block is outside the eigensystem: {block!r}")
    coordinates = eigenvectors[:, start:end] * np.sqrt(
        np.maximum(eigenvalues[start:end], 0.0)
    )
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)
    return np.nan_to_num(coordinates)


def separate_adaptive_cosine_space(
    data: pd.DataFrame | np.ndarray,
    *,
    weighting: str,
    max_rank: int,
    min_segment_length: int,
    max_segments: int,
) -> AdaptiveCosineSpace:
    """Run the complete adaptive cosine space-separation method."""

    values = weight_feature_matrix(data, weighting)
    eigenvalues, eigenvectors = cosine_eigendecomposition(values, max_rank=max_rank)
    blocks, diagnostics = adaptive_spectral_blocks(
        eigenvalues,
        min_segment_length=min_segment_length,
        max_segments=max_segments,
    )
    return AdaptiveCosineSpace(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        blocks=tuple(blocks),
        diagnostics=diagnostics,
        weighting=weighting,
    )
