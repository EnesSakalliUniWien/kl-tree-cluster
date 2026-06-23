#!/usr/bin/env python3
"""Diagnostic for the local identity Marchenko-Pastur spectral law.

This script tests the evidence behind the current production MP edge:

    lambda_+ = (1 + sqrt(d / m))^2

That edge is the identity-population MP law after null whitening. The
diagnostic materializes the same node-local null-whitened tangent matrices used
by the production spectral worker, compares their positive spectra with the
conditional identity-MP law, and writes descriptive summaries. It does not add
bootstrap thresholding, production fallback behavior, or a deformed-MP method.

For continuous empirical-Gaussian blocks, the diagnostic also reports the
finite-rank self-whitening reference eigenvalue (m - 1) / m. That is not an MP
edge. It tests whether node-local continuous spectra are dominated by whitening
with the same node's empirical covariance rather than by a fresh identity-MP
sample-covariance law.

For Bernoulli and categorical selected spectra, the diagnostic also reports a
finite-sample identity-null top-eigenvalue quantile with the same row count and
active feature count. That simulated Gaussian Wishart reference is diagnostic
only: it separates ordinary finite-size MP edge fluctuation from selected-tree
spectral inflation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.decomposition import (
    eigendecompose_covariance,
)

from benchmarks.diagnostics.spectral.profile_spectral_backends import (
    SpectralMatrixRecord,
    _build_matrix_records,
)
from benchmarks.shared.cases import get_default_test_cases

SCHEMA_VERSION = "local_mp_identity_law_diagnostic/v1"
GENERATED_BY = "benchmarks.diagnostics.spectral.local_mp_identity_law_diagnostic"
DIAGNOSTIC_ROLE = "descriptive_identity_mp_law_screen_not_production_threshold"
DEFAULT_CASE_NAMES = (
    "gauss_null_large",
    "binary_many_features",
    "cat_highcard_20cat_4c",
    "cat_highd_3cat_500feat",
    "dim_diffuse_6c_136f",
    "gauss_null_large_continuous",
    "dim_diffuse_6c_136f_continuous",
)
DEFAULT_QUANTILES = (0.5, 0.9, 0.95)
DEFAULT_FINITE_IDENTITY_NULL_REPS = 80
DEFAULT_FINITE_IDENTITY_NULL_QUANTILE = 0.95
DEFAULT_FINITE_IDENTITY_NULL_SEED = 20260603
SELECTED_TREE_SPECTRAL_TARGET = "log_top_eigenvalue_over_finite_identity_null_quantile"
SELECTED_TREE_SPECTRAL_COVARIATES = (
    "log_node_size_fraction",
    "log_aspect_ratio",
    "log_matrix_rows",
    "log_active_feature_count",
)


@dataclass(frozen=True)
class IdentityMpSupport:
    """Support of the identity-population MP law for a positive aspect ratio."""

    aspect_ratio: float
    lower_edge: float
    upper_edge: float
    positive_mass: float


@dataclass(frozen=True)
class IdentityMpGrid:
    """Numerical positive-spectrum CDF for the identity MP law."""

    support: IdentityMpSupport
    grid: np.ndarray
    cdf: np.ndarray


def identity_mp_support(aspect_ratio: float) -> IdentityMpSupport:
    """Return the positive support edges for the identity MP law."""
    c = float(aspect_ratio)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"aspect_ratio must be positive and finite; got {aspect_ratio!r}.")
    root_c = float(np.sqrt(c))
    lower_edge = float((1.0 - root_c) ** 2)
    upper_edge = float((1.0 + root_c) ** 2)
    positive_mass = 1.0 if c <= 1.0 else float(1.0 / c)
    return IdentityMpSupport(
        aspect_ratio=c,
        lower_edge=lower_edge,
        upper_edge=upper_edge,
        positive_mass=positive_mass,
    )


def identity_mp_positive_grid(
    aspect_ratio: float,
    *,
    grid_size: int = 8192,
) -> IdentityMpGrid:
    """Build a conditional positive-spectrum CDF for the identity MP law."""
    if grid_size < 128:
        raise ValueError(f"grid_size must be at least 128; got {grid_size!r}.")
    support = identity_mp_support(aspect_ratio)
    lower = support.lower_edge
    upper = support.upper_edge
    if upper <= lower:
        raise ValueError(
            "Identity MP support requires upper_edge > lower_edge; "
            f"got lower={lower}, upper={upper}."
        )

    start = lower
    if start <= 0.0:
        start = upper / float(grid_size * grid_size)
    grid = np.linspace(start, upper, int(grid_size), dtype=np.float64)
    density = _identity_mp_density(grid, support)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "Identity MP positive CDF has non-positive integral; "
            f"aspect_ratio={aspect_ratio!r}."
        )
    cdf = cdf / total
    cdf[-1] = 1.0
    return IdentityMpGrid(support=support, grid=grid, cdf=cdf)


def identity_mp_positive_quantiles(
    aspect_ratio: float,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    *,
    grid_size: int = 8192,
) -> dict[float, float]:
    """Return conditional positive-spectrum identity-MP quantiles."""
    for q in quantiles:
        if not np.isfinite(float(q)) or not 0.0 < float(q) < 1.0:
            raise ValueError(f"MP quantiles must lie in (0, 1); got {q!r}.")
    mp_grid = identity_mp_positive_grid(aspect_ratio, grid_size=grid_size)
    return {
        float(q): float(np.interp(float(q), mp_grid.cdf, mp_grid.grid))
        for q in quantiles
    }


def centered_self_whitening_reference_eigenvalue(matrix_rows: int) -> float:
    """Return the finite-rank eigenvalue after whitening by the same covariance."""
    row_count = int(matrix_rows)
    if row_count < 2:
        raise ValueError(
            "Centered self-whitening reference requires at least two rows; "
            f"got {matrix_rows!r}."
        )
    return float(row_count - 1) / float(row_count)


@lru_cache(maxsize=None)
def finite_identity_null_top_eigenvalue_quantile(
    matrix_rows: int,
    active_feature_count: int,
    reps: int,
    quantile: float,
    seed: int,
) -> float:
    """Return a finite-sample top-eigenvalue quantile under identity null."""
    row_count = int(matrix_rows)
    feature_count = int(active_feature_count)
    replicate_count = int(reps)
    q = float(quantile)
    if row_count < 2:
        raise ValueError(
            "Finite identity-null top edge requires at least two rows; "
            f"got {matrix_rows!r}."
        )
    if feature_count < 1:
        raise ValueError(
            "Finite identity-null top edge requires at least one active feature; "
            f"got {active_feature_count!r}."
        )
    if replicate_count < 1:
        raise ValueError(f"reps must be positive; got {reps!r}.")
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile!r}.")

    rng = np.random.default_rng(
        int(seed) + 1_000_003 * row_count + 9_176 * feature_count + 37 * replicate_count
    )
    top_eigenvalues = np.empty(replicate_count, dtype=np.float64)
    for rep_index in range(replicate_count):
        top_eigenvalues[rep_index] = _identity_null_top_eigenvalue(
            row_count,
            feature_count,
            rng,
        )
    return float(np.quantile(top_eigenvalues, q))


def summarize_node_spectrum(
    record: SpectralMatrixRecord,
    *,
    min_positive_eigenvalues: int = 4,
    grid_size: int = 8192,
    finite_null_reps: int = DEFAULT_FINITE_IDENTITY_NULL_REPS,
    finite_null_quantile: float = DEFAULT_FINITE_IDENTITY_NULL_QUANTILE,
    finite_null_seed: int = DEFAULT_FINITE_IDENTITY_NULL_SEED,
) -> dict[str, object]:
    """Summarize one production node matrix against the identity MP law."""
    if min_positive_eigenvalues < 1:
        raise ValueError(
            "min_positive_eigenvalues must be positive; "
            f"got {min_positive_eigenvalues!r}."
        )
    matrix = np.asarray(record.matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Node matrix for {record.node_id!r} must be 2-D; got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Node matrix for {record.node_id!r} contains non-finite values.")

    base_row = {
        "node_id": record.node_id,
        "matrix_rows": int(matrix.shape[0]),
        "feature_count": int(record.feature_count),
        "descendant_leaf_rows": int(record.descendant_leaf_rows),
        "active_feature_count": 0,
        "aspect_ratio": np.nan,
        "mp_lower_edge": np.nan,
        "mp_upper_edge": np.nan,
        "positive_eigenvalue_count": 0,
        "raw_mp_signal_count": 0,
        "top_eigenvalue": np.nan,
        "top_eigenvalue_over_mp_upper": np.nan,
        "finite_identity_null_top_eigenvalue_quantile": np.nan,
        "finite_identity_null_quantile": float(finite_null_quantile),
        "finite_identity_null_reps": int(finite_null_reps),
        "finite_identity_null_top_over_mp_upper": np.nan,
        "top_eigenvalue_over_finite_identity_null_quantile": np.nan,
        "above_finite_identity_null_top_quantile": False,
        "median_eigenvalue_over_identity_mp_median": np.nan,
        "q90_eigenvalue_over_identity_mp_q90": np.nan,
        "q95_eigenvalue_over_identity_mp_q95": np.nan,
        "identity_mp_positive_ks_distance": np.nan,
        "centered_self_whitening_reference_eigenvalue": np.nan,
        "median_eigenvalue_over_centered_self_whitening_reference": np.nan,
        "top_eigenvalue_over_centered_self_whitening_reference": np.nan,
        "positive_rank_fraction": np.nan,
        "diagnostic_status": "not_evaluated",
    }
    if matrix.shape[0] < 2:
        return {**base_row, "diagnostic_status": "insufficient_rows"}

    eig = eigendecompose_covariance(matrix, compute_eigenvectors=False)
    if eig is None or eig.active_feature_count <= 0:
        return {**base_row, "diagnostic_status": "no_active_features"}
    eigenvalues = np.asarray(eig.eigenvalues, dtype=np.float64)
    if eigenvalues.ndim != 1:
        raise ValueError(
            f"Eigenvalues for {record.node_id!r} must be 1-D; got {eigenvalues.shape}."
        )
    if not np.isfinite(eigenvalues).all():
        raise ValueError(f"Eigenvalues for {record.node_id!r} contain non-finite values.")

    tolerance = (
        np.finfo(np.float64).eps
        * max(eigenvalues.shape[0], 1)
        * max(float(np.max(eigenvalues)) if eigenvalues.size else 0.0, 1.0)
    )
    positive_eigenvalues = np.sort(eigenvalues[eigenvalues > tolerance])
    aspect_ratio = float(eig.active_feature_count) / float(matrix.shape[0])
    support = identity_mp_support(aspect_ratio)
    self_whitening_reference = centered_self_whitening_reference_eigenvalue(
        matrix.shape[0]
    )
    finite_null_top_quantile = finite_identity_null_top_eigenvalue_quantile(
        matrix.shape[0],
        int(eig.active_feature_count),
        int(finite_null_reps),
        float(finite_null_quantile),
        int(finite_null_seed),
    )
    raw_signal_count = int(np.count_nonzero(eigenvalues > support.upper_edge))
    top_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else np.nan
    centered_rank = min(int(eig.active_feature_count), max(int(matrix.shape[0]) - 1, 0))
    row = {
        **base_row,
        "active_feature_count": int(eig.active_feature_count),
        "aspect_ratio": aspect_ratio,
        "mp_lower_edge": support.lower_edge,
        "mp_upper_edge": support.upper_edge,
        "positive_eigenvalue_count": int(positive_eigenvalues.size),
        "raw_mp_signal_count": raw_signal_count,
        "top_eigenvalue": top_eigenvalue,
        "top_eigenvalue_over_mp_upper": (
            float(top_eigenvalue / support.upper_edge)
            if support.upper_edge > 0.0 and np.isfinite(top_eigenvalue)
            else np.nan
        ),
        "finite_identity_null_top_eigenvalue_quantile": finite_null_top_quantile,
        "finite_identity_null_quantile": float(finite_null_quantile),
        "finite_identity_null_reps": int(finite_null_reps),
        "finite_identity_null_top_over_mp_upper": _ratio(
            finite_null_top_quantile,
            support.upper_edge,
        ),
        "top_eigenvalue_over_finite_identity_null_quantile": _ratio(
            top_eigenvalue,
            finite_null_top_quantile,
        ),
        "above_finite_identity_null_top_quantile": bool(
            np.isfinite(top_eigenvalue)
            and np.isfinite(finite_null_top_quantile)
            and top_eigenvalue > finite_null_top_quantile
        ),
        "centered_self_whitening_reference_eigenvalue": self_whitening_reference,
        "top_eigenvalue_over_centered_self_whitening_reference": _ratio(
            top_eigenvalue,
            self_whitening_reference,
        ),
        "positive_rank_fraction": _ratio(
            float(positive_eigenvalues.size),
            float(centered_rank),
        ),
    }
    if positive_eigenvalues.size < min_positive_eigenvalues:
        return {**row, "diagnostic_status": "insufficient_positive_spectrum"}

    theoretical_quantiles = identity_mp_positive_quantiles(
        aspect_ratio,
        DEFAULT_QUANTILES,
        grid_size=grid_size,
    )
    empirical_quantiles = {
        q: float(np.quantile(positive_eigenvalues, q)) for q in DEFAULT_QUANTILES
    }
    mp_grid = identity_mp_positive_grid(aspect_ratio, grid_size=grid_size)
    ks_distance = _positive_spectrum_ks_distance(positive_eigenvalues, mp_grid)
    return {
        **row,
        "median_eigenvalue_over_identity_mp_median": _ratio(
            empirical_quantiles[0.5], theoretical_quantiles[0.5]
        ),
        "q90_eigenvalue_over_identity_mp_q90": _ratio(
            empirical_quantiles[0.9], theoretical_quantiles[0.9]
        ),
        "q95_eigenvalue_over_identity_mp_q95": _ratio(
            empirical_quantiles[0.95], theoretical_quantiles[0.95]
        ),
        "identity_mp_positive_ks_distance": ks_distance,
        "median_eigenvalue_over_centered_self_whitening_reference": _ratio(
            empirical_quantiles[0.5],
            self_whitening_reference,
        ),
        "diagnostic_status": "evaluated",
    }


def run_local_mp_identity_law_diagnostic(
    *,
    case_names: Sequence[str],
    max_cases: int | None,
    max_nodes: int | None,
    min_positive_eigenvalues: int,
    grid_size: int,
    finite_null_reps: int,
    finite_null_quantile: float,
    finite_null_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the diagnostic for selected benchmark cases."""
    cases = _resolve_cases(case_names, max_cases)
    node_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for case in cases:
        records = _records_for_case(case, max_nodes=max_nodes)
        max_matrix_rows = max((record.matrix.shape[0] for record in records), default=0)
        source_spectral_family = _source_spectral_family(case)
        for record in records:
            row = summarize_node_spectrum(
                record,
                min_positive_eigenvalues=min_positive_eigenvalues,
                grid_size=grid_size,
                finite_null_reps=finite_null_reps,
                finite_null_quantile=finite_null_quantile,
                finite_null_seed=finite_null_seed,
            )
            node_rows.append(
                {
                    "case_name": str(case["name"]),
                    "case_category": str(case.get("category", "")),
                    "generator": str(case.get("generator", "")),
                    "source_spectral_family": source_spectral_family,
                    "node_size_fraction": _ratio(
                        float(record.matrix.shape[0]),
                        float(max_matrix_rows),
                    ),
                    **row,
                }
            )
        case_rows.append(_summarize_case(case, node_rows_for_case=node_rows, records=records))
    node_spectrum = _with_selected_tree_spectral_covariates(pd.DataFrame(node_rows))
    return pd.DataFrame(case_rows), node_spectrum


def write_outputs(
    output_dir: Path,
    *,
    case_summary: pd.DataFrame,
    node_spectrum: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write diagnostic CSVs and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_law_relationships = summarize_spectral_law_relationships(node_spectrum)
    categorical_extreme_nodes = summarize_categorical_extreme_nodes(node_spectrum)
    case_summary.to_csv(output_dir / "case_summary.csv", index=False)
    node_spectrum.to_csv(output_dir / "node_spectrum.csv", index=False)
    spectral_law_relationships.to_csv(
        output_dir / "spectral_law_relationships.csv",
        index=False,
    )
    categorical_extreme_nodes.to_csv(
        output_dir / "categorical_extreme_nodes.csv",
        index=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "created_utc": _utc_now(),
        "code_commit": _git_commit(),
        "git_worktree_status": _git_status_short(),
        "run_command": " ".join(sys.argv),
        "case_names": list(_parse_case_names(args.case_names)),
        "max_cases": args.max_cases,
        "max_nodes": args.max_nodes,
        "min_positive_eigenvalues": args.min_positive_eigenvalues,
        "grid_size": args.grid_size,
        "finite_identity_null_reps": args.finite_null_reps,
        "finite_identity_null_quantile": args.finite_null_quantile,
        "finite_identity_null_seed": args.finite_null_seed,
        "outputs": {
            "case_summary": "case_summary.csv",
            "node_spectrum": "node_spectrum.csv",
            "spectral_law_relationships": "spectral_law_relationships.csv",
            "categorical_extreme_nodes": "categorical_extreme_nodes.csv",
        },
        "limitations": [
            "Descriptive screen on production selected node spectra.",
            "Does not estimate a deformed-MP edge.",
            "Does not validate selected-hierarchy sibling calibration.",
            "Does not add bootstrap or resampling threshold behavior.",
            "The centered self-whitening columns are diagnostic explanations, not thresholds.",
            "The finite identity-null top-edge columns are diagnostic simulations, not thresholds.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_spectral_law_relationships(node_spectrum: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-tree spectral target relationships by source family."""
    required_columns = {
        "case_name",
        "source_spectral_family",
        SELECTED_TREE_SPECTRAL_TARGET,
        *SELECTED_TREE_SPECTRAL_COVARIATES,
    }
    missing = sorted(required_columns - set(node_spectrum.columns))
    if missing:
        raise KeyError(f"node_spectrum is missing required columns: {missing!r}.")

    evaluated = node_spectrum[
        node_spectrum["diagnostic_status"].eq("evaluated")
        & np.isfinite(node_spectrum[SELECTED_TREE_SPECTRAL_TARGET].to_numpy(dtype=float))
    ].copy()
    rows: list[dict[str, object]] = []
    group_keys = ["source_spectral_family", "case_name"]
    for group_values, group in evaluated.groupby(group_keys, dropna=False):
        source_family, case_name = group_values
        rows.extend(
            _relationship_rows_for_group(
                source_spectral_family=str(source_family),
                case_name=str(case_name),
                group=group,
            )
        )
    for source_family, group in evaluated.groupby("source_spectral_family", dropna=False):
        rows.extend(
            _relationship_rows_for_group(
                source_spectral_family=str(source_family),
                case_name="__all__",
                group=group,
            )
        )
    return pd.DataFrame(rows)


def summarize_categorical_extreme_nodes(node_spectrum: pd.DataFrame) -> pd.DataFrame:
    """Return categorical nodes above the finite identity-null top edge."""
    required_columns = {
        "source_spectral_family",
        "above_finite_identity_null_top_quantile",
        "top_eigenvalue_over_finite_identity_null_quantile",
    }
    missing = sorted(required_columns - set(node_spectrum.columns))
    if missing:
        raise KeyError(f"node_spectrum is missing required columns: {missing!r}.")
    columns = [
        "case_name",
        "node_id",
        "matrix_rows",
        "node_size_fraction",
        "active_feature_count",
        "aspect_ratio",
        "raw_mp_signal_count",
        "top_eigenvalue_over_mp_upper",
        "top_eigenvalue_over_finite_identity_null_quantile",
        "finite_identity_null_top_over_mp_upper",
    ]
    extreme_nodes = node_spectrum[
        node_spectrum["source_spectral_family"].eq("categorical")
        & node_spectrum["above_finite_identity_null_top_quantile"].astype(bool)
    ].copy()
    if extreme_nodes.empty:
        return pd.DataFrame(columns=columns)
    return extreme_nodes.sort_values(
        "top_eigenvalue_over_finite_identity_null_quantile",
        ascending=False,
    )[columns]


def _with_selected_tree_spectral_covariates(
    node_spectrum: pd.DataFrame,
) -> pd.DataFrame:
    """Attach explicit selected-tree spectral-law coordinates."""
    result = node_spectrum.copy()
    required_columns = {
        "top_eigenvalue_over_finite_identity_null_quantile",
        "node_size_fraction",
        "aspect_ratio",
        "matrix_rows",
        "active_feature_count",
    }
    missing = sorted(required_columns - set(result.columns))
    if missing:
        raise KeyError(f"node_spectrum is missing required columns: {missing!r}.")

    _positive_log_column(
        result,
        source_column="top_eigenvalue_over_finite_identity_null_quantile",
        output_column=SELECTED_TREE_SPECTRAL_TARGET,
    )
    _positive_log_column(
        result,
        source_column="node_size_fraction",
        output_column="log_node_size_fraction",
    )
    _positive_log_column(
        result,
        source_column="aspect_ratio",
        output_column="log_aspect_ratio",
    )
    _positive_log_column(
        result,
        source_column="matrix_rows",
        output_column="log_matrix_rows",
    )
    _positive_log_column(
        result,
        source_column="active_feature_count",
        output_column="log_active_feature_count",
    )
    result["selected_spectral_excess_over_finite_identity_null"] = (
        result["top_eigenvalue_over_finite_identity_null_quantile"].astype(float) - 1.0
    )
    return result


def _relationship_rows_for_group(
    *,
    source_spectral_family: str,
    case_name: str,
    group: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = group[SELECTED_TREE_SPECTRAL_TARGET].to_numpy(dtype=np.float64)
    for covariate in SELECTED_TREE_SPECTRAL_COVARIATES:
        x = group[covariate].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(target)
        n = int(np.count_nonzero(finite))
        relationship_status = "evaluated"
        pearson = float("nan")
        spearman = float("nan")
        slope = float("nan")
        intercept = float("nan")
        r_squared = float("nan")
        if n < 3:
            relationship_status = "insufficient_pairs"
        else:
            x_finite = x[finite]
            y_finite = target[finite]
            if np.allclose(x_finite, x_finite[0]):
                relationship_status = "constant_covariate"
            elif np.allclose(y_finite, y_finite[0]):
                relationship_status = "constant_target"
            else:
                pearson = _pearson_correlation(x_finite, y_finite)
                spearman = _spearman_correlation(x_finite, y_finite)
                slope, intercept, r_squared = _linear_relationship(x_finite, y_finite)
        rows.append(
            {
                "source_spectral_family": source_spectral_family,
                "case_name": case_name,
                "target": SELECTED_TREE_SPECTRAL_TARGET,
                "covariate": covariate,
                "relationship_status": relationship_status,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
                "linear_slope": slope,
                "linear_intercept": intercept,
                "linear_r_squared": r_squared,
            }
        )
    return rows


def _source_spectral_family(case: dict[str, object]) -> str:
    generator = str(case.get("generator", ""))
    if generator in {"blobs_continuous", "dimensional_gaussian_continuous"}:
        return "continuous_self_whitened"
    if generator in {"categorical", "phylogenetic", "temporal_evolution"}:
        return "categorical"
    if generator in {"binary", "sbm"}:
        return "bernoulli"
    if generator in {
        "blobs",
        "blobs_quantile",
        "dimensional_gaussian",
        "gaussian_outliers",
    }:
        return "bernoulli_discretized"
    return "other"


def _positive_log_column(
    frame: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
) -> None:
    values = frame[source_column].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    invalid = finite & (values <= 0.0)
    if np.any(invalid):
        invalid_values = values[invalid][:5].tolist()
        raise ValueError(
            f"{source_column} must be positive wherever finite; "
            f"examples={invalid_values!r}."
        )
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[finite] = np.log(values[finite])
    frame[output_column] = output


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_correlation(_average_ranks(x), _average_ranks(y))


def _linear_relationship(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.dot(x_centered, y - y_mean) / denominator)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    residual_ss = float(np.dot(y - fitted, y - fitted))
    total_ss = float(np.dot(y - y_mean, y - y_mean))
    r_squared = float("nan") if total_ss == 0.0 else float(1.0 - residual_ss / total_ss)
    return slope, intercept, r_squared


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _identity_mp_density(x: np.ndarray, support: IdentityMpSupport) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    density = np.zeros_like(x)
    inside = (x > support.lower_edge) & (x < support.upper_edge) & (x > 0.0)
    if np.any(inside):
        numerator = np.sqrt(
            np.maximum(support.upper_edge - x[inside], 0.0)
            * np.maximum(x[inside] - support.lower_edge, 0.0)
        )
        density[inside] = numerator / (2.0 * np.pi * support.aspect_ratio * x[inside])
    return density


def _identity_null_top_eigenvalue(
    matrix_rows: int,
    active_feature_count: int,
    rng: np.random.Generator,
) -> float:
    null_matrix = rng.normal(size=(int(matrix_rows), int(active_feature_count)))
    centered = null_matrix - np.mean(null_matrix, axis=0)
    if active_feature_count > matrix_rows:
        covariance = centered @ centered.T / float(matrix_rows)
    else:
        covariance = centered.T @ centered / float(matrix_rows)
    return float(np.linalg.eigvalsh(covariance)[-1])


def _positive_spectrum_ks_distance(
    positive_eigenvalues: np.ndarray,
    mp_grid: IdentityMpGrid,
) -> float:
    values = np.sort(np.asarray(positive_eigenvalues, dtype=np.float64))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("positive_eigenvalues must be a non-empty 1-D array.")
    theoretical_cdf = np.interp(
        np.clip(values, mp_grid.grid[0], mp_grid.grid[-1]),
        mp_grid.grid,
        mp_grid.cdf,
        left=0.0,
        right=1.0,
    )
    n = float(values.size)
    empirical_upper = np.arange(1, values.size + 1, dtype=np.float64) / n
    empirical_lower = np.arange(0, values.size, dtype=np.float64) / n
    return float(
        np.max(
            np.maximum(
                np.abs(empirical_upper - theoretical_cdf),
                np.abs(theoretical_cdf - empirical_lower),
            )
        )
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def _records_for_case(
    case: dict[str, object],
    *,
    max_nodes: int | None,
) -> list[SpectralMatrixRecord]:
    (
        records,
        _leaf_feature_matrix,
        _tasks,
        _feature_space,
        _tree_seconds,
        _materialize_seconds,
        _vectorized_materialize_seconds,
        _vectorized_max_abs_diff,
        _tree_distance_metric,
        _tree_distance_source,
    ) = _build_matrix_records(case, max_nodes=max_nodes)
    return records


def _summarize_case(
    case: dict[str, object],
    *,
    node_rows_for_case: list[dict[str, object]],
    records: Sequence[SpectralMatrixRecord],
) -> dict[str, object]:
    case_name = str(case["name"])
    rows = [row for row in node_rows_for_case if row["case_name"] == case_name]
    evaluated = [row for row in rows if row["diagnostic_status"] == "evaluated"]
    return {
        "case_name": case_name,
        "case_category": str(case.get("category", "")),
        "generator": str(case.get("generator", "")),
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "nodes_total": int(len(records)),
        "nodes_evaluated": int(len(evaluated)),
        "nodes_insufficient_positive_spectrum": _count_status(
            rows, "insufficient_positive_spectrum"
        ),
        "nodes_no_active_features": _count_status(rows, "no_active_features"),
        "median_aspect_ratio": _median(evaluated, "aspect_ratio"),
        "median_active_feature_count": _median(evaluated, "active_feature_count"),
        "raw_mp_signal_node_fraction": _mean_positive(evaluated, "raw_mp_signal_count"),
        "mean_raw_mp_signal_count": _mean(evaluated, "raw_mp_signal_count"),
        "median_top_eigenvalue_over_mp_upper": _median(
            evaluated, "top_eigenvalue_over_mp_upper"
        ),
        "q95_top_eigenvalue_over_mp_upper": _quantile(
            evaluated, "top_eigenvalue_over_mp_upper", 0.95
        ),
        "median_q95_eigenvalue_over_identity_mp_q95": _median(
            evaluated, "q95_eigenvalue_over_identity_mp_q95"
        ),
        "median_identity_mp_positive_ks_distance": _median(
            evaluated, "identity_mp_positive_ks_distance"
        ),
        "finite_identity_null_quantile": _first_finite(
            evaluated, "finite_identity_null_quantile"
        ),
        "finite_identity_null_reps": _first_finite(
            evaluated, "finite_identity_null_reps"
        ),
        "above_finite_identity_null_top_quantile_node_fraction": _mean_bool(
            evaluated, "above_finite_identity_null_top_quantile"
        ),
        "median_finite_identity_null_top_over_mp_upper": _median(
            evaluated, "finite_identity_null_top_over_mp_upper"
        ),
        "median_top_eigenvalue_over_finite_identity_null_quantile": _median(
            evaluated, "top_eigenvalue_over_finite_identity_null_quantile"
        ),
        "median_top_eigenvalue_over_centered_self_whitening_reference": _median(
            evaluated, "top_eigenvalue_over_centered_self_whitening_reference"
        ),
        "median_eigenvalue_over_centered_self_whitening_reference": _median(
            evaluated, "median_eigenvalue_over_centered_self_whitening_reference"
        ),
        "median_positive_rank_fraction": _median(evaluated, "positive_rank_fraction"),
    }


def _count_status(rows: Sequence[dict[str, object]], status: str) -> int:
    return int(sum(row["diagnostic_status"] == status for row in rows))


def _finite_values(rows: Sequence[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _median(rows: Sequence[dict[str, object]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.median(values)) if values.size else float("nan")


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.mean(values)) if values.size else float("nan")


def _quantile(rows: Sequence[dict[str, object]], key: str, q: float) -> float:
    values = _finite_values(rows, key)
    return float(np.quantile(values, q)) if values.size else float("nan")


def _mean_positive(rows: Sequence[dict[str, object]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.mean(values > 0.0)) if values.size else float("nan")


def _mean_bool(rows: Sequence[dict[str, object]], key: str) -> float:
    if not rows:
        return float("nan")
    values = np.asarray([bool(row[key]) for row in rows], dtype=np.float64)
    return float(np.mean(values))


def _first_finite(rows: Sequence[dict[str, object]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(values[0]) if values.size else float("nan")


def _resolve_cases(
    case_names: Sequence[str],
    max_cases: int | None,
) -> list[dict[str, object]]:
    requested_names = list(case_names)
    if not requested_names:
        raise ValueError("At least one case name is required.")
    requested_set = set(requested_names)
    cases = [case for case in get_default_test_cases() if str(case["name"]) in requested_set]
    found_names = {str(case["name"]) for case in cases}
    missing = sorted(requested_set - found_names)
    if missing:
        raise ValueError(f"Unknown case names: {', '.join(missing)}")
    ordered_cases = sorted(cases, key=lambda case: requested_names.index(str(case["name"])))
    if max_cases is not None:
        ordered_cases = ordered_cases[: max(int(max_cases), 0)]
    return ordered_cases


def _parse_case_names(raw_names: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in raw_names.split(",") if name.strip())
    if not names:
        raise ValueError("case_names must contain at least one non-empty case name.")
    return names


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("raw/assets/benchmark-results") / f"local_mp_identity_law_{stamp}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen local production spectra against the identity MP law."
    )
    parser.add_argument(
        "--case-names",
        default=",".join(DEFAULT_CASE_NAMES),
        help="Comma-separated benchmark case names.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--min-positive-eigenvalues", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=8192)
    parser.add_argument(
        "--finite-null-reps",
        type=int,
        default=DEFAULT_FINITE_IDENTITY_NULL_REPS,
    )
    parser.add_argument(
        "--finite-null-quantile",
        type=float,
        default=DEFAULT_FINITE_IDENTITY_NULL_QUANTILE,
    )
    parser.add_argument(
        "--finite-null-seed",
        type=int,
        default=DEFAULT_FINITE_IDENTITY_NULL_SEED,
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    case_summary, node_spectrum = run_local_mp_identity_law_diagnostic(
        case_names=_parse_case_names(args.case_names),
        max_cases=args.max_cases,
        max_nodes=args.max_nodes,
        min_positive_eigenvalues=args.min_positive_eigenvalues,
        grid_size=args.grid_size,
        finite_null_reps=args.finite_null_reps,
        finite_null_quantile=args.finite_null_quantile,
        finite_null_seed=args.finite_null_seed,
    )
    write_outputs(
        output_dir,
        case_summary=case_summary,
        node_spectrum=node_spectrum,
        args=args,
    )
    print(f"Wrote {output_dir}")
    print(case_summary.to_string(index=False))


if __name__ == "__main__":
    main()
