"""Third-generation analytic inflation model with no permutations or sample splitting.

This experiment replaces the permutation target with a purely analytic,
full-data calibration strategy built only from null-like sibling pairs.

Design:
1. Fit a pooled power-law baseline on null-like pairs:
      log(c) = a + b * log(n_parent)
2. Fit a ridge residual model on the same null-like pairs using simple
   node-structure features.
3. Select a conservative residual scale and safety offset on training
   null-like rows only.
4. Evaluate leave-one-case-out on:
   - approximate-null calibration (null-like rows)
   - focal-node aggressiveness (focal rows)

This is intentionally conservative. The v3 prediction can only increase
deflation relative to the pooled power-law baseline; it never reduces it.

Usage:
    python debug_scripts/enhancement_lab/exp_parametric_inflation_v3.py
    python debug_scripts/enhancement_lab/exp_parametric_inflation_v3.py --limit 2
    python debug_scripts/enhancement_lab/exp_parametric_inflation_v3.py --cases gauss_moderate_3c
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp_parametric_inflation import DIAGNOSTIC_CASES  # noqa: E402
from lab_helpers import build_tree_and_data, compute_ari, run_decomposition  # noqa: E402

from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.core_utils.data_utils import extract_node_sample_size  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)


EPS = 1e-9
POWER_RIDGE = 0.01
FEATURE_RIDGE = 4.0
SCALE_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
OFFSET_QUANTILES = (0.25, 0.33, 0.50, 0.67, 0.80, 0.90)


@dataclass(frozen=True)
class PairRow:
    case_name: str
    parent: str
    n_parent: int
    n_left: int
    n_right: int
    k: int
    depth: int
    branch_length_sum: float
    sibling_null_prior: float
    is_null_like: bool
    is_binary_case: float
    is_null_case: float
    t_obs: float
    c_global: float
    p_global: float
    true_k: int | None
    found_k: int
    ari: float


@dataclass(frozen=True)
class PowerLawModel:
    intercept: float
    slope: float
    min_log_c: float
    max_log_c: float
    n_train: int

    def predict_log(self, row: PairRow) -> float:
        log_c = self.intercept + self.slope * math.log(max(row.n_parent, 1))
        return float(np.clip(log_c, self.min_log_c, self.max_log_c))

    def predict(self, row: PairRow) -> float:
        return max(1.0, math.exp(self.predict_log(row)))


@dataclass(frozen=True)
class ResidualModel:
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...]
    n_train: int

    def predict(self, row: PairRow) -> float:
        raw = _feature_vector(row)
        standardized = (raw - self.means) / self.scales
        return float(np.dot(standardized, self.coefficients))


@dataclass(frozen=True)
class V3Model:
    power_model: PowerLawModel
    residual_model: ResidualModel
    residual_scale: float
    offset_quantile: float
    safety_offset: float

    def predict_log(self, row: PairRow) -> float:
        baseline_log = self.power_model.predict_log(row)
        residual_term = self.residual_scale * self.residual_model.predict(row) + self.safety_offset
        extra_log = max(0.0, residual_term)
        return float(np.clip(baseline_log + extra_log, self.power_model.min_log_c, self.power_model.max_log_c))

    def predict(self, row: PairRow) -> float:
        return max(1.0, math.exp(self.predict_log(row)))


@dataclass(frozen=True)
class CandidateScore:
    residual_scale: float
    offset_quantile: float
    safety_offset: float
    bh_null_rejects: int
    raw_null_rejects: int
    bh_focal_rejects: int
    raw_focal_rejects: int
    mae_log_ratio: float
    mean_log_c: float


@dataclass(frozen=True)
class CaseEvaluation:
    case_name: str
    true_k: int | None
    found_k: int
    ari: float
    residual_scale: float
    offset_quantile: float
    safety_offset: float
    n_null: int
    n_focal: int
    raw_null_global: int
    raw_null_power: int
    raw_null_v3: int
    bh_null_global: int
    bh_null_power: int
    bh_null_v3: int
    raw_focal_global: int
    raw_focal_power: int
    raw_focal_v3: int
    bh_focal_global: int
    bh_focal_power: int
    bh_focal_v3: int


def _case_flags(case_name: str) -> tuple[float, float]:
    is_binary_case = 1.0 if case_name.startswith("binary_") else 0.0
    is_null_case = 1.0 if "_null_" in case_name else 0.0
    return is_binary_case, is_null_case


def _balance_ratio(n_left: int, n_right: int) -> float:
    total = max(n_left + n_right, 1)
    return min(n_left, n_right) / total


def _feature_vector(row: PairRow) -> np.ndarray:
    log_n_parent = math.log(max(row.n_parent, 1))
    log_k = math.log(max(row.k, 1))
    balance = _balance_ratio(row.n_left, row.n_right)
    log_branch = math.log1p(max(row.branch_length_sum, 0.0))
    return np.array(
        [
            1.0,
            log_n_parent,
            log_k,
            balance,
            float(row.depth),
            log_branch,
            row.is_binary_case,
            row.is_null_case,
            log_n_parent * row.is_binary_case,
            log_n_parent * row.is_null_case,
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = (
    "intercept",
    "log_n_parent",
    "log_k",
    "balance",
    "depth",
    "log1p_branch_sum",
    "is_binary_case",
    "is_null_case",
    "log_n_parent_x_binary",
    "log_n_parent_x_null",
)


def _standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    means[0] = 0.0
    scales[0] = 1.0
    scales = np.where(scales < EPS, 1.0, scales)
    standardized = (matrix - means) / scales
    standardized[:, 0] = 1.0
    return standardized, means, scales


def _fit_weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    sqrt_w = np.sqrt(np.clip(weights, 0.05, None))[:, None]
    weighted_design = design * sqrt_w
    weighted_target = target * sqrt_w[:, 0]
    penalty = ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_target
    return np.linalg.solve(lhs, rhs)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("Cannot compute a weighted quantile of an empty array.")
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.clip(weights[order], 0.0, None)
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float(np.quantile(sorted_values, quantile))
    cumulative = np.cumsum(sorted_weights) / total_weight
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _predict_p_value(t_obs: float, k: int, c_value: float) -> float:
    adjusted_stat = t_obs / max(c_value, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _bh_reject_flags(p_values: np.ndarray) -> np.ndarray:
    rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
    return rejected.astype(bool)


def fit_pooled_power_law(rows: list[PairRow]) -> PowerLawModel:
    if not rows:
        raise ValueError("Cannot fit pooled power-law model with no rows.")

    log_n = np.array([math.log(max(row.n_parent, 1)) for row in rows], dtype=np.float64)
    log_ratio = np.array([math.log(max(row.t_obs / row.k, 1.0)) for row in rows], dtype=np.float64)
    weights = np.array([max(row.sibling_null_prior, 0.05) for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(rows), dtype=np.float64), log_n])
    coefficients = _fit_weighted_ridge(design, log_ratio, weights, ridge_lambda=POWER_RIDGE)

    return PowerLawModel(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        min_log_c=0.0,
        max_log_c=float(np.max(log_ratio) + 0.25),
        n_train=len(rows),
    )


def fit_residual_model(rows: list[PairRow], power_model: PowerLawModel) -> ResidualModel:
    if not rows:
        raise ValueError("Cannot fit residual model with no rows.")

    raw_design = np.vstack([_feature_vector(row) for row in rows])
    design, means, scales = _standardize_matrix(raw_design)
    target = np.array(
        [
            math.log(max(row.t_obs / row.k, 1.0)) - power_model.predict_log(row)
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = np.array([max(row.sibling_null_prior, 0.05) for row in rows], dtype=np.float64)
    coefficients = _fit_weighted_ridge(design, target, weights, ridge_lambda=FEATURE_RIDGE)

    return ResidualModel(
        coefficients=coefficients,
        means=means,
        scales=scales,
        feature_names=FEATURE_NAMES,
        n_train=len(rows),
    )


def _score_candidate(
    null_rows: list[PairRow],
    eval_rows: list[PairRow],
    power_model: PowerLawModel,
    residual_model: ResidualModel,
    residual_scale: float,
    offset_quantile: float,
) -> CandidateScore:
    weights = np.array([max(row.sibling_null_prior, 0.05) for row in null_rows], dtype=np.float64)
    baseline_logs = np.array([power_model.predict_log(row) for row in null_rows], dtype=np.float64)
    residual_predictions = np.array([residual_model.predict(row) for row in null_rows], dtype=np.float64)
    target_logs = np.array(
        [math.log(max(row.t_obs / row.k, 1.0)) for row in null_rows],
        dtype=np.float64,
    )

    residual_errors = target_logs - (baseline_logs + residual_scale * residual_predictions)
    safety_offset = max(0.0, _weighted_quantile(residual_errors, weights, offset_quantile))
    extra_logs = np.maximum(0.0, residual_scale * residual_predictions + safety_offset)
    predicted_logs = np.clip(baseline_logs + extra_logs, power_model.min_log_c, power_model.max_log_c)
    null_p_values = np.array(
        [
            _predict_p_value(row.t_obs, row.k, math.exp(predicted_log))
            for row, predicted_log in zip(null_rows, predicted_logs, strict=False)
        ],
        dtype=np.float64,
    )

    eval_p_values = np.array(
        [
            _predict_p_value(
                row.t_obs,
                row.k,
                math.exp(
                    np.clip(
                        power_model.predict_log(row)
                        + max(0.0, residual_scale * residual_model.predict(row) + safety_offset),
                        power_model.min_log_c,
                        power_model.max_log_c,
                    )
                ),
            )
            for row in eval_rows
        ],
        dtype=np.float64,
    )

    focal_eval_p_values = np.array(
        [p for row, p in zip(eval_rows, eval_p_values, strict=False) if not row.is_null_like],
        dtype=np.float64,
    )

    raw_null_rejects = int(np.sum(null_p_values < config.SIBLING_ALPHA))
    bh_null_rejects = int(np.sum(_bh_reject_flags(null_p_values)))
    raw_focal_rejects = int(np.sum(focal_eval_p_values < config.SIBLING_ALPHA))
    bh_focal_rejects = int(np.sum(_bh_reject_flags(focal_eval_p_values)))
    mae_log_ratio = float(np.mean(np.abs(predicted_logs - target_logs)))
    mean_log_c = float(np.mean(predicted_logs))
    return CandidateScore(
        residual_scale=residual_scale,
        offset_quantile=offset_quantile,
        safety_offset=safety_offset,
        bh_null_rejects=bh_null_rejects,
        raw_null_rejects=raw_null_rejects,
        bh_focal_rejects=bh_focal_rejects,
        raw_focal_rejects=raw_focal_rejects,
        mae_log_ratio=mae_log_ratio,
        mean_log_c=mean_log_c,
    )


def select_v3_model(train_null_rows: list[PairRow], train_eval_rows: list[PairRow] | None = None) -> V3Model:
    power_model = fit_pooled_power_law(train_null_rows)
    residual_model = fit_residual_model(train_null_rows, power_model)
    eval_rows = train_eval_rows if train_eval_rows is not None else train_null_rows

    candidates: list[CandidateScore] = []
    for residual_scale in SCALE_GRID:
        for offset_quantile in OFFSET_QUANTILES:
            candidates.append(
                _score_candidate(
                    train_null_rows,
                    eval_rows,
                    power_model,
                    residual_model,
                    residual_scale,
                    offset_quantile,
                )
            )

    zero_raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.bh_null_rejects == 0 and candidate.raw_null_rejects == 0
    ]
    if zero_raw_candidates:
        candidate_pool = zero_raw_candidates
    else:
        candidate_pool = candidates

    best = min(
        candidate_pool,
        key=lambda score: (
            score.bh_null_rejects,
            score.raw_null_rejects,
            -score.bh_focal_rejects,
            -score.raw_focal_rejects,
            score.mae_log_ratio,
            score.mean_log_c,
        ),
    )

    return V3Model(
        power_model=power_model,
        residual_model=residual_model,
        residual_scale=best.residual_scale,
        offset_quantile=best.offset_quantile,
        safety_offset=best.safety_offset,
    )


def collect_case_rows(case_name: str) -> list[PairRow]:
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    decomp = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

    true_k = tc.get("n_clusters")
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_true) if y_true is not None else float("nan")
    c_global = float(annotations_df.attrs.get("sibling_divergence_audit", {}).get("global_inflation_factor", 1.0))

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    sibling_dims = derive_sibling_spectral_dims(tree, annotations_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(annotations_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, annotations_df, sibling_dims)
    records, _ = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca,
        pca_eigenvalues=sibling_eig,
        whitening=config.SIBLING_WHITENING,
    )

    root = tree.root()
    is_binary_case, is_null_case = _case_flags(case_name)
    rows: list[PairRow] = []

    for record in records:
        if record.degrees_of_freedom <= 0 or not np.isfinite(record.stat):
            continue
        children = list(tree.successors(record.parent))
        if len(children) != 2:
            continue

        left, right = children
        n_left = extract_node_sample_size(tree, left)
        n_right = extract_node_sample_size(tree, right)
        depth = int(nx.shortest_path_length(tree, root, record.parent))
        p_global = _predict_p_value(record.stat, int(record.degrees_of_freedom), c_global)

        rows.append(
            PairRow(
                case_name=case_name,
                parent=record.parent,
                n_parent=int(record.n_parent),
                n_left=int(n_left),
                n_right=int(n_right),
                k=int(record.degrees_of_freedom),
                depth=depth,
                branch_length_sum=float(max(record.branch_length_sum, 0.0)),
                sibling_null_prior=float(record.sibling_null_prior_from_edge_pvalue),
                is_null_like=bool(record.is_null_like),
                is_binary_case=is_binary_case,
                is_null_case=is_null_case,
                t_obs=float(record.stat),
                c_global=c_global,
                p_global=p_global,
                true_k=true_k,
                found_k=found_k,
                ari=ari,
            )
        )

    return rows


def build_rows(case_names: list[str]) -> list[PairRow]:
    rows: list[PairRow] = []
    for case_name in case_names:
        case_rows = collect_case_rows(case_name)
        rows.extend(case_rows)
        case_null = sum(1 for row in case_rows if row.is_null_like)
        case_focal = len(case_rows) - case_null
        if case_rows:
            exemplar = case_rows[0]
            print(
                f"{case_name:<28} K={exemplar.found_k}/{exemplar.true_k} "
                f"ARI={exemplar.ari:>5.3f} rows={len(case_rows):>3} "
                f"null={case_null:>3} focal={case_focal:>3} c_global={exemplar.c_global:>8.3f}"
            )
        else:
            print(f"{case_name:<28} no valid sibling rows")
    return rows


def evaluate_case(case_name: str, rows: list[PairRow], model: V3Model) -> CaseEvaluation:
    case_rows = [row for row in rows if row.case_name == case_name]
    if not case_rows:
        raise ValueError(f"No rows found for case {case_name!r}.")

    null_rows = [row for row in case_rows if row.is_null_like]
    focal_rows = [row for row in case_rows if not row.is_null_like]
    exemplar = case_rows[0]

    def _p_values(target_rows: list[PairRow], mode: str) -> np.ndarray:
        if not target_rows:
            return np.array([], dtype=np.float64)
        if mode == "global":
            return np.array([row.p_global for row in target_rows], dtype=np.float64)
        if mode == "power":
            return np.array(
                [_predict_p_value(row.t_obs, row.k, model.power_model.predict(row)) for row in target_rows],
                dtype=np.float64,
            )
        if mode == "v3":
            return np.array(
                [_predict_p_value(row.t_obs, row.k, model.predict(row)) for row in target_rows],
                dtype=np.float64,
            )
        raise ValueError(f"Unsupported mode: {mode}")

    null_global = _p_values(null_rows, "global")
    null_power = _p_values(null_rows, "power")
    null_v3 = _p_values(null_rows, "v3")
    focal_global = _p_values(focal_rows, "global")
    focal_power = _p_values(focal_rows, "power")
    focal_v3 = _p_values(focal_rows, "v3")

    return CaseEvaluation(
        case_name=case_name,
        true_k=exemplar.true_k,
        found_k=exemplar.found_k,
        ari=exemplar.ari,
        residual_scale=model.residual_scale,
        offset_quantile=model.offset_quantile,
        safety_offset=model.safety_offset,
        n_null=len(null_rows),
        n_focal=len(focal_rows),
        raw_null_global=int(np.sum(null_global < config.SIBLING_ALPHA)),
        raw_null_power=int(np.sum(null_power < config.SIBLING_ALPHA)),
        raw_null_v3=int(np.sum(null_v3 < config.SIBLING_ALPHA)),
        bh_null_global=int(np.sum(_bh_reject_flags(null_global))) if len(null_global) else 0,
        bh_null_power=int(np.sum(_bh_reject_flags(null_power))) if len(null_power) else 0,
        bh_null_v3=int(np.sum(_bh_reject_flags(null_v3))) if len(null_v3) else 0,
        raw_focal_global=int(np.sum(focal_global < config.SIBLING_ALPHA)),
        raw_focal_power=int(np.sum(focal_power < config.SIBLING_ALPHA)),
        raw_focal_v3=int(np.sum(focal_v3 < config.SIBLING_ALPHA)),
        bh_focal_global=int(np.sum(_bh_reject_flags(focal_global))) if len(focal_global) else 0,
        bh_focal_power=int(np.sum(_bh_reject_flags(focal_power))) if len(focal_power) else 0,
        bh_focal_v3=int(np.sum(_bh_reject_flags(focal_v3))) if len(focal_v3) else 0,
    )


def evaluate_loco(rows: list[PairRow], case_names: list[str]) -> None:
    print("\n" + "=" * 116)
    print("LEAVE-ONE-CASE-OUT V3: global ĉ vs pooled power-law vs conservative analytic v3")
    print("=" * 116)
    print("Approx-null rows are null-like pairs only. Focal rows are reported as aggressiveness, not accuracy.")

    evaluations: list[CaseEvaluation] = []
    total_null = 0
    total_focal = 0
    total_raw_null_global = 0
    total_raw_null_power = 0
    total_raw_null_v3 = 0
    total_bh_null_global = 0
    total_bh_null_power = 0
    total_bh_null_v3 = 0
    total_raw_focal_global = 0
    total_raw_focal_power = 0
    total_raw_focal_v3 = 0
    total_bh_focal_global = 0
    total_bh_focal_power = 0
    total_bh_focal_v3 = 0

    for case_name in case_names:
        train_null_rows = [row for row in rows if row.case_name != case_name and row.is_null_like]
        train_eval_rows = [row for row in rows if row.case_name != case_name]
        if not train_null_rows:
            train_null_rows = [row for row in rows if row.case_name == case_name and row.is_null_like]
            train_eval_rows = [row for row in rows if row.case_name == case_name]
        if not train_null_rows:
            raise ValueError(f"No null-like training rows available for case {case_name!r}.")

        model = select_v3_model(train_null_rows, train_eval_rows=train_eval_rows)
        evaluation = evaluate_case(case_name, rows, model)
        evaluations.append(evaluation)

        total_null += evaluation.n_null
        total_focal += evaluation.n_focal
        total_raw_null_global += evaluation.raw_null_global
        total_raw_null_power += evaluation.raw_null_power
        total_raw_null_v3 += evaluation.raw_null_v3
        total_bh_null_global += evaluation.bh_null_global
        total_bh_null_power += evaluation.bh_null_power
        total_bh_null_v3 += evaluation.bh_null_v3
        total_raw_focal_global += evaluation.raw_focal_global
        total_raw_focal_power += evaluation.raw_focal_power
        total_raw_focal_v3 += evaluation.raw_focal_v3
        total_bh_focal_global += evaluation.bh_focal_global
        total_bh_focal_power += evaluation.bh_focal_power
        total_bh_focal_v3 += evaluation.bh_focal_v3

        print(
            f"{case_name:<28} null={evaluation.n_null:>3} focal={evaluation.n_focal:>3} "
            f"K={evaluation.found_k}/{evaluation.true_k} ARI={evaluation.ari:>5.3f} | "
            f"scale={evaluation.residual_scale:>4.2f} q={evaluation.offset_quantile:>4.2f} off={evaluation.safety_offset:>5.3f} | "
            f"null raw G/P/V3={evaluation.raw_null_global:>2}/{evaluation.raw_null_power:>2}/{evaluation.raw_null_v3:>2} | "
            f"focal BH G/P/V3={evaluation.bh_focal_global:>2}/{evaluation.bh_focal_power:>2}/{evaluation.bh_focal_v3:>2}"
        )

    print("\n" + "-" * 116)
    print(f"Total null-like rows: {total_null}")
    if total_null > 0:
        print(
            "Approx-null raw rejection rate: "
            f"global={total_raw_null_global / total_null:.1%}, "
            f"power-law={total_raw_null_power / total_null:.1%}, "
            f"v3={total_raw_null_v3 / total_null:.1%}"
        )
        print(
            "Approx-null BH rejection rate: "
            f"global={total_bh_null_global / total_null:.1%}, "
            f"power-law={total_bh_null_power / total_null:.1%}, "
            f"v3={total_bh_null_v3 / total_null:.1%}"
        )

    print(f"\nTotal focal rows: {total_focal}")
    if total_focal > 0:
        print(
            "Focal raw rejection rate: "
            f"global={total_raw_focal_global / total_focal:.1%}, "
            f"power-law={total_raw_focal_power / total_focal:.1%}, "
            f"v3={total_raw_focal_v3 / total_focal:.1%}"
        )
        print(
            "Focal BH rejection rate: "
            f"global={total_bh_focal_global / total_focal:.1%}, "
            f"power-law={total_bh_focal_power / total_focal:.1%}, "
            f"v3={total_bh_focal_v3 / total_focal:.1%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Specific cases to evaluate. Defaults to the shared diagnostic battery.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prefix length of the selected case list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_names = list(args.cases) if args.cases else list(DIAGNOSTIC_CASES)
    if args.limit is not None:
        case_names = case_names[: args.limit]
    if not case_names:
        raise ValueError("At least one case is required.")

    print(
        f"Config: METHOD={config.SIBLING_TEST_METHOD}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, EDGE_ALPHA={config.EDGE_ALPHA}"
    )
    print("V3 model: pooled power-law + conservative residual uplift from null-like pairs only")
    print("No permutations. No sample splitting.")
    print(f"Cases: {len(case_names)}\n")

    rows = build_rows(case_names)
    evaluate_loco(rows, case_names)


if __name__ == "__main__":
    main()
