"""Second-generation parametric inflation model with leave-one-case-out testing.

This script improves on the first parametric experiment in two ways:
1. It evaluates models out-of-case using leave-one-case-out validation.
2. It adds simple node-structure features beyond n_parent alone.

The target remains the node-local permutation inflation proxy c_perm from
exp_parametric_inflation.py, but the model is now tested more rigorously.

Usage:
    python debug_scripts/enhancement_lab/exp_parametric_inflation_v2.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import chi2, spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp_parametric_inflation import DIAGNOSTIC_CASES, run_case as run_case_v1  # noqa: E402
from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402

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
RIDGE_LAMBDA = 1.0


@dataclass(frozen=True)
class FeatureRow:
    case_name: str
    parent: str
    n_parent: int
    n_left: int
    n_right: int
    k: int
    depth: int
    branch_length_sum: float
    edge_weight: float
    is_null_like: bool
    is_binary_case: float
    is_null_case: float
    t_obs: float
    c_perm: float
    p_perm: float
    c_global: float
    p_global: float


@dataclass(frozen=True)
class PowerLawModel:
    intercept: float
    slope: float
    max_log_c: float
    min_log_c: float
    n_train: int

    def predict(self, row: FeatureRow) -> float:
        log_c = self.intercept + self.slope * math.log(max(row.n_parent, 1))
        bounded = min(max(log_c, self.min_log_c), self.max_log_c)
        return max(1.0, math.exp(bounded))


@dataclass(frozen=True)
class FeatureModel:
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...]
    max_log_c: float
    min_log_c: float
    n_train: int

    def predict(self, row: FeatureRow) -> float:
        raw = _feature_vector(row)
        standardized = (raw - self.means) / self.scales
        log_c = float(np.dot(standardized, self.coefficients))
        bounded = min(max(log_c, self.min_log_c), self.max_log_c)
        return max(1.0, math.exp(bounded))


@dataclass(frozen=True)
class CaseEvaluation:
    case_name: str
    blend_weight: float
    n_focal: int
    raw_perm_rejects: int
    raw_global_rejects: int
    raw_power_rejects: int
    raw_v2_rejects: int
    bh_perm_rejects: int
    bh_global_rejects: int
    bh_power_rejects: int
    bh_v2_rejects: int
    raw_agreement_global: float
    raw_agreement_power: float
    raw_agreement_v2: float
    bh_agreement_global: float
    bh_agreement_power: float
    bh_agreement_v2: float


def _case_flags(case_name: str) -> tuple[float, float]:
    is_binary_case = 1.0 if case_name.startswith("binary_") else 0.0
    is_null_case = 1.0 if "_null_" in case_name else 0.0
    return is_binary_case, is_null_case


def _balance_ratio(n_left: int, n_right: int) -> float:
    total = max(n_left + n_right, 1)
    return min(n_left, n_right) / total


def _feature_vector(row: FeatureRow) -> np.ndarray:
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


def fit_pooled_power_law(rows: list[FeatureRow]) -> PowerLawModel:
    if not rows:
        raise ValueError("Cannot fit pooled power-law model with no training rows.")

    log_n = np.array([math.log(max(row.n_parent, 1)) for row in rows], dtype=np.float64)
    log_c = np.array([math.log(max(row.c_perm, 1.0)) for row in rows], dtype=np.float64)
    weights = np.array([max(row.edge_weight, 0.05) for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(rows), dtype=np.float64), log_n])
    coefficients = _fit_weighted_ridge(design, log_c, weights, ridge_lambda=0.01)
    return PowerLawModel(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        max_log_c=float(np.max(log_c) + 0.25),
        min_log_c=0.0,
        n_train=len(rows),
    )


def fit_feature_model(rows: list[FeatureRow]) -> FeatureModel:
    if not rows:
        raise ValueError("Cannot fit feature model with no training rows.")

    raw_design = np.vstack([_feature_vector(row) for row in rows])
    design, means, scales = _standardize_matrix(raw_design)
    target = np.array([math.log(max(row.c_perm, 1.0)) for row in rows], dtype=np.float64)
    weights = np.array([max(row.edge_weight, 0.05) for row in rows], dtype=np.float64)
    coefficients = _fit_weighted_ridge(design, target, weights, ridge_lambda=RIDGE_LAMBDA)
    return FeatureModel(
        coefficients=coefficients,
        means=means,
        scales=scales,
        feature_names=FEATURE_NAMES,
        max_log_c=float(np.max(target) + 0.25),
        min_log_c=0.0,
        n_train=len(rows),
    )


def _collect_case_metadata(case_name: str) -> dict[str, dict[str, float | int | bool]]:
    tree, data_df, _, _ = build_tree_and_data(case_name)
    _ = run_decomposition(tree, data_df)
    annotations_df = tree.annotations_df

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
    metadata: dict[str, dict[str, float | int | bool]] = {}
    for record in records:
        children = list(tree.successors(record.parent))
        if len(children) != 2:
            continue
        left, right = children
        depth = int(nx.shortest_path_length(tree, root, record.parent))
        metadata[record.parent] = {
            "n_left": extract_node_sample_size(tree, left),
            "n_right": extract_node_sample_size(tree, right),
            "depth": depth,
            "branch_length_sum": float(record.branch_length_sum),
            "edge_weight": float(record.edge_weight),
            "is_null_like": bool(record.is_null_like),
            "k": int(record.degrees_of_freedom),
        }
    return metadata


def build_feature_rows(case_names: list[str]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for case_name in case_names:
        case_label, nodes, _, _ = run_case_v1(case_name)
        metadata_by_parent = _collect_case_metadata(case_name)
        is_binary_case, is_null_case = _case_flags(case_name)

        for node in nodes:
            meta = metadata_by_parent.get(node.parent)
            if meta is None:
                raise ValueError(f"Missing metadata for parent {node.parent!r} in case {case_name!r}.")

            rows.append(
                FeatureRow(
                    case_name=case_label,
                    parent=node.parent,
                    n_parent=node.n_parent,
                    n_left=int(meta["n_left"]),
                    n_right=int(meta["n_right"]),
                    k=node.k,
                    depth=int(meta["depth"]),
                    branch_length_sum=float(meta["branch_length_sum"]),
                    edge_weight=float(meta["edge_weight"]),
                    is_null_like=node.is_null_like,
                    is_binary_case=is_binary_case,
                    is_null_case=is_null_case,
                    t_obs=node.t_obs,
                    c_perm=node.c_perm,
                    p_perm=node.p_perm,
                    c_global=node.c_global,
                    p_global=node.p_global,
                )
            )
    return rows


def _predict_p_value(t_obs: float, k: int, c_value: float) -> float:
    adjusted_stat = t_obs / max(c_value, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _blend_c(power_c: float, feature_c: float, blend_weight: float) -> float:
    power_log = math.log(max(power_c, 1.0))
    feature_log = math.log(max(feature_c, 1.0))
    blended = blend_weight * power_log + (1.0 - blend_weight) * feature_log
    return max(1.0, math.exp(blended))


def _bh_reject_flags(p_values: np.ndarray) -> np.ndarray:
    rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
    return rejected.astype(bool)


def _score_blend_weight(
    train_rows: list[FeatureRow],
    power_model: PowerLawModel,
    feature_model: FeatureModel,
    blend_weight: float,
) -> tuple[float, float, int, int]:
    focal_rows = [row for row in train_rows if not row.is_null_like]
    if not focal_rows:
        return float("-inf"), float("-inf"), 0, 0

    raw_perm_all: list[np.ndarray] = []
    raw_blend_all: list[np.ndarray] = []
    bh_perm_all: list[np.ndarray] = []
    bh_blend_all: list[np.ndarray] = []

    for case_name in sorted({row.case_name for row in focal_rows}):
        case_rows = [row for row in focal_rows if row.case_name == case_name]
        p_perm = np.array([row.p_perm for row in case_rows], dtype=np.float64)
        p_blend = np.array(
            [
                _predict_p_value(
                    row.t_obs,
                    row.k,
                    _blend_c(power_model.predict(row), feature_model.predict(row), blend_weight),
                )
                for row in case_rows
            ],
            dtype=np.float64,
        )
        raw_perm_all.append(p_perm < config.SIBLING_ALPHA)
        raw_blend_all.append(p_blend < config.SIBLING_ALPHA)
        bh_perm_all.append(_bh_reject_flags(p_perm))
        bh_blend_all.append(_bh_reject_flags(p_blend))

    raw_perm = np.concatenate(raw_perm_all)
    raw_blend = np.concatenate(raw_blend_all)
    bh_perm = np.concatenate(bh_perm_all)
    bh_blend = np.concatenate(bh_blend_all)
    bh_agreement = float(np.mean(bh_perm == bh_blend))
    raw_agreement = float(np.mean(raw_perm == raw_blend))
    bh_false_positives = int(np.sum(~bh_perm & bh_blend))
    raw_false_positives = int(np.sum(~raw_perm & raw_blend))
    return bh_agreement, raw_agreement, bh_false_positives, raw_false_positives


def select_blend_weight(
    train_rows: list[FeatureRow],
    power_model: PowerLawModel,
    feature_model: FeatureModel,
) -> float:
    candidate_weights = np.linspace(0.0, 1.0, 9)
    scored_candidates: list[tuple[float, float, int, int, float]] = []
    for weight in candidate_weights:
        bh_agreement, raw_agreement, bh_fp, raw_fp = _score_blend_weight(
            train_rows,
            power_model,
            feature_model,
            float(weight),
        )
        scored_candidates.append((bh_agreement, raw_agreement, -bh_fp, -raw_fp, float(weight)))

    scored_candidates.sort(reverse=True)
    return scored_candidates[0][-1]


def evaluate_case(
    case_name: str,
    rows: list[FeatureRow],
    power_model: PowerLawModel,
    feature_model: FeatureModel,
    blend_weight: float,
) -> CaseEvaluation:
    focal_rows = [row for row in rows if row.case_name == case_name and not row.is_null_like]
    if not focal_rows:
        return CaseEvaluation(
            case_name=case_name,
            blend_weight=blend_weight,
            n_focal=0,
            raw_perm_rejects=0,
            raw_global_rejects=0,
            raw_power_rejects=0,
            raw_v2_rejects=0,
            bh_perm_rejects=0,
            bh_global_rejects=0,
            bh_power_rejects=0,
            bh_v2_rejects=0,
            raw_agreement_global=float("nan"),
            raw_agreement_power=float("nan"),
            raw_agreement_v2=float("nan"),
            bh_agreement_global=float("nan"),
            bh_agreement_power=float("nan"),
            bh_agreement_v2=float("nan"),
        )

    p_perm = np.array([row.p_perm for row in focal_rows], dtype=np.float64)
    p_global = np.array([row.p_global for row in focal_rows], dtype=np.float64)
    p_power = np.array([
        _predict_p_value(row.t_obs, row.k, power_model.predict(row)) for row in focal_rows
    ])
    p_v2 = np.array([
        _predict_p_value(
            row.t_obs,
            row.k,
            _blend_c(power_model.predict(row), feature_model.predict(row), blend_weight),
        )
        for row in focal_rows
    ])

    alpha = config.SIBLING_ALPHA
    raw_perm = p_perm < alpha
    raw_global = p_global < alpha
    raw_power = p_power < alpha
    raw_v2 = p_v2 < alpha

    bh_perm = _bh_reject_flags(p_perm)
    bh_global = _bh_reject_flags(p_global)
    bh_power = _bh_reject_flags(p_power)
    bh_v2 = _bh_reject_flags(p_v2)

    return CaseEvaluation(
        case_name=case_name,
        blend_weight=blend_weight,
        n_focal=len(focal_rows),
        raw_perm_rejects=int(raw_perm.sum()),
        raw_global_rejects=int(raw_global.sum()),
        raw_power_rejects=int(raw_power.sum()),
        raw_v2_rejects=int(raw_v2.sum()),
        bh_perm_rejects=int(bh_perm.sum()),
        bh_global_rejects=int(bh_global.sum()),
        bh_power_rejects=int(bh_power.sum()),
        bh_v2_rejects=int(bh_v2.sum()),
        raw_agreement_global=float(np.mean(raw_perm == raw_global)),
        raw_agreement_power=float(np.mean(raw_perm == raw_power)),
        raw_agreement_v2=float(np.mean(raw_perm == raw_v2)),
        bh_agreement_global=float(np.mean(bh_perm == bh_global)),
        bh_agreement_power=float(np.mean(bh_perm == bh_power)),
        bh_agreement_v2=float(np.mean(bh_perm == bh_v2)),
    )


def evaluate_loco(rows: list[FeatureRow]) -> None:
    print("\n" + "=" * 100)
    print("LEAVE-ONE-CASE-OUT EVALUATION: Global ĉ vs pooled power-law vs feature model v2")
    print("=" * 100)

    all_focal = [row for row in rows if not row.is_null_like]
    c_perm_all = np.array([row.c_perm for row in all_focal], dtype=np.float64)
    c_global_all = np.array([row.c_global for row in all_focal], dtype=np.float64)
    c_power_all: list[float] = []
    c_v2_all: list[float] = []

    raw_perm_flags: list[np.ndarray] = []
    raw_global_flags: list[np.ndarray] = []
    raw_power_flags: list[np.ndarray] = []
    raw_v2_flags: list[np.ndarray] = []
    bh_perm_flags: list[np.ndarray] = []
    bh_global_flags: list[np.ndarray] = []
    bh_power_flags: list[np.ndarray] = []
    bh_v2_flags: list[np.ndarray] = []

    case_evaluations: list[CaseEvaluation] = []

    for case_name in DIAGNOSTIC_CASES:
        train_null_rows = [row for row in rows if row.case_name != case_name and row.is_null_like]
        train_rows = [row for row in rows if row.case_name != case_name]
        if not train_null_rows:
            raise ValueError(f"No training rows available for held-out case {case_name!r}.")

        power_model = fit_pooled_power_law(train_null_rows)
        feature_model = fit_feature_model(train_null_rows)
        blend_weight = select_blend_weight(train_rows, power_model, feature_model)
        case_eval = evaluate_case(case_name, rows, power_model, feature_model, blend_weight)
        case_evaluations.append(case_eval)

        test_rows = [row for row in rows if row.case_name == case_name and not row.is_null_like]
        if test_rows:
            p_perm = np.array([row.p_perm for row in test_rows], dtype=np.float64)
            p_global = np.array([row.p_global for row in test_rows], dtype=np.float64)
            p_power = np.array([
                _predict_p_value(row.t_obs, row.k, power_model.predict(row)) for row in test_rows
            ])
            p_v2 = np.array([
                _predict_p_value(
                    row.t_obs,
                    row.k,
                    _blend_c(power_model.predict(row), feature_model.predict(row), blend_weight),
                )
                for row in test_rows
            ])
            c_power_all.extend(power_model.predict(row) for row in test_rows)
            c_v2_all.extend(
                _blend_c(power_model.predict(row), feature_model.predict(row), blend_weight)
                for row in test_rows
            )

            raw_perm_flags.append(p_perm < config.SIBLING_ALPHA)
            raw_global_flags.append(p_global < config.SIBLING_ALPHA)
            raw_power_flags.append(p_power < config.SIBLING_ALPHA)
            raw_v2_flags.append(p_v2 < config.SIBLING_ALPHA)
            bh_perm_flags.append(_bh_reject_flags(p_perm))
            bh_global_flags.append(_bh_reject_flags(p_global))
            bh_power_flags.append(_bh_reject_flags(p_power))
            bh_v2_flags.append(_bh_reject_flags(p_v2))

        print(
            f"{case_name:<28} focal={case_eval.n_focal:>3} | "
            f"w={case_eval.blend_weight:>4.2f} | "
            f"raw agree G/P/V2 = {case_eval.raw_agreement_global:>5.1%} / "
            f"{case_eval.raw_agreement_power:>5.1%} / {case_eval.raw_agreement_v2:>5.1%} | "
            f"BH agree G/P/V2 = {case_eval.bh_agreement_global:>5.1%} / "
            f"{case_eval.bh_agreement_power:>5.1%} / {case_eval.bh_agreement_v2:>5.1%}"
        )

    raw_perm = np.concatenate(raw_perm_flags) if raw_perm_flags else np.array([], dtype=bool)
    raw_global = np.concatenate(raw_global_flags) if raw_global_flags else np.array([], dtype=bool)
    raw_power = np.concatenate(raw_power_flags) if raw_power_flags else np.array([], dtype=bool)
    raw_v2 = np.concatenate(raw_v2_flags) if raw_v2_flags else np.array([], dtype=bool)
    bh_perm = np.concatenate(bh_perm_flags) if bh_perm_flags else np.array([], dtype=bool)
    bh_global = np.concatenate(bh_global_flags) if bh_global_flags else np.array([], dtype=bool)
    bh_power = np.concatenate(bh_power_flags) if bh_power_flags else np.array([], dtype=bool)
    bh_v2 = np.concatenate(bh_v2_flags) if bh_v2_flags else np.array([], dtype=bool)

    print("\n" + "-" * 100)
    print(f"Total focal nodes: {len(all_focal)}")
    print(
        f"Raw agreement with permutation: "
        f"global={np.mean(raw_perm == raw_global):.1%}, "
        f"power-law={np.mean(raw_perm == raw_power):.1%}, "
        f"v2={np.mean(raw_perm == raw_v2):.1%}"
    )
    print(
        f"BH agreement with permutation: "
        f"global={np.mean(bh_perm == bh_global):.1%}, "
        f"power-law={np.mean(bh_perm == bh_power):.1%}, "
        f"v2={np.mean(bh_perm == bh_v2):.1%}"
    )
    print(
        f"Raw false negatives: global={np.sum(raw_perm & ~raw_global)}, "
        f"power-law={np.sum(raw_perm & ~raw_power)}, v2={np.sum(raw_perm & ~raw_v2)}"
    )
    print(
        f"Raw false positives: global={np.sum(~raw_perm & raw_global)}, "
        f"power-law={np.sum(~raw_perm & raw_power)}, v2={np.sum(~raw_perm & raw_v2)}"
    )
    print(
        f"BH false negatives: global={np.sum(bh_perm & ~bh_global)}, "
        f"power-law={np.sum(bh_perm & ~bh_power)}, v2={np.sum(bh_perm & ~bh_v2)}"
    )
    print(
        f"BH false positives: global={np.sum(~bh_perm & bh_global)}, "
        f"power-law={np.sum(~bh_perm & bh_power)}, v2={np.sum(~bh_perm & bh_v2)}"
    )

    c_power = np.array(c_power_all, dtype=np.float64)
    c_v2 = np.array(c_v2_all, dtype=np.float64)
    rho_global, _ = spearmanr(c_global_all, c_perm_all)
    rho_power, _ = spearmanr(c_power, c_perm_all)
    rho_v2, _ = spearmanr(c_v2, c_perm_all)
    print("\n" + "-" * 100)
    print("c prediction vs permutation (focal rows only):")
    print(f"  Spearman ρ: global={rho_global:.3f}, power-law={rho_power:.3f}, v2={rho_v2:.3f}")
    print(
        f"  Median c_pred/c_perm: global={np.median(c_global_all / c_perm_all):.3f}, "
        f"power-law={np.median(c_power / c_perm_all):.3f}, "
        f"v2={np.median(c_v2 / c_perm_all):.3f}"
    )
    print(
        f"  MAE: global={np.mean(np.abs(c_global_all - c_perm_all)):.3f}, "
        f"power-law={np.mean(np.abs(c_power - c_perm_all)):.3f}, "
        f"v2={np.mean(np.abs(c_v2 - c_perm_all)):.3f}"
    )


def main() -> None:
    print(
        f"Config: METHOD={config.SIBLING_TEST_METHOD}, "
        f"SIBLING_ALPHA={config.SIBLING_ALPHA}, EDGE_ALPHA={config.EDGE_ALPHA}"
    )
    print("V2 model: pooled leave-one-case-out log-linear feature model")
    print(f"Cases: {len(DIAGNOSTIC_CASES)}\n")

    rows = build_feature_rows(DIAGNOSTIC_CASES)
    evaluate_loco(rows)


if __name__ == "__main__":
    main()
