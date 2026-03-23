"""Experiment 31 — Oracle decision dataset and equation discovery.

This experiment builds a row-level dataset of sibling-test nodes using the
permutation benchmark as an offline oracle, then fits simple interpretable
equations that describe:

1. The gap between node-local permutation calibration and the current global
   c-hat.
2. The conditions under which the permutation oracle wants a split but the
   current global calibration does not.

Outputs:
- a row-level CSV for inspection
- a disagreement-only CSV with fitted probabilities for the interesting nodes
- a ranked false-global-split report using the family-held-out probability
- a compact markdown report summarizing the top ranked false-global splits by family
- a case-family portability CSV using leave-one-family-out evaluation
- a Fourier summary CSV for the calibration-gap signal and disagreement signal
- saved spectra plots per family for the Fourier diagnostics
- an OLS equation for log(c_perm_mean / c_global)
- a logistic equation for oracle-vs-global BH disagreement
- a shallow decision tree rule summary for the same disagreement target

This is an offline analysis only. It does not modify production code.

Usage:
    python debug_scripts/enhancement_lab/exp31_oracle_policy_equations.py
    python debug_scripts/enhancement_lab/exp31_oracle_policy_equations.py --n-permutations 199
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LAB))

from exp29_low_ess_permutation_benchmark import discover_low_ess_cases  # noqa: E402
from exp_parametric_inflation import DIAGNOSTIC_CASES, permutation_c  # noqa: E402
from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_child_pca_projections,
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)


@dataclass(frozen=True)
class FourierSummary:
    case_name: str
    case_family: str
    signal_name: str
    n_points: int
    dominant_frequency: float
    dominant_period_nodes: float
    low_band_power_fraction: float
    spectral_entropy: float


@dataclass(frozen=True)
class ExtremeNoiseDiagnostic:
    focused_rows: pd.DataFrame
    case_summary: pd.DataFrame
    driver_summary: pd.DataFrame
    driver_depth_summary: pd.DataFrame
    depth_summary: pd.DataFrame
    node_drivers: pd.DataFrame
    highd_trace: pd.DataFrame
    highd_ablation: pd.DataFrame


@dataclass(frozen=True)
class GuardBenchmark:
    hit_rows: pd.DataFrame
    subfamily_summary: pd.DataFrame
    case_summary: pd.DataFrame


def _predict_p_chi2(stat: float, k: int, c_hat: float) -> float:
    adjusted_stat = stat / max(c_hat, 1.0)
    return float(chi2.sf(adjusted_stat, df=k))


def _bh_flags(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return np.array([], dtype=bool)
    rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
    return rejected.astype(bool)


def _infer_case_family(case_name: str, category: str | None) -> str:
    label = f"{case_name} {category or ''}".lower()
    if "gaussian" in label or "gauss" in label:
        return "gaussian"
    if "binary" in label or "sparse" in label:
        return "binary"
    if "phylo" in label:
        return "phylo"
    if "cat" in label:
        return "categorical"
    if "sbm" in label:
        return "sbm"
    if "overlap" in label:
        return "overlap"
    return (category or "other").lower().replace(" ", "_")


def _infer_case_subfamily(case_name: str, category: str | None, family: str) -> str:
    label = f"{case_name} {category or ''}".lower()
    if family != "gaussian":
        return family
    if "extreme_noise" in label:
        return "gaussian_extreme_noise"
    if "overlap" in label:
        return "gaussian_overlap"
    if "null" in label:
        return "gaussian_null"
    if "clear" in label:
        return "gaussian_clear"
    if "moderate" in label:
        return "gaussian_moderate"
    if "noisy" in label:
        return "gaussian_noisy"
    if "outlier" in label:
        return "gaussian_outlier"
    if "dimensionality" in label:
        return "gaussian_dimensionality"
    return "gaussian_other"


def _seed_case_names_by_family(limit_per_family: int = 2) -> list[str]:
    seeded: list[str] = []
    family_counts = {"gaussian": 0, "binary": 0, "phylo": 0}
    for case in get_default_test_cases():
        case_name = str(case["name"])
        family = _infer_case_family(case_name, str(case.get("category", "")))
        if family not in family_counts:
            continue
        if family_counts[family] >= limit_per_family:
            continue
        seeded.append(case_name)
        family_counts[family] += 1
    return seeded


def _apply_casewise_bh(frame: pd.DataFrame, column: str, output_column: str) -> None:
    flags = np.zeros(len(frame), dtype=bool)
    for _case_name, case_frame in frame.groupby("case_name", sort=False):
        indices = case_frame.index.to_numpy(dtype=np.int64)
        p_values = case_frame[column].to_numpy(dtype=np.float64)
        flags[indices] = _bh_flags(p_values)
    frame[output_column] = flags


def _case_names(top_k: int) -> list[str]:
    low_ess_names = [
        case.name
        for case in discover_low_ess_cases(
            top_k=top_k,
            require_focal=True,
            require_null_like=True,
        )
    ]
    ordered: list[str] = []
    for case_name in list(DIAGNOSTIC_CASES) + _seed_case_names_by_family() + low_ess_names:
        if case_name not in ordered:
            ordered.append(case_name)
    return ordered


def _collect_case_frame(case_name: str, *, n_permutations: int) -> pd.DataFrame:
    tree, data_df, _y_true, tc = build_tree_and_data(case_name)
    run_decomposition(tree, data_df)
    stats_df = tree.stats_df
    if stats_df is None:
        raise ValueError(f"stats_df not populated for case {case_name}")

    audit = stats_df.attrs.get("sibling_divergence_audit", {})
    diagnostics = audit.get("diagnostics", {})
    c_global = float(audit.get("global_inflation_factor", 1.0))
    effective_n = float(diagnostics.get("effective_n", 0.0))
    case_category = str(tc.get("category", "unknown"))
    case_family = _infer_case_family(case_name, case_category)
    case_subfamily = _infer_case_subfamily(case_name, case_category, case_family)

    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    sibling_dims = derive_sibling_spectral_dims(tree, stats_df)
    sibling_pca, sibling_eig = derive_sibling_pca_projections(stats_df, sibling_dims)
    sibling_child_pca = derive_sibling_child_pca_projections(tree, stats_df, sibling_dims)
    records, _ = collect_sibling_pair_records(
        tree,
        stats_df,
        mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=sibling_pca,
        pca_eigenvalues=sibling_eig,
        child_pca_projections=sibling_child_pca,
        whitening=config.SIBLING_WHITENING,
    )

    root = tree.root()
    rows: list[dict[str, float | int | str | bool]] = []
    for record in records:
        if record.degrees_of_freedom <= 0 or not np.isfinite(record.stat):
            continue
        children = list(tree.successors(record.parent))
        if len(children) != 2:
            continue
        left, right = children
        pca_proj = sibling_pca.get(record.parent) if sibling_pca else None
        pca_eig = sibling_eig.get(record.parent) if sibling_eig else None
        child_pca = sibling_child_pca.get(record.parent) if sibling_child_pca else None
        bl_left = tree.edges[record.parent, left].get("branch_length")
        bl_right = tree.edges[record.parent, right].get("branch_length")
        branch_length_sum = 0.0
        perm_branch_length_sum = None
        if mean_bl is not None and bl_left is not None and bl_right is not None:
            branch_length_sum = float(bl_left) + float(bl_right)
            perm_branch_length_sum = branch_length_sum if branch_length_sum > 0.0 else None

        c_perm_mean, c_perm_median, p_perm = permutation_c(
            tree,
            record.parent,
            left,
            right,
            data_df,
            mean_branch_length=mean_bl,
            branch_length_sum=perm_branch_length_sum,
            spectral_k=int(record.degrees_of_freedom),
            pca_projection=pca_proj,
            pca_eigenvalues=pca_eig,
            child_pca_projections=child_pca,
            whitening=config.SIBLING_WHITENING,
            n_permutations=n_permutations,
        )

        p_global = _predict_p_chi2(float(record.stat), int(record.degrees_of_freedom), c_global)
        depth = int(nx.shortest_path_length(tree, root, record.parent))
        edge_weight = float(record.edge_weight)
        stability = 1.0 / (1.0 + abs(math.log(max(c_perm_mean, 1.0) / max(c_perm_median, 1.0))))
        rows.append(
            {
                "case_name": case_name,
                "case_category": case_category,
                "case_family": case_family,
                "case_subfamily": case_subfamily,
                "parent": str(record.parent),
                "depth": depth,
                "n_parent": int(record.n_parent),
                "k": int(record.degrees_of_freedom),
                "edge_weight": edge_weight,
                "branch_length_sum": branch_length_sum,
                "effective_n": effective_n,
                "is_null_like": bool(record.is_null_like),
                "c_global": c_global,
                "c_perm_mean": float(max(c_perm_mean, 1.0)),
                "c_perm_median": float(max(c_perm_median, 1.0)),
                "stability": float(stability),
                "stat": float(record.stat),
                "p_global": p_global,
                "p_perm": float(p_perm),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.sort_values(["depth", "n_parent", "parent"], kind="stable").reset_index(drop=True)

    frame = frame.assign(
        node_order=lambda df: np.arange(len(df), dtype=np.int64),
        n_null_case=lambda df: int(df["is_null_like"].sum()),
        n_focal_case=lambda df: int((~df["is_null_like"]).sum()),
        log_n_parent=lambda df: np.log(np.maximum(df["n_parent"].to_numpy(dtype=np.float64), 1.0)),
        log_k=lambda df: np.log(np.maximum(df["k"].to_numpy(dtype=np.float64), 1.0)),
        log_branch=lambda df: np.log1p(
            np.maximum(df["branch_length_sum"].to_numpy(dtype=np.float64), 0.0)
        ),
        log_c_ratio=lambda df: np.log(
            df["c_perm_mean"].to_numpy(dtype=np.float64)
            / np.maximum(df["c_global"].to_numpy(dtype=np.float64), 1.0)
        ),
        gap_log=lambda df: np.abs(
            np.log(
                df["c_perm_mean"].to_numpy(dtype=np.float64)
                / np.maximum(df["c_global"].to_numpy(dtype=np.float64), 1.0)
            )
        ),
        global_raw_reject=lambda df: df["p_global"] < config.SIBLING_ALPHA,
        perm_raw_reject=lambda df: df["p_perm"] < config.SIBLING_ALPHA,
    )
    _apply_casewise_bh(frame, "p_global", "global_bh_reject")
    _apply_casewise_bh(frame, "p_perm", "perm_bh_reject")
    frame["oracle_prefers_local_bh"] = frame["perm_bh_reject"] & (~frame["global_bh_reject"])
    frame["oracle_prefers_global_bh"] = frame["global_bh_reject"] & (~frame["perm_bh_reject"])
    frame["any_bh_disagreement"] = (
        frame["oracle_prefers_local_bh"] | frame["oracle_prefers_global_bh"]
    )
    frame["signed_bh_disagreement"] = frame["oracle_prefers_local_bh"].astype(np.int64) - frame[
        "oracle_prefers_global_bh"
    ].astype(np.int64)
    return frame


def _fit_ols_equation(frame: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    predictors = _prepare_design_matrix(
        frame,
        ["log_n_parent", "log_k", "edge_weight", "log_branch", "depth", "stability"],
    )
    model = sm.OLS(frame["log_c_ratio"], predictors)
    return model.fit()


def _usable_feature_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        if float(np.std(values)) <= 1e-12:
            continue
        kept.append(column)
    return kept


def _prepare_design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    kept = _usable_feature_columns(frame, columns)
    predictors = frame[kept].copy()
    return sm.add_constant(predictors, has_constant="add")


def _fit_logit_equation(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> sm.discrete.discrete_model.BinaryResultsWrapper | None:
    focal = frame.loc[~frame["is_null_like"]].copy()
    positives = int(focal[target_column].sum())
    negatives = int((~focal[target_column]).sum())
    if positives == 0 or negatives == 0:
        return None
    predictors = _prepare_design_matrix(
        focal,
        ["log_n_parent", "log_k", "edge_weight", "log_branch", "depth", "stability", "gap_log"],
    )
    target = focal[target_column].astype(int)
    model = sm.Logit(target, predictors)
    try:
        return model.fit(disp=False)
    except Exception:
        return model.fit_regularized(disp=False, alpha=1e-4)


def _fit_tree_rule(
    frame: pd.DataFrame, *, target_column: str
) -> tuple[str, float] | tuple[None, None]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    positives = int(focal[target_column].sum())
    negatives = int((~focal[target_column]).sum())
    if positives == 0 or negatives == 0 or focal["case_name"].nunique() < 2:
        return None, None

    feature_names = [
        "log_n_parent",
        "log_k",
        "edge_weight",
        "log_branch",
        "depth",
        "stability",
        "gap_log",
    ]
    usable_features = _usable_feature_columns(focal, feature_names)
    x = focal[usable_features].to_numpy(dtype=np.float64)
    y = focal[target_column].astype(int).to_numpy(dtype=np.int64)
    groups = focal["case_name"].to_numpy()
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    scores = cross_val_score(tree, x, y, groups=groups, cv=cv, scoring="balanced_accuracy")
    tree.fit(x, y)
    return export_text(tree, feature_names=usable_features), float(np.mean(scores))


def _fit_tree_rule_with_groups(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
) -> tuple[str, float] | tuple[None, None]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    positives = int(focal[target_column].sum())
    negatives = int((~focal[target_column]).sum())
    if positives == 0 or negatives == 0 or focal[group_column].nunique() < 2:
        return None, None

    feature_names = [
        "log_n_parent",
        "log_k",
        "edge_weight",
        "log_branch",
        "depth",
        "stability",
        "gap_log",
    ]
    usable_features = _usable_feature_columns(focal, feature_names)
    x = focal[usable_features].to_numpy(dtype=np.float64)
    y = focal[target_column].astype(int).to_numpy(dtype=np.int64)
    groups = focal[group_column].to_numpy()
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    scores = cross_val_score(tree, x, y, groups=groups, cv=cv, scoring="balanced_accuracy")
    tree.fit(x, y)
    return export_text(tree, feature_names=usable_features), float(np.mean(scores))


def _select_disagreement_target(frame: pd.DataFrame) -> str | None:
    focal = frame.loc[~frame["is_null_like"]]
    for column in ("oracle_prefers_local_bh", "oracle_prefers_global_bh"):
        positives = int(focal[column].sum())
        negatives = int((~focal[column]).sum())
        if positives > 0 and negatives > 0:
            return column
    return None


def _predict_statsmodels_probabilities(
    fitted_model: sm.discrete.discrete_model.BinaryResultsWrapper | None,
    frame: pd.DataFrame,
) -> pd.Series:
    probabilities = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if fitted_model is None or frame.empty:
        return probabilities
    exog_names = [name for name in fitted_model.model.exog_names if name != "const"]
    predictors = sm.add_constant(frame[exog_names].copy(), has_constant="add")
    probabilities.loc[frame.index] = fitted_model.predict(predictors)
    return probabilities


def _fit_family_portability_model(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.Series, pd.DataFrame]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    probabilities = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if focal.empty or focal["case_family"].nunique() < 2:
        return probabilities, pd.DataFrame()

    feature_names = _usable_feature_columns(
        focal,
        ["log_n_parent", "log_k", "edge_weight", "log_branch", "depth", "stability", "gap_log"],
    )
    if not feature_names:
        return probabilities, pd.DataFrame()

    x = focal[feature_names].to_numpy(dtype=np.float64)
    y = focal[target_column].astype(int).to_numpy(dtype=np.int64)
    groups = focal["case_family"].to_numpy()
    splitter = LeaveOneGroupOut()
    rows: list[dict[str, float | int | str]] = []

    for train_idx, test_idx in splitter.split(x, y, groups):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        test_family = str(groups[test_idx][0])
        if len(np.unique(y_train)) < 2:
            continue

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(np.int64)
        test_indices = focal.iloc[test_idx].index
        probabilities.loc[test_indices] = test_prob

        roc_auc = math.nan
        if len(np.unique(y_test)) > 1:
            roc_auc = float(roc_auc_score(y_test, test_prob))

        rows.append(
            {
                "held_out_family": test_family,
                "n_test": int(len(test_idx)),
                "positive_rate": float(np.mean(y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
                "roc_auc": roc_auc,
            }
        )

    summary = pd.DataFrame(rows)
    return probabilities, summary


def _fit_group_portability_model(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    positive_label: str,
) -> tuple[pd.Series, pd.DataFrame]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    probabilities = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if focal.empty or focal[group_column].nunique() < 2:
        return probabilities, pd.DataFrame()

    feature_names = _usable_feature_columns(
        focal,
        ["log_n_parent", "log_k", "edge_weight", "log_branch", "depth", "stability", "gap_log"],
    )
    if not feature_names:
        return probabilities, pd.DataFrame()

    x = focal[feature_names].to_numpy(dtype=np.float64)
    y = focal[target_column].astype(int).to_numpy(dtype=np.int64)
    groups = focal[group_column].to_numpy()
    splitter = LeaveOneGroupOut()
    rows: list[dict[str, float | int | str]] = []

    for train_idx, test_idx in splitter.split(x, y, groups):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        held_out = str(groups[test_idx][0])
        if len(np.unique(y_train)) < 2:
            continue

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(np.int64)
        test_indices = focal.iloc[test_idx].index
        probabilities.loc[test_indices] = test_prob

        roc_auc = math.nan
        if len(np.unique(y_test)) > 1:
            roc_auc = float(roc_auc_score(y_test, test_prob))

        rows.append(
            {
                positive_label: held_out,
                "n_test": int(len(test_idx)),
                "positive_rate": float(np.mean(y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
                "roc_auc": roc_auc,
            }
        )

    return probabilities, pd.DataFrame(rows)


def _normalized_spectral_entropy(power: np.ndarray) -> float:
    if len(power) == 0:
        return 0.0
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return 0.0
    probs = power / total_power
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(entropy / math.log(len(probs))) if len(probs) > 1 else 0.0


def _compute_fft_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values.astype(np.float64) - float(np.mean(values))
    n_points = int(len(centered))
    if n_points < 4:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    fft_values = np.fft.rfft(centered)
    power = np.abs(fft_values[1:]) ** 2
    frequencies = np.fft.rfftfreq(n_points, d=1.0)[1:]
    if len(power) == 0 or float(np.sum(power)) <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    normalized_power = power / np.sum(power)
    return frequencies.astype(np.float64), normalized_power.astype(np.float64)


def _summarize_fft(
    case_name: str, case_family: str, signal_name: str, values: np.ndarray
) -> FourierSummary:
    n_points = int(len(values))
    if n_points < 4:
        return FourierSummary(
            case_name=case_name,
            case_family=case_family,
            signal_name=signal_name,
            n_points=n_points,
            dominant_frequency=math.nan,
            dominant_period_nodes=math.nan,
            low_band_power_fraction=math.nan,
            spectral_entropy=math.nan,
        )

    frequencies, normalized_power = _compute_fft_components(values)
    if len(normalized_power) == 0:
        return FourierSummary(
            case_name=case_name,
            case_family=case_family,
            signal_name=signal_name,
            n_points=n_points,
            dominant_frequency=0.0,
            dominant_period_nodes=math.inf,
            low_band_power_fraction=0.0,
            spectral_entropy=0.0,
        )

    dominant_idx = int(np.argmax(normalized_power))
    dominant_frequency = float(frequencies[dominant_idx])
    dominant_period_nodes = float(
        math.inf if dominant_frequency <= 0.0 else 1.0 / dominant_frequency
    )
    low_band_end = max(1, len(normalized_power) // 4)
    low_band_fraction = float(np.sum(normalized_power[:low_band_end]))
    return FourierSummary(
        case_name=case_name,
        case_family=case_family,
        signal_name=signal_name,
        n_points=n_points,
        dominant_frequency=dominant_frequency,
        dominant_period_nodes=dominant_period_nodes,
        low_band_power_fraction=low_band_fraction,
        spectral_entropy=_normalized_spectral_entropy(normalized_power),
    )


def _compute_fourier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    focal = frame.loc[~frame["is_null_like"]].copy()
    rows: list[dict[str, float | int | str]] = []
    for case_name, case_frame in focal.groupby("case_name", sort=False):
        ordered = case_frame.sort_values(
            ["node_order", "depth", "n_parent", "parent"], kind="stable"
        )
        case_family = str(ordered["case_family"].iloc[0])
        signals = {
            "log_c_ratio": ordered["log_c_ratio"].to_numpy(dtype=np.float64),
            "signed_bh_disagreement": ordered["signed_bh_disagreement"].to_numpy(dtype=np.float64),
        }
        for signal_name, values in signals.items():
            rows.append(_summarize_fft(case_name, case_family, signal_name, values).__dict__)
    return pd.DataFrame(rows)


def _save_family_spectra_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    frequency_grid = np.linspace(0.0, 0.5, 256, dtype=np.float64)
    signal_names = ("log_c_ratio", "signed_bh_disagreement")

    for case_family, family_frame in focal.groupby("case_family", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        fig.suptitle(f"Experiment 31 Fourier Spectra — {case_family}")

        for axis, signal_name in zip(axes, signal_names, strict=False):
            stacked: list[np.ndarray] = []
            n_cases_with_signal = 0
            for _case_name, case_frame in family_frame.groupby("case_name", sort=False):
                ordered = case_frame.sort_values(
                    ["node_order", "depth", "n_parent", "parent"], kind="stable"
                )
                frequencies, normalized_power = _compute_fft_components(
                    ordered[signal_name].to_numpy(dtype=np.float64)
                )
                if len(normalized_power) == 0:
                    continue
                interpolated = np.interp(
                    frequency_grid,
                    frequencies,
                    normalized_power,
                    left=0.0,
                    right=0.0,
                )
                stacked.append(interpolated)
                n_cases_with_signal += 1
                axis.plot(
                    frequency_grid,
                    interpolated,
                    color="#8da0cb",
                    alpha=0.22,
                    linewidth=0.9,
                )
                dominant_frequency = float(frequencies[int(np.argmax(normalized_power))])
                axis.axvline(dominant_frequency, color="#8da0cb", alpha=0.06, linewidth=0.8)

            if stacked:
                mean_spectrum = np.mean(np.vstack(stacked), axis=0)
                dominant_frequency = float(frequency_grid[int(np.argmax(mean_spectrum))])
                axis.plot(
                    frequency_grid,
                    mean_spectrum,
                    color="#1b4f72",
                    linewidth=2.2,
                    label="family mean",
                )
                axis.fill_between(
                    frequency_grid,
                    0.0,
                    mean_spectrum,
                    color="#1b4f72",
                    alpha=0.12,
                )
                axis.axvspan(0.0, 0.125, color="#f6c85f", alpha=0.12)
                axis.axvline(dominant_frequency, color="#d95f02", linestyle="--", linewidth=1.2)
                axis.text(
                    0.98,
                    0.95,
                    f"cases={n_cases_with_signal}\ndom={dominant_frequency:.3f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No usable spectrum",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )

            axis.set_title(signal_name.replace("_", " "))
            axis.set_xlabel("frequency (cycles per node)")
            axis.set_ylabel("normalized power")
            axis.set_xlim(0.0, 0.5)
            axis.grid(alpha=0.2)

        plot_path = output_dir / f"{case_family}_spectra.png"
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def _build_ranked_false_global_split_report(frame: pd.DataFrame) -> pd.DataFrame:
    disagreement_rows = frame.loc[frame["oracle_prefers_global_bh"]].copy()
    if disagreement_rows.empty:
        return disagreement_rows

    ranked = disagreement_rows.assign(
        family_probability_rank=lambda df: df["family_portability_probability"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(np.int64),
        probability_gap=lambda df: df["family_portability_probability"]
        - df["statsmodels_fitted_probability"],
    )
    columns = [
        "family_probability_rank",
        "case_family",
        "case_name",
        "parent",
        "depth",
        "n_parent",
        "k",
        "edge_weight",
        "effective_n",
        "c_global",
        "c_perm_mean",
        "p_global",
        "p_perm",
        "gap_log",
        "stability",
        "statsmodels_fitted_probability",
        "family_portability_probability",
        "probability_gap",
    ]
    return ranked.sort_values(
        ["family_portability_probability", "statsmodels_fitted_probability", "p_global"],
        ascending=[False, False, True],
    )[columns]


def _write_ranked_false_global_markdown(
    report_frame: pd.DataFrame, output_path: Path, *, top_n: int = 20
) -> None:
    top_rows = report_frame.head(top_n).copy()
    lines: list[str] = [
        "# Experiment 31: Top False-Global Splits",
        "",
        f"Top {min(len(top_rows), top_n)} ranked rows from the family-held-out false-global-split report.",
        "",
    ]

    if top_rows.empty:
        lines.append("No false-global split rows were identified.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    family_counts = (
        top_rows.groupby("case_family", as_index=False)
        .agg(rows=("case_family", "size"))
        .sort_values(["rows", "case_family"], ascending=[False, True])
    )
    lines.extend(
        [
            "## Family Breakdown",
            "",
            "| Family | Rows In Top 20 |",
            "| --- | ---: |",
        ]
    )
    for row in family_counts.itertuples(index=False):
        lines.append(f"| {row.case_family} | {row.rows} |")

    for case_family, family_frame in top_rows.groupby("case_family", sort=False):
        lines.extend(
            [
                "",
                f"## {case_family.title()}",
                "",
                "| Rank | Case | Parent | Depth | n_parent | k | Family Prob | Model Prob | Gap | p_global | p_perm |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in family_frame.itertuples(index=False):
            lines.append(
                "| "
                f"{int(row.family_probability_rank)} | {row.case_name} | {row.parent} | {int(row.depth)} | "
                f"{int(row.n_parent)} | {int(row.k)} | {float(row.family_portability_probability):.6f} | "
                f"{float(row.statsmodels_fitted_probability):.6f} | {float(row.probability_gap):.6f} | "
                f"{float(row.p_global):.3e} | {float(row.p_perm):.3g} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_subtree(
    tree: nx.DiGraph,
    rows_by_parent: pd.DataFrame,
    node: str,
) -> tuple[int, int, float]:
    internal_nodes = {node, *nx.descendants(tree, node)}
    subtree_rows = rows_by_parent.loc[rows_by_parent.index.intersection(internal_nodes)]
    false_global = int(subtree_rows["oracle_prefers_global_bh"].sum()) if not subtree_rows.empty else 0
    row_count = int(len(subtree_rows))
    risk_sum = float(subtree_rows["risk_score"].sum()) if not subtree_rows.empty else 0.0
    return false_global, row_count, risk_sum


def _build_highd_failure_trace(frame: pd.DataFrame) -> pd.DataFrame:
    highd_rows = frame.loc[frame["case_name"] == "gauss_extreme_noise_highd"].copy()
    if highd_rows.empty:
        return pd.DataFrame()

    tree, data_df, _y_true, _tc = build_tree_and_data("gauss_extreme_noise_highd")
    run_decomposition(tree, data_df)
    root = str(tree.root())
    rows_by_parent = highd_rows.set_index("parent", drop=False)
    trace_rows: list[dict[str, float | int | str | bool]] = []
    current = root
    step = 0

    while True:
        if current not in rows_by_parent.index:
            break
        current_row = rows_by_parent.loc[current]
        children = list(tree.successors(current))
        if len(children) != 2:
            break

        child_summaries: list[tuple[str, int, int, float]] = []
        for child in children:
            false_global, row_count, risk_sum = _summarize_subtree(tree, rows_by_parent, str(child))
            child_summaries.append((str(child), false_global, row_count, risk_sum))

        child_summaries.sort(key=lambda item: (item[1], item[3], item[2], item[0]), reverse=True)
        chosen_child, chosen_false_global, chosen_rows, chosen_risk = child_summaries[0]
        sibling_child, sibling_false_global, sibling_rows, sibling_risk = child_summaries[1]

        trace_rows.append(
            {
                "step": step,
                "node": current,
                "depth": int(current_row["depth"]),
                "node_false_global": bool(current_row["oracle_prefers_global_bh"]),
                "node_driver_type": str(current_row["driver_type"]),
                "node_gap_log": float(current_row["gap_log"]),
                "node_risk_score": float(current_row["risk_score"]),
                "left_child": str(children[0]),
                "right_child": str(children[1]),
                "chosen_child": chosen_child,
                "chosen_child_false_global_subtree": chosen_false_global,
                "chosen_child_row_subtree": chosen_rows,
                "chosen_child_risk_subtree": chosen_risk,
                "sibling_child": sibling_child,
                "sibling_child_false_global_subtree": sibling_false_global,
                "sibling_child_row_subtree": sibling_rows,
                "sibling_child_risk_subtree": sibling_risk,
                "split_concentrates_downstream": chosen_false_global > sibling_false_global,
            }
        )

        if chosen_child not in rows_by_parent.index:
            break
        if chosen_false_global <= 0 and chosen_rows <= 0:
            break

        current = chosen_child
        step += 1

    return pd.DataFrame(trace_rows)


def _build_highd_ablation(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()

    total_false_global = int(trace.iloc[0]["chosen_child_false_global_subtree"])
    total_false_global += int(trace.iloc[0]["sibling_child_false_global_subtree"])
    total_false_global += int(trace.iloc[0]["node_false_global"])

    ablation = trace.assign(
        removed_false_global=lambda df: df["chosen_child_false_global_subtree"].astype(np.int64),
        remaining_false_global_if_cut_here=lambda df: (
            total_false_global
            - df["chosen_child_false_global_subtree"].astype(np.int64)
        ),
        removed_false_global_share=lambda df: df["chosen_child_false_global_subtree"].astype(np.float64)
        / max(total_false_global, 1),
        sibling_share_after_cut=lambda df: df["sibling_child_false_global_subtree"].astype(np.float64)
        / max(total_false_global, 1),
    ).loc[
        :,
        [
            "step",
            "node",
            "depth",
            "node_driver_type",
            "node_false_global",
            "chosen_child",
            "chosen_child_false_global_subtree",
            "sibling_child",
            "sibling_child_false_global_subtree",
            "removed_false_global",
            "removed_false_global_share",
            "remaining_false_global_if_cut_here",
            "sibling_share_after_cut",
            "split_concentrates_downstream",
        ],
    ]
    return ablation


def _build_gaussian_guard_counterfactual(frame: pd.DataFrame) -> GuardBenchmark:
    gaussian = frame.loc[(frame["case_family"] == "gaussian") & (~frame["is_null_like"])].copy()
    if gaussian.empty:
        empty = pd.DataFrame()
        return GuardBenchmark(hit_rows=empty, subfamily_summary=empty, case_summary=empty)

    edge_eps = 1e-12
    gaussian = gaussian.assign(
        guard_v1_hit=lambda df: (
            (df["depth"] <= 3)
            & (df["gap_log"] >= 4.0)
            & (df["edge_weight"] <= edge_eps)
            & (df["k"] == 1)
        ),
        guard_v1_driver=lambda df: np.select(
            [
                (df["depth"] <= 1) & (df["gap_log"] >= 4.0),
                df["edge_weight"] <= edge_eps,
                df["k"] == 1,
            ],
            [
                "top_level_high_gap",
                "zero_edge_weight",
                "k1_pair",
            ],
            default="other",
        ),
        guard_v1_reason=lambda df: np.where(
            (df["depth"] <= 3)
            & (df["gap_log"] >= 4.0)
            & (df["edge_weight"] <= edge_eps)
            & (df["k"] == 1),
            "depth<=3 & gap>=4 & edge≈0 & k=1",
            "",
        ),
    )

    hits = gaussian.loc[gaussian["guard_v1_hit"]].copy()
    if hits.empty:
        empty = pd.DataFrame()
        return GuardBenchmark(hit_rows=hits, subfamily_summary=empty, case_summary=empty)

    subfamily_totals = (
        gaussian.groupby("case_subfamily", as_index=False)
        .agg(
            focal_rows=("case_subfamily", "size"),
            false_global_total=("oracle_prefers_global_bh", "sum"),
        )
        .set_index("case_subfamily")
    )
    case_totals = (
        gaussian.groupby("case_name", as_index=False)
        .agg(
            focal_rows=("case_name", "size"),
            false_global_total=("oracle_prefers_global_bh", "sum"),
        )
        .set_index("case_name")
    )

    subfamily_summary = (
        hits.groupby("case_subfamily", as_index=False)
        .agg(
            guard_hits=("case_subfamily", "size"),
            false_global_guarded=("oracle_prefers_global_bh", "sum"),
            mean_gap_log=("gap_log", "mean"),
            max_depth=("depth", "max"),
        )
        .assign(
            focal_rows=lambda df: df["case_subfamily"].map(subfamily_totals["focal_rows"]),
            false_global_total=lambda df: df["case_subfamily"].map(
                subfamily_totals["false_global_total"]
            ),
        )
        .assign(
            collateral_hits=lambda df: df["guard_hits"] - df["false_global_guarded"],
            false_global_removed_share=lambda df: df["false_global_guarded"]
            / np.maximum(df["false_global_total"], 1),
            collateral_rate=lambda df: df["collateral_hits"] / np.maximum(df["guard_hits"], 1),
            remaining_false_global=lambda df: df["false_global_total"] - df["false_global_guarded"],
        )
        .sort_values(["false_global_guarded", "collateral_hits"], ascending=[False, True])
        .reset_index(drop=True)
    )

    case_summary = (
        hits.groupby(["case_subfamily", "case_name"], as_index=False)
        .agg(
            guard_hits=("case_name", "size"),
            false_global_guarded=("oracle_prefers_global_bh", "sum"),
            mean_gap_log=("gap_log", "mean"),
            min_depth=("depth", "min"),
            max_depth=("depth", "max"),
        )
        .assign(
            focal_rows=lambda df: df["case_name"].map(case_totals["focal_rows"]),
            false_global_total=lambda df: df["case_name"].map(case_totals["false_global_total"]),
        )
        .assign(
            collateral_hits=lambda df: df["guard_hits"] - df["false_global_guarded"],
            false_global_removed_share=lambda df: df["false_global_guarded"]
            / np.maximum(df["false_global_total"], 1),
            remaining_false_global=lambda df: df["false_global_total"] - df["false_global_guarded"],
        )
        .sort_values(["false_global_guarded", "collateral_hits"], ascending=[False, True])
        .reset_index(drop=True)
    )

    hits = hits.loc[
        :,
        [
            "case_subfamily",
            "case_name",
            "parent",
            "depth",
            "n_parent",
            "k",
            "edge_weight",
            "gap_log",
            "oracle_prefers_global_bh",
            "guard_v1_driver",
            "guard_v1_reason",
        ],
    ].sort_values(["oracle_prefers_global_bh", "gap_log", "depth"], ascending=[False, False, True])

    return GuardBenchmark(hit_rows=hits, subfamily_summary=subfamily_summary, case_summary=case_summary)


def _build_extreme_noise_diagnostic(frame: pd.DataFrame) -> ExtremeNoiseDiagnostic:
    focused = frame.loc[
        (frame["case_subfamily"] == "gaussian_extreme_noise") & (~frame["is_null_like"])
    ].copy()
    if focused.empty:
        empty = pd.DataFrame()
        return ExtremeNoiseDiagnostic(
            focused_rows=empty,
            case_summary=empty,
            driver_summary=empty,
            driver_depth_summary=empty,
            depth_summary=empty,
            node_drivers=empty,
            highd_trace=empty,
            highd_ablation=empty,
        )

    focused = focused.assign(
        risk_score=lambda df: (
            df["gaussian_subfamily_probability"].fillna(df["family_portability_probability"])
            .fillna(df["statsmodels_fitted_probability"])
            .fillna(0.0)
        )
        * (1.0 + df["gap_log"].to_numpy(dtype=np.float64))
        * (1.0 + df["oracle_prefers_global_bh"].astype(np.float64)),
        driver_type=lambda df: np.select(
            [
                (df["depth"] <= 1) & (df["gap_log"] >= 4.0),
                (df["depth"] >= 3) & (df["n_parent"] <= 4),
                df["edge_weight"] <= 1e-9,
                df["k"] == 1,
            ],
            [
                "top_level_high_gap",
                "tiny_deep_node",
                "zero_edge_weight",
                "k1_pair",
            ],
            default="mid_tree_gap",
        ),
    )

    case_summary = (
        focused.groupby("case_name", as_index=False)
        .agg(
            rows=("case_name", "size"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            disagreement_rate=("oracle_prefers_global_bh", "mean"),
            mean_gap_log=("gap_log", "mean"),
            max_gap_log=("gap_log", "max"),
            mean_n_parent=("n_parent", "mean"),
            max_risk_score=("risk_score", "max"),
            mean_family_probability=("family_portability_probability", "mean"),
            mean_subfamily_probability=("gaussian_subfamily_probability", "mean"),
        )
        .sort_values(
            ["global_prefers", "mean_gap_log", "max_risk_score"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    driver_summary = (
        focused.groupby("driver_type", as_index=False)
        .agg(
            rows=("driver_type", "size"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            disagreement_rate=("oracle_prefers_global_bh", "mean"),
            mean_gap_log=("gap_log", "mean"),
            mean_risk_score=("risk_score", "mean"),
        )
        .assign(
            false_global_share=lambda df: df["global_prefers"]
            / max(int(focused["oracle_prefers_global_bh"].sum()), 1)
        )
        .sort_values(["global_prefers", "mean_gap_log"], ascending=[False, False])
        .reset_index(drop=True)
    )

    driver_depth_summary = (
        focused.groupby(["driver_type", "depth"], as_index=False)
        .agg(
            rows=("depth", "size"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            disagreement_rate=("oracle_prefers_global_bh", "mean"),
            mean_gap_log=("gap_log", "mean"),
            median_n_parent=("n_parent", "median"),
            mean_risk_score=("risk_score", "mean"),
        )
        .assign(
            false_global_share=lambda df: df["global_prefers"]
            / max(int(focused["oracle_prefers_global_bh"].sum()), 1)
        )
        .sort_values(["global_prefers", "driver_type", "depth"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    depth_summary = (
        focused.groupby("depth", as_index=False)
        .agg(
            rows=("depth", "size"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            disagreement_rate=("oracle_prefers_global_bh", "mean"),
            mean_gap_log=("gap_log", "mean"),
            median_n_parent=("n_parent", "median"),
            mean_risk_score=("risk_score", "mean"),
        )
        .sort_values("depth")
        .reset_index(drop=True)
    )

    node_drivers = focused.loc[
        :,  # keep order explicit for downstream report generation
        [
            "case_name",
            "parent",
            "depth",
            "n_parent",
            "k",
            "edge_weight",
            "gap_log",
            "c_global",
            "c_perm_mean",
            "p_global",
            "p_perm",
            "oracle_prefers_global_bh",
            "statsmodels_fitted_probability",
            "family_portability_probability",
            "gaussian_subfamily_probability",
            "risk_score",
            "driver_type",
        ],
    ].sort_values(
        [
            "oracle_prefers_global_bh",
            "gaussian_subfamily_probability",
            "risk_score",
            "gap_log",
            "p_global",
        ],
        ascending=[False, False, False, False, True],
    )

    highd_trace = _build_highd_failure_trace(focused)
    highd_ablation = _build_highd_ablation(highd_trace)

    return ExtremeNoiseDiagnostic(
        focused_rows=focused,
        case_summary=case_summary,
        driver_summary=driver_summary,
        driver_depth_summary=driver_depth_summary,
        depth_summary=depth_summary,
        node_drivers=node_drivers,
        highd_trace=highd_trace,
        highd_ablation=highd_ablation,
    )


def _write_extreme_noise_markdown(diagnostic: ExtremeNoiseDiagnostic, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31: Gaussian Extreme-Noise Diagnostic",
        "",
    ]

    if diagnostic.focused_rows.empty:
        lines.append("No gaussian_extreme_noise focal rows were found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    focused = diagnostic.focused_rows
    lines.extend(
        [
            f"Rows: {len(focused)} focal nodes across {focused['case_name'].nunique()} cases.",
            f"False-global splits: {int(focused['oracle_prefers_global_bh'].sum())} "
            f"({float(focused['oracle_prefers_global_bh'].mean()):.3f} of focal rows).",
            "",
            "## Case Burden",
            "",
            "| Case | Rows | False-Global Splits | Rate | Mean Gap | Max Gap | Mean Subfamily Prob |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic.case_summary.itertuples(index=False):
        lines.append(
            f"| {row.case_name} | {int(row.rows)} | {int(row.global_prefers)} | "
            f"{float(row.disagreement_rate):.3f} | {float(row.mean_gap_log):.3f} | "
            f"{float(row.max_gap_log):.3f} | {float(row.mean_subfamily_probability):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Driver Burden",
            "",
            "| Driver | Rows | False-Global Splits | Share Of False-Global | Rate | Mean Gap | Mean Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic.driver_summary.itertuples(index=False):
        lines.append(
            f"| {row.driver_type} | {int(row.rows)} | {int(row.global_prefers)} | "
            f"{float(row.false_global_share):.3f} | {float(row.disagreement_rate):.3f} | "
            f"{float(row.mean_gap_log):.3f} | {float(row.mean_risk_score):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Driver By Depth",
            "",
            "| Driver | Depth | Rows | False-Global Splits | Share Of False-Global | Rate | Mean Gap | Median n_parent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic.driver_depth_summary.head(20).itertuples(index=False):
        lines.append(
            f"| {row.driver_type} | {int(row.depth)} | {int(row.rows)} | {int(row.global_prefers)} | "
            f"{float(row.false_global_share):.3f} | {float(row.disagreement_rate):.3f} | "
            f"{float(row.mean_gap_log):.3f} | {float(row.median_n_parent):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Depth Burden",
            "",
            "| Depth | Rows | False-Global Splits | Rate | Mean Gap | Median n_parent | Mean Risk |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic.depth_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.depth)} | {int(row.rows)} | {int(row.global_prefers)} | "
            f"{float(row.disagreement_rate):.3f} | {float(row.mean_gap_log):.3f} | "
            f"{float(row.median_n_parent):.1f} | {float(row.mean_risk_score):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top Driver Nodes",
            "",
            "| Case | Parent | Depth | n_parent | k | Driver | False-Global | Gap | p_global | p_perm | Subfamily Prob | Risk |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic.node_drivers.head(20).itertuples(index=False):
        lines.append(
            f"| {row.case_name} | {row.parent} | {int(row.depth)} | {int(row.n_parent)} | {int(row.k)} | "
            f"{row.driver_type} | {int(row.oracle_prefers_global_bh)} | {float(row.gap_log):.3f} | "
            f"{float(row.p_global):.3e} | {float(row.p_perm):.3g} | "
            f"{float(row.gaussian_subfamily_probability):.3f} | {float(row.risk_score):.3f} |"
        )

    lines.extend(
        [
            "",
            "## gauss_extreme_noise_highd Trace From N78",
            "",
        ]
    )
    if diagnostic.highd_trace.empty:
        lines.append("No highd trace could be reconstructed.")
    else:
        lines.extend(
            [
                "| Step | Node | Depth | Driver | False-Global | Gap | Chosen Child | Chosen Subtree False-Global | Sibling Subtree False-Global | Concentrates Downstream |",
                "| ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in diagnostic.highd_trace.itertuples(index=False):
            lines.append(
                f"| {int(row.step)} | {row.node} | {int(row.depth)} | {row.node_driver_type} | "
                f"{int(row.node_false_global)} | {float(row.node_gap_log):.3f} | {row.chosen_child} | "
                f"{int(row.chosen_child_false_global_subtree)} | {int(row.sibling_child_false_global_subtree)} | "
                f"{str(bool(row.split_concentrates_downstream))} |"
            )

    lines.extend(
        [
            "",
            "## gauss_extreme_noise_highd Branch Ablation",
            "",
        ]
    )
    if diagnostic.highd_ablation.empty:
        lines.append("No highd ablation table could be reconstructed.")
    else:
        lines.extend(
            [
                "| Step | Node | Driver | Cut Child | Removed False-Global | Removed Share | Remaining False-Global | Sibling Residual |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in diagnostic.highd_ablation.itertuples(index=False):
            lines.append(
                f"| {int(row.step)} | {row.node} | {row.node_driver_type} | {row.chosen_child} | "
                f"{int(row.removed_false_global)} | {float(row.removed_false_global_share):.3f} | "
                f"{int(row.remaining_false_global_if_cut_here)} | {int(row.sibling_child_false_global_subtree)} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gaussian_guard_markdown(benchmark: GuardBenchmark, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31: Gaussian Guard Counterfactual",
        "",
        "Candidate guard v1: `depth <= 3 and gap_log >= 4.0 and edge_weight ~= 0 and k = 1`.",
        "",
    ]

    if benchmark.hit_rows.empty:
        lines.append("No gaussian rows were hit by guard v1.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Subfamily Summary",
            "",
            "| Subfamily | Guard Hits | False-Global Guarded | Removed Share | Collateral Hits | Remaining False-Global |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in benchmark.subfamily_summary.itertuples(index=False):
        lines.append(
            f"| {row.case_subfamily} | {int(row.guard_hits)} | {int(row.false_global_guarded)} | "
            f"{float(row.false_global_removed_share):.3f} | {int(row.collateral_hits)} | "
            f"{int(row.remaining_false_global)} |"
        )

    lines.extend(
        [
            "",
            "## Case Summary",
            "",
            "| Case | Guard Hits | False-Global Guarded | Removed Share | Collateral Hits | Depth Range |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in benchmark.case_summary.itertuples(index=False):
        lines.append(
            f"| {row.case_name} | {int(row.guard_hits)} | {int(row.false_global_guarded)} | "
            f"{float(row.false_global_removed_share):.3f} | {int(row.collateral_hits)} | "
            f"{int(row.min_depth)}-{int(row.max_depth)} |"
        )

    lines.extend(
        [
            "",
            "## Guarded Nodes",
            "",
            "| Case | Parent | Depth | False-Global | Driver | Gap | Reason |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in benchmark.hit_rows.itertuples(index=False):
        lines.append(
            f"| {row.case_name} | {row.parent} | {int(row.depth)} | {int(row.oracle_prefers_global_bh)} | "
            f"{row.guard_v1_driver} | {float(row.gap_log):.3f} | {row.guard_v1_reason} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-low-ess", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=49)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_rows.csv",
    )
    parser.add_argument(
        "--disagreement-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_disagreements.csv",
    )
    parser.add_argument(
        "--family-portability-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_family_portability.csv",
    )
    parser.add_argument(
        "--gaussian-subfamily-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_subfamily_portability.csv",
    )
    parser.add_argument(
        "--ranked-false-global-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_ranked_false_global_splits.csv",
    )
    parser.add_argument(
        "--ranked-false-global-markdown",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_ranked_false_global_splits.md",
    )
    parser.add_argument(
        "--fourier-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_fourier_summary.csv",
    )
    parser.add_argument(
        "--fourier-plot-dir",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_fourier_plots",
    )
    parser.add_argument(
        "--extreme-noise-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_slice.csv",
    )
    parser.add_argument(
        "--extreme-noise-markdown",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_slice.md",
    )
    parser.add_argument(
        "--extreme-noise-driver-depth-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_driver_depth.csv",
    )
    parser.add_argument(
        "--extreme-noise-highd-trace-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_highd_trace.csv",
    )
    parser.add_argument(
        "--extreme-noise-highd-ablation-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_highd_ablation.csv",
    )
    parser.add_argument(
        "--gaussian-guard-summary-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_guard_summary.csv",
    )
    parser.add_argument(
        "--gaussian-guard-case-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_guard_case_hits.csv",
    )
    parser.add_argument(
        "--gaussian-guard-hit-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_guard_hits.csv",
    )
    parser.add_argument(
        "--gaussian-guard-markdown",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_guard_counterfactual.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_names = _case_names(args.top_k_low_ess)
    frames = [
        _collect_case_frame(case_name, n_permutations=args.n_permutations)
        for case_name in case_names
    ]
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if combined.empty:
        raise ValueError("No oracle rows collected.")

    ols_result = _fit_ols_equation(combined)
    target_column = _select_disagreement_target(combined)
    logit_result = (
        None
        if target_column is None
        else _fit_logit_equation(combined, target_column=target_column)
    )
    tree_text, tree_score = (
        (None, None)
        if target_column is None
        else _fit_tree_rule(combined, target_column=target_column)
    )
    family_probabilities, family_portability = (
        (pd.Series(np.nan, index=combined.index, dtype=np.float64), pd.DataFrame())
        if target_column is None
        else _fit_family_portability_model(combined, target_column=target_column)
    )
    combined["statsmodels_fitted_probability"] = np.nan
    combined["family_portability_probability"] = family_probabilities.to_numpy(dtype=np.float64)

    gaussian_frame = combined.loc[combined["case_family"] == "gaussian"].copy()
    gaussian_target_column = _select_disagreement_target(gaussian_frame)
    gaussian_ols_result = _fit_ols_equation(gaussian_frame) if not gaussian_frame.empty else None
    gaussian_logit_result = (
        None
        if gaussian_target_column is None
        else _fit_logit_equation(gaussian_frame, target_column=gaussian_target_column)
    )
    gaussian_tree_text, gaussian_tree_score = (
        (None, None)
        if gaussian_target_column is None
        else _fit_tree_rule_with_groups(
            gaussian_frame,
            target_column=gaussian_target_column,
            group_column="case_subfamily",
        )
    )
    gaussian_subfamily_probabilities, gaussian_subfamily_portability = (
        (pd.Series(np.nan, index=combined.index, dtype=np.float64), pd.DataFrame())
        if gaussian_target_column is None
        else _fit_group_portability_model(
            gaussian_frame,
            target_column=gaussian_target_column,
            group_column="case_subfamily",
            positive_label="held_out_subfamily",
        )
    )
    combined["gaussian_subfamily_probability"] = np.nan
    if not gaussian_subfamily_probabilities.empty:
        combined.loc[
            gaussian_subfamily_probabilities.index,
            "gaussian_subfamily_probability",
        ] = gaussian_subfamily_probabilities.to_numpy(dtype=np.float64)
    if target_column is not None:
        focal_mask = ~combined["is_null_like"]
        combined.loc[focal_mask, "statsmodels_fitted_probability"] = (
            _predict_statsmodels_probabilities(
                logit_result,
                combined.loc[focal_mask],
            ).to_numpy(dtype=np.float64)
        )
        combined["fitted_target_column"] = target_column
    else:
        combined["fitted_target_column"] = "none"

    disagreement_path = (_ROOT / args.disagreement_output_csv).resolve()
    disagreement_rows = combined.loc[combined["any_bh_disagreement"]].copy()
    disagreement_rows.to_csv(disagreement_path, index=False)

    ranked_false_global_path = (_ROOT / args.ranked_false_global_output_csv).resolve()
    ranked_false_global = _build_ranked_false_global_split_report(combined)
    ranked_false_global.to_csv(ranked_false_global_path, index=False)
    ranked_false_global_markdown_path = (_ROOT / args.ranked_false_global_markdown).resolve()
    _write_ranked_false_global_markdown(ranked_false_global, ranked_false_global_markdown_path)

    family_portability_path = (_ROOT / args.family_portability_output_csv).resolve()
    family_portability.to_csv(family_portability_path, index=False)

    gaussian_subfamily_path = (_ROOT / args.gaussian_subfamily_output_csv).resolve()
    gaussian_subfamily_portability.to_csv(gaussian_subfamily_path, index=False)

    fourier_summary = _compute_fourier_summary(combined)
    fourier_path = (_ROOT / args.fourier_output_csv).resolve()
    fourier_summary.to_csv(fourier_path, index=False)
    fourier_plot_dir = (_ROOT / args.fourier_plot_dir).resolve()
    saved_fourier_plots = _save_family_spectra_plots(combined, fourier_plot_dir)

    extreme_noise_diagnostic = _build_extreme_noise_diagnostic(combined)
    gaussian_guard = _build_gaussian_guard_counterfactual(combined)
    extreme_noise_path = (_ROOT / args.extreme_noise_output_csv).resolve()
    extreme_noise_markdown_path = (_ROOT / args.extreme_noise_markdown).resolve()
    extreme_noise_driver_depth_path = (_ROOT / args.extreme_noise_driver_depth_csv).resolve()
    extreme_noise_highd_trace_path = (_ROOT / args.extreme_noise_highd_trace_csv).resolve()
    extreme_noise_highd_ablation_path = (_ROOT / args.extreme_noise_highd_ablation_csv).resolve()
    gaussian_guard_summary_path = (_ROOT / args.gaussian_guard_summary_csv).resolve()
    gaussian_guard_case_path = (_ROOT / args.gaussian_guard_case_csv).resolve()
    gaussian_guard_hit_path = (_ROOT / args.gaussian_guard_hit_csv).resolve()
    gaussian_guard_markdown_path = (_ROOT / args.gaussian_guard_markdown).resolve()
    extreme_noise_diagnostic.node_drivers.to_csv(extreme_noise_path, index=False)
    extreme_noise_diagnostic.driver_depth_summary.to_csv(extreme_noise_driver_depth_path, index=False)
    extreme_noise_diagnostic.highd_trace.to_csv(extreme_noise_highd_trace_path, index=False)
    extreme_noise_diagnostic.highd_ablation.to_csv(extreme_noise_highd_ablation_path, index=False)
    _write_extreme_noise_markdown(extreme_noise_diagnostic, extreme_noise_markdown_path)
    gaussian_guard.subfamily_summary.to_csv(gaussian_guard_summary_path, index=False)
    gaussian_guard.case_summary.to_csv(gaussian_guard_case_path, index=False)
    gaussian_guard.hit_rows.to_csv(gaussian_guard_hit_path, index=False)
    _write_gaussian_guard_markdown(gaussian_guard, gaussian_guard_markdown_path)

    output_path = (_ROOT / args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    focal = combined.loc[~combined["is_null_like"]]
    print("=" * 116)
    print("Experiment 31: oracle policy equations")
    print("=" * 116)
    print(f"Cases: {len(case_names)}")
    print(
        f"Rows: {len(combined)} total, {len(focal)} focal, {int(combined['is_null_like'].sum())} null-like"
    )
    print(f"Permutation rows CSV: {output_path}")
    print(f"Disagreement rows CSV: {disagreement_path}")
    print(f"Ranked false-global report CSV: {ranked_false_global_path}")
    print(f"Ranked false-global markdown: {ranked_false_global_markdown_path}")
    print(f"Family portability CSV: {family_portability_path}")
    print(f"Gaussian subfamily CSV: {gaussian_subfamily_path}")
    print(f"Fourier summary CSV: {fourier_path}")
    print(f"Fourier plot directory: {fourier_plot_dir}")
    print(f"Extreme-noise node CSV: {extreme_noise_path}")
    print(f"Extreme-noise driver-depth CSV: {extreme_noise_driver_depth_path}")
    print(f"Extreme-noise highd trace CSV: {extreme_noise_highd_trace_path}")
    print(f"Extreme-noise highd ablation CSV: {extreme_noise_highd_ablation_path}")
    print(f"Extreme-noise markdown: {extreme_noise_markdown_path}")
    print(f"Gaussian guard summary CSV: {gaussian_guard_summary_path}")
    print(f"Gaussian guard case CSV: {gaussian_guard_case_path}")
    print(f"Gaussian guard hit CSV: {gaussian_guard_hit_path}")
    print(f"Gaussian guard markdown: {gaussian_guard_markdown_path}")
    print()
    print("Outcome counts:")
    print(f"  oracle_prefers_local_bh : {int(combined['oracle_prefers_local_bh'].sum())}")
    print(f"  oracle_prefers_global_bh: {int(combined['oracle_prefers_global_bh'].sum())}")
    print()

    per_case_summary = (
        combined.groupby("case_name", as_index=False)
        .agg(
            case_family=("case_family", "first"),
            rows=("case_name", "size"),
            focal_rows=("is_null_like", lambda s: int((~s).sum())),
            local_prefers=("oracle_prefers_local_bh", "sum"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            mean_log_c_ratio=("log_c_ratio", "mean"),
        )
        .sort_values(
            ["global_prefers", "local_prefers", "focal_rows"], ascending=[False, False, False]
        )
    )
    print("Per-case disagreement summary:")
    with pd.option_context("display.max_rows", 20, "display.width", 160):
        print(per_case_summary.to_string(index=False))
    print()

    disagreements = combined.loc[
        combined["oracle_prefers_local_bh"] | combined["oracle_prefers_global_bh"],
        [
            "case_name",
            "parent",
            "is_null_like",
            "n_parent",
            "k",
            "edge_weight",
            "effective_n",
            "c_global",
            "c_perm_mean",
            "p_global",
            "p_perm",
            "oracle_prefers_local_bh",
            "oracle_prefers_global_bh",
        ],
    ].sort_values(
        ["oracle_prefers_local_bh", "oracle_prefers_global_bh", "p_perm"],
        ascending=[False, False, True],
    )
    print("Top disagreement rows:")
    if disagreements.empty:
        print("  none")
    else:
        with pd.option_context(
            "display.max_rows", 12, "display.max_columns", None, "display.width", 160
        ):
            print(disagreements.head(12).to_string(index=False))
    print()

    print("Top likely false global splits:")
    if ranked_false_global.empty:
        print("  none")
    else:
        with pd.option_context(
            "display.max_rows", 12, "display.max_columns", None, "display.width", 180
        ):
            print(ranked_false_global.head(12).to_string(index=False))
    print()

    family_summary = (
        combined.groupby("case_family", as_index=False)
        .agg(
            cases=("case_name", "nunique"),
            rows=("case_name", "size"),
            focal_rows=("is_null_like", lambda s: int((~s).sum())),
            local_prefers=("oracle_prefers_local_bh", "sum"),
            global_prefers=("oracle_prefers_global_bh", "sum"),
            mean_log_c_ratio=("log_c_ratio", "mean"),
        )
        .sort_values(
            ["global_prefers", "local_prefers", "focal_rows"], ascending=[False, False, False]
        )
    )
    print("Per-family summary:")
    with pd.option_context("display.max_rows", 12, "display.width", 160):
        print(family_summary.to_string(index=False))
    print()

    if not gaussian_frame.empty:
        gaussian_subfamily_summary = (
            gaussian_frame.groupby("case_subfamily", as_index=False)
            .agg(
                cases=("case_name", "nunique"),
                rows=("case_name", "size"),
                focal_rows=("is_null_like", lambda s: int((~s).sum())),
                global_prefers=("oracle_prefers_global_bh", "sum"),
                mean_log_c_ratio=("log_c_ratio", "mean"),
            )
            .sort_values(["global_prefers", "focal_rows"], ascending=[False, False])
        )
        print("Gaussian subfamily summary:")
        with pd.option_context("display.max_rows", 20, "display.width", 160):
            print(gaussian_subfamily_summary.to_string(index=False))
        print()

    print("OLS equation for log(c_perm_mean / c_global):")
    print(ols_result.summary())
    print()

    if gaussian_ols_result is not None:
        print("Gaussian-only OLS equation for log(c_perm_mean / c_global):")
        print(gaussian_ols_result.summary())
        print()

    if logit_result is None or target_column is None:
        print("Logit equation for disagreement target: insufficient class variation")
    else:
        print(f"Logit equation for {target_column} on focal rows:")
        print(logit_result.summary())
    print()

    if gaussian_logit_result is None or gaussian_target_column is None:
        print("Gaussian-only logit equation: insufficient class variation")
    else:
        print(f"Gaussian-only logit for {gaussian_target_column} on focal rows:")
        print(gaussian_logit_result.summary())
    print()

    if tree_text is None or target_column is None:
        print("Decision-tree rule summary: insufficient class variation")
    else:
        print(f"Decision-tree grouped balanced accuracy: {tree_score:.3f}")
        print(f"Decision-tree target: {target_column}")
        print("Decision-tree rule summary:")
        print(tree_text)

    print()
    if gaussian_tree_text is None or gaussian_target_column is None:
        print("Gaussian-only decision-tree rule summary: insufficient subfamily variation")
    else:
        print(f"Gaussian-only grouped balanced accuracy: {gaussian_tree_score:.3f}")
        print(f"Gaussian-only target: {gaussian_target_column}")
        print("Gaussian-only decision-tree rule summary:")
        print(gaussian_tree_text)

    print()
    if family_portability.empty or target_column is None:
        print("Case-family portability model: insufficient family variation")
    else:
        print(f"Case-family portability model target: {target_column}")
        with pd.option_context("display.max_rows", 12, "display.width", 160):
            print(family_portability.to_string(index=False))
        print(
            "Mean leave-one-family-out balanced accuracy: "
            f"{family_portability['balanced_accuracy'].mean():.3f}"
        )

    print()
    if gaussian_subfamily_portability.empty or gaussian_target_column is None:
        print("Gaussian subfamily portability model: insufficient subfamily variation")
    else:
        print(f"Gaussian subfamily portability target: {gaussian_target_column}")
        with pd.option_context("display.max_rows", 20, "display.width", 160):
            print(gaussian_subfamily_portability.to_string(index=False))
        print(
            "Mean leave-one-subfamily-out balanced accuracy: "
            f"{gaussian_subfamily_portability['balanced_accuracy'].mean():.3f}"
        )

    print()
    if extreme_noise_diagnostic.focused_rows.empty:
        print("Gaussian extreme-noise diagnostic: no focal rows")
    else:
        print("Gaussian extreme-noise case summary:")
        with pd.option_context("display.max_rows", 10, "display.width", 180):
            print(extreme_noise_diagnostic.case_summary.to_string(index=False))
        print()
        print("Gaussian extreme-noise driver summary:")
        with pd.option_context("display.max_rows", 10, "display.width", 180):
            print(extreme_noise_diagnostic.driver_summary.to_string(index=False))
        print()
        print("Gaussian extreme-noise driver-by-depth summary:")
        with pd.option_context("display.max_rows", 20, "display.width", 180):
            print(extreme_noise_diagnostic.driver_depth_summary.head(20).to_string(index=False))
        print()
        print("Gaussian extreme-noise depth summary:")
        with pd.option_context("display.max_rows", 10, "display.width", 180):
            print(extreme_noise_diagnostic.depth_summary.to_string(index=False))
        print()
        print("Gaussian extreme-noise top driver nodes:")
        with pd.option_context(
            "display.max_rows", 12, "display.max_columns", None, "display.width", 200
        ):
            print(extreme_noise_diagnostic.node_drivers.head(12).to_string(index=False))
        print()
        print("gauss_extreme_noise_highd trace from N78:")
        with pd.option_context(
            "display.max_rows", 20, "display.max_columns", None, "display.width", 220
        ):
            print(extreme_noise_diagnostic.highd_trace.to_string(index=False))
        print()
        print("gauss_extreme_noise_highd branch ablation:")
        with pd.option_context(
            "display.max_rows", 20, "display.max_columns", None, "display.width", 220
        ):
            print(extreme_noise_diagnostic.highd_ablation.to_string(index=False))

    print()
    if gaussian_guard.subfamily_summary.empty:
        print("Gaussian guard counterfactual: no gaussian rows hit")
    else:
        print("Gaussian guard counterfactual summary:")
        with pd.option_context("display.max_rows", 12, "display.width", 180):
            print(gaussian_guard.subfamily_summary.to_string(index=False))
        print()
        print("Gaussian guard case hits:")
        with pd.option_context("display.max_rows", 20, "display.width", 180):
            print(gaussian_guard.case_summary.to_string(index=False))

    print()
    if fourier_summary.empty:
        print("Fourier summary: no focal rows available")
    else:
        family_fourier = (
            fourier_summary.groupby(["case_family", "signal_name"], as_index=False)
            .agg(
                cases=("case_name", "nunique"),
                mean_low_band_power_fraction=("low_band_power_fraction", "mean"),
                mean_spectral_entropy=("spectral_entropy", "mean"),
                mean_dominant_period_nodes=("dominant_period_nodes", "mean"),
            )
            .sort_values(["case_family", "signal_name"])
        )
        print("Fourier family summary:")
        with pd.option_context("display.max_rows", 20, "display.width", 160):
            print(family_fourier.to_string(index=False))
        if saved_fourier_plots:
            print()
            print("Saved Fourier spectra plots:")
            for plot_path in saved_fourier_plots:
                print(f"  {plot_path}")


if __name__ == "__main__":
    main()
