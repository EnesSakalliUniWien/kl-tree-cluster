#!/usr/bin/env python3
"""
Experiment 31f: hurdle-style runtime deflation student via symbolic regression.

This follows exp31e, but handles the zero-inflated target explicitly:

1. Gate model:
   predict whether extra deflation is needed at all
   gate_target = 1{ extra_log_deflation > threshold }

2. Amount model:
   predict the positive extra deflation magnitude on gate-positive rows only

Deployable forms:
    gate_prob = sigmoid(gate_equation(x))
    amount_hat = max(0, amount_equation(x))
    extra_soft = gate_prob * amount_hat
    extra_hard = 1{gate_prob >= gate_threshold} * amount_hat
    c_student = c_global * exp(extra_hat)

Inputs:
- exp31 row exports already written to disk

Outputs:
- hurdle summary CSV
- gate/amount equations CSV
- row-level predictions CSV
- markdown summary
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from scipy.stats import chi2
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from debug_scripts.enhancement_lab.lab_helpers import (
    enhancement_lab_results_relative,
    resolve_enhancement_lab_artifact_path,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (
    benjamini_hochberg_correction,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GROUP_COLUMN = "case_family"
_RUNTIME_FEATURE_COLUMNS = [
    "log_n_parent",
    "log_k",
    "depth",
    "neglog_edge_weight",
    "neglog_p_global",
    "log_c_global",
    "log_stat_per_k",
    "log_effective_n",
    "log_n_null_case",
    "log_n_focal_case",
]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    rows_csv: str


_DEFAULT_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        name="baseline",
        rows_csv=enhancement_lab_results_relative("_oracle_policy_rows.csv"),
    ),
    ModeSpec(
        name="one_active_1d",
        rows_csv=enhancement_lab_results_relative("_oracle_policy_rows_one_active_1d.csv"),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a hurdle-style symbolic deflation student from exp31 row exports."
    )
    parser.add_argument("--modes", default="baseline,one_active_1d")
    parser.add_argument(
        "--summary-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_hurdle_deflation_summary.csv"),
    )
    parser.add_argument(
        "--equation-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_hurdle_deflation_equations.csv"),
    )
    parser.add_argument(
        "--prediction-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_hurdle_deflation_predictions.csv"),
    )
    parser.add_argument(
        "--summary-markdown",
        default=enhancement_lab_results_relative("_oracle_policy_hurdle_deflation_summary.md"),
    )
    parser.add_argument("--group-column", default=_DEFAULT_GROUP_COLUMN)
    parser.add_argument("--positive-threshold", type=float, default=1e-9)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--parsimony", type=float, default=0.002)
    parser.add_argument("--tournament-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _resolve_modes(arg: str) -> list[ModeSpec]:
    wanted = {token.strip() for token in arg.split(",") if token.strip()}
    resolved = [mode for mode in _DEFAULT_MODES if mode.name in wanted]
    if not resolved:
        raise ValueError(f"No known modes requested from {sorted(wanted)}")
    return resolved


def _safe_log(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(values.astype(np.float64), floor))


def _safe_neglog10(values: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    return -np.log10(np.maximum(values.astype(np.float64), floor))


def _apply_casewise_bh(frame: pd.DataFrame, p_column: str, out_column: str) -> None:
    flags = np.zeros(len(frame), dtype=bool)
    for _case_name, case_frame in frame.groupby("case_name", sort=False):
        idx = case_frame.index.to_numpy(dtype=np.int64)
        p_values = case_frame[p_column].to_numpy(dtype=np.float64)
        rejected, _, _ = benjamini_hochberg_correction(p_values, alpha=config.SIBLING_ALPHA)
        flags[idx] = rejected.astype(bool)
    frame[out_column] = flags


def _usable_feature_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    kept: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        if float(np.std(values)) <= 1e-12:
            continue
        kept.append(column)
    return kept


def _prepare_frame(path: Path, *, positive_threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[~frame["is_null_like"]].copy().reset_index(drop=True)

    stat = frame["stat"].to_numpy(dtype=np.float64)
    k = np.maximum(frame["k"].to_numpy(dtype=np.float64), 1.0)
    c_global = np.maximum(frame["c_global"].to_numpy(dtype=np.float64), 1.0)
    p_global = frame["p_global"].to_numpy(dtype=np.float64)
    edge_weight = frame["edge_weight"].to_numpy(dtype=np.float64)

    frame["log_c_global"] = _safe_log(c_global, floor=1.0)
    frame["log_stat_per_k"] = _safe_log(stat / k)
    frame["neglog_edge_weight"] = _safe_neglog10(edge_weight, floor=1e-12)
    frame["neglog_p_global"] = _safe_neglog10(p_global, floor=1e-300)
    frame["log_effective_n"] = np.log1p(
        np.maximum(frame["effective_n"].to_numpy(dtype=np.float64), 0.0)
    )
    frame["log_n_null_case"] = np.log1p(
        np.maximum(frame["n_null_case"].to_numpy(dtype=np.float64), 0.0)
    )
    frame["log_n_focal_case"] = np.log1p(
        np.maximum(frame["n_focal_case"].to_numpy(dtype=np.float64), 0.0)
    )
    frame["extra_log_deflation"] = np.maximum(frame["log_c_ratio"].to_numpy(dtype=np.float64), 0.0)
    frame["gate_target"] = frame["extra_log_deflation"] > positive_threshold
    return frame


def _make_symbolic_classifier(
    *,
    feature_names: list[str],
    population_size: int,
    generations: int,
    parsimony: float,
    tournament_size: int,
    random_state: int,
) -> SymbolicClassifier:
    return SymbolicClassifier(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        stopping_criteria=0.0,
        const_range=(-5.0, 5.0),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "log", "sqrt", "abs", "neg"),
        transformer="sigmoid",
        metric="log loss",
        parsimony_coefficient=parsimony,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=0.9,
        verbose=0,
        random_state=random_state,
        feature_names=feature_names,
    )


def _make_symbolic_regressor(
    *,
    feature_names: list[str],
    population_size: int,
    generations: int,
    parsimony: float,
    tournament_size: int,
    random_state: int,
) -> SymbolicRegressor:
    return SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        stopping_criteria=0.0,
        const_range=(-5.0, 5.0),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "log", "sqrt", "abs", "neg"),
        metric="mse",
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=0.9,
        parsimony_coefficient=parsimony,
        verbose=0,
        random_state=random_state,
        feature_names=feature_names,
    )


def _fit_classifier(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    args: argparse.Namespace,
    random_state: int,
) -> SymbolicClassifier:
    x = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame["gate_target"].astype(int).to_numpy(dtype=np.int64)
    model = _make_symbolic_classifier(
        feature_names=feature_names,
        population_size=args.population_size,
        generations=args.generations,
        parsimony=args.parsimony,
        tournament_size=args.tournament_size,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def _fit_regressor(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    args: argparse.Namespace,
    random_state: int,
) -> SymbolicRegressor | None:
    positive = frame.loc[frame["gate_target"]].copy()
    if len(positive) < 10:
        return None
    x = positive[feature_names].to_numpy(dtype=np.float64)
    y = positive["extra_log_deflation"].to_numpy(dtype=np.float64)
    if float(np.std(y)) <= 1e-12:
        return None
    model = _make_symbolic_regressor(
        feature_names=feature_names,
        population_size=args.population_size,
        generations=args.generations,
        parsimony=args.parsimony,
        tournament_size=args.tournament_size,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def _predict_classifier_prob(
    model: SymbolicClassifier,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    x = frame[feature_names].to_numpy(dtype=np.float64)
    return model.predict_proba(x)[:, 1].astype(np.float64)


def _predict_regressor_amount(
    model: SymbolicRegressor | None,
    frame: pd.DataFrame,
    feature_names: list[str],
    fallback: float,
) -> np.ndarray:
    if model is None:
        return np.full(len(frame), max(float(fallback), 0.0), dtype=np.float64)
    x = frame[feature_names].to_numpy(dtype=np.float64)
    return np.maximum(model.predict(x).astype(np.float64), 0.0)


def _safe_program_int(program: object, attr_name: str) -> float | int:
    value = getattr(program, attr_name, math.nan)
    if value is None:
        return math.nan
    try:
        if not np.isfinite(float(value)):
            return math.nan
    except Exception:
        return math.nan
    return int(value)


def _render_protected_equation(program: str) -> str:
    rendered = str(program)
    rendered = rendered.replace("div(", "pdiv(")
    rendered = rendered.replace("sqrt(", "psqrt(")
    rendered = rendered.replace("log(", "plog(")
    return rendered


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(balanced_accuracy_score(y_true, y_pred))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _student_p_values(
    frame: pd.DataFrame, extra_log_deflation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c_global = np.maximum(frame["c_global"].to_numpy(dtype=np.float64), 1.0)
    stat = frame["stat"].to_numpy(dtype=np.float64)
    k = frame["k"].to_numpy(dtype=np.int64)
    c_student = c_global * np.exp(np.maximum(extra_log_deflation, 0.0))
    p_student = np.array(
        [
            float(chi2.sf(float(s) / max(float(c_hat), 1.0), df=int(df)))
            for s, c_hat, df in zip(stat, c_student, k, strict=False)
        ],
        dtype=np.float64,
    )
    return c_student, p_student


def _evaluate_student_bh(
    frame: pd.DataFrame,
    *,
    soft_extra: np.ndarray,
    hard_extra: np.ndarray,
) -> pd.DataFrame:
    evaluated = frame.copy()
    c_soft, p_soft = _student_p_values(evaluated, soft_extra)
    c_hard, p_hard = _student_p_values(evaluated, hard_extra)
    evaluated["student_soft_extra_log_deflation"] = soft_extra
    evaluated["student_hard_extra_log_deflation"] = hard_extra
    evaluated["c_student_soft"] = c_soft
    evaluated["c_student_hard"] = c_hard
    evaluated["p_student_soft"] = p_soft
    evaluated["p_student_hard"] = p_hard
    _apply_casewise_bh(evaluated, "p_student_soft", "student_soft_bh_reject")
    _apply_casewise_bh(evaluated, "p_student_hard", "student_hard_bh_reject")
    evaluated["baseline_false_global"] = evaluated["global_bh_reject"] & (
        ~evaluated["perm_bh_reject"]
    )
    evaluated["soft_false_global"] = evaluated["student_soft_bh_reject"] & (
        ~evaluated["perm_bh_reject"]
    )
    evaluated["hard_false_global"] = evaluated["student_hard_bh_reject"] & (
        ~evaluated["perm_bh_reject"]
    )
    return evaluated


def _leave_group_out_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    groups = frame[args.group_column].to_numpy()
    splitter = LeaveOneGroupOut()
    gate_prob = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    amount_pred = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    soft_extra = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(
            frame[feature_names].to_numpy(dtype=np.float64),
            frame["gate_target"].astype(int).to_numpy(),
            groups,
        )
    ):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        held_out = str(test[args.group_column].iloc[0])
        classifier = _fit_classifier(
            train,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + fold_idx,
        )
        regressor = _fit_regressor(
            train,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + 10_000 + fold_idx,
        )
        fallback_amount = float(train.loc[train["gate_target"], "extra_log_deflation"].mean())
        if not np.isfinite(fallback_amount):
            fallback_amount = 0.0
        test_gate = _predict_classifier_prob(classifier, test, feature_names)
        test_amount = _predict_regressor_amount(
            regressor, test, feature_names, fallback=fallback_amount
        )
        test_soft = test_gate * test_amount

        gate_prob.loc[test.index] = test_gate
        amount_pred.loc[test.index] = test_amount
        soft_extra.loc[test.index] = test_soft

        y_gate_true = test["gate_target"].astype(int).to_numpy(dtype=np.int64)
        y_gate_pred = (test_gate >= args.gate_threshold).astype(np.int64)
        y_amount_true = test["extra_log_deflation"].to_numpy(dtype=np.float64)
        fold_regression = _regression_metrics(y_amount_true, test_soft)
        rows.append(
            {
                "held_out_group": held_out,
                "n_test": int(len(test)),
                "gate_positive_rate": float(np.mean(y_gate_true)),
                "gate_balanced_accuracy": _safe_balanced_accuracy(y_gate_true, y_gate_pred),
                "gate_roc_auc": _safe_roc_auc(y_gate_true, test_gate),
                **fold_regression,
            }
        )

    return gate_prob, amount_pred, soft_extra, pd.DataFrame(rows)


def _summary_row(
    *,
    mode_name: str,
    frame: pd.DataFrame,
    gate_prob: pd.Series,
    amount_pred: pd.Series,
    soft_extra: pd.Series,
    args: argparse.Namespace,
    feature_names: list[str],
    fold_summary: pd.DataFrame,
    gate_model: SymbolicClassifier,
    amount_model: SymbolicRegressor | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    valid = frame.loc[gate_prob.notna()].copy()
    gate_p = gate_prob.loc[valid.index].to_numpy(dtype=np.float64)
    amount_hat = amount_pred.loc[valid.index].to_numpy(dtype=np.float64)
    soft_hat = soft_extra.loc[valid.index].to_numpy(dtype=np.float64)
    hard_hat = (gate_p >= args.gate_threshold).astype(np.float64) * amount_hat
    evaluated = _evaluate_student_bh(valid, soft_extra=soft_hat, hard_extra=hard_hat)

    gate_true = valid["gate_target"].astype(int).to_numpy(dtype=np.int64)
    gate_pred = (gate_p >= args.gate_threshold).astype(np.int64)
    all_regression = _regression_metrics(
        valid["extra_log_deflation"].to_numpy(dtype=np.float64), soft_hat
    )
    positive = valid.loc[valid["gate_target"]].copy()
    if not positive.empty:
        positive_soft = soft_extra.loc[positive.index].to_numpy(dtype=np.float64)
        positive_regression = _regression_metrics(
            positive["extra_log_deflation"].to_numpy(dtype=np.float64),
            positive_soft,
        )
    else:
        positive_regression = {"mae": math.nan, "rmse": math.nan, "r2": math.nan}

    gate_program = str(gate_model._program)
    amount_program = str(amount_model._program) if amount_model is not None else "constant_fallback"
    equation_frame = pd.DataFrame(
        [
            {
                "mode": mode_name,
                "equation_type": "gate",
                "equation": gate_program,
                "equation_rendered": f"sigmoid({_render_protected_equation(gate_program)})",
                "program_length": _safe_program_int(gate_model._program, "length_"),
                "program_depth": _safe_program_int(gate_model._program, "depth_"),
                "raw_fitness": float(getattr(gate_model._program, "raw_fitness_", math.nan)),
                "feature_names": ",".join(feature_names),
            },
            {
                "mode": mode_name,
                "equation_type": "amount",
                "equation": amount_program,
                "equation_rendered": _render_protected_equation(amount_program),
                "program_length": (
                    _safe_program_int(amount_model._program, "length_")
                    if amount_model is not None
                    else math.nan
                ),
                "program_depth": (
                    _safe_program_int(amount_model._program, "depth_")
                    if amount_model is not None
                    else math.nan
                ),
                "raw_fitness": (
                    float(getattr(amount_model._program, "raw_fitness_", math.nan))
                    if amount_model is not None
                    else math.nan
                ),
                "feature_names": ",".join(feature_names),
            },
        ]
    )

    summary = {
        "mode": mode_name,
        "n_rows": int(len(frame)),
        "n_gate_positive": int(frame["gate_target"].sum()),
        "gate_balanced_accuracy": _safe_balanced_accuracy(gate_true, gate_pred),
        "gate_roc_auc": _safe_roc_auc(gate_true, gate_p),
        "all_mae_soft": all_regression["mae"],
        "all_rmse_soft": all_regression["rmse"],
        "all_r2_soft": all_regression["r2"],
        "positive_mae_soft": positive_regression["mae"],
        "positive_rmse_soft": positive_regression["rmse"],
        "positive_r2_soft": positive_regression["r2"],
        "mean_fold_gate_balanced_accuracy": (
            float(fold_summary["gate_balanced_accuracy"].mean())
            if not fold_summary.empty
            else math.nan
        ),
        "mean_fold_gate_roc_auc": (
            float(fold_summary["gate_roc_auc"].mean()) if not fold_summary.empty else math.nan
        ),
        "baseline_false_global": int(evaluated["baseline_false_global"].sum()),
        "soft_false_global": int(evaluated["soft_false_global"].sum()),
        "hard_false_global": int(evaluated["hard_false_global"].sum()),
        "mean_c_global": float(evaluated["c_global"].mean()),
        "mean_c_student_soft": float(evaluated["c_student_soft"].mean()),
        "mean_c_student_hard": float(evaluated["c_student_hard"].mean()),
        "gate_equation_rendered": equation_frame.loc[
            equation_frame["equation_type"] == "gate", "equation_rendered"
        ].iloc[0],
        "amount_equation_rendered": equation_frame.loc[
            equation_frame["equation_type"] == "amount", "equation_rendered"
        ].iloc[0],
    }
    return (
        summary,
        evaluated.assign(
            mode=mode_name,
            gate_probability=gate_p,
            amount_prediction=amount_hat,
        ),
        equation_frame,
    )


def _write_markdown(summary: pd.DataFrame, equations: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31f: Hurdle Runtime Deflation Student",
        "",
        "Two-stage symbolic model for extra deflation:",
        "- gate equation decides whether extra deflation is needed",
        "- amount equation predicts the positive magnitude",
        "",
        "Protected operators:",
        "",
        "- `pdiv(a, b) = a / b` if `|b| > 1e-3`, else `1`",
        "- `psqrt(x) = sqrt(|x|)`",
        "- `plog(x) = log(|x|)` if `|x| > 1e-3`, else `0`",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"## {row.mode}",
                "",
                f"- rows: `{int(row.n_rows)}`",
                f"- gate positives: `{int(row.n_gate_positive)}`",
                f"- gate balanced accuracy: `{float(row.gate_balanced_accuracy):.6f}`",
                f"- gate ROC AUC: `{float(row.gate_roc_auc):.6f}`",
                f"- soft all-row R²: `{float(row.all_r2_soft):.6f}`",
                f"- soft positive-row R²: `{float(row.positive_r2_soft):.6f}`",
                f"- false-global count: baseline `{int(row.baseline_false_global)}`, soft `{int(row.soft_false_global)}`, hard `{int(row.hard_false_global)}`",
                "",
                "Gate equation:",
                "",
                "```text",
                str(row.gate_equation_rendered),
                "```",
                "",
                "Amount equation:",
                "",
                "```text",
                str(row.amount_equation_rendered),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    modes = _resolve_modes(args.modes)

    summary_rows: list[dict[str, object]] = []
    equation_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for mode in modes:
        path = resolve_enhancement_lab_artifact_path(mode.rows_csv, for_input=True)
        frame = _prepare_frame(path, positive_threshold=args.positive_threshold)
        feature_names = _usable_feature_columns(frame, _RUNTIME_FEATURE_COLUMNS)
        if not feature_names:
            raise ValueError(f"No usable runtime features for mode {mode.name}")

        gate_prob, amount_pred, soft_extra, fold_summary = _leave_group_out_predictions(
            frame,
            feature_names=feature_names,
            args=args,
        )
        gate_model = _fit_classifier(
            frame,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + 20_000,
        )
        amount_model = _fit_regressor(
            frame,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + 30_000,
        )
        summary_row, evaluated, equation_frame = _summary_row(
            mode_name=mode.name,
            frame=frame,
            gate_prob=gate_prob,
            amount_pred=amount_pred,
            soft_extra=soft_extra,
            args=args,
            feature_names=feature_names,
            fold_summary=fold_summary,
            gate_model=gate_model,
            amount_model=amount_model,
        )
        summary_rows.append(summary_row)
        prediction_frames.append(evaluated)
        equation_frames.append(equation_frame)

    summary_df = pd.DataFrame(summary_rows)
    equations_df = pd.concat(equation_frames, axis=0, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, axis=0, ignore_index=True)

    summary_path = resolve_enhancement_lab_artifact_path(args.summary_output_csv)
    equation_path = resolve_enhancement_lab_artifact_path(args.equation_output_csv)
    prediction_path = resolve_enhancement_lab_artifact_path(args.prediction_output_csv)
    markdown_path = resolve_enhancement_lab_artifact_path(args.summary_markdown)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_path, index=False)
    equations_df.to_csv(equation_path, index=False)
    predictions_df.to_csv(prediction_path, index=False)
    _write_markdown(summary_df, equations_df, markdown_path)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote equations to {equation_path}")
    print(f"Wrote predictions to {prediction_path}")
    print(f"Wrote markdown to {markdown_path}")


if __name__ == "__main__":
    main()
