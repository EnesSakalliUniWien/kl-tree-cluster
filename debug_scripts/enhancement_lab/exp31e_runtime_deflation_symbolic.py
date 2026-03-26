#!/usr/bin/env python3
"""
Experiment 31e: runtime deflation student via symbolic regression.

Purpose:
- use permutation only as an offline teacher for extra deflation
- fit a runtime-only symbolic equation for the positive deflation gap
- evaluate portability with leave-one-family-out prediction
- replay casewise BH decisions using the student c-hat

Teacher target:
    extra_log_deflation = max(log(c_perm_mean / c_global), 0)

Deployable form:
    c_student = c_global * exp(y_hat_runtime)

Inputs:
- exp31 row exports already written to disk

Outputs:
- per-mode summary CSV
- symbolic equation CSV
- row-level prediction CSV
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
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from scipy.stats import chi2
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
        description="Fit a runtime-only symbolic deflation student from exp31 row exports."
    )
    parser.add_argument(
        "--modes",
        default="baseline,one_active_1d",
        help="Comma-separated mode names to run.",
    )
    parser.add_argument(
        "--summary-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_symbolic_deflation_summary.csv"),
    )
    parser.add_argument(
        "--equation-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_symbolic_deflation_equations.csv"),
    )
    parser.add_argument(
        "--prediction-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_symbolic_deflation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--summary-markdown",
        default=enhancement_lab_results_relative("_oracle_policy_symbolic_deflation_summary.md"),
    )
    parser.add_argument(
        "--include-null-like",
        action="store_true",
        help="Use all rows instead of focal rows only.",
    )
    parser.add_argument(
        "--group-column",
        default=_DEFAULT_GROUP_COLUMN,
        help="Group column for held-out evaluation.",
    )
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


def _prepare_frame(path: Path, *, include_null_like: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not include_null_like:
        frame = frame.loc[~frame["is_null_like"]].copy()
    else:
        frame = frame.copy()

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
    frame["extra_log_deflation"] = np.maximum(
        frame["log_c_ratio"].to_numpy(dtype=np.float64),
        0.0,
    )
    frame["student_feature_scope"] = "all_rows" if include_null_like else "focal_only"
    return frame.reset_index(drop=True)


def _make_symbolic_model(
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


def _fit_symbolic(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    args: argparse.Namespace,
    random_state: int,
) -> SymbolicRegressor:
    x = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame["extra_log_deflation"].to_numpy(dtype=np.float64)
    model = _make_symbolic_model(
        feature_names=feature_names,
        population_size=args.population_size,
        generations=args.generations,
        parsimony=args.parsimony,
        tournament_size=args.tournament_size,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def _predict_symbolic(
    model: SymbolicRegressor, frame: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    x = frame[feature_names].to_numpy(dtype=np.float64)
    prediction = model.predict(x).astype(np.float64)
    return np.maximum(prediction, 0.0)


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(balanced_accuracy_score(y_true, y_pred))


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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _symbolic_to_sympy(program: str, feature_names: list[str]) -> str:
    locals_map: dict[str, object] = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "abs": sp.Abs,
        "neg": lambda a: -a,
    }
    for name in feature_names:
        locals_map[name] = sp.Symbol(name)
    try:
        expr = sp.sympify(program, locals=locals_map)
        return str(sp.simplify(expr))
    except Exception:
        return program


def _render_protected_equation(program: str) -> str:
    rendered = str(program)
    rendered = rendered.replace("div(", "pdiv(")
    rendered = rendered.replace("sqrt(", "psqrt(")
    rendered = rendered.replace("log(", "plog(")
    return rendered


def _student_p_values(
    frame: pd.DataFrame, student_log_deflation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c_global = np.maximum(frame["c_global"].to_numpy(dtype=np.float64), 1.0)
    stat = frame["stat"].to_numpy(dtype=np.float64)
    k = frame["k"].to_numpy(dtype=np.int64)
    c_student = c_global * np.exp(np.maximum(student_log_deflation, 0.0))
    p_student = np.array(
        [
            float(chi2.sf(float(s) / max(float(c_hat), 1.0), df=int(df)))
            for s, c_hat, df in zip(stat, c_student, k, strict=False)
        ],
        dtype=np.float64,
    )
    return c_student, p_student


def _leave_group_out_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.DataFrame]:
    groups = frame[args.group_column].to_numpy()
    splitter = LeaveOneGroupOut()
    predictions = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    x_full = frame[feature_names].to_numpy(dtype=np.float64)
    y_full = frame["extra_log_deflation"].to_numpy(dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_full, y_full, groups)):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        held_out = str(test[args.group_column].iloc[0])
        model = _fit_symbolic(
            train,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + fold_idx,
        )
        test_pred = _predict_symbolic(model, test, feature_names)
        predictions.loc[test.index] = test_pred
        fold_metrics = _regression_metrics(
            test["extra_log_deflation"].to_numpy(dtype=np.float64),
            test_pred,
        )
        rows.append(
            {
                "held_out_group": held_out,
                "n_test": int(len(test)),
                "target_mean": float(test["extra_log_deflation"].mean()),
                "prediction_mean": float(np.mean(test_pred)),
                **fold_metrics,
            }
        )

    return predictions, pd.DataFrame(rows)


def _evaluate_student_bh(frame: pd.DataFrame, *, student_log_deflation: np.ndarray) -> pd.DataFrame:
    evaluated = frame.copy()
    c_student, p_student = _student_p_values(evaluated, student_log_deflation)
    evaluated["student_extra_log_deflation"] = student_log_deflation
    evaluated["c_student"] = c_student
    evaluated["p_student"] = p_student
    _apply_casewise_bh(evaluated, "p_student", "student_bh_reject")
    evaluated["student_false_global"] = evaluated["student_bh_reject"] & (
        ~evaluated["perm_bh_reject"]
    )
    evaluated["student_false_local"] = evaluated["perm_bh_reject"] & (
        ~evaluated["student_bh_reject"]
    )
    evaluated["baseline_false_global"] = evaluated["global_bh_reject"] & (
        ~evaluated["perm_bh_reject"]
    )
    evaluated["baseline_false_local"] = evaluated["perm_bh_reject"] & (
        ~evaluated["global_bh_reject"]
    )
    return evaluated


def _summary_row(
    *,
    mode_name: str,
    group_column: str,
    frame: pd.DataFrame,
    feature_names: list[str],
    cv_predictions: pd.Series,
    fold_summary: pd.DataFrame,
    final_model: SymbolicRegressor,
    program_text: str,
    rendered_text: str,
    simplified_text: str,
) -> dict[str, object]:
    valid = frame.loc[cv_predictions.notna()].copy()
    y_true = valid["extra_log_deflation"].to_numpy(dtype=np.float64)
    y_pred = cv_predictions.loc[valid.index].to_numpy(dtype=np.float64)
    regression = _regression_metrics(y_true, y_pred)
    evaluated = _evaluate_student_bh(valid, student_log_deflation=y_pred)
    perm_target = evaluated["perm_bh_reject"].astype(np.int64).to_numpy()
    student_pred = evaluated["student_bh_reject"].astype(np.int64).to_numpy()
    global_pred = evaluated["global_bh_reject"].astype(np.int64).to_numpy()
    return {
        "mode": mode_name,
        "feature_scope": str(frame["student_feature_scope"].iloc[0]),
        "group_column": group_column,
        "n_rows": int(len(frame)),
        "n_valid_cv": int(len(valid)),
        "n_positive_target": int((frame["extra_log_deflation"] > 0.0).sum()),
        "perm_bh_positive": int(valid["perm_bh_reject"].sum()),
        "target_mean": float(frame["extra_log_deflation"].mean()),
        "prediction_mean": float(np.mean(y_pred)),
        "mean_fold_mae": float(fold_summary["mae"].mean()) if not fold_summary.empty else math.nan,
        "mean_fold_rmse": (
            float(fold_summary["rmse"].mean()) if not fold_summary.empty else math.nan
        ),
        **regression,
        "global_bh_balanced_accuracy": _safe_balanced_accuracy(perm_target, global_pred),
        "student_bh_balanced_accuracy": _safe_balanced_accuracy(perm_target, student_pred),
        "baseline_false_global": int(evaluated["baseline_false_global"].sum()),
        "student_false_global": int(evaluated["student_false_global"].sum()),
        "baseline_false_local": int(evaluated["baseline_false_local"].sum()),
        "student_false_local": int(evaluated["student_false_local"].sum()),
        "mean_c_global": float(evaluated["c_global"].mean()),
        "mean_c_student": float(evaluated["c_student"].mean()),
        "program_length": _safe_program_int(final_model._program, "length_"),
        "program_depth": _safe_program_int(final_model._program, "depth_"),
        "raw_fitness": float(getattr(final_model._program, "raw_fitness_", math.nan)),
        "equation": program_text,
        "equation_rendered": rendered_text,
        "equation_simplified": simplified_text,
        "feature_names": ",".join(feature_names),
    }


def _write_markdown(summary: pd.DataFrame, equations: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31e: Runtime Deflation Student",
        "",
        "Symbolic regression fit to permutation-taught extra deflation using runtime-only features.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"## {row.mode}",
                "",
                f"- rows: `{int(row.n_rows)}`",
                f"- positive target rows: `{int(row.n_positive_target)}`",
                f"- permutation BH positives in evaluation: `{int(row.perm_bh_positive)}`",
                f"- CV MAE: `{float(row.mae):.6f}`",
                f"- CV RMSE: `{float(row.rmse):.6f}`",
                f"- CV R²: `{float(row.r2):.6f}`",
                f"- global BH balanced accuracy vs permutation: `{float(row.global_bh_balanced_accuracy):.6f}`",
                f"- student BH balanced accuracy vs permutation: `{float(row.student_bh_balanced_accuracy):.6f}`",
                f"- false-global count: `{int(row.baseline_false_global)} -> {int(row.student_false_global)}`",
                f"- false-local count: `{int(row.baseline_false_local)} -> {int(row.student_false_local)}`",
                "",
            ]
        )
        equation_row = equations.loc[equations["mode"] == row.mode].iloc[0]
        lines.extend(
            [
                "Protected operators:",
                "",
                "- `pdiv(a, b) = a / b` if `|b| > 1e-3`, else `1`",
                "- `psqrt(x) = sqrt(|x|)`",
                "- `plog(x) = log(|x|)` if `|x| > 1e-3`, else `0`",
                "",
                "Equation:",
                "",
                "```text",
                str(equation_row["equation_rendered"]),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    modes = _resolve_modes(args.modes)

    summary_rows: list[dict[str, object]] = []
    equation_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for mode in modes:
        path = resolve_enhancement_lab_artifact_path(mode.rows_csv, for_input=True)
        frame = _prepare_frame(path, include_null_like=args.include_null_like)
        feature_names = _usable_feature_columns(frame, _RUNTIME_FEATURE_COLUMNS)
        if not feature_names:
            raise ValueError(f"No usable runtime features for mode {mode.name}")

        cv_predictions, fold_summary = _leave_group_out_predictions(
            frame,
            feature_names=feature_names,
            args=args,
        )
        final_model = _fit_symbolic(
            frame,
            feature_names=feature_names,
            args=args,
            random_state=args.random_seed + 10_000,
        )
        program_text = str(final_model._program)
        rendered_text = _render_protected_equation(program_text)
        simplified_text = _symbolic_to_sympy(program_text, feature_names)
        evaluated = _evaluate_student_bh(
            frame,
            student_log_deflation=cv_predictions.fillna(0.0).to_numpy(dtype=np.float64),
        )
        evaluated.insert(0, "mode", mode.name)
        prediction_frames.append(evaluated)

        summary_rows.append(
            _summary_row(
                mode_name=mode.name,
                frame=frame,
                feature_names=feature_names,
                cv_predictions=cv_predictions,
                fold_summary=fold_summary,
                final_model=final_model,
                program_text=program_text,
                rendered_text=rendered_text,
                simplified_text=simplified_text,
                group_column=args.group_column,
            )
        )
        equation_rows.append(
            {
                "mode": mode.name,
                "engine": "gplearn",
                "feature_scope": str(frame["student_feature_scope"].iloc[0]),
                "group_column": args.group_column,
                "population_size": args.population_size,
                "generations": args.generations,
                "parsimony": args.parsimony,
                "feature_names": ",".join(feature_names),
                "equation": program_text,
                "equation_rendered": rendered_text,
                "equation_simplified": simplified_text,
                "program_length": _safe_program_int(final_model._program, "length_"),
                "program_depth": _safe_program_int(final_model._program, "depth_"),
                "raw_fitness": float(getattr(final_model._program, "raw_fitness_", math.nan)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    equations_df = pd.DataFrame(equation_rows)
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
