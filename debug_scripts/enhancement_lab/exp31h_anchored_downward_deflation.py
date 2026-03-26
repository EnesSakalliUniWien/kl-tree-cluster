#!/usr/bin/env python3
"""
Experiment 31h: anchored downward-only deflation.

Goal
----
Test a safe node-wise deflation family that never exceeds the current global
adjustment:

    c_student = 1 + (c_global - 1) * r_student

with

    0 <= r_student <= 1

This guarantees:

    1 <= c_student <= c_global

so the student can only *reduce* global deflation. It cannot create the
catastrophic over-deflation failures seen in the df-proximity replacement.

Teacher target
--------------
We define a downward-only oracle target by capping the permutation teacher at
the current global baseline:

    c_teacher_down = min(c_global, max(c_perm_mean, 1))

and fit the bounded ratio

    r_target = (c_teacher_down - 1) / (c_global - 1)

clipped into [0, 1].

Evaluation
----------
For each exp31 row export:

1. Fit a leave-one-family-out bounded student on runtime-only features.
2. Replay BH using:
   - baseline global c_global
   - oracle teacher_down
   - student prediction
3. Sweep shrink strengths eta in [0, 1]:

      r_eta = 1 - eta * (1 - r)

   so eta=0 is baseline and eta=1 is full downward correction.

Outputs
-------
- summary CSV
- coefficients CSV
- predictions CSV
- case delta CSV
- markdown summary
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from debug_scripts.enhancement_lab.lab_helpers import (
    enhancement_lab_results_relative,
    resolve_enhancement_lab_artifact_path,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (
    benjamini_hochberg_correction,
)

_ROOT = Path(__file__).resolve().parents[2]
_FEATURES = [
    "neglog_p_global",
    "neglog_edge_weight",
    "log_n_parent",
    "log_k",
    "log_effective_n",
    "log_n_null_case",
    "log_stat_per_k",
    "abs_log_k_mismatch",
]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    rows_csv: str


_DEFAULT_MODES: tuple[ModeSpec, ...] = (
    ModeSpec("baseline", enhancement_lab_results_relative("_oracle_policy_rows.csv")),
    ModeSpec(
        "one_active_1d",
        enhancement_lab_results_relative("_oracle_policy_rows_one_active_1d.csv"),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate anchored downward-only runtime deflation against exp31 oracle rows."
    )
    parser.add_argument("--modes", default="baseline,one_active_1d")
    parser.add_argument("--group-column", default="case_family")
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--logit-eps", type=float, default=1e-4)
    parser.add_argument("--shrink-strengths", default="0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--summary-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_anchored_downward_summary.csv"),
    )
    parser.add_argument(
        "--coefficients-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_anchored_downward_coefficients.csv"
        ),
    )
    parser.add_argument(
        "--predictions-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_anchored_downward_predictions.csv"
        ),
    )
    parser.add_argument(
        "--case-delta-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_anchored_downward_case_deltas.csv"
        ),
    )
    parser.add_argument(
        "--summary-markdown",
        default=enhancement_lab_results_relative("_oracle_policy_anchored_downward_summary.md"),
    )
    return parser.parse_args()


def _resolve_modes(arg: str) -> list[ModeSpec]:
    requested = {token.strip() for token in arg.split(",") if token.strip()}
    resolved = [mode for mode in _DEFAULT_MODES if mode.name in requested]
    if not resolved:
        raise ValueError(f"No known modes requested from {sorted(requested)}")
    return resolved


def _parse_strengths(arg: str) -> list[float]:
    strengths = sorted({float(token.strip()) for token in arg.split(",") if token.strip()})
    if not strengths:
        raise ValueError("At least one shrink strength is required.")
    for value in strengths:
        if value < 0.0 or value > 1.0:
            raise ValueError("Shrink strengths must lie in [0, 1].")
    return strengths


def _clip_unit(values: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(values, eps, 1.0 - eps)


def _compute_case_log_k_center(full_frame: pd.DataFrame) -> pd.DataFrame:
    full = full_frame.copy()
    full["weight"] = np.maximum(full["edge_weight"].to_numpy(dtype=np.float64), 1e-12)
    full["log_k_full"] = np.log(np.maximum(full["k"].to_numpy(dtype=np.float64), 1.0))

    rows: list[dict[str, float | str]] = []
    for case_name, case_frame in full.groupby("case_name", sort=False):
        null_like = case_frame.loc[case_frame["is_null_like"]].copy()
        pool = null_like if not null_like.empty else case_frame
        weights = pool["weight"].to_numpy(dtype=np.float64)
        log_k = pool["log_k_full"].to_numpy(dtype=np.float64)
        center = float(np.average(log_k, weights=weights))
        variance = float(np.average((log_k - center) ** 2, weights=weights))
        rows.append(
            {
                "case_name": case_name,
                "case_log_k_center": center,
                "case_log_k_sd": float(np.sqrt(max(variance, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _prepare_frame(path: Path, *, logit_eps: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    case_k = _compute_case_log_k_center(frame)
    focal = frame.loc[~frame["is_null_like"]].copy().reset_index(drop=True)
    focal = focal.merge(case_k, on="case_name", how="left")

    focal["neglog_p_global"] = -np.log10(
        np.maximum(focal["p_global"].to_numpy(dtype=np.float64), 1e-300)
    )
    focal["neglog_edge_weight"] = -np.log10(
        np.maximum(focal["edge_weight"].to_numpy(dtype=np.float64), 1e-300)
    )
    focal["log_effective_n"] = np.log1p(
        np.maximum(focal["effective_n"].to_numpy(dtype=np.float64), 0.0)
    )
    focal["log_n_null_case"] = np.log1p(
        np.maximum(focal["n_null_case"].to_numpy(dtype=np.float64), 0.0)
    )
    focal["log_k"] = np.log(np.maximum(focal["k"].to_numpy(dtype=np.float64), 1.0))
    focal["log_n_parent"] = np.log(np.maximum(focal["n_parent"].to_numpy(dtype=np.float64), 1.0))
    focal["log_stat_per_k"] = np.log(
        np.maximum(
            focal["stat"].to_numpy(dtype=np.float64)
            / np.maximum(focal["k"].to_numpy(dtype=np.float64), 1.0),
            1e-300,
        )
    )
    focal["abs_log_k_mismatch"] = np.abs(
        focal["log_k"].to_numpy(dtype=np.float64)
        - focal["case_log_k_center"].to_numpy(dtype=np.float64)
    )

    c_global = np.maximum(focal["c_global"].to_numpy(dtype=np.float64), 1.0)
    c_perm = np.maximum(focal["c_perm_mean"].to_numpy(dtype=np.float64), 1.0)
    c_teacher_down = np.minimum(c_global, c_perm)
    denom = np.maximum(c_global - 1.0, 1e-12)
    r_target = np.where(c_global <= 1.0 + 1e-12, 1.0, (c_teacher_down - 1.0) / denom)
    r_target = np.clip(r_target, 0.0, 1.0)
    focal["c_teacher_down"] = c_teacher_down
    focal["r_target"] = r_target
    focal["z_target"] = logit(_clip_unit(r_target, logit_eps))
    return focal


def _fit_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _extract_raw_coefficients(model: Pipeline) -> tuple[float, dict[str, float]]:
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    coef = ridge.coef_
    intercept = float(ridge.intercept_)
    raw_coef = coef / scaler.scale_
    raw_intercept = intercept - float(np.sum(coef * scaler.mean_ / scaler.scale_))
    return raw_intercept, dict(zip(_FEATURES, raw_coef, strict=False))


def _apply_casewise_bh(frame: pd.DataFrame, p_column: str, out_column: str) -> None:
    flags = np.zeros(len(frame), dtype=bool)
    for _case_name, case_frame in frame.groupby("case_name", sort=False):
        idx = case_frame.index.to_numpy(dtype=np.int64)
        rejected, _, _ = benjamini_hochberg_correction(
            case_frame[p_column].to_numpy(dtype=np.float64),
            alpha=config.SIBLING_ALPHA,
        )
        flags[idx] = rejected.astype(bool)
    frame[out_column] = flags


def _compute_p_values(stat: np.ndarray, k: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(chi2.sf(float(s) / max(float(c), 1.0), df=int(df)))
            for s, df, c in zip(stat, k, c_hat, strict=False)
        ],
        dtype=np.float64,
    )


def _evaluate_method(
    frame: pd.DataFrame,
    *,
    method_name: str,
    c_values: np.ndarray,
) -> dict[str, float | int | str]:
    evaluated = frame.copy()
    evaluated["c_method"] = c_values
    evaluated["p_method"] = _compute_p_values(
        evaluated["stat"].to_numpy(dtype=np.float64),
        evaluated["k"].to_numpy(dtype=np.int64),
        evaluated["c_method"].to_numpy(dtype=np.float64),
    )
    _apply_casewise_bh(evaluated, "p_method", "bh_method")
    false_global = evaluated["bh_method"] & (~evaluated["perm_bh_reject"])
    false_local = (~evaluated["bh_method"]) & evaluated["perm_bh_reject"]
    return {
        "method": method_name,
        "n_reject": int(evaluated["bh_method"].sum()),
        "false_global": int(false_global.sum()),
        "false_local": int(false_local.sum()),
        "mean_c": float(np.mean(c_values)),
        "median_c": float(np.median(c_values)),
        "mean_p": float(np.mean(evaluated["p_method"].to_numpy(dtype=np.float64))),
    }


def _evaluate_mode(
    frame: pd.DataFrame,
    *,
    group_column: str,
    ridge_alpha: float,
    logit_eps: float,
    strengths: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = frame[_FEATURES].to_numpy(dtype=np.float64)
    y = frame["z_target"].to_numpy(dtype=np.float64)
    groups = frame[group_column].to_numpy()

    pred_z = np.full(len(frame), np.nan, dtype=np.float64)
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, groups=groups):
        model = _fit_pipeline(ridge_alpha)
        model.fit(x[train_idx], y[train_idx])
        pred_z[test_idx] = model.predict(x[test_idx])

    pred_r = expit(pred_z)
    pred_r = np.clip(pred_r, 0.0, 1.0)

    evaluated = frame.copy()
    evaluated["r_pred"] = pred_r
    evaluated["c_baseline"] = np.maximum(evaluated["c_global"].to_numpy(dtype=np.float64), 1.0)
    evaluated["c_teacher_down"] = np.maximum(
        evaluated["c_teacher_down"].to_numpy(dtype=np.float64),
        1.0,
    )

    method_rows: list[dict[str, float | int | str]] = []
    case_rows: list[dict[str, float | int | str]] = []

    baseline_eval = _evaluate_method(
        evaluated,
        method_name="baseline",
        c_values=evaluated["c_baseline"].to_numpy(dtype=np.float64),
    )
    teacher_eval = _evaluate_method(
        evaluated,
        method_name="teacher_down_eta1.00",
        c_values=evaluated["c_teacher_down"].to_numpy(dtype=np.float64),
    )
    method_rows.extend([baseline_eval, teacher_eval])

    for eta in strengths:
        r_teacher_eta = 1.0 - eta * (1.0 - evaluated["r_target"].to_numpy(dtype=np.float64))
        r_student_eta = 1.0 - eta * (1.0 - evaluated["r_pred"].to_numpy(dtype=np.float64))
        c_global = evaluated["c_baseline"].to_numpy(dtype=np.float64)
        c_teacher_eta = 1.0 + (c_global - 1.0) * r_teacher_eta
        c_student_eta = 1.0 + (c_global - 1.0) * r_student_eta

        evaluated[f"r_teacher_eta_{eta:.2f}"] = r_teacher_eta
        evaluated[f"r_student_eta_{eta:.2f}"] = r_student_eta
        evaluated[f"c_student_eta_{eta:.2f}"] = c_student_eta

        teacher_method = _evaluate_method(
            evaluated,
            method_name=f"teacher_down_eta{eta:.2f}",
            c_values=c_teacher_eta,
        )
        student_method = _evaluate_method(
            evaluated,
            method_name=f"student_eta{eta:.2f}",
            c_values=c_student_eta,
        )
        method_rows.extend([teacher_method, student_method])

        case_eval = evaluated[["case_name", "perm_bh_reject"]].copy()
        case_eval["baseline_bh"] = False
        case_eval["student_bh"] = False

        baseline_p = _compute_p_values(
            evaluated["stat"].to_numpy(dtype=np.float64),
            evaluated["k"].to_numpy(dtype=np.int64),
            c_global,
        )
        evaluated["p_baseline"] = baseline_p
        _apply_casewise_bh(evaluated, "p_baseline", "bh_baseline")

        student_p = _compute_p_values(
            evaluated["stat"].to_numpy(dtype=np.float64),
            evaluated["k"].to_numpy(dtype=np.int64),
            c_student_eta,
        )
        evaluated["p_student_tmp"] = student_p
        _apply_casewise_bh(evaluated, "p_student_tmp", "bh_student_tmp")

        for case_name, case_frame in evaluated.groupby("case_name", sort=False):
            base_false_global = int(
                (case_frame["bh_baseline"] & (~case_frame["perm_bh_reject"])).sum()
            )
            student_false_global = int(
                (case_frame["bh_student_tmp"] & (~case_frame["perm_bh_reject"])).sum()
            )
            case_rows.append(
                {
                    "case_name": case_name,
                    "eta": eta,
                    "baseline_false_global": base_false_global,
                    "student_false_global": student_false_global,
                    "delta_false_global": student_false_global - base_false_global,
                    "mean_r_target": float(case_frame["r_target"].mean()),
                    "mean_r_pred": float(case_frame["r_pred"].mean()),
                    "mean_c_global": float(case_frame["c_baseline"].mean()),
                    "mean_c_student": float(c_student_eta[case_frame.index].mean()),
                }
            )

    final_model = _fit_pipeline(ridge_alpha)
    final_model.fit(x, y)
    intercept, coef_map = _extract_raw_coefficients(final_model)

    coeff_rows = [
        {"term": "intercept", "value": intercept},
        *({"term": feature, "value": float(coef_map[feature])} for feature in _FEATURES),
    ]

    metrics = pd.DataFrame(method_rows)
    metrics["target_r_mae"] = math.nan
    metrics["target_r_r2"] = math.nan
    metrics["target_logc_mae"] = math.nan

    for eta in strengths:
        method_name = f"student_eta{eta:.2f}"
        mask = metrics["method"] == method_name
        r_student_eta = evaluated[f"r_student_eta_{eta:.2f}"].to_numpy(dtype=np.float64)
        c_student_eta = evaluated[f"c_student_eta_{eta:.2f}"].to_numpy(dtype=np.float64)
        metrics.loc[mask, "target_r_mae"] = float(
            mean_absolute_error(evaluated["r_target"], r_student_eta)
        )
        metrics.loc[mask, "target_r_r2"] = float(r2_score(evaluated["r_target"], r_student_eta))
        metrics.loc[mask, "target_logc_mae"] = float(
            mean_absolute_error(
                np.log(evaluated["c_teacher_down"].to_numpy(dtype=np.float64)),
                np.log(np.maximum(c_student_eta, 1.0)),
            )
        )

        teacher_name = f"teacher_down_eta{eta:.2f}"
        teacher_mask = metrics["method"] == teacher_name
        metrics.loc[teacher_mask, "target_r_mae"] = float(
            mean_absolute_error(
                evaluated["r_target"],
                evaluated[f"r_teacher_eta_{eta:.2f}"].to_numpy(dtype=np.float64),
            )
        )
        metrics.loc[teacher_mask, "target_r_r2"] = float(
            r2_score(
                evaluated["r_target"],
                evaluated[f"r_teacher_eta_{eta:.2f}"].to_numpy(dtype=np.float64),
            )
        )
        metrics.loc[teacher_mask, "target_logc_mae"] = float(
            mean_absolute_error(
                np.log(evaluated["c_teacher_down"].to_numpy(dtype=np.float64)),
                np.log(
                    np.maximum(
                        1.0
                        + (evaluated["c_baseline"].to_numpy(dtype=np.float64) - 1.0)
                        * evaluated[f"r_teacher_eta_{eta:.2f}"].to_numpy(dtype=np.float64),
                        1.0,
                    )
                ),
            )
        )

    baseline_mask = metrics["method"] == "baseline"
    metrics.loc[baseline_mask, "target_r_mae"] = float(
        mean_absolute_error(evaluated["r_target"], np.ones(len(evaluated)))
    )
    metrics.loc[baseline_mask, "target_r_r2"] = float(
        r2_score(evaluated["r_target"], np.ones(len(evaluated)))
    )
    metrics.loc[baseline_mask, "target_logc_mae"] = float(
        mean_absolute_error(
            np.log(evaluated["c_teacher_down"].to_numpy(dtype=np.float64)),
            np.log(evaluated["c_baseline"].to_numpy(dtype=np.float64)),
        )
    )

    return metrics, pd.DataFrame(coeff_rows), pd.DataFrame(case_rows), evaluated


def _summary_markdown(summary: pd.DataFrame, case_deltas: pd.DataFrame) -> str:
    lines = [
        "# Anchored Downward Deflation",
        "",
    ]
    for mode_name, mode_frame in summary.groupby("mode", sort=False):
        lines.append(f"## {mode_name}")
        mode_sorted = mode_frame.sort_values(["method"])
        for row in mode_sorted.itertuples(index=False):
            lines.append(
                f"- `{row.method}`: rejects={int(row.n_reject)}, false_global={int(row.false_global)}, "
                f"target_r_mae={float(row.target_r_mae):.4f}, target_r_r2={float(row.target_r_r2):.4f}, "
                f"logc_mae={float(row.target_logc_mae):.4f}"
            )
        lines.append("")

        student_rows = mode_frame.loc[mode_frame["method"].str.startswith("student_eta")].copy()
        if not student_rows.empty:
            best = student_rows.sort_values(["false_global", "target_logc_mae"]).iloc[0]
            lines.append(
                f"Best student tradeoff: `{best['method']}` with false_global={int(best['false_global'])} "
                f"and logc_mae={float(best['target_logc_mae']):.4f}."
            )
            lines.append("")

    worst_case = case_deltas.sort_values("delta_false_global", ascending=False).head(10)
    if not worst_case.empty:
        lines.append("## Largest Case False-Global Increases")
        for row in worst_case.itertuples(index=False):
            lines.append(
                f"- `{row.case_name}` eta={row.eta:.2f}: baseline={int(row.baseline_false_global)}, "
                f"student={int(row.student_false_global)}, delta={int(row.delta_false_global)}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    modes = _resolve_modes(args.modes)
    strengths = _parse_strengths(args.shrink_strengths)

    summary_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    case_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for mode in modes:
        frame = _prepare_frame(
            resolve_enhancement_lab_artifact_path(mode.rows_csv, for_input=True),
            logit_eps=args.logit_eps,
        )
        summary, coefficients, cases, evaluated = _evaluate_mode(
            frame,
            group_column=args.group_column,
            ridge_alpha=args.ridge_alpha,
            logit_eps=args.logit_eps,
            strengths=strengths,
        )
        summary.insert(0, "mode", mode.name)
        coefficients.insert(0, "mode", mode.name)
        cases.insert(0, "mode", mode.name)
        evaluated.insert(0, "mode", mode.name)
        summary_rows.append(summary)
        coefficient_rows.append(coefficients)
        case_rows.append(cases)
        prediction_rows.append(evaluated)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    coefficients_df = pd.concat(coefficient_rows, ignore_index=True)
    case_df = pd.concat(case_rows, ignore_index=True)

    summary_path = resolve_enhancement_lab_artifact_path(args.summary_output_csv)
    coeff_path = resolve_enhancement_lab_artifact_path(args.coefficients_output_csv)
    pred_path = resolve_enhancement_lab_artifact_path(args.predictions_output_csv)
    case_path = resolve_enhancement_lab_artifact_path(args.case_delta_output_csv)
    md_path = resolve_enhancement_lab_artifact_path(args.summary_markdown)
    for path in (summary_path, coeff_path, pred_path, case_path, md_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_path, index=False)
    coefficients_df.to_csv(coeff_path, index=False)
    case_df.to_csv(case_path, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(pred_path, index=False)

    md_path.write_text(_summary_markdown(summary_df, case_df), encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote coefficients: {coeff_path}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote case deltas: {case_path}")
    print(f"Wrote markdown: {md_path}")


if __name__ == "__main__":
    main()
