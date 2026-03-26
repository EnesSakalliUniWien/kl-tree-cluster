#!/usr/bin/env python3
"""
Experiment 31g: distilled runtime deflation equation.

This script turns the deflation-student findings into a compact candidate
runtime equation aligned with the live adjusted-Wald code path.

Design:
- fit a leave-one-family-out logistic gate for whether extra deflation is needed
- use a constant uplift equal to the training-fold median positive extra deflation
- replay BH with

    c_student = c_global * exp(gate_prob * uplift)

The feature set is intentionally restricted to quantities that are already
available, or trivial to expose, in the live adjusted-Wald resolver:
- neglog_p_global
- log_n_null_case
- log_effective_n
- log_n_parent

Outputs:
- summary CSV
- coefficients CSV
- predictions CSV
- markdown summary
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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
    "log_n_null_case",
    "log_effective_n",
    "log_n_parent",
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
        description="Fit a distilled runtime deflation gate + constant uplift equation."
    )
    parser.add_argument("--modes", default="baseline,one_active_1d")
    parser.add_argument("--group-column", default="case_family")
    parser.add_argument("--positive-threshold", type=float, default=1e-9)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--summary-output-csv",
        default=enhancement_lab_results_relative("_oracle_policy_distilled_deflation_summary.csv"),
    )
    parser.add_argument(
        "--coefficients-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_distilled_deflation_coefficients.csv"
        ),
    )
    parser.add_argument(
        "--predictions-output-csv",
        default=enhancement_lab_results_relative(
            "_oracle_policy_distilled_deflation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--model-output-json",
        default=enhancement_lab_results_relative(
            "_oracle_policy_distilled_deflation_coefficients.json"
        ),
    )
    parser.add_argument(
        "--summary-markdown",
        default=enhancement_lab_results_relative("_oracle_policy_distilled_deflation_summary.md"),
    )
    return parser.parse_args()


def _resolve_modes(arg: str) -> list[ModeSpec]:
    requested = {token.strip() for token in arg.split(",") if token.strip()}
    resolved = [mode for mode in _DEFAULT_MODES if mode.name in requested]
    if not resolved:
        raise ValueError(f"No known modes requested from {sorted(requested)}")
    return resolved


def _prepare_frame(path: Path, *, positive_threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[~frame["is_null_like"]].copy().reset_index(drop=True)
    frame["neglog_p_global"] = -np.log10(
        np.maximum(frame["p_global"].to_numpy(dtype=np.float64), 1e-300)
    )
    frame["log_effective_n"] = np.log1p(
        np.maximum(frame["effective_n"].to_numpy(dtype=np.float64), 0.0)
    )
    frame["log_n_null_case"] = np.log1p(
        np.maximum(frame["n_null_case"].to_numpy(dtype=np.float64), 0.0)
    )
    frame["extra_log_deflation"] = np.maximum(
        frame["log_c_ratio"].to_numpy(dtype=np.float64),
        0.0,
    )
    frame["gate_target"] = frame["extra_log_deflation"] > positive_threshold
    return frame


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


def _fit_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _extract_raw_coefficients(model: Pipeline) -> tuple[float, dict[str, float]]:
    scaler = model.named_steps["scaler"]
    logit = model.named_steps["logit"]
    coef = logit.coef_[0]
    intercept = float(logit.intercept_[0])
    raw_coef = coef / scaler.scale_
    raw_intercept = intercept - float(np.sum(coef * scaler.mean_ / scaler.scale_))
    return raw_intercept, dict(zip(_FEATURES, raw_coef, strict=False))


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(balanced_accuracy_score(y_true, y_pred))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def _equation_string(intercept: float, coef_map: dict[str, float]) -> str:
    terms = [f"{intercept:.6f}"]
    for feature in _FEATURES:
        coef = float(coef_map[feature])
        sign = "+" if coef >= 0 else "-"
        terms.append(f" {sign} {abs(coef):.6f}*{feature}")
    linear = "".join(terms)
    return f"sigmoid({linear})"


def _student_p_values(
    frame: pd.DataFrame, extra_log_deflation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c_global = np.maximum(frame["c_global"].to_numpy(dtype=np.float64), 1.0)
    c_student = c_global * np.exp(np.maximum(extra_log_deflation, 0.0))
    p_student = np.array(
        [
            float(chi2.sf(float(stat) / max(float(c_hat), 1.0), df=int(k)))
            for stat, c_hat, k in zip(
                frame["stat"].to_numpy(dtype=np.float64),
                c_student,
                frame["k"].to_numpy(dtype=np.int64),
                strict=False,
            )
        ],
        dtype=np.float64,
    )
    return c_student, p_student


def _evaluate_mode(
    frame: pd.DataFrame, *, group_column: str, gate_threshold: float
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    x = frame[_FEATURES].to_numpy(dtype=np.float64)
    y = frame["gate_target"].astype(int).to_numpy(dtype=np.int64)
    groups = frame[group_column].to_numpy()

    gate_prob = np.full(len(frame), np.nan, dtype=np.float64)
    uplift = np.full(len(frame), np.nan, dtype=np.float64)
    gate_ba_scores: list[float] = []
    gate_auc_scores: list[float] = []

    for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
        train = frame.iloc[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
        model = _fit_pipeline()
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(test_x)[:, 1]
        gate_prob[test_idx] = prob
        gate_ba_scores.append(
            _safe_balanced_accuracy(test_y, (prob >= gate_threshold).astype(np.int64))
        )
        gate_auc_scores.append(_safe_roc_auc(test_y, prob))
        positive_train = train.loc[train["gate_target"], "extra_log_deflation"]
        uplift[test_idx] = float(positive_train.median()) if not positive_train.empty else 0.0

    evaluated = frame.copy()
    evaluated["gate_probability"] = gate_prob
    evaluated["fold_uplift"] = uplift
    evaluated["student_extra_soft"] = gate_prob * uplift
    evaluated["student_extra_hard"] = (gate_prob >= gate_threshold).astype(np.float64) * uplift

    c_soft, p_soft = _student_p_values(
        evaluated, evaluated["student_extra_soft"].to_numpy(dtype=np.float64)
    )
    c_hard, p_hard = _student_p_values(
        evaluated, evaluated["student_extra_hard"].to_numpy(dtype=np.float64)
    )
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

    final_model = _fit_pipeline()
    final_model.fit(x, y)
    intercept, coef_map = _extract_raw_coefficients(final_model)
    overall_uplift = float(frame.loc[frame["gate_target"], "extra_log_deflation"].median())

    summary = {
        "n_rows": int(len(frame)),
        "n_gate_positive": int(frame["gate_target"].sum()),
        "gate_balanced_accuracy": float(np.nanmean(gate_ba_scores)),
        "gate_roc_auc": float(np.nanmean(gate_auc_scores)),
        "baseline_false_global": int(evaluated["baseline_false_global"].sum()),
        "soft_false_global": int(evaluated["soft_false_global"].sum()),
        "hard_false_global": int(evaluated["hard_false_global"].sum()),
        "mean_c_global": float(evaluated["c_global"].mean()),
        "mean_c_student_soft": float(evaluated["c_student_soft"].mean()),
        "mean_c_student_hard": float(evaluated["c_student_hard"].mean()),
        "uplift_median_positive": overall_uplift,
        "equation": _equation_string(intercept, coef_map),
        "intercept": intercept,
        **{f"coef_{feature}": coef_map[feature] for feature in _FEATURES},
    }

    coefficients = {
        "intercept": intercept,
        "uplift_median_positive": overall_uplift,
        **coef_map,
    }
    return summary, evaluated, coefficients


def _write_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31g: Distilled Runtime Deflation Equation",
        "",
        "Deployable form:",
        "",
        "```text",
        "gate_prob = sigmoid(beta0 + beta1*neglog_p_global + beta2*log_n_null_case + beta3*log_effective_n + beta4*log_n_parent)",
        "extra_soft = gate_prob * uplift_median_positive",
        "c_student = c_global * exp(extra_soft)",
        "```",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.mode}",
                "",
                f"- rows: `{int(row.n_rows)}`",
                f"- gate positives: `{int(row.n_gate_positive)}`",
                f"- gate balanced accuracy: `{float(row.gate_balanced_accuracy):.6f}`",
                f"- gate ROC AUC: `{float(row.gate_roc_auc):.6f}`",
                f"- false-global count: baseline `{int(row.baseline_false_global)}`, soft `{int(row.soft_false_global)}`, hard `{int(row.hard_false_global)}`",
                f"- uplift median positive: `{float(row.uplift_median_positive):.6f}`",
                "",
                "Equation:",
                "",
                "```text",
                str(row.equation),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    modes = _resolve_modes(args.modes)

    summary_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for mode in modes:
        path = resolve_enhancement_lab_artifact_path(mode.rows_csv, for_input=True)
        frame = _prepare_frame(path, positive_threshold=args.positive_threshold)
        summary, evaluated, coefficients = _evaluate_mode(
            frame,
            group_column=args.group_column,
            gate_threshold=args.gate_threshold,
        )
        summary_rows.append({"mode": mode.name, **summary})
        coefficient_rows.append({"mode": mode.name, **coefficients})
        prediction_frames.append(evaluated.assign(mode=mode.name))

    summary_df = pd.DataFrame(summary_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)
    predictions_df = pd.concat(prediction_frames, axis=0, ignore_index=True)

    summary_path = resolve_enhancement_lab_artifact_path(args.summary_output_csv)
    coefficients_path = resolve_enhancement_lab_artifact_path(args.coefficients_output_csv)
    predictions_path = resolve_enhancement_lab_artifact_path(args.predictions_output_csv)
    model_json_path = resolve_enhancement_lab_artifact_path(args.model_output_json)
    markdown_path = resolve_enhancement_lab_artifact_path(args.summary_markdown)
    for path in [summary_path, coefficients_path, predictions_path, model_json_path, markdown_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    model_payload = {
        ("off" if row["mode"] == "baseline" else "per_tree_load_guard"): {
            "intercept": float(row["intercept"]),
            "coef_neglog_p_global": float(row["neglog_p_global"]),
            "coef_log_n_null_case": float(row["log_n_null_case"]),
            "coef_log_effective_n": float(row["log_effective_n"]),
            "coef_log_n_parent": float(row["log_n_parent"]),
            "uplift_median_positive": float(row["uplift_median_positive"]),
        }
        for row in coefficient_rows
    }
    model_json_path.write_text(
        json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(summary_df, markdown_path)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote coefficients to {coefficients_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote model JSON to {model_json_path}")
    print(f"Wrote markdown to {markdown_path}")


if __name__ == "__main__":
    main()
