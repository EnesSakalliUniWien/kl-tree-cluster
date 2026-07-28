"""Descriptive candidate-law analysis for selected-edge geometry rows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS_ROLE = "diagnostic_candidate_law_search"
TARGET_COLUMN = "edge_rejected"
REQUIRED_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        "edge_bh_action",
        "edge_statistic_margin",
        "selected_eigenvalue_over_mp",
        "tree_balance",
        "sample_ratio",
        "path_length_from_root",
    }
)
CANDIDATE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "edge_action": ("edge_bh_action",),
    "edge_margin": ("edge_statistic_margin",),
    "sampling_geometry": ("sample_ratio", "tree_balance", "path_length_from_root"),
    "spectral": ("selected_eigenvalue_over_mp",),
    "edge_action_plus_spectral": ("edge_bh_action", "selected_eigenvalue_over_mp"),
    "combined": (
        "edge_bh_action",
        "edge_statistic_margin",
        "sample_ratio",
        "tree_balance",
        "path_length_from_root",
        "selected_eigenvalue_over_mp",
    ),
}


def _validate_edge_rows(edge_rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(edge_rows.columns))
    if missing:
        raise ValueError(f"Selected-edge geometry rows are missing columns: {missing!r}.")
    target = edge_rows[TARGET_COLUMN].astype(bool).astype(int)
    if target.nunique() < 2:
        raise ValueError(
            "Selected-edge geometry analysis requires both rejected and non-rejected edge rows."
        )


def _clean_model_table(
    edge_rows: pd.DataFrame, features: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    columns = list(features)
    table = edge_rows[columns + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    if table.empty:
        raise ValueError(f"No finite rows available for features {columns!r}.")
    target = table[TARGET_COLUMN].astype(bool).astype(int)
    if target.nunique() < 2:
        raise ValueError(f"Features {columns!r} have only one target class after cleaning.")
    return table[columns], target


def _safe_cv_predictions(model, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    min_class = int(y.value_counts().min())
    if min_class < 2:
        fitted = model.fit(x, y)
        return fitted.predict_proba(x)[:, 1]
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260604)
    return cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]


def _score_predictions(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    result = {
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }
    if y.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability))
    else:
        result["roc_auc"] = float("nan")
    return result


def _spearman_rows(edge_rows: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = edge_rows[TARGET_COLUMN].astype(bool).astype(int).to_numpy(dtype=float)
    for feature in sorted(REQUIRED_COLUMNS - {TARGET_COLUMN}):
        values = edge_rows[feature].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 3 or np.unique(values[mask]).size < 2:
            statistic = float("nan")
            p_value = float("nan")
        else:
            statistic, p_value = spearmanr(values[mask], target[mask])
        rows.append(
            {
                "model_name": f"spearman::{feature}",
                "model_family": "spearman",
                "features": feature,
                "n_rows": int(mask.sum()),
                "score": float(statistic),
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "brier_score": float("nan"),
                "spearman_p_value": float(p_value),
                "analysis_role": ANALYSIS_ROLE,
            }
        )
    return rows


def _model_rows(edge_rows: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_name, features in CANDIDATE_FEATURE_GROUPS.items():
        x, y = _clean_model_table(edge_rows, features)
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=20260604),
        )
        logistic_probability = _safe_cv_predictions(logistic, x, y)
        logistic_scores = _score_predictions(y, logistic_probability)
        rows.append(
            {
                "model_name": f"logistic::{group_name}",
                "model_family": "logistic",
                "features": ",".join(features),
                "n_rows": int(len(x)),
                "score": logistic_scores["roc_auc"],
                "roc_auc": logistic_scores["roc_auc"],
                "average_precision": logistic_scores["average_precision"],
                "brier_score": logistic_scores["brier_score"],
                "spearman_p_value": float("nan"),
                "analysis_role": ANALYSIS_ROLE,
            }
        )

        if len(x) >= 8 and y.value_counts().min() >= 2:
            gradient = GradientBoostingClassifier(random_state=20260604)
            gradient_probability = _safe_cv_predictions(gradient, x, y)
            gradient_scores = _score_predictions(y, gradient_probability)
            fitted = gradient.fit(x, y)
            importance = permutation_importance(
                fitted,
                x,
                y,
                n_repeats=5,
                random_state=20260604,
                scoring="roc_auc",
            )
            feature_importance = ";".join(
                f"{feature}:{value:.6g}"
                for feature, value in zip(features, importance.importances_mean, strict=True)
            )
            rows.append(
                {
                    "model_name": f"gradient_boosting::{group_name}",
                    "model_family": "gradient_boosting",
                    "features": ",".join(features),
                    "n_rows": int(len(x)),
                    "score": gradient_scores["roc_auc"],
                    "roc_auc": gradient_scores["roc_auc"],
                    "average_precision": gradient_scores["average_precision"],
                    "brier_score": gradient_scores["brier_score"],
                    "spearman_p_value": float("nan"),
                    "permutation_importance": feature_importance,
                    "analysis_role": ANALYSIS_ROLE,
                }
            )
    return rows


def analyze_selected_edge_geometry(*, edge_path: Path, output_path: Path) -> pd.DataFrame:
    """Rank descriptive selected-edge candidate laws."""
    edge_rows = pd.read_csv(edge_path)
    _validate_edge_rows(edge_rows)
    rows = _spearman_rows(edge_rows) + _model_rows(edge_rows)
    result = pd.DataFrame.from_records(rows)
    if "permutation_importance" not in result.columns:
        result["permutation_importance"] = ""
    result = result.sort_values(
        ["model_family", "score"],
        ascending=[True, False],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_selected_edge_geometry(edge_path=args.edge_rows, output_path=args.output)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
