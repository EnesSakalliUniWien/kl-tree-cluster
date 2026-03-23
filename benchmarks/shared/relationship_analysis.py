"""Post-run statistical analysis for benchmark relationships."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression

from benchmarks.shared.plots.cover_page import GROUP_ORDER, category_group
from benchmarks.shared.util.pdf.layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "test_case": ("test_case", "Test"),
    "case_id": ("case_id", "Case_Name"),
    "case_category": ("case_category", "Case_Category"),
    "method": ("method", "Method"),
    "params": ("params", "Params"),
    "true_clusters": ("true_clusters", "True"),
    "found_clusters": ("found_clusters", "Found"),
    "samples": ("samples", "Samples"),
    "features": ("features", "Features"),
    "noise": ("noise", "Noise"),
    "ari": ("ari", "ARI"),
    "nmi": ("nmi", "NMI"),
    "purity": ("purity", "Purity"),
    "macro_recall": ("macro_recall", "Macro_Recall"),
    "macro_f1": ("macro_f1", "Macro_F1"),
    "worst_cluster_recall": ("worst_cluster_recall", "Worst_Cluster_Recall"),
    "outlier_precision": ("outlier_precision", "Outlier_Precision"),
    "outlier_recall": ("outlier_recall", "Outlier_Recall"),
    "outlier_f1": ("outlier_f1", "Outlier_F1"),
    "singleton_outlier_isolated": (
        "singleton_outlier_isolated",
        "Singleton_Outlier_Isolated",
    ),
    "grouped_outlier_cluster_recovered": (
        "grouped_outlier_cluster_recovered",
        "Grouped_Outlier_Cluster_Recovered",
    ),
    "cluster_count_abs_error": ("cluster_count_abs_error", "Cluster_Count_Abs_Error"),
    "over_split": ("over_split", "Over_Split"),
    "under_split": ("under_split", "Under_Split"),
    "status": ("status", "Status"),
    "skip_reason": ("skip_reason", "Skip_Reason"),
    "labels_length": ("labels_length", "Labels_Length"),
}

_NUMERIC_COLUMNS = (
    "test_case",
    "true_clusters",
    "found_clusters",
    "samples",
    "features",
    "noise",
    "ari",
    "nmi",
    "purity",
    "macro_recall",
    "macro_f1",
    "worst_cluster_recall",
    "outlier_precision",
    "outlier_recall",
    "outlier_f1",
    "singleton_outlier_isolated",
    "grouped_outlier_cluster_recovered",
    "cluster_count_abs_error",
    "over_split",
    "under_split",
    "labels_length",
)

_CONTINUOUS_EFFECT_LABELS = {
    "noise_z": "noise",
    "log_features_z": "feature count",
    "log_samples_z": "sample count",
    "log_true_clusters_z": "true cluster count",
    "log_samples_per_cluster_z": "samples per cluster",
    "log_features_per_cluster_z": "features per cluster",
    "log_samples_per_feature_z": "samples per feature",
    "noise_cluster_interaction_z": "noise x cluster-count interaction",
    "noise_missing": "missing noise metadata",
    "audit_available": "audit metadata available",
    "audit_root_split_rejected": "root split rejected",
    "audit_root_sibling_neglog10_p_z": "root split evidence",
    "audit_sig_sibling_fraction_z": "significant sibling-split fraction",
    "audit_sig_edge_fraction_z": "significant edge-split fraction",
    "audit_internal_nodes_z": "internal node count",
    "audit_mean_branch_length_z": "mean branch length",
}

_MODEL_CONTINUOUS_TERMS = (
    "noise_z",
    "log_features_z",
    "log_samples_z",
    "log_true_clusters_z",
    "log_samples_per_cluster_z",
    "log_features_per_cluster_z",
    "log_samples_per_feature_z",
    "noise_cluster_interaction_z",
    "noise_missing",
    "audit_available",
    "audit_root_split_rejected",
    "audit_root_sibling_neglog10_p_z",
    "audit_sig_sibling_fraction_z",
    "audit_sig_edge_fraction_z",
    "audit_internal_nodes_z",
    "audit_mean_branch_length_z",
)


@dataclass(frozen=True)
class BenchmarkRelationshipArtifacts:
    """Paths emitted by the relationship analysis pipeline."""

    report_md: Path
    augmented_rows_csv: Path
    method_summary_csv: Path
    section_summary_csv: Path
    method_section_summary_csv: Path
    correlation_summary_csv: Path
    pairwise_method_summary_csv: Path
    regression_ari_csv: Path | None
    regression_exact_k_csv: Path | None
    regression_over_split_csv: Path | None
    regression_under_split_csv: Path | None
    plots_pdf: Path | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "report_md": str(self.report_md),
            "augmented_rows_csv": str(self.augmented_rows_csv),
            "method_summary_csv": str(self.method_summary_csv),
            "section_summary_csv": str(self.section_summary_csv),
            "method_section_summary_csv": str(self.method_section_summary_csv),
            "correlation_summary_csv": str(self.correlation_summary_csv),
            "pairwise_method_summary_csv": str(self.pairwise_method_summary_csv),
            "regression_ari_csv": str(self.regression_ari_csv) if self.regression_ari_csv else None,
            "regression_exact_k_csv": (
                str(self.regression_exact_k_csv) if self.regression_exact_k_csv else None
            ),
            "regression_over_split_csv": (
                str(self.regression_over_split_csv) if self.regression_over_split_csv else None
            ),
            "regression_under_split_csv": (
                str(self.regression_under_split_csv) if self.regression_under_split_csv else None
            ),
            "plots_pdf": str(self.plots_pdf) if self.plots_pdf else None,
        }


def normalize_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy and current result columns to a canonical schema."""
    normalized = pd.DataFrame(index=df.index)
    used: set[str] = set()

    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                normalized[canonical] = df[alias]
                used.add(alias)
                break

    for canonical in _COLUMN_ALIASES:
        if canonical not in normalized.columns:
            normalized[canonical] = np.nan

    for col in df.columns:
        if col not in used and col not in normalized.columns:
            normalized[col] = df[col]

    for col in _NUMERIC_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in ("case_id", "case_category", "method", "params", "skip_reason"):
        normalized[col] = normalized[col].fillna("").astype(str)

    normalized["status"] = normalized["status"].fillna("").astype(str).str.lower()
    normalized["case_category"] = normalized["case_category"].replace("", "unknown")
    normalized["method"] = normalized["method"].replace("", "unknown")
    return normalized


def prepare_relationship_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned benchmark frame used for summaries and models."""
    normalized = normalize_results_dataframe(df)
    ok_mask = normalized["status"].eq("ok") if "status" in normalized.columns else True
    frame = normalized.loc[ok_mask].copy()
    frame = frame[frame["ari"].notna()].copy()
    if frame.empty:
        return frame

    frame["section"] = frame["case_category"].map(lambda value: category_group(str(value)))
    frame["section"] = frame["section"].replace("", "unknown")
    frame["exact_k"] = np.where(
        frame["true_clusters"].notna() & frame["found_clusters"].notna(),
        (frame["true_clusters"] == frame["found_clusters"]).astype(float),
        np.nan,
    )
    frame["cluster_error_signed"] = frame["found_clusters"] - frame["true_clusters"]
    frame["abs_cluster_error"] = frame["cluster_error_signed"].abs()
    frame["over_split_flag"] = np.where(
        frame["true_clusters"] > 0,
        (frame["cluster_error_signed"] > 0).astype(float),
        np.nan,
    )
    frame["under_split_flag"] = np.where(
        frame["true_clusters"] > 0,
        (frame["cluster_error_signed"] < 0).astype(float),
        np.nan,
    )

    noise = pd.to_numeric(frame["noise"], errors="coerce")
    noise_median = float(noise.dropna().median()) if noise.notna().any() else 0.0
    frame["noise_missing"] = noise.isna().astype(float)
    frame["noise_filled"] = noise.fillna(noise_median)

    frame["log_samples"] = np.log1p(frame["samples"].clip(lower=0))
    frame["log_features"] = np.log1p(frame["features"].clip(lower=0))
    frame["log_true_clusters"] = np.log1p(frame["true_clusters"].clip(lower=0))
    safe_true_clusters = frame["true_clusters"].replace(0, np.nan)
    safe_features = frame["features"].replace(0, np.nan)
    frame["samples_per_cluster"] = frame["samples"] / safe_true_clusters
    frame["features_per_cluster"] = frame["features"] / safe_true_clusters
    frame["samples_per_feature"] = frame["samples"] / safe_features
    frame["log_samples_per_cluster"] = np.log1p(frame["samples_per_cluster"].clip(lower=0))
    frame["log_features_per_cluster"] = np.log1p(frame["features_per_cluster"].clip(lower=0))
    frame["log_samples_per_feature"] = np.log1p(frame["samples_per_feature"].clip(lower=0))
    frame["noise_cluster_interaction"] = frame["noise_filled"] * frame["log_true_clusters"]

    for source, target in (
        ("noise_filled", "noise_z"),
        ("log_samples", "log_samples_z"),
        ("log_features", "log_features_z"),
        ("log_true_clusters", "log_true_clusters_z"),
        ("log_samples_per_cluster", "log_samples_per_cluster_z"),
        ("log_features_per_cluster", "log_features_per_cluster_z"),
        ("log_samples_per_feature", "log_samples_per_feature_z"),
        ("noise_cluster_interaction", "noise_cluster_interaction_z"),
    ):
        frame[target] = _zscore(frame[source])

    return frame


def analyze_benchmark_relationships(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    source_path: Path | None = None,
    include_plots: bool = True,
) -> BenchmarkRelationshipArtifacts:
    """Write relationship summaries, statistical models, and optional plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_relationship_frame(df)
    frame = _attach_audit_factors(frame, output_dir / "audit")

    report_md = output_dir / "benchmark_relationship_report.md"
    augmented_rows_csv = output_dir / "benchmark_relationship_augmented_rows.csv"
    method_summary_csv = output_dir / "benchmark_relationship_method_summary.csv"
    section_summary_csv = output_dir / "benchmark_relationship_section_summary.csv"
    method_section_summary_csv = output_dir / "benchmark_relationship_method_section_summary.csv"
    correlation_summary_csv = output_dir / "benchmark_relationship_correlations.csv"
    pairwise_method_summary_csv = output_dir / "benchmark_relationship_pairwise_methods.csv"
    regression_ari_csv = output_dir / "benchmark_relationship_regression_ari.csv"
    regression_exact_k_csv = output_dir / "benchmark_relationship_regression_exact_k.csv"
    regression_over_split_csv = output_dir / "benchmark_relationship_regression_over_split.csv"
    regression_under_split_csv = output_dir / "benchmark_relationship_regression_under_split.csv"
    plots_pdf = output_dir / "benchmark_relationship_plots.pdf"

    if frame.empty:
        _write_empty_report(report_md, source_path)
        pd.DataFrame().to_csv(augmented_rows_csv, index=False)
        pd.DataFrame().to_csv(method_summary_csv, index=False)
        pd.DataFrame().to_csv(section_summary_csv, index=False)
        pd.DataFrame().to_csv(method_section_summary_csv, index=False)
        pd.DataFrame().to_csv(correlation_summary_csv, index=False)
        pd.DataFrame().to_csv(pairwise_method_summary_csv, index=False)
        return BenchmarkRelationshipArtifacts(
            report_md=report_md,
            augmented_rows_csv=augmented_rows_csv,
            method_summary_csv=method_summary_csv,
            section_summary_csv=section_summary_csv,
            method_section_summary_csv=method_section_summary_csv,
            correlation_summary_csv=correlation_summary_csv,
            pairwise_method_summary_csv=pairwise_method_summary_csv,
            regression_ari_csv=None,
            regression_exact_k_csv=None,
            regression_over_split_csv=None,
            regression_under_split_csv=None,
            plots_pdf=None,
        )

    frame.to_csv(augmented_rows_csv, index=False)
    method_summary = _summarize(frame, ["method"]).sort_values(
        ["mean_ari", "exact_k_rate"], ascending=[False, False]
    )
    section_summary = _summarize(frame, ["section"])
    section_summary["section_order"] = section_summary["section"].map(_section_order)
    section_summary = section_summary.sort_values(["section_order", "section"]).drop(
        columns=["section_order"]
    )
    method_section_summary = _summarize(frame, ["method", "section"])
    method_section_summary["section_order"] = method_section_summary["section"].map(_section_order)
    method_section_summary = method_section_summary.sort_values(
        ["method", "section_order", "section"]
    ).drop(columns=["section_order"])
    correlations = _compute_correlations(frame)
    pairwise_method_summary = _compute_pairwise_method_summary(frame)

    method_summary.to_csv(method_summary_csv, index=False)
    section_summary.to_csv(section_summary_csv, index=False)
    method_section_summary.to_csv(method_section_summary_csv, index=False)
    correlations.to_csv(correlation_summary_csv, index=False)
    pairwise_method_summary.to_csv(pairwise_method_summary_csv, index=False)

    regression_tables: dict[str, pd.DataFrame] = {}
    model_stats: dict[str, dict[str, Any]] = {}
    for target, family in (
        ("ari", "ols"),
        ("exact_k", "binomial"),
        ("over_split_flag", "binomial"),
        ("under_split_flag", "binomial"),
    ):
        subset = frame if target == "ari" or target == "exact_k" else frame[frame["true_clusters"] > 1]
        table, stats = _fit_relationship_model(subset, target=target, family=family)
        if table is not None:
            regression_tables[target] = table
            model_stats[target] = stats

    if "ari" in regression_tables:
        regression_tables["ari"].to_csv(regression_ari_csv, index=False)
    else:
        regression_ari_csv = None
    if "exact_k" in regression_tables:
        regression_tables["exact_k"].to_csv(regression_exact_k_csv, index=False)
    else:
        regression_exact_k_csv = None
    if "over_split_flag" in regression_tables:
        regression_tables["over_split_flag"].to_csv(regression_over_split_csv, index=False)
    else:
        regression_over_split_csv = None
    if "under_split_flag" in regression_tables:
        regression_tables["under_split_flag"].to_csv(regression_under_split_csv, index=False)
    else:
        regression_under_split_csv = None

    _write_relationship_report(
        report_md=report_md,
        frame=frame,
        method_summary=method_summary,
        section_summary=section_summary,
        method_section_summary=method_section_summary,
        correlations=correlations,
        pairwise_method_summary=pairwise_method_summary,
        regression_tables=regression_tables,
        model_stats=model_stats,
        source_path=source_path,
    )

    if include_plots:
        _write_relationship_plots_pdf(
            plots_pdf=plots_pdf,
            frame=frame,
            method_summary=method_summary,
            section_summary=section_summary,
            method_section_summary=method_section_summary,
            pairwise_method_summary=pairwise_method_summary,
        )
    else:
        plots_pdf = None

    return BenchmarkRelationshipArtifacts(
        report_md=report_md,
        augmented_rows_csv=augmented_rows_csv,
        method_summary_csv=method_summary_csv,
        section_summary_csv=section_summary_csv,
        method_section_summary_csv=method_section_summary_csv,
        correlation_summary_csv=correlation_summary_csv,
        pairwise_method_summary_csv=pairwise_method_summary_csv,
        regression_ari_csv=regression_ari_csv,
        regression_exact_k_csv=regression_exact_k_csv,
        regression_over_split_csv=regression_over_split_csv,
        regression_under_split_csv=regression_under_split_csv,
        plots_pdf=plots_pdf,
    )


def _summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("ari", "size"),
            n_cases=("case_id", "nunique"),
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            mean_nmi=("nmi", "mean"),
            mean_purity=("purity", "mean"),
            mean_outlier_precision=("outlier_precision", "mean"),
            mean_outlier_recall=("outlier_recall", "mean"),
            mean_outlier_f1=("outlier_f1", "mean"),
            singleton_hit_rate=("singleton_outlier_isolated", "mean"),
            grouped_recovery_rate=("grouped_outlier_cluster_recovered", "mean"),
            exact_k_rate=("exact_k", "mean"),
            over_split_rate=("over_split_flag", "mean"),
            under_split_rate=("under_split_flag", "mean"),
            mean_abs_cluster_error=("abs_cluster_error", "mean"),
            mean_cluster_error_signed=("cluster_error_signed", "mean"),
            mean_true_clusters=("true_clusters", "mean"),
            mean_found_clusters=("found_clusters", "mean"),
        )
        .reset_index()
    )
    return grouped


def _compute_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "noise_filled",
        "log_features",
        "log_samples",
        "log_true_clusters",
        "log_samples_per_cluster",
        "log_features_per_cluster",
        "log_samples_per_feature",
        "noise_cluster_interaction",
        "audit_root_split_rejected",
        "audit_root_sibling_neglog10_p",
        "audit_sig_sibling_fraction",
        "audit_sig_edge_fraction",
        "audit_internal_nodes",
        "audit_mean_branch_length",
    ]
    predictors = [
        predictor
        for predictor in predictors
        if predictor in frame.columns and frame[predictor].notna().any()
    ]
    targets = ["ari", "nmi", "exact_k", "over_split_flag", "under_split_flag", "abs_cluster_error"]
    for outlier_target in (
        "outlier_precision",
        "outlier_recall",
        "outlier_f1",
        "singleton_outlier_isolated",
        "grouped_outlier_cluster_recovered",
    ):
        if outlier_target in frame.columns and frame[outlier_target].notna().any():
            targets.append(outlier_target)
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]

    if frame["method"].nunique() > 1:
        for method, method_df in frame.groupby("method"):
            scopes.append((f"method:{method}", method_df))

    rows: list[dict[str, Any]] = []
    for scope_name, scope_df in scopes:
        for predictor in predictors:
            for target in targets:
                sub = scope_df[[predictor, target]].dropna()
                if len(sub) < 5:
                    continue
                if sub[predictor].nunique() < 2 or sub[target].nunique() < 2:
                    continue
                rho, pvalue = spearmanr(sub[predictor], sub[target])
                rows.append(
                    {
                        "scope": scope_name,
                        "predictor": predictor,
                        "target": target,
                        "n": len(sub),
                        "spearman_rho": float(rho),
                        "pvalue": float(pvalue),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["scope", "target", "pvalue", "predictor"])


def _compute_pairwise_method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["method"].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "method_a",
                "method_b",
                "n_cases",
                "win_rate_a",
                "tie_rate",
                "mean_ari_delta_a_minus_b",
                "mean_exact_k_delta_a_minus_b",
            ]
        )

    ari = frame.pivot(index="case_id", columns="method", values="ari")
    exact = frame.pivot(index="case_id", columns="method", values="exact_k")
    methods = ari.columns.tolist()
    rows: list[dict[str, Any]] = []
    for method_a in methods:
        for method_b in methods:
            comparable_ari = ari[[method_a, method_b]].dropna()
            comparable_exact = exact[[method_a, method_b]].dropna()
            n_cases = len(comparable_ari)
            if method_a == method_b and n_cases > 0:
                win_rate = 0.5
                tie_rate = 1.0
                mean_delta = 0.0
            elif n_cases > 0:
                left = comparable_ari[method_a]
                right = comparable_ari[method_b]
                win_rate = float((left > right).mean())
                tie_rate = float((left == right).mean())
                mean_delta = float((left - right).mean())
            else:
                win_rate = np.nan
                tie_rate = np.nan
                mean_delta = np.nan

            if method_a == method_b and len(comparable_exact) > 0:
                mean_exact_delta = 0.0
            elif len(comparable_exact) > 0:
                mean_exact_delta = float(
                    (comparable_exact[method_a] - comparable_exact[method_b]).mean()
                )
            else:
                mean_exact_delta = np.nan

            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_cases": n_cases,
                    "win_rate_a": win_rate,
                    "tie_rate": tie_rate,
                    "mean_ari_delta_a_minus_b": mean_delta,
                    "mean_exact_k_delta_a_minus_b": mean_exact_delta,
                }
            )
    return pd.DataFrame(rows)


def _attach_audit_factors(frame: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["audit_available"] = 0.0
    frame["audit_root_split_rejected"] = np.nan
    frame["audit_root_sibling_p"] = np.nan
    frame["audit_root_sibling_neglog10_p"] = np.nan
    frame["audit_sig_sibling_fraction"] = np.nan
    frame["audit_sig_edge_fraction"] = np.nan
    frame["audit_internal_nodes"] = np.nan
    frame["audit_mean_branch_length"] = np.nan

    for col in (
        "audit_root_sibling_neglog10_p_z",
        "audit_sig_sibling_fraction_z",
        "audit_sig_edge_fraction_z",
        "audit_internal_nodes_z",
        "audit_mean_branch_length_z",
    ):
        frame[col] = 0.0

    if frame.empty or not audit_dir.exists():
        return frame

    file_index = _index_audit_files(audit_dir)
    if not file_index:
        return frame

    cache: dict[Path, dict[str, float] | None] = {}
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        case_num = pd.to_numeric(row.get("test_case"), errors="coerce")
        if not np.isfinite(case_num):
            rows.append(_empty_audit_summary())
            continue
        audit_path = _resolve_audit_file(
            file_index=file_index,
            case_num=int(case_num),
            method=str(row.get("method", "")),
        )
        if audit_path is None:
            rows.append(_empty_audit_summary())
            continue
        if audit_path not in cache:
            cache[audit_path] = _extract_audit_summary(audit_path)
        rows.append(cache[audit_path] or _empty_audit_summary())

    audit_frame = pd.DataFrame(rows, index=frame.index)
    for col in audit_frame.columns:
        frame[col] = audit_frame[col]

    available_mask = frame["audit_available"] > 0
    for source, target in (
        ("audit_root_sibling_neglog10_p", "audit_root_sibling_neglog10_p_z"),
        ("audit_sig_sibling_fraction", "audit_sig_sibling_fraction_z"),
        ("audit_sig_edge_fraction", "audit_sig_edge_fraction_z"),
        ("audit_internal_nodes", "audit_internal_nodes_z"),
        ("audit_mean_branch_length", "audit_mean_branch_length_z"),
    ):
        z = pd.Series(np.nan, index=frame.index, dtype=float)
        if available_mask.any():
            z.loc[available_mask] = _zscore(frame.loc[available_mask, source])
        frame[target] = z.fillna(0.0)

    frame["audit_root_split_rejected"] = frame["audit_root_split_rejected"].fillna(0.0)
    return frame


def _empty_audit_summary() -> dict[str, float]:
    return {
        "audit_available": 0.0,
        "audit_root_split_rejected": np.nan,
        "audit_root_sibling_p": np.nan,
        "audit_root_sibling_neglog10_p": np.nan,
        "audit_sig_sibling_fraction": np.nan,
        "audit_sig_edge_fraction": np.nan,
        "audit_internal_nodes": np.nan,
        "audit_mean_branch_length": np.nan,
    }


def _index_audit_files(audit_dir: Path) -> dict[int, dict[str, Path]]:
    index: dict[int, dict[str, Path]] = {}
    for path in sorted(audit_dir.glob("case_*_stats.csv")):
        stem = path.stem
        if not stem.startswith("case_") or not stem.endswith("_stats"):
            continue
        body = stem[len("case_") : -len("_stats")]
        case_str, _, method_slug = body.partition("_")
        if not case_str.isdigit() or not method_slug:
            continue
        index.setdefault(int(case_str), {})[_normalize_method_key(method_slug)] = path
    return index


def _resolve_audit_file(
    *,
    file_index: dict[int, dict[str, Path]],
    case_num: int,
    method: str,
) -> Path | None:
    case_files = file_index.get(case_num, {})
    if not case_files:
        return None

    aliases = _method_aliases(method)
    for alias in aliases:
        candidate = case_files.get(alias)
        if candidate is not None:
            return candidate
    return None


def _method_aliases(method: str) -> list[str]:
    raw = str(method).strip().lower()
    aliases = {
        _normalize_method_key(raw),
        _normalize_method_key(raw.replace("_", " ")),
        _normalize_method_key(raw.replace("_", "-")),
    }
    if raw == "kl":
        aliases.update(
            {
                _normalize_method_key("kl_divergence"),
                _normalize_method_key("kl divergence"),
            }
        )
    if raw.startswith("kl_"):
        suffix = raw.split("_", 1)[1]
        aliases.update(
            {
                _normalize_method_key(f"kl_{suffix}"),
                _normalize_method_key(f"kl ({suffix})"),
                _normalize_method_key(f"kl-{suffix}"),
            }
        )
    return [alias for alias in aliases if alias]


def _normalize_method_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _extract_audit_summary(audit_path: Path) -> dict[str, float] | None:
    try:
        df = pd.read_csv(audit_path)
    except Exception:
        return None

    required = {"node_id", "leaf_count", "parent_node"}
    if not required.issubset(df.columns):
        return None

    try:
        root_idx = df["leaf_count"].astype(float).idxmax()
        root_id = df.loc[root_idx, "node_id"]
    except Exception:
        return None

    children = df[df["parent_node"] == root_id].copy()
    root_p = _select_root_split_pvalue(children)
    root_split_rejected = _infer_root_split_rejected(children, root_p)

    sibling_valid = pd.Series(True, index=df.index, dtype=bool)
    if "Sibling_Divergence_Skipped" in df.columns:
        sibling_valid &= ~_coerce_bool_series(df["Sibling_Divergence_Skipped"])
    if "Sibling_Divergence_Invalid" in df.columns:
        sibling_valid &= ~_coerce_bool_series(df["Sibling_Divergence_Invalid"])
    if "Sibling_BH_Different" in df.columns:
        sibling_valid &= df["Sibling_BH_Different"].notna()
    sibling_total = int(sibling_valid.sum())
    if "Sibling_BH_Different" in df.columns and sibling_total > 0:
        sig_sibling_fraction = float(_coerce_bool_series(df.loc[sibling_valid, "Sibling_BH_Different"]).mean())
    else:
        sig_sibling_fraction = np.nan

    edge_valid = pd.Series(df["parent_node"].notna(), index=df.index, dtype=bool)
    if "Child_Parent_Divergence_Invalid" in df.columns:
        edge_valid &= ~_coerce_bool_series(df["Child_Parent_Divergence_Invalid"])
    if "Child_Parent_Divergence_Significant" in df.columns:
        edge_valid &= df["Child_Parent_Divergence_Significant"].notna()
    edge_total = int(edge_valid.sum())
    if "Child_Parent_Divergence_Significant" in df.columns and edge_total > 0:
        sig_edge_fraction = float(
            _coerce_bool_series(df.loc[edge_valid, "Child_Parent_Divergence_Significant"]).mean()
        )
    else:
        sig_edge_fraction = np.nan

    is_leaf = _coerce_bool_series(df["is_leaf"]) if "is_leaf" in df.columns else df["leaf_count"].eq(1)
    internal_nodes = float((~is_leaf).sum())
    if "branch_length" in df.columns:
        branch_lengths = pd.to_numeric(df["branch_length"], errors="coerce")
        mean_branch_length = (
            float(branch_lengths.dropna().mean()) if branch_lengths.notna().any() else np.nan
        )
    else:
        mean_branch_length = np.nan
    neglog_root_p = (
        float(-np.log10(max(root_p, 1e-12)))
        if np.isfinite(root_p) and root_p > 0.0
        else np.nan
    )

    return {
        "audit_available": 1.0,
        "audit_root_split_rejected": float(root_split_rejected),
        "audit_root_sibling_p": float(root_p) if np.isfinite(root_p) else np.nan,
        "audit_root_sibling_neglog10_p": neglog_root_p,
        "audit_sig_sibling_fraction": sig_sibling_fraction,
        "audit_sig_edge_fraction": sig_edge_fraction,
        "audit_internal_nodes": internal_nodes,
        "audit_mean_branch_length": mean_branch_length,
    }


def _select_root_split_pvalue(children: pd.DataFrame) -> float:
    if children.empty:
        return np.nan
    for col in (
        "Sibling_Divergence_P_Value_Corrected",
        "Sibling_Divergence_P_Value",
    ):
        if col in children.columns:
            values = pd.to_numeric(children[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.min())
    return np.nan


def _infer_root_split_rejected(children: pd.DataFrame, root_p: float) -> bool:
    if children.empty:
        return True
    if "Sibling_BH_Different" in children.columns and children["Sibling_BH_Different"].notna().any():
        return not bool(_coerce_bool_series(children["Sibling_BH_Different"]).any())
    if np.isfinite(root_p):
        return bool(root_p >= 0.05)
    return True


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _fit_relationship_model(
    frame: pd.DataFrame,
    *,
    target: str,
    family: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if frame.empty or target not in frame.columns:
        return None, {}

    sub = frame.copy()
    sub = sub[sub[target].notna()].copy()
    if sub.empty:
        return None, {}

    design = pd.DataFrame(index=sub.index)
    continuous_terms: list[str] = []
    for term in _MODEL_CONTINUOUS_TERMS:
        if term not in sub.columns:
            continue
        values = pd.to_numeric(sub[term], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue
        design[term] = values
        continuous_terms.append(term)

    if sub["section"].nunique(dropna=True) > 1:
        design = pd.concat([design, _encode_categorical(sub["section"], "C(section)")], axis=1)
    if sub["method"].nunique(dropna=True) > 1:
        design = pd.concat([design, _encode_categorical(sub["method"], "C(method)")], axis=1)

    if design.empty:
        return None, {}

    if len(sub) < 4 * max(1, design.shape[1]):
        ordered_cols = [term for term in _MODEL_CONTINUOUS_TERMS if term in design.columns]
        ordered_cols.extend(col for col in design.columns if col not in ordered_cols)
        keep_cols = ordered_cols[: max(4, len(sub) // 4)]
        design = design[keep_cols]
        continuous_terms = [term for term in continuous_terms if term in keep_cols]

    fit_df = pd.concat([sub[[target]], design], axis=1).dropna()
    if fit_df.empty or len(fit_df) < max(6, design.shape[1] + 2):
        return None, {}

    y = pd.to_numeric(fit_df[target], errors="coerce").to_numpy(dtype=float)
    X = fit_df.drop(columns=[target])
    if family == "binomial" and pd.Series(y).nunique() < 2:
        return None, {}

    try:
        point, score = _fit_point_estimate(X, y, family=family)
        std_err, conf_low, conf_high, pvalues = _bootstrap_interval_summary(
            X,
            y,
            family=family,
            point=point,
        )
    except Exception:
        return None, {}

    terms = ["Intercept", *X.columns.tolist()]
    table = pd.DataFrame(
        {
            "term": terms,
            "coef": point,
            "std_err": std_err,
            "pvalue": pvalues,
            "conf_low": conf_low,
            "conf_high": conf_high,
            "model": target,
            "family": family,
            "n_obs": int(len(fit_df)),
            "formula": f"{target} ~ " + " + ".join(X.columns.tolist()),
            "continuous_term": np.array([False] + [term in continuous_terms for term in X.columns]),
        }
    )
    if family == "binomial":
        table["odds_ratio"] = _safe_exp(table["coef"])
        table["or_conf_low"] = _safe_exp(table["conf_low"])
        table["or_conf_high"] = _safe_exp(table["conf_high"])

    model_stats = {
        "formula": f"{target} ~ " + " + ".join(X.columns.tolist()),
        "family": family,
        "n_obs": int(len(fit_df)),
        "score_name": "r_squared" if family == "ols" else "pseudo_r_squared",
        "score": score,
    }
    return table, model_stats


def _write_empty_report(report_md: Path, source_path: Path | None) -> None:
    lines = [
        "# Benchmark Relationship Report",
        "",
        "No successful benchmark rows were available for relationship analysis.",
    ]
    if source_path is not None:
        lines.insert(2, f"- Source: `{source_path}`")
        lines.insert(3, "")
    report_md.write_text("\n".join(lines) + "\n")


def _write_relationship_report(
    *,
    report_md: Path,
    frame: pd.DataFrame,
    method_summary: pd.DataFrame,
    section_summary: pd.DataFrame,
    method_section_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    pairwise_method_summary: pd.DataFrame,
    regression_tables: dict[str, pd.DataFrame],
    model_stats: dict[str, dict[str, Any]],
    source_path: Path | None,
) -> None:
    lines: list[str] = ["# Benchmark Relationship Report", ""]

    if source_path is not None:
        lines.append(f"- Source: `{source_path}`")
    lines.extend(
        [
            f"- Rows analyzed: `{len(frame)}`",
            f"- Unique cases: `{frame['case_id'].nunique()}`",
            f"- Methods: `{', '.join(sorted(frame['method'].unique()))}`",
            f"- Sections: `{', '.join(sorted(frame['section'].unique()))}`",
            f"- Audit-backed rows: `{int(frame.get('audit_available', pd.Series(dtype=float)).fillna(0).sum())}`",
            "",
        ]
    )

    best_method = method_summary.iloc[0]
    hardest_section = section_summary.sort_values("mean_ari", ascending=True).iloc[0]
    easiest_section = section_summary.sort_values("mean_ari", ascending=False).iloc[0]
    lines.extend(
        [
            "## Headline Findings",
            "",
            (
                f"- Best average ARI: `{best_method['method']}` at `{best_method['mean_ari']:.3f}` "
                f"with exact-K rate `{best_method['exact_k_rate']:.3f}`."
            ),
            (
                f"- Easiest section: `{easiest_section['section']}` "
                f"(mean ARI `{easiest_section['mean_ari']:.3f}`)."
            ),
            (
                f"- Hardest section: `{hardest_section['section']}` "
                f"(mean ARI `{hardest_section['mean_ari']:.3f}`)."
            ),
        ]
    )

    best_cell = method_section_summary.sort_values("mean_ari", ascending=False).iloc[0]
    worst_cell = method_section_summary.sort_values("mean_ari", ascending=True).iloc[0]
    lines.extend(
        [
            (
                f"- Strongest method/section cell: `{best_cell['method']}` on `{best_cell['section']}` "
                f"(mean ARI `{best_cell['mean_ari']:.3f}`, exact-K `{best_cell['exact_k_rate']:.3f}`)."
            ),
            (
                f"- Weakest method/section cell: `{worst_cell['method']}` on `{worst_cell['section']}` "
                f"(mean ARI `{worst_cell['mean_ari']:.3f}`, exact-K `{worst_cell['exact_k_rate']:.3f}`)."
            ),
            "",
        ]
    )

    outlier_rows = frame[frame["outlier_f1"].notna()].copy()
    if not outlier_rows.empty:
        best_outlier_method = (
            outlier_rows.groupby("method", dropna=False)
            .agg(
                mean_outlier_f1=("outlier_f1", "mean"),
                singleton_hit_rate=("singleton_outlier_isolated", "mean"),
                grouped_recovery_rate=("grouped_outlier_cluster_recovered", "mean"),
            )
            .reset_index()
            .sort_values(
                ["mean_outlier_f1", "grouped_recovery_rate", "singleton_hit_rate"],
                ascending=[False, False, False],
            )
            .iloc[0]
        )
        lines.extend(
            [
                (
                    f"- Best outlier recovery: `{best_outlier_method['method']}` "
                    f"(mean outlier F1 `{best_outlier_method['mean_outlier_f1']:.3f}`, "
                    f"singleton hit rate `{best_outlier_method['singleton_hit_rate']:.3f}`, "
                    f"grouped recovery `{best_outlier_method['grouped_recovery_rate']:.3f}`)."
                ),
                "",
            ]
        )

    pairwise_lines = _summarize_pairwise_findings(pairwise_method_summary)
    if pairwise_lines:
        lines.append("## Pairwise Method Contrasts")
        lines.append("")
        lines.extend(pairwise_lines)
        lines.append("")

    lines.extend(
        [
            "## Method Summary",
            "",
            "```text",
            _format_table(
                method_summary[
                    [
                        "method",
                        "n_rows",
                        "mean_ari",
                        "mean_nmi",
                        "mean_outlier_f1",
                        "singleton_hit_rate",
                        "grouped_recovery_rate",
                        "exact_k_rate",
                        "over_split_rate",
                        "under_split_rate",
                    ]
                ]
            ),
            "```",
            "",
            "## Section Summary",
            "",
            "```text",
            _format_table(
                section_summary[
                    [
                        "section",
                        "n_rows",
                        "mean_ari",
                        "mean_nmi",
                        "mean_outlier_f1",
                        "singleton_hit_rate",
                        "grouped_recovery_rate",
                        "exact_k_rate",
                        "over_split_rate",
                        "under_split_rate",
                    ]
                ]
            ),
            "```",
            "",
        ]
    )

    correlation_lines = _summarize_correlation_findings(correlations)
    if correlation_lines:
        lines.append("## Correlation Highlights")
        lines.append("")
        lines.extend(correlation_lines)
        lines.append("")

    effect_lines = _summarize_regression_findings(regression_tables, model_stats)
    if effect_lines:
        lines.append("## Model Highlights")
        lines.append("")
        lines.extend(effect_lines)
        lines.append("")

    audit_lines = _summarize_audit_findings(frame)
    if audit_lines:
        lines.append("## Audit Factor Highlights")
        lines.append("")
        lines.extend(audit_lines)
        lines.append("")

    if regression_tables:
        lines.append("## Regression Coefficients")
        lines.append("")
        for target, table in regression_tables.items():
            lines.append(f"### `{target}`")
            lines.append("")
            keep_cols = ["term", "coef", "std_err", "pvalue", "conf_low", "conf_high"]
            if "odds_ratio" in table.columns:
                keep_cols.extend(["odds_ratio", "or_conf_low", "or_conf_high"])
            lines.append("```text")
            lines.append(_format_table(table[keep_cols]))
            lines.append("```")
            lines.append("")

    report_md.write_text("\n".join(lines).rstrip() + "\n")


def _summarize_correlation_findings(correlations: pd.DataFrame) -> list[str]:
    if correlations.empty:
        return []

    overall = correlations[correlations["scope"] == "all"].copy()
    if overall.empty:
        return []

    lines: list[str] = []
    for target in ("ari", "exact_k", "over_split_flag", "under_split_flag"):
        subset = overall[overall["target"] == target].copy()
        if subset.empty:
            continue
        best = subset.iloc[subset["spearman_rho"].abs().argmax()]
        relation = "positive" if best["spearman_rho"] > 0 else "negative"
        lines.append(
            (
                f"- Strongest monotonic `{target}` relationship: `{best['predictor']}` "
                f"(`rho={best['spearman_rho']:.3f}`, `{relation}`, `p={best['pvalue']:.3g}`, `n={int(best['n'])}`)."
            )
        )
    return lines


def _summarize_regression_findings(
    regression_tables: dict[str, pd.DataFrame],
    model_stats: dict[str, dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for target, table in regression_tables.items():
        stats = model_stats.get(target, {})
        score_name = stats.get("score_name")
        score = stats.get("score")
        if score_name and score is not None and np.isfinite(score):
            lines.append(
                f"- `{target}` model: `{score_name}={float(score):.3f}` over `{stats.get('n_obs', '?')}` rows."
            )

        subset = table[(table["continuous_term"]) & (table["pvalue"] <= 0.05)].copy()
        if subset.empty:
            continue
        subset = subset.reindex(subset["coef"].abs().sort_values(ascending=False).index)
        for _, row in subset.head(3).iterrows():
            label = _CONTINUOUS_EFFECT_LABELS.get(row["term"], row["term"])
            if row["family"] == "ols":
                direction = "higher" if row["coef"] > 0 else "lower"
                lines.append(
                    (
                        f"- `{target}`: more `{label}` is associated with `{direction}` values "
                        f"(`beta={row['coef']:.3f}`, `p={row['pvalue']:.3g}`)."
                    )
                )
            else:
                direction = "higher" if row["coef"] > 0 else "lower"
                lines.append(
                    (
                        f"- `{target}`: more `{label}` is associated with `{direction}` odds "
                        f"(`OR={row['odds_ratio']:.3f}`, `p={row['pvalue']:.3g}`)."
                    )
                )
    return lines


def _summarize_pairwise_findings(pairwise_method_summary: pd.DataFrame) -> list[str]:
    if pairwise_method_summary.empty:
        return []

    lines: list[str] = []
    if "kl" in set(pairwise_method_summary["method_b"]):
        vs_kl = pairwise_method_summary[
            (pairwise_method_summary["method_b"] == "kl")
            & (pairwise_method_summary["method_a"] != "kl")
        ].copy()
        if not vs_kl.empty:
            best = vs_kl.sort_values("mean_ari_delta_a_minus_b", ascending=False).iloc[0]
            worst = vs_kl.sort_values("mean_ari_delta_a_minus_b", ascending=True).iloc[0]
            lines.append(
                (
                    f"- Largest ARI gain over `kl`: `{best['method_a']}` "
                    f"(`delta={best['mean_ari_delta_a_minus_b']:.3f}`, "
                    f"`win_rate={best['win_rate_a']:.3f}`, `n={int(best['n_cases'])}`)."
                )
            )
            lines.append(
                (
                    f"- Weakest method relative to `kl`: `{worst['method_a']}` "
                    f"(`delta={worst['mean_ari_delta_a_minus_b']:.3f}`, "
                    f"`win_rate={worst['win_rate_a']:.3f}`, `n={int(worst['n_cases'])}`)."
                )
            )

    best_overall = pairwise_method_summary[
        pairwise_method_summary["method_a"] != pairwise_method_summary["method_b"]
    ].copy()
    if not best_overall.empty:
        top_win = best_overall.sort_values("win_rate_a", ascending=False).iloc[0]
        lines.append(
            (
                f"- Strongest head-to-head win rate: `{top_win['method_a']}` over `{top_win['method_b']}` "
                f"(`win_rate={top_win['win_rate_a']:.3f}`, `delta={top_win['mean_ari_delta_a_minus_b']:.3f}`)."
            )
        )
    return lines


def _summarize_audit_findings(frame: pd.DataFrame) -> list[str]:
    if "audit_available" not in frame.columns:
        return []

    audit_frame = frame[frame["audit_available"] > 0].copy()
    if audit_frame.empty:
        return []

    lines: list[str] = []
    if "audit_root_split_rejected" in audit_frame.columns:
        grouped = audit_frame.groupby("audit_root_split_rejected").agg(
            mean_ari=("ari", "mean"),
            exact_k_rate=("exact_k", "mean"),
            n_rows=("ari", "size"),
        )
        if 0.0 in grouped.index and 1.0 in grouped.index:
            lines.append(
                (
                    f"- Root split accepted rows: mean ARI `{grouped.loc[0.0, 'mean_ari']:.3f}` "
                    f"vs rejected rows `{grouped.loc[1.0, 'mean_ari']:.3f}`."
                )
            )
            lines.append(
                (
                    f"- Exact-K with root split accepted: `{grouped.loc[0.0, 'exact_k_rate']:.3f}` "
                    f"vs rejected `{grouped.loc[1.0, 'exact_k_rate']:.3f}`."
                )
            )

    audit_corr = _compute_correlations(audit_frame)
    if not audit_corr.empty:
        audit_only = audit_corr[
            audit_corr["predictor"].isin(
                {
                    "audit_root_sibling_neglog10_p",
                    "audit_sig_sibling_fraction",
                    "audit_sig_edge_fraction",
                    "audit_internal_nodes",
                    "audit_mean_branch_length",
                }
            )
            & (audit_corr["scope"] == "all")
            & (audit_corr["target"] == "ari")
        ]
        if not audit_only.empty:
            best = audit_only.iloc[audit_only["spearman_rho"].abs().argmax()]
            relation = "positive" if best["spearman_rho"] > 0 else "negative"
            lines.append(
                (
                    f"- Strongest audit-vs-ARI relationship: `{best['predictor']}` "
                    f"(`rho={best['spearman_rho']:.3f}`, `{relation}`, `p={best['pvalue']:.3g}`)."
                )
            )
    return lines


def _write_relationship_plots_pdf(
    *,
    plots_pdf: Path,
    frame: pd.DataFrame,
    method_summary: pd.DataFrame,
    section_summary: pd.DataFrame,
    method_section_summary: pd.DataFrame,
    pairwise_method_summary: pd.DataFrame,
) -> None:
    with PdfPages(plots_pdf) as pdf:
        fig1 = _build_summary_figure(frame, method_summary, section_summary, method_section_summary)
        prepare_pdf_figure(fig1)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = _build_relationship_figure(frame)
        prepare_pdf_figure(fig2)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig2b = _build_factor_relationship_figure(frame)
        prepare_pdf_figure(fig2b)
        pdf.savefig(fig2b)
        plt.close(fig2b)

        if frame["method"].nunique() > 1:
            fig3 = _build_method_relationship_figure(frame, method_summary)
            if fig3 is not None:
                prepare_pdf_figure(fig3)
                pdf.savefig(fig3)
                plt.close(fig3)

            fig4 = _build_pairwise_contrast_figure(pairwise_method_summary)
            if fig4 is not None:
                prepare_pdf_figure(fig4)
                pdf.savefig(fig4)
                plt.close(fig4)

        fig_audit = _build_audit_relationship_figure(frame)
        if fig_audit is not None:
            prepare_pdf_figure(fig_audit)
            pdf.savefig(fig_audit)
            plt.close(fig_audit)


def _build_summary_figure(
    frame: pd.DataFrame,
    method_summary: pd.DataFrame,
    section_summary: pd.DataFrame,
    method_section_summary: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Benchmark Relationship Summary", fontsize=16, weight="bold")

    if frame["method"].nunique() > 1:
        ordered_methods = method_summary["method"].tolist()
        axes[0, 0].bar(
            ordered_methods,
            method_summary["mean_ari"],
            color="#4C72B0",
            edgecolor="black",
        )
        axes[0, 0].set_title("Mean ARI by Method", weight="bold")
        axes[0, 0].set_ylabel("Mean ARI")
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].tick_params(axis="x", rotation=35)
        axes[0, 0].grid(True, axis="y", alpha=0.25)

        axes[0, 1].bar(
            ordered_methods,
            method_summary["exact_k_rate"],
            color="#55A868",
            edgecolor="black",
        )
        axes[0, 1].set_title("Exact-K Rate by Method", weight="bold")
        axes[0, 1].set_ylabel("Rate")
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].tick_params(axis="x", rotation=35)
        axes[0, 1].grid(True, axis="y", alpha=0.25)

        ari_pivot = _metric_pivot(method_section_summary, "mean_ari")
        exact_pivot = _metric_pivot(method_section_summary, "exact_k_rate")
        _draw_heatmap(axes[1, 0], ari_pivot, title="Mean ARI by Method and Section")
        _draw_heatmap(axes[1, 1], exact_pivot, title="Exact-K Rate by Method and Section")
    else:
        ordered_sections = section_summary["section"].tolist()
        axes[0, 0].bar(
            ordered_sections,
            section_summary["mean_ari"],
            color="#4C72B0",
            edgecolor="black",
        )
        axes[0, 0].set_title("Mean ARI by Section", weight="bold")
        axes[0, 0].set_ylabel("Mean ARI")
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].tick_params(axis="x", rotation=35)
        axes[0, 0].grid(True, axis="y", alpha=0.25)

        axes[0, 1].bar(
            ordered_sections,
            section_summary["exact_k_rate"],
            color="#55A868",
            edgecolor="black",
        )
        axes[0, 1].set_title("Exact-K Rate by Section", weight="bold")
        axes[0, 1].set_ylabel("Rate")
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].tick_params(axis="x", rotation=35)
        axes[0, 1].grid(True, axis="y", alpha=0.25)

        x = np.arange(len(section_summary))
        axes[1, 0].bar(
            x,
            section_summary["over_split_rate"],
            label="Over-split",
            color="#DD8452",
            edgecolor="black",
        )
        axes[1, 0].bar(
            x,
            section_summary["under_split_rate"],
            bottom=section_summary["over_split_rate"],
            label="Under-split",
            color="#C44E52",
            edgecolor="black",
        )
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(ordered_sections, rotation=35, ha="right")
        axes[1, 0].set_ylim(0.0, 1.05)
        axes[1, 0].set_title("Split Failure Rates by Section", weight="bold")
        axes[1, 0].legend(frameon=False)
        axes[1, 0].grid(True, axis="y", alpha=0.25)

        axes[1, 1].bar(
            ordered_sections,
            section_summary["mean_abs_cluster_error"],
            color="#8172B3",
            edgecolor="black",
        )
        axes[1, 1].set_title("Mean Absolute Cluster Error", weight="bold")
        axes[1, 1].set_ylabel("|Found - True|")
        axes[1, 1].tick_params(axis="x", rotation=35)
        axes[1, 1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    return fig


def _build_relationship_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Benchmark Difficulty Relationships", fontsize=16, weight="bold")

    _scatter_with_fit(
        axes[0, 0],
        frame=frame,
        x_col="noise_filled",
        y_col="ari",
        x_label="Noise",
        y_label="ARI",
        title="ARI vs Noise",
    )
    _scatter_with_fit(
        axes[0, 1],
        frame=frame,
        x_col="log_features",
        y_col="ari",
        x_label="log(1 + features)",
        y_label="ARI",
        title="ARI vs Feature Count",
    )

    by_k = (
        frame.groupby("true_clusters")
        .agg(
            exact_k_rate=("exact_k", "mean"),
            over_split_rate=("over_split_flag", "mean"),
            under_split_rate=("under_split_flag", "mean"),
        )
        .reset_index()
        .sort_values("true_clusters")
    )
    axes[1, 0].plot(
        by_k["true_clusters"],
        by_k["exact_k_rate"],
        marker="o",
        color="#55A868",
        linewidth=2,
    )
    axes[1, 0].set_title("Exact-K Rate vs True Cluster Count", weight="bold")
    axes[1, 0].set_xlabel("True Clusters")
    axes[1, 0].set_ylabel("Exact-K Rate")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(
        by_k["true_clusters"],
        by_k["over_split_rate"],
        marker="o",
        color="#DD8452",
        linewidth=2,
        label="Over-split",
    )
    axes[1, 1].plot(
        by_k["true_clusters"],
        by_k["under_split_rate"],
        marker="s",
        color="#C44E52",
        linewidth=2,
        label="Under-split",
    )
    axes[1, 1].set_title("Split Rates vs True Cluster Count", weight="bold")
    axes[1, 1].set_xlabel("True Clusters")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    return fig


def _build_factor_relationship_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Derived Difficulty Factors", fontsize=16, weight="bold")

    _scatter_with_fit(
        axes[0, 0],
        frame=frame,
        x_col="log_samples_per_cluster",
        y_col="ari",
        x_label="log(1 + samples per cluster)",
        y_label="ARI",
        title="ARI vs Samples per Cluster",
    )
    _scatter_with_fit(
        axes[0, 1],
        frame=frame,
        x_col="log_features_per_cluster",
        y_col="ari",
        x_label="log(1 + features per cluster)",
        y_label="ARI",
        title="ARI vs Features per Cluster",
    )
    _scatter_with_fit(
        axes[1, 0],
        frame=frame,
        x_col="log_samples_per_feature",
        y_col="exact_k",
        x_label="log(1 + samples per feature)",
        y_label="Exact-K",
        title="Exact-K vs Samples per Feature",
    )
    _scatter_with_fit(
        axes[1, 1],
        frame=frame,
        x_col="noise_cluster_interaction",
        y_col="ari",
        x_label="Noise x log(1 + true clusters)",
        y_label="ARI",
        title="ARI vs Noise-Cluster Interaction",
    )

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    return fig


def _build_method_relationship_figure(
    frame: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> plt.Figure | None:
    display_methods = _select_display_methods(method_summary)
    if len(display_methods) <= 1:
        return None

    fig, axes = plt.subplots(2, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Method-Specific Relationships", fontsize=16, weight="bold")
    palette = plt.cm.tab10(np.linspace(0, 1, len(display_methods)))

    for color, method in zip(palette, display_methods):
        sub = frame[frame["method"] == method].copy()
        if sub.empty:
            continue
        by_k = (
            sub.groupby("true_clusters")
            .agg(
                mean_ari=("ari", "mean"),
                exact_k_rate=("exact_k", "mean"),
                mean_cluster_error=("cluster_error_signed", "mean"),
            )
            .reset_index()
            .sort_values("true_clusters")
        )
        axes[0, 0].plot(
            by_k["true_clusters"],
            by_k["mean_ari"],
            marker="o",
            linewidth=2,
            color=color,
            label=method,
        )
        axes[0, 1].plot(
            by_k["true_clusters"],
            by_k["exact_k_rate"],
            marker="o",
            linewidth=2,
            color=color,
            label=method,
        )
        axes[1, 0].plot(
            by_k["true_clusters"],
            by_k["mean_cluster_error"],
            marker="o",
            linewidth=2,
            color=color,
            label=method,
        )

        by_noise = (
            _bin_continuous(sub, "noise_filled")
            .groupby("bin_mid", observed=False)
            .agg(mean_ari=("ari", "mean"))
            .reset_index()
            .sort_values("bin_mid")
        )
        if not by_noise.empty:
            axes[1, 1].plot(
                by_noise["bin_mid"],
                by_noise["mean_ari"],
                marker="o",
                linewidth=2,
                color=color,
                label=method,
            )

    axes[0, 0].set_title("ARI vs True Cluster Count", weight="bold")
    axes[0, 1].set_title("Exact-K Rate vs True Cluster Count", weight="bold")
    axes[1, 0].set_title("Mean Signed Cluster Error", weight="bold")
    axes[1, 1].set_title("Binned ARI vs Noise", weight="bold")

    axes[0, 0].set_xlabel("True Clusters")
    axes[0, 1].set_xlabel("True Clusters")
    axes[1, 0].set_xlabel("True Clusters")
    axes[1, 1].set_xlabel("Noise (bin midpoint)")

    axes[0, 0].set_ylabel("ARI")
    axes[0, 1].set_ylabel("Exact-K Rate")
    axes[1, 0].set_ylabel("Found - True")
    axes[1, 1].set_ylabel("ARI")

    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 0].axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False)

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    return fig


def _build_pairwise_contrast_figure(
    pairwise_method_summary: pd.DataFrame,
) -> plt.Figure | None:
    if pairwise_method_summary.empty:
        return None

    ari_delta = pairwise_method_summary.pivot(
        index="method_a",
        columns="method_b",
        values="mean_ari_delta_a_minus_b",
    )
    win_rate = pairwise_method_summary.pivot(
        index="method_a",
        columns="method_b",
        values="win_rate_a",
    )
    exact_delta = pairwise_method_summary.pivot(
        index="method_a",
        columns="method_b",
        values="mean_exact_k_delta_a_minus_b",
    )
    ties = pairwise_method_summary.pivot(
        index="method_a",
        columns="method_b",
        values="tie_rate",
    )

    fig, axes = plt.subplots(2, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Pairwise Method Contrasts", fontsize=16, weight="bold")
    _draw_heatmap_signed(
        axes[0, 0],
        ari_delta,
        title="Mean ARI Delta (row - column)",
        vmin=-1.0,
        vmax=1.0,
    )
    _draw_heatmap(
        axes[0, 1],
        win_rate,
        title="Head-to-Head Win Rate",
    )
    _draw_heatmap_signed(
        axes[1, 0],
        exact_delta,
        title="Mean Exact-K Delta (row - column)",
        vmin=-1.0,
        vmax=1.0,
    )
    _draw_heatmap(
        axes[1, 1],
        ties,
        title="Tie Rate",
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    return fig


def _build_audit_relationship_figure(frame: pd.DataFrame) -> plt.Figure | None:
    if "audit_available" not in frame.columns:
        return None

    audit_frame = frame[frame["audit_available"] > 0].copy()
    if audit_frame.empty:
        return None

    fig, axes = plt.subplots(3, 2, figsize=PDF_PAGE_SIZE_INCHES)
    fig.suptitle("Audit-Derived Structural Factors", fontsize=16, weight="bold")

    grouped = (
        audit_frame.groupby("audit_root_split_rejected")
        .agg(mean_ari=("ari", "mean"), exact_k_rate=("exact_k", "mean"))
        .reset_index()
    )
    x = np.arange(len(grouped))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        grouped["mean_ari"],
        width=width,
        color="#4C72B0",
        edgecolor="black",
        label="Mean ARI",
    )
    axes[0, 0].bar(
        x + width / 2,
        grouped["exact_k_rate"],
        width=width,
        color="#55A868",
        edgecolor="black",
        label="Exact-K Rate",
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(
        [
            "Root Accepted" if value == 0 else "Root Rejected"
            for value in grouped["audit_root_split_rejected"]
        ]
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Root Split Outcome vs Performance", weight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    _scatter_with_fit(
        axes[0, 1],
        frame=audit_frame,
        x_col="audit_sig_sibling_fraction",
        y_col="ari",
        x_label="Significant sibling fraction",
        y_label="ARI",
        title="ARI vs Significant Sibling Fraction",
    )
    _scatter_with_fit(
        axes[1, 0],
        frame=audit_frame,
        x_col="audit_sig_edge_fraction",
        y_col="ari",
        x_label="Significant edge fraction",
        y_label="ARI",
        title="ARI vs Significant Edge Fraction",
    )
    _scatter_with_fit(
        axes[1, 1],
        frame=audit_frame,
        x_col="audit_root_sibling_neglog10_p",
        y_col="ari",
        x_label="-log10(root sibling p)",
        y_label="ARI",
        title="ARI vs Root Split Evidence",
    )
    _scatter_with_fit(
        axes[2, 0],
        frame=audit_frame,
        x_col="audit_internal_nodes",
        y_col="ari",
        x_label="Internal nodes",
        y_label="ARI",
        title="ARI vs Internal Node Count",
    )
    _scatter_with_fit(
        axes[2, 1],
        frame=audit_frame,
        x_col="audit_mean_branch_length",
        y_col="ari",
        x_label="Mean branch length",
        y_label="ARI",
        title="ARI vs Mean Branch Length",
    )

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    return fig


def _metric_pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = summary.pivot(index="method", columns="section", values=metric)
    ordered_cols = [section for section in GROUP_ORDER if section in pivot.columns]
    ordered_cols.extend(sorted(col for col in pivot.columns if col not in ordered_cols))
    return pivot.reindex(columns=ordered_cols)


def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, *, title: str) -> None:
    if pivot.empty:
        ax.axis("off")
        ax.set_title(title)
        return

    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if np.isfinite(val) and val < 0.5 else "black"
            text = f"{val:.2f}" if np.isfinite(val) else "NA"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_heatmap_signed(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    if pivot.empty:
        ax.axis("off")
        ax.set_title(title)
        return

    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    threshold = max(abs(vmin), abs(vmax)) * 0.35
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if np.isfinite(val) and abs(val) >= threshold else "black"
            text = f"{val:.2f}" if np.isfinite(val) else "NA"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _scatter_with_fit(
    ax: plt.Axes,
    *,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, frame["section"].nunique())))
    color_map = dict(zip(sorted(frame["section"].unique()), colors))
    for section, section_df in frame.groupby("section"):
        ax.scatter(
            section_df[x_col],
            section_df[y_col],
            s=36,
            alpha=0.55,
            edgecolors="none",
            color=color_map[section],
            label=section,
        )

    sub = frame[[x_col, y_col]].dropna()
    if len(sub) >= 3 and sub[x_col].nunique() >= 2:
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        x_grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_grid, intercept + slope * x_grid, color="black", linewidth=2)

    ax.set_title(title, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_col in {"ari", "exact_k"}:
        ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, ncol=2)


def _bin_continuous(frame: pd.DataFrame, col: str, bins: int = 5) -> pd.DataFrame:
    sub = frame[[col]].copy()
    if sub[col].nunique(dropna=True) < 2:
        frame = frame.copy()
        frame["bin_mid"] = frame[col]
        return frame

    quantiles = min(bins, int(sub[col].nunique()))
    labels = pd.qcut(sub[col], q=quantiles, duplicates="drop")
    frame = frame.copy()
    frame["_bin"] = labels
    frame["bin_mid"] = frame["_bin"].map(
        lambda interval: float((interval.left + interval.right) / 2.0)
        if pd.notna(interval)
        else np.nan
    )
    return frame.drop(columns=["_bin"])


def _format_table(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}" if np.isfinite(value) else "nan",
    )


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    mean = float(values.mean())
    return (values - mean) / std


def _safe_exp(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.exp(np.clip(arr, -50.0, 50.0))


def _section_order(section: str) -> int:
    try:
        return GROUP_ORDER.index(section)
    except ValueError:
        return len(GROUP_ORDER)


def _select_display_methods(method_summary: pd.DataFrame) -> list[str]:
    ordered = method_summary["method"].tolist()
    selected = ordered[:5]
    if "kl" in ordered and "kl" not in selected:
        selected.append("kl")
    return selected[:6]


def _encode_categorical(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = series.fillna("unknown").astype(str)
    dummies = pd.get_dummies(values, drop_first=True, dtype=float)
    dummies.columns = [f"{prefix}[T.{column}]" for column in dummies.columns]
    return dummies


def _fit_point_estimate(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    family: str,
) -> tuple[np.ndarray, float]:
    if family == "ols":
        model = LinearRegression()
        model.fit(X, y)
        point = np.concatenate(([float(model.intercept_)], model.coef_.astype(float)))
        score = float(model.score(X, y))
        return point, score

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(X, y.astype(int))
    point = np.concatenate(([float(model.intercept_[0])], model.coef_[0].astype(float)))

    probs = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    y_mean = float(np.mean(y))
    if y_mean <= 0.0 or y_mean >= 1.0:
        score = float("nan")
    else:
        ll_full = float(np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
        ll_null = float(np.sum(y * np.log(y_mean) + (1.0 - y) * np.log(1.0 - y_mean)))
        score = 1.0 - (ll_full / ll_null) if ll_null != 0.0 else float("nan")
    return point, score


def _bootstrap_interval_summary(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    family: str,
    point: np.ndarray,
    n_boot: int = 160,
    min_success: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    draws: list[np.ndarray] = []
    n_obs = len(X)
    if n_obs == 0:
        raise ValueError("Cannot bootstrap an empty design matrix")

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_obs, size=n_obs)
        X_boot = X.iloc[sample_idx]
        y_boot = y[sample_idx]
        if family == "binomial" and pd.Series(y_boot).nunique() < 2:
            continue
        try:
            draw, _ = _fit_point_estimate(X_boot, y_boot, family=family)
        except Exception:
            continue
        draws.append(draw)

    if len(draws) < min_success:
        std_err = np.full_like(point, np.nan, dtype=float)
        conf_low = np.full_like(point, np.nan, dtype=float)
        conf_high = np.full_like(point, np.nan, dtype=float)
        pvalues = np.full_like(point, np.nan, dtype=float)
        return std_err, conf_low, conf_high, pvalues

    arr = np.vstack(draws)
    std_err = arr.std(axis=0, ddof=1)
    conf_low = np.quantile(arr, 0.025, axis=0)
    conf_high = np.quantile(arr, 0.975, axis=0)
    z_scores = np.divide(
        point,
        std_err,
        out=np.full_like(point, np.nan, dtype=float),
        where=np.isfinite(std_err) & (std_err > 0.0),
    )
    pvalues = 2.0 * norm.sf(np.abs(z_scores))
    return std_err, conf_low, conf_high, pvalues


__all__ = [
    "BenchmarkRelationshipArtifacts",
    "analyze_benchmark_relationships",
    "normalize_results_dataframe",
    "prepare_relationship_frame",
]
