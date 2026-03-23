"""Compare calibration strategies for deterministic one-active-feature testing."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from debug_scripts.diagnostics.compare_one_active_feature_1d_strength import (
    _normalize_benchmark_df,
    _patch_one_active_feature_to_1d,
    _run_kl_suite,
    _summarize,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.types import (
    CalibrationModel,
)


def _is_canonical_one_feature_basis(pca_projection: np.ndarray | None) -> bool:
    if pca_projection is None:
        return False
    basis = np.asarray(pca_projection, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != 1:
        return False
    nonzero = np.flatnonzero(np.abs(basis[0]) > 1e-12)
    if nonzero.size != 1:
        return False
    return bool(np.isclose(np.linalg.norm(basis[0]), 1.0))


def _regime_for_projection(pca_projection: np.ndarray | None) -> str:
    return "one_active" if _is_canonical_one_feature_basis(pca_projection) else "multivariate"


def _clone_model(method: str, diagnostics: dict[str, object], c_hat: float) -> CalibrationModel:
    return CalibrationModel(
        method=method,
        n_calibration=int(diagnostics.get("n_contributing", diagnostics.get("n_calibration", 0)) or 0),
        global_inflation_factor=float(c_hat),
        max_observed_ratio=float(diagnostics.get("max_observed_ratio", c_hat)),
        beta=np.array([np.log(max(float(c_hat), 1e-12)), 0.0, 0.0], dtype=np.float64),
        diagnostics=diagnostics,
    )


@contextmanager
def _collect_one_active_records(collector: list[dict[str, float]]) -> None:
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
        adjusted_wald_annotation as awa,
    )

    original_collect = awa.collect_sibling_pair_records

    def wrapped_collect(*args, **kwargs):
        records, non_binary = original_collect(*args, **kwargs)
        pca_projections = kwargs.get("pca_projections")
        for record in records:
            regime = _regime_for_projection(
                pca_projections.get(record.parent) if pca_projections is not None else None
            )
            if regime == "one_active":
                collector.append(
                    {
                        "stat": float(record.stat),
                        "df": float(record.degrees_of_freedom),
                        "edge_weight": float(record.edge_weight),
                        "is_null_like": bool(record.is_null_like),
                    }
                )
        return records, non_binary

    awa.collect_sibling_pair_records = wrapped_collect
    try:
        yield
    finally:
        awa.collect_sibling_pair_records = original_collect


@contextmanager
def _patch_one_active_calibration(
    *,
    mode: str,
    global_one_active_c: float | None = None,
    shrink_tau: float = 10.0,
) -> None:
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
        adjusted_wald_annotation as awa,
    )
    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
        _compute_weighted_inflation,
    )

    if mode not in {"split", "offline", "shrink_to_multivariate"}:
        raise ValueError(f"Unsupported mode: {mode}")

    original_collect = awa.collect_sibling_pair_records
    original_fit = awa.fit_inflation_model
    original_deflate = awa._deflate_and_test

    def wrapped_collect(*args, **kwargs):
        records, non_binary = original_collect(*args, **kwargs)
        pca_projections = kwargs.get("pca_projections")
        for record in records:
            record.projection_regime = _regime_for_projection(
                pca_projections.get(record.parent) if pca_projections is not None else None
            )
        return records, non_binary

    def wrapped_fit(records):
        one_active_records = [
            record
            for record in records
            if getattr(record, "projection_regime", "multivariate") == "one_active"
        ]
        multivariate_records = [
            record
            for record in records
            if getattr(record, "projection_regime", "multivariate") != "one_active"
        ]

        one_active_model = (
            _compute_weighted_inflation(one_active_records)
            if one_active_records
            else CalibrationModel(
                method="weighted_mean",
                n_calibration=0,
                global_inflation_factor=1.0,
                diagnostics={"fit_status": "empty_one_active"},
            )
        )
        multivariate_model = (
            _compute_weighted_inflation(multivariate_records)
            if multivariate_records
            else CalibrationModel(
                method="weighted_mean",
                n_calibration=0,
                global_inflation_factor=1.0,
                diagnostics={"fit_status": "empty_multivariate"},
            )
        )

        if mode == "split":
            one_active_c = float(one_active_model.global_inflation_factor)
        elif mode == "offline":
            one_active_c = float(global_one_active_c if global_one_active_c is not None else 1.0)
        else:
            n_one_active = float(one_active_model.n_calibration)
            w = n_one_active / (n_one_active + float(shrink_tau)) if n_one_active > 0 else 0.0
            log_c_1d = np.log(max(float(one_active_model.global_inflation_factor), 1e-12))
            log_c_multi = np.log(max(float(multivariate_model.global_inflation_factor), 1e-12))
            one_active_c = float(np.exp(w * log_c_1d + (1.0 - w) * log_c_multi))

        return CalibrationModel(
            method=f"regime_{mode}",
            n_calibration=int(one_active_model.n_calibration + multivariate_model.n_calibration),
            global_inflation_factor=float(multivariate_model.global_inflation_factor),
            max_observed_ratio=max(
                float(one_active_model.max_observed_ratio),
                float(multivariate_model.max_observed_ratio),
            ),
            beta=np.array(
                [np.log(max(multivariate_model.global_inflation_factor, 1e-12)), 0.0, 0.0],
                dtype=np.float64,
            ),
            diagnostics={
                "fit_status": f"regime_{mode}",
                "regime_models": {
                    "one_active": {
                        "n_calibration": int(one_active_model.n_calibration),
                        "c_hat": float(one_active_c),
                        "source_c_hat": float(one_active_model.global_inflation_factor),
                        "diagnostics": one_active_model.diagnostics,
                    },
                    "multivariate": {
                        "n_calibration": int(multivariate_model.n_calibration),
                        "c_hat": float(multivariate_model.global_inflation_factor),
                        "diagnostics": multivariate_model.diagnostics,
                    },
                },
                "global_one_active_c": (
                    None if global_one_active_c is None else float(global_one_active_c)
                ),
                "shrink_tau": float(shrink_tau),
            },
        )

    def wrapped_deflate(records, model):
        regime_models = model.diagnostics["regime_models"]
        focal_parents: list[str] = []
        focal_results: list[tuple[float, float, float]] = []
        methods: list[str] = []

        for record in records:
            if record.is_null_like:
                continue

            if not np.isfinite(record.stat) or record.degrees_of_freedom <= 0:
                focal_parents.append(record.parent)
                focal_results.append((np.nan, np.nan, np.nan))
                methods.append("invalid")
                continue

            regime = getattr(record, "projection_regime", "multivariate")
            regime_key = "one_active" if regime == "one_active" else "multivariate"
            inflation_factor = float(regime_models[regime_key]["c_hat"])

            t_adj = record.stat / inflation_factor
            p_adj = float(chi2.sf(t_adj, df=record.degrees_of_freedom))

            focal_parents.append(record.parent)
            focal_results.append((t_adj, float(record.degrees_of_freedom), p_adj))
            methods.append(f"adjusted_{regime_key}_{mode}")

        return focal_parents, focal_results, methods

    awa.collect_sibling_pair_records = wrapped_collect
    awa.fit_inflation_model = wrapped_fit
    awa._deflate_and_test = wrapped_deflate
    try:
        yield
    finally:
        awa.collect_sibling_pair_records = original_collect
        awa.fit_inflation_model = original_fit
        awa._deflate_and_test = original_deflate


def _compare(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    compare = reference.merge(
        candidate,
        on=["test_case", "case_id"],
        how="outer",
        suffixes=("_baseline", "_candidate"),
    )
    compare["ari_delta"] = compare["ari_candidate"] - compare["ari_baseline"]
    compare["found_delta"] = (
        compare["found_clusters_candidate"] - compare["found_clusters_baseline"]
    )

    changed = compare[
        (compare["found_delta"].fillna(0) != 0)
        | (compare["ari_delta"].abs().fillna(0) > 1e-12)
    ].copy()
    changed = changed.sort_values(["ari_delta", "case_id"], ascending=[True, True])

    summary = {
        "changed_cases": int(len(changed)),
        "improved_cases": int((changed["ari_delta"] > 1e-12).sum()),
        "worsened_cases": int((changed["ari_delta"] < -1e-12).sum()),
        "top_improvements": changed.sort_values("ari_delta", ascending=False)
        .head(10)[
            [
                "case_id",
                "ari_baseline",
                "ari_candidate",
                "ari_delta",
                "found_clusters_baseline",
                "found_clusters_candidate",
            ]
        ]
        .to_dict(orient="records"),
        "top_regressions": changed.sort_values("ari_delta", ascending=True)
        .head(10)[
            [
                "case_id",
                "ari_baseline",
                "ari_candidate",
                "ari_delta",
                "found_clusters_baseline",
                "found_clusters_candidate",
            ]
        ]
        .to_dict(orient="records"),
    }
    return compare, summary


def _run_suite_with_contexts(*contexts) -> pd.DataFrame:
    managers = [ctx for ctx in contexts]
    entered = []
    try:
        for manager in managers:
            entered.append(manager.__enter__())
        return _normalize_benchmark_df(_run_kl_suite())
    finally:
        while managers:
            manager = managers.pop()
            manager.__exit__(None, None, None)


def _global_one_active_c() -> tuple[float, int]:
    collector: list[dict[str, float]] = []
    with _patch_one_active_feature_to_1d():
        with _collect_one_active_records(collector):
            _run_kl_suite()

    ratios = np.array([row["stat"] / row["df"] for row in collector if np.isfinite(row["stat"]) and row["df"] > 0 and (row["stat"] / row["df"]) > 0], dtype=np.float64)
    weights = np.array([row["edge_weight"] for row in collector if np.isfinite(row["stat"]) and row["df"] > 0 and (row["stat"] / row["df"]) > 0], dtype=np.float64)
    if len(ratios) == 0 or np.sum(weights) <= 0:
        return 1.0, 0
    return float(np.average(ratios, weights=weights)), int(np.sum(weights > 0))


def main() -> None:
    output_dir = Path("/tmp/kl_one_active_feature_calibration_modes")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _normalize_benchmark_df(_run_kl_suite())
    pooled = _run_suite_with_contexts(_patch_one_active_feature_to_1d())
    split = _run_suite_with_contexts(
        _patch_one_active_feature_to_1d(),
        _patch_one_active_calibration(mode="split"),
    )

    global_one_active_c, global_one_active_n = _global_one_active_c()

    offline = _run_suite_with_contexts(
        _patch_one_active_feature_to_1d(),
        _patch_one_active_calibration(
            mode="offline",
            global_one_active_c=global_one_active_c,
        ),
    )
    shrink = _run_suite_with_contexts(
        _patch_one_active_feature_to_1d(),
        _patch_one_active_calibration(
            mode="shrink_to_multivariate",
            global_one_active_c=global_one_active_c,
            shrink_tau=10.0,
        ),
    )

    results = {
        "pooled_one_active_1d": pooled,
        "split_one_active_1d": split,
        "offline_one_active_1d": offline,
        "shrink_one_active_1d": shrink,
    }

    summary: dict[str, object] = {
        "baseline": _summarize(baseline),
        "global_one_active_c": global_one_active_c,
        "global_one_active_n": global_one_active_n,
    }

    baseline_summary = _summarize(baseline)
    baseline.to_csv(output_dir / "baseline.csv", index=False)

    for label, df in results.items():
        df.to_csv(output_dir / f"{label}.csv", index=False)
        compare_df, delta_summary = _compare(baseline, df)
        compare_df.to_csv(output_dir / f"{label}_diff.csv", index=False)
        candidate_summary = _summarize(df)
        summary[label] = candidate_summary
        summary[f"{label}_delta_vs_baseline"] = {
            "mean_ari": float(candidate_summary["mean_ari"] - baseline_summary["mean_ari"]),
            "exact_k": int(candidate_summary["exact_k"] - baseline_summary["exact_k"]),
            "k_equals_1": int(candidate_summary["k_equals_1"] - baseline_summary["k_equals_1"]),
            **delta_summary,
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
