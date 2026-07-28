"""Fixed cosine-band coherence comparator.

This diagnostic ports the useful part of the old c2ef cosine-subspace scripts
without restoring them as production method paths. It keeps the predeclared
cosine eigen-bands, runs the current gate/decomposition stack on each band tree,
and reports biological/coherence summaries for the resulting clusters.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.space_separation import (
    SpectralBlock,
    cosine_eigendecomposition,
    weight_feature_matrix,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
    continuous_feature_space_from_columns,
)

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    SCHEMA_VERSION as ADAPTIVE_SCHEMA_VERSION,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    labels_from_assignments,
    run_block_tree,
    sibling_method_counts,
)
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "cosine_band_coherence_comparator/v1"
STUDY_ROLE = "diagnostic_legacy_cosine_band_coherence_comparator_not_calibration"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "weighting",
    "block_id",
    "block_name",
    "block_type",
    "block_start",
    "block_end",
    "run_id",
    "subspace_dimensions",
    "block_energy_fraction",
    "edge_alpha",
    "sibling_alpha",
    "status",
    "n_clusters",
    "largest_cluster_fraction",
    "singleton_fraction",
    "coherent_cluster_count",
    "coherent_cluster_fraction",
    "median_cluster_significant_terms_q05",
    "median_cluster_mean_within_tfidf_cosine",
    "ari",
    "nmi",
    "runtime_sec",
    "sibling_test_method_counts",
    "production_status",
    "error",
)

COHERENCE_COLUMNS = (
    "schema_version",
    "study_role",
    "run_id",
    "weighting",
    "block_name",
    "cluster_id",
    "cluster_size",
    "n_significant_terms_q05",
    "min_q_value",
    "top_term",
    "top_term_cluster_prevalence",
    "top_term_rest_prevalence",
    "top_term_prevalence_delta",
    "mean_within_tfidf_cosine",
    "coherent_by_rule",
)

SPECTRUM_COLUMNS = (
    "schema_version",
    "study_role",
    "weighting",
    "component",
    "eigenvalue",
    "fraction_of_kept_operator_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed c2ef cosine-band coherence diagnostics."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weightings", nargs="+", default=["binary", "tfidf"])
    parser.add_argument("--band-names", nargs="+", default=None)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--min-significant-terms", type=int, default=3)
    parser.add_argument("--min-prevalence-delta", type=float, default=0.25)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def legacy_cosine_bands(rank: int) -> list[SpectralBlock]:
    """Return the fixed cosine bands used by the c2ef comparator scripts."""
    candidates = (
        ("common_mode_01", 1, 1, "legacy_common_mode"),
        ("variation_02_05", 2, 5, "legacy_variation_band"),
        ("variation_06_15", 6, 15, "legacy_variation_band"),
        ("variation_16_35", 16, 35, "legacy_variation_band"),
        ("variation_36_80", 36, 80, "legacy_variation_band"),
        ("broad_variation_02_35", 2, 35, "legacy_broad_variation"),
        ("broad_variation_02_80", 2, 80, "legacy_broad_variation"),
        ("all_modes_01_80", 1, 80, "legacy_all_modes"),
    )
    bands: list[SpectralBlock] = []
    for name, start, end, block_type in candidates:
        if start > int(rank):
            continue
        block_end = min(end, int(rank))
        bands.append(
            SpectralBlock(
                block_id=len(bands),
                block_name=name,
                block_start=start,
                block_end=block_end,
                block_type=block_type,
            )
        )
    return bands


def infer_comparator_feature_space(data: pd.DataFrame) -> FeatureSpace:
    """Infer a conservative feature-space contract for comparator diagnostics."""
    values = data.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.all() and np.all((values == 0.0) | (values == 1.0)):
        return bernoulli_feature_space_from_columns(data.columns)
    return continuous_feature_space_from_columns(data.columns)


def load_comparator_matrix(path: Path) -> pd.DataFrame:
    """Load a CSV or TSV feature matrix with samples in rows."""
    first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    sep = "\t" if "\t" in first_line else ","
    data = pd.read_csv(path, sep=sep, index_col=0)
    zero_columns = data.columns[(data.sum(axis=0) == 0).to_numpy()]
    if len(zero_columns):
        data = data.drop(columns=zero_columns)
    zero_rows = data.index[(data.sum(axis=1) == 0).to_numpy()]
    if len(zero_rows):
        raise ValueError(
            "Rows with zero feature mass cannot enter cosine/KAK analysis: "
            f"{list(zero_rows[:10])!r}"
        )
    return data.astype(float)


def _bh_q_values(p_values: Sequence[float]) -> np.ndarray:
    if len(p_values) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(multipletests(p_values, method="fdr_bh")[1], dtype=float)


def _one_sided_enrichment_p_values(
    *,
    cluster_counts: pd.Series,
    rest_counts: pd.Series,
    cluster_size: int,
    rest_size: int,
) -> np.ndarray:
    """Return vectorized Fisher-exact greater-tail enrichment p-values."""
    present_cluster = cluster_counts.to_numpy(dtype=int)
    present_rest = rest_counts.to_numpy(dtype=int)
    population_size = int(cluster_size) + int(rest_size)
    present_total = present_cluster + present_rest
    return np.asarray(
        hypergeom.sf(
            present_cluster - 1,
            population_size,
            present_total,
            int(cluster_size),
        ),
        dtype=float,
    )


def _mean_within_tfidf_cosine_by_cluster(
    data: pd.DataFrame,
    labels: pd.Series,
) -> dict[int, float]:
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(
        data.to_numpy(dtype=float)
    )
    out: dict[int, float] = {}
    label_values = labels.astype(int).to_numpy()
    for cluster_id in sorted(set(label_values)):
        idx = np.where(label_values == cluster_id)[0]
        if len(idx) < 2:
            out[int(cluster_id)] = math.nan
            continue
        gram = (tfidf[idx] @ tfidf[idx].T).toarray()
        out[int(cluster_id)] = float((gram.sum() - np.trace(gram)) / (len(idx) * (len(idx) - 1)))
    return out


def cluster_biological_coherence(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    min_significant_terms: int = 3,
    min_prevalence_delta: float = 0.25,
) -> pd.DataFrame:
    """Return old-style enrichment/coherence summaries for cluster labels."""
    aligned_labels = labels.reindex(data.index).astype(int)
    within_cosine = _mean_within_tfidf_cosine_by_cluster(data, aligned_labels)
    feature_presence = data.gt(0).astype(int)
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(aligned_labels.unique()):
        mask = aligned_labels.eq(cluster_id)
        cluster_size = int(mask.sum())
        rest_size = int((~mask).sum())
        if cluster_size == 0 or rest_size == 0:
            continue
        cluster = feature_presence.loc[mask]
        rest = feature_presence.loc[~mask]
        cluster_count = cluster.sum(axis=0).astype(int)
        rest_count = rest.sum(axis=0).astype(int)
        p_values = _one_sided_enrichment_p_values(
            cluster_counts=cluster_count,
            rest_counts=rest_count,
            cluster_size=cluster_size,
            rest_size=rest_size,
        )
        q_values = _bh_q_values(p_values)
        cluster_prev = cluster_count / cluster_size
        rest_prev = rest_count / max(rest_size, 1)
        delta = cluster_prev - rest_prev
        frame = pd.DataFrame(
            {
                "term": data.columns.astype(str),
                "q_value": q_values,
                "cluster_prevalence": cluster_prev.to_numpy(dtype=float),
                "rest_prevalence": rest_prev.to_numpy(dtype=float),
                "prevalence_delta": delta.to_numpy(dtype=float),
            }
        ).sort_values(["q_value", "prevalence_delta"], ascending=[True, False])
        significant = frame[frame["q_value"] < 0.05]
        top = frame.iloc[0]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "n_significant_terms_q05": int(significant.shape[0]),
                "min_q_value": float(top["q_value"]),
                "top_term": str(top["term"]),
                "top_term_cluster_prevalence": float(top["cluster_prevalence"]),
                "top_term_rest_prevalence": float(top["rest_prevalence"]),
                "top_term_prevalence_delta": float(top["prevalence_delta"]),
                "mean_within_tfidf_cosine": within_cosine.get(int(cluster_id), math.nan),
                "coherent_by_rule": bool(
                    cluster_size >= 3
                    and int(significant.shape[0]) >= int(min_significant_terms)
                    and float(top["prevalence_delta"]) >= float(min_prevalence_delta)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _spectrum_rows(weighting: str, eigvals: np.ndarray) -> list[dict[str, object]]:
    total = float(np.sum(eigvals))
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "weighting": weighting,
            "component": int(component),
            "eigenvalue": float(eigenvalue),
            "fraction_of_kept_operator_energy": (
                float(eigenvalue / total) if total > 0.0 else math.nan
            ),
        }
        for component, eigenvalue in enumerate(eigvals, start=1)
    ]


def build_cosine_band_coherence_rows(
    *,
    data: pd.DataFrame,
    weightings: Sequence[str] = ("binary", "tfidf"),
    max_rank: int = 80,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    feature_space: FeatureSpace | None = None,
    min_significant_terms: int = 3,
    min_prevalence_delta: float = 0.25,
    checkpoint_dir: Path | None = None,
    band_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run fixed cosine-band trees and return row/coherence/spectrum tables."""
    resolved_feature_space = feature_space or infer_comparator_feature_space(data)
    rows: list[dict[str, object]] = []
    coherence_rows: list[pd.DataFrame] = []
    spectrum_rows: list[dict[str, object]] = []

    def write_checkpoint() -> None:
        if checkpoint_dir is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(rows, columns=ROW_COLUMNS).to_csv(
            checkpoint_dir / "cosine_band_comparator_rows.partial.csv",
            index=False,
        )
        (
            pd.concat(coherence_rows, ignore_index=True)
            if coherence_rows
            else pd.DataFrame(columns=COHERENCE_COLUMNS)
        ).to_csv(
            checkpoint_dir / "cosine_band_comparator_cluster_coherence.partial.csv",
            index=False,
        )
        pd.DataFrame.from_records(spectrum_rows, columns=SPECTRUM_COLUMNS).to_csv(
            checkpoint_dir / "cosine_band_comparator_spectrum.partial.csv",
            index=False,
        )

    for weighting in weightings:
        try:
            values = weight_feature_matrix(data, weighting)
            eigvals, eigvecs = cosine_eigendecomposition(values, max_rank=max_rank)
            spectrum_rows.extend(_spectrum_rows(str(weighting), eigvals))
            total_energy = float(np.sum(eigvals))
            bands = legacy_cosine_bands(len(eigvals))
            if band_names is not None:
                requested_bands = {str(name) for name in band_names}
                bands = [band for band in bands if str(band.block_name) in requested_bands]
                missing_bands = requested_bands - {str(band.block_name) for band in bands}
                if missing_bands:
                    raise ValueError(
                        f"Unknown or unavailable band_names: {sorted(missing_bands)!r}"
                    )
        except Exception as exc:  # noqa: BLE001 - diagnostic table records failures.
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "weighting": str(weighting),
                    "block_id": pd.NA,
                    "block_name": pd.NA,
                    "block_type": pd.NA,
                    "block_start": pd.NA,
                    "block_end": pd.NA,
                    "run_id": f"{weighting}__spectrum_failed",
                    "subspace_dimensions": pd.NA,
                    "block_energy_fraction": math.nan,
                    "edge_alpha": float(edge_alpha),
                    "sibling_alpha": float(sibling_alpha),
                    "status": "failed_spectrum",
                    "n_clusters": pd.NA,
                    "largest_cluster_fraction": math.nan,
                    "singleton_fraction": math.nan,
                    "coherent_cluster_count": 0,
                    "coherent_cluster_fraction": math.nan,
                    "median_cluster_significant_terms_q05": math.nan,
                    "median_cluster_mean_within_tfidf_cosine": math.nan,
                    "ari": math.nan,
                    "nmi": math.nan,
                    "runtime_sec": 0.0,
                    "sibling_test_method_counts": "",
                    "production_status": "diagnostic_only_not_production_calibration",
                    "error": repr(exc),
                }
            )
            write_checkpoint()
            continue

        for block in bands:
            run_id = f"{weighting}__{block.block_name}"
            block_energy = (
                float(np.sum(eigvals[block.block_start - 1 : block.block_end]) / total_energy)
                if total_energy > 0.0
                else math.nan
            )
            start = time.perf_counter()
            try:
                assignments, _, annotations_df = run_block_tree(
                    data=data,
                    feature_space=resolved_feature_space,
                    eigvals=eigvals,
                    eigvecs=eigvecs,
                    block=block,
                    edge_alpha=edge_alpha,
                    sibling_alpha=sibling_alpha,
                )
                labels = labels_from_assignments(assignments, data.index)
                label_series = pd.Series(labels, index=data.index, name="cluster_id")
                coherence = cluster_biological_coherence(
                    data,
                    label_series,
                    min_significant_terms=min_significant_terms,
                    min_prevalence_delta=min_prevalence_delta,
                )
                if not coherence.empty:
                    enriched = coherence.assign(
                        schema_version=SCHEMA_VERSION,
                        study_role=STUDY_ROLE,
                        run_id=run_id,
                        weighting=str(weighting),
                        block_name=block.block_name,
                    )
                    coherence_rows.append(enriched.loc[:, COHERENCE_COLUMNS])
                cluster_sizes = pd.Series(labels).value_counts()
                coherent_count = (
                    int(coherence["coherent_by_rule"].sum()) if not coherence.empty else 0
                )
                n_clusters = int(len(cluster_sizes))
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "weighting": str(weighting),
                        "block_id": int(block.block_id),
                        "block_name": block.block_name,
                        "block_type": block.block_type,
                        "block_start": int(block.block_start),
                        "block_end": int(block.block_end),
                        "run_id": run_id,
                        "subspace_dimensions": int(block.block_end - block.block_start + 1),
                        "block_energy_fraction": block_energy,
                        "edge_alpha": float(edge_alpha),
                        "sibling_alpha": float(sibling_alpha),
                        "status": "ok",
                        "n_clusters": n_clusters,
                        "largest_cluster_fraction": float(cluster_sizes.max() / len(labels)),
                        "singleton_fraction": float(
                            (cluster_sizes == 1).sum() / max(n_clusters, 1)
                        ),
                        "coherent_cluster_count": coherent_count,
                        "coherent_cluster_fraction": float(coherent_count / max(n_clusters, 1)),
                        "median_cluster_significant_terms_q05": (
                            float(coherence["n_significant_terms_q05"].median())
                            if not coherence.empty
                            else math.nan
                        ),
                        "median_cluster_mean_within_tfidf_cosine": (
                            float(coherence["mean_within_tfidf_cosine"].median())
                            if not coherence.empty
                            else math.nan
                        ),
                        "ari": math.nan,
                        "nmi": math.nan,
                        "runtime_sec": float(time.perf_counter() - start),
                        "sibling_test_method_counts": sibling_method_counts(annotations_df),
                        "production_status": "diagnostic_only_not_production_calibration",
                        "error": "",
                    }
                )
                if "true_label" in data.attrs:
                    true_labels = np.asarray(data.attrs["true_label"])
                    rows[-1]["ari"] = float(adjusted_rand_score(true_labels, labels))
                    rows[-1]["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
                write_checkpoint()
            except Exception as exc:  # noqa: BLE001 - diagnostic table records failures.
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "weighting": str(weighting),
                        "block_id": int(block.block_id),
                        "block_name": block.block_name,
                        "block_type": block.block_type,
                        "block_start": int(block.block_start),
                        "block_end": int(block.block_end),
                        "run_id": run_id,
                        "subspace_dimensions": int(block.block_end - block.block_start + 1),
                        "block_energy_fraction": block_energy,
                        "edge_alpha": float(edge_alpha),
                        "sibling_alpha": float(sibling_alpha),
                        "status": "failed_gate",
                        "n_clusters": pd.NA,
                        "largest_cluster_fraction": math.nan,
                        "singleton_fraction": math.nan,
                        "coherent_cluster_count": 0,
                        "coherent_cluster_fraction": math.nan,
                        "median_cluster_significant_terms_q05": math.nan,
                        "median_cluster_mean_within_tfidf_cosine": math.nan,
                        "ari": math.nan,
                        "nmi": math.nan,
                        "runtime_sec": float(time.perf_counter() - start),
                        "sibling_test_method_counts": "",
                        "production_status": "diagnostic_only_not_production_calibration",
                        "error": repr(exc),
                    }
                )
                write_checkpoint()

    rows_df = pd.DataFrame.from_records(rows, columns=ROW_COLUMNS)
    coherence_df = (
        pd.concat(coherence_rows, ignore_index=True)
        if coherence_rows
        else pd.DataFrame(columns=COHERENCE_COLUMNS)
    )
    spectrum_df = pd.DataFrame.from_records(spectrum_rows, columns=SPECTRUM_COLUMNS)
    return rows_df, coherence_df, spectrum_df


def run_cosine_band_coherence_comparator(
    *,
    input_path: Path,
    output_dir: Path,
    weightings: Sequence[str] = ("binary", "tfidf"),
    max_rank: int = 80,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    min_significant_terms: int = 3,
    min_prevalence_delta: float = 0.25,
    checkpoint_dir: Path | None = None,
    band_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Run the comparator and write rows, coherence, spectrum, and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_comparator_matrix(input_path)
    rows, coherence, spectrum = build_cosine_band_coherence_rows(
        data=data,
        weightings=weightings,
        max_rank=max_rank,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        min_significant_terms=min_significant_terms,
        min_prevalence_delta=min_prevalence_delta,
        checkpoint_dir=checkpoint_dir,
        band_names=band_names,
    )
    rows_path = output_dir / "cosine_band_comparator_rows.csv"
    coherence_path = output_dir / "cosine_band_comparator_cluster_coherence.csv"
    spectrum_path = output_dir / "cosine_band_comparator_spectrum.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    coherence.to_csv(coherence_path, index=False)
    spectrum.to_csv(spectrum_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "generated_by": __name__,
                "generated_at_utc": format_timestamp_utc(),
                "input_path": str(input_path),
                "band_names": list(band_names) if band_names is not None else None,
                "adaptive_schema_reused": ADAPTIVE_SCHEMA_VERSION,
                "outputs": {
                    "rows": str(rows_path),
                    "cluster_coherence": str(coherence_path),
                    "spectrum": str(spectrum_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_path,
        "cluster_coherence": coherence_path,
        "spectrum": spectrum_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    run_cosine_band_coherence_comparator(
        input_path=args.input,
        output_dir=args.output_dir,
        weightings=tuple(args.weightings),
        max_rank=int(args.max_rank),
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        min_significant_terms=int(args.min_significant_terms),
        min_prevalence_delta=float(args.min_prevalence_delta),
        checkpoint_dir=args.checkpoint_dir,
        band_names=args.band_names,
    )


if __name__ == "__main__":
    main()
