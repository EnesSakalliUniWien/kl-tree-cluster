"""Audit benchmark generator geometry before method interpretation.

This diagnostic answers a narrow question: did the generated representation
already lose or distort the target clustering signal before any TBS method runs?
It is intentionally read-only and writes compact CSV/Markdown evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tree_break_selection.tree.feature_space import resolve_feature_space, validate_feature_matrix

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data

LOW_SIGNAL_NEAREST_NEIGHBOR_THRESHOLD = 0.55
LOW_SIGNAL_SEPARATION_RATIO_THRESHOLD = 1.05
DUPLICATE_HEAVY_BLOCK_THRESHOLD = 8


@dataclass(frozen=True)
class GeneratorGeometryAuditRow:
    case_id: str
    case_category: str
    generator: str
    source_family: str
    feature_representation: str
    simulation_model: str
    observation_model: str
    benchmark_intent: str
    metadata_scientific_caution: str
    n_samples: int
    n_features: int
    true_clusters: int
    true_label_clusters: int
    feature_space_in_metadata: bool
    resolved_feature_family: str
    requires_precomputed_tbs_distance: bool
    distance_metric: str
    matrix_binary_valued: bool
    unique_rows: int
    duplicate_rows: int
    max_duplicate_count: int
    mixed_label_duplicate_blocks: int
    within_distance_mean: float
    between_distance_mean: float
    separation_ratio: float
    nearest_neighbor_same_label_fraction: float
    precomputed_distance_metric: str
    precomputed_within_distance_mean: float
    precomputed_between_distance_mean: float
    precomputed_separation_ratio: float
    precomputed_nearest_neighbor_same_label_fraction: float
    geometry_flags: str
    scientific_assessment: str
    recommended_simulation_family: str


def _nan() -> float:
    return float("nan")


def _pairwise_signal(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
) -> tuple[float, float, float, float]:
    if len(values) < 2 or len(set(labels.tolist())) <= 1:
        return _nan(), _nan(), _nan(), _nan()
    distances = squareform(pdist(values, metric=metric))
    return _distance_signal(distances, labels)


def _distance_signal(distances: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    same_label = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    within = distances[upper & same_label]
    between = distances[upper & ~same_label]
    within_mean = float(np.mean(within)) if len(within) else _nan()
    between_mean = float(np.mean(between)) if len(between) else _nan()
    separation_ratio = (
        float(between_mean / within_mean)
        if np.isfinite(within_mean) and within_mean > 0.0 and np.isfinite(between_mean)
        else _nan()
    )
    nearest = np.argsort(distances + np.eye(len(distances)) * 1e9, axis=1)[:, 0]
    nearest_same = float(np.mean(labels[nearest] == labels))
    return within_mean, between_mean, separation_ratio, nearest_same


def _duplicate_evidence(values: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    unique_rows, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    mixed_label_blocks = 0
    for block_id, count in enumerate(counts):
        if int(count) <= 1:
            continue
        if len(set(labels[inverse == block_id].tolist())) > 1:
            mixed_label_blocks += 1
    return (
        int(len(unique_rows)),
        int(len(values) - len(unique_rows)),
        int(counts.max(initial=0)),
        int(mixed_label_blocks),
    )


def _recommended_simulation_family(metadata: dict[str, object]) -> str:
    representation = str(metadata["feature_representation"])
    source_family = str(metadata["source_family"])
    generator = str(metadata["generator"])
    if representation == "graph_adjacency":
        return "graph_sbm_or_lfr_with_graph_native_distances"
    if representation == "continuous":
        if "single_cell" in source_family or generator in {"scrna", "scanpy"}:
            return "single_cell_count_model_splatter_scdesign3_or_zinb"
        return "gaussian_mixture_or_sparse_subspace_gaussian"
    if representation in {"categorical_one_hot", "quantile_one_hot"}:
        if source_family.startswith("phylogenetic") or generator == "phylogenetic":
            return "phylogenetic_substitution_indel_sequence_simulator"
        return "categorical_or_dirichlet_multinomial_blocks"
    if representation == "median_binary":
        return "explicit_bernoulli_threshold_model_or_continuous_gaussian_variant"
    if representation == "binary":
        return "bernoulli_template_or_latent_class_binary_model"
    return "case_specific_model_required"


def _scientific_assessment(
    *,
    metadata: dict[str, object],
    separation_ratio: float,
    nearest_same: float,
    max_duplicate_count: int,
    mixed_label_duplicate_blocks: int,
) -> tuple[str, str]:
    flags: list[str] = []
    notes: list[str] = []
    representation = str(metadata["feature_representation"])
    metadata_caution = str(metadata.get("scientific_caution", "none"))

    if (
        np.isfinite(nearest_same)
        and nearest_same < LOW_SIGNAL_NEAREST_NEIGHBOR_THRESHOLD
        and np.isfinite(separation_ratio)
        and separation_ratio < LOW_SIGNAL_SEPARATION_RATIO_THRESHOLD
    ):
        flags.append("low_geometry_signal")
        notes.append(
            "Generated representation has weak within/between distance separation before methods run."
        )
    if max_duplicate_count >= DUPLICATE_HEAVY_BLOCK_THRESHOLD:
        flags.append("duplicate_heavy")
        notes.append(
            "Exact duplicate blocks can create tied linkage topology and pydiffmap bandwidth failures."
        )
    if mixed_label_duplicate_blocks:
        flags.append("mixed_label_duplicates")
        notes.append("At least one exact duplicate row contains multiple true labels.")
    if representation == "median_binary":
        flags.append("discretized_continuous_as_bernoulli")
        notes.append(metadata_caution)
    if representation == "graph_adjacency":
        flags.append("graph_rows_as_feature_matrix")
        notes.append(metadata_caution)
    if not flags:
        flags.append("geometry_contract_clean")
        notes.append("No generator-level geometry warning under current audit thresholds.")
    return ";".join(flags), " ".join(notes)


def audit_case(case: dict[str, object]) -> GeneratorGeometryAuditRow:
    data, labels, _original, metadata = generate_case_data(case)
    labels = np.asarray(labels, dtype=int)
    values = data.to_numpy(dtype=float)
    feature_space = metadata.get("feature_space")
    resolved_feature_space = resolve_feature_space(tuple(data.columns), feature_space)
    validate_feature_matrix(values, resolved_feature_space, value_name=str(metadata["name"]))

    unique_rows, duplicate_rows, max_duplicate_count, mixed_duplicate_blocks = _duplicate_evidence(
        values,
        labels,
    )
    default_metric = "euclidean" if resolved_feature_space.family_label == "continuous" else "hamming"
    within, between, ratio, nearest_same = _pairwise_signal(
        values,
        labels,
        metric=default_metric,
    )

    pre_metric = ""
    pre_within = pre_between = pre_ratio = pre_nearest_same = _nan()
    precomputed = metadata.get("precomputed_distance_condensed")
    if precomputed is not None and len(labels) > 1 and len(set(labels.tolist())) > 1:
        pre_metric = str(metadata.get("distance_metric", "precomputed"))
        pre_distances = squareform(np.asarray(precomputed, dtype=float))
        pre_within, pre_between, pre_ratio, pre_nearest_same = _distance_signal(
            pre_distances,
            labels,
        )

    flags, assessment = _scientific_assessment(
        metadata=metadata,
        separation_ratio=ratio,
        nearest_same=nearest_same,
        max_duplicate_count=max_duplicate_count,
        mixed_label_duplicate_blocks=mixed_duplicate_blocks,
    )
    return GeneratorGeometryAuditRow(
        case_id=str(metadata["name"]),
        case_category=str(case.get("category", "")),
        generator=str(metadata["generator"]),
        source_family=str(metadata["source_family"]),
        feature_representation=str(metadata["feature_representation"]),
        simulation_model=str(metadata["simulation_model"]),
        observation_model=str(metadata["observation_model"]),
        benchmark_intent=str(metadata["benchmark_intent"]),
        metadata_scientific_caution=str(metadata["scientific_caution"]),
        n_samples=int(data.shape[0]),
        n_features=int(data.shape[1]),
        true_clusters=int(metadata["n_clusters"]),
        true_label_clusters=int(len(set(labels.tolist()))),
        feature_space_in_metadata=feature_space is not None,
        resolved_feature_family=str(resolved_feature_space.family_label),
        requires_precomputed_tbs_distance=bool(metadata["requires_precomputed_tbs_distance"]),
        distance_metric=default_metric,
        matrix_binary_valued=bool(np.isin(values, (0.0, 1.0)).all()),
        unique_rows=unique_rows,
        duplicate_rows=duplicate_rows,
        max_duplicate_count=max_duplicate_count,
        mixed_label_duplicate_blocks=mixed_duplicate_blocks,
        within_distance_mean=within,
        between_distance_mean=between,
        separation_ratio=ratio,
        nearest_neighbor_same_label_fraction=nearest_same,
        precomputed_distance_metric=pre_metric,
        precomputed_within_distance_mean=pre_within,
        precomputed_between_distance_mean=pre_between,
        precomputed_separation_ratio=pre_ratio,
        precomputed_nearest_neighbor_same_label_fraction=pre_nearest_same,
        geometry_flags=flags,
        scientific_assessment=assessment,
        recommended_simulation_family=str(
            metadata.get("recommended_simulation_family") or _recommended_simulation_family(metadata)
        ),
    )


def build_generator_geometry_audit() -> pd.DataFrame:
    """Return one generator-geometry audit row per default benchmark case."""
    rows = [asdict(audit_case(case)) for case in get_default_test_cases()]
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    flag_counts: dict[str, int] = {}
    for raw in frame["geometry_flags"].astype(str):
        for flag in raw.split(";"):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "case_count": int(len(frame)),
        "flag_counts": dict(sorted(flag_counts.items())),
        "intent_counts": frame["benchmark_intent"].value_counts().to_dict(),
        "representation_counts": frame["feature_representation"].value_counts().to_dict(),
        "recommended_simulation_counts": frame[
            "recommended_simulation_family"
        ].value_counts().to_dict(),
    }


def write_generator_geometry_audit(
    frame: pd.DataFrame,
    *,
    output: Path,
    markdown_output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = _summary(frame)
    flagged = frame[~frame["geometry_flags"].str.contains("geometry_contract_clean")]
    low_signal = frame[frame["geometry_flags"].str.contains("low_geometry_signal")]
    duplicate_heavy = frame[frame["geometry_flags"].str.contains("duplicate_heavy")]

    lines = [
        "# Generator geometry audit",
        "",
        f"- Cases audited: `{summary['case_count']}`",
        f"- CSV: `{output}`",
        "",
        "## Flag counts",
        "",
    ]
    for flag, count in summary["flag_counts"].items():
        lines.append(f"- `{flag}`: {count}")
    lines.extend(["", "## Low-signal generated representations", ""])
    if low_signal.empty:
        lines.append("None under current thresholds.")
    else:
        for row in low_signal.sort_values("nearest_neighbor_same_label_fraction").itertuples():
            lines.append(
                "- "
                f"`{row.case_id}`: nn_same={row.nearest_neighbor_same_label_fraction:.3f}, "
                f"separation_ratio={row.separation_ratio:.3f}, "
                f"recommended={row.recommended_simulation_family}"
            )
    lines.extend(["", "## Duplicate-heavy generated representations", ""])
    if duplicate_heavy.empty:
        lines.append("None under current thresholds.")
    else:
        for row in duplicate_heavy.sort_values("max_duplicate_count", ascending=False).itertuples():
            lines.append(
                "- "
                f"`{row.case_id}`: max_duplicate_count={row.max_duplicate_count}, "
                f"mixed_label_duplicate_blocks={row.mixed_label_duplicate_blocks}"
            )
    lines.extend(["", "## Scientific-assumption warnings", ""])
    for row in flagged.sort_values(["feature_representation", "case_id"]).itertuples():
        lines.append(f"- `{row.case_id}`: {row.geometry_flags} — {row.scientific_assessment}")
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/audits/generated/generator-geometry.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/audits/generated/generator-geometry.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    frame = build_generator_geometry_audit()
    write_generator_geometry_audit(
        frame,
        output=args.output,
        markdown_output=args.markdown_output,
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(_summary(frame), indent=2), encoding="utf-8")
    print(f"Generator geometry audit: {args.output}")
    print(f"Generator geometry markdown: {args.markdown_output}")
    print(
        "Generator flags: "
        + ", ".join(f"{key}={value}" for key, value in _summary(frame)["flag_counts"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
