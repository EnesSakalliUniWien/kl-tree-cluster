#!/usr/bin/env python3
"""Traversal-aligned sibling FDR null diagnostics.

This module generates validation evidence only. It does not change production
FDR, alpha defaults, empirical-null inflation, or selected-tree calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.projection_dimension_annotation import (
    write_record_projection_dimensions,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

from benchmarks.validation.selected_edge_type1_geometry import (
    _annotate_edges,
    _case_contract,
    _prepare_tree_for_mode,
    _select_cases,
)

SCHEMA_VERSION = "traversal_sibling_fdr_null/v1"
GENERATED_BY = "benchmarks.validation.traversal_sibling_fdr_null"
SIMULATION_OUTPUT_NAME = "traversal_sibling_fdr_simulations.csv"
SUMMARY_OUTPUT_NAME = "traversal_sibling_fdr_summary.csv"
MANIFEST_OUTPUT_NAME = "traversal_sibling_fdr_manifest.json"


class FdrLayer(StrEnum):
    """Layer of the sibling-FDR calibration stack under diagnostic study."""

    SYNTHETIC_VALID_P = "synthetic_valid_p"
    FIXED_TREE_WALD = "fixed_tree_wald"
    SELECTED_TREE_WALD = "selected_tree_wald"
    SELECTED_TREE_INFLATED = "selected_tree_inflated"


@dataclass(frozen=True)
class TraversalSiblingFdrConfig:
    """Runtime contract for one traversal sibling-FDR diagnostic layer."""

    layer: FdrLayer
    case_names: tuple[str, ...]
    replicates: int
    alpha: float
    base_seed: int
    suite: str = "binary"
    edge_alpha: float = DEFAULT_EDGE_ALPHA

    def __post_init__(self) -> None:
        if self.replicates <= 0:
            raise ValueError(f"replicates must be positive; got {self.replicates!r}.")
        if not np.isfinite(self.alpha) or not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be finite and in (0, 1]; got {self.alpha!r}.")
        if not np.isfinite(self.edge_alpha) or not (0.0 < float(self.edge_alpha) <= 1.0):
            raise ValueError(
                f"edge_alpha must be finite and in (0, 1]; got {self.edge_alpha!r}."
            )
        if not self.case_names:
            raise ValueError("case_names must contain at least one case.")


def parse_fdr_layers(raw: str) -> tuple[FdrLayer, ...]:
    """Parse a comma-separated list of FDR diagnostic layers."""
    layers = tuple(FdrLayer(item.strip()) for item in raw.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one FDR layer is required.")
    return layers


def parse_names(raw: str | None) -> tuple[str, ...]:
    """Parse optional comma-separated names."""
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def estimate_fdr(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    """Estimate FDR-style summaries without turning support failures into null decisions."""
    materialized_rows = list(rows)
    if not materialized_rows:
        raise ValueError("Cannot estimate FDR from zero simulation rows.")

    ok_rows = [row for row in materialized_rows if str(row.get("status", "ok")) == "ok"]
    support_failures = [
        row for row in materialized_rows if str(row.get("status", "")) == "support_failure"
    ]
    runtime_failures = [
        row for row in materialized_rows if str(row.get("status", "")) == "runtime_error"
    ]
    if ok_rows:
        fdps = [
            float(int(row["n_false_rejections"])) / float(max(1, int(row["n_rejections"])))
            for row in ok_rows
        ]
        false_rejection_indicators = [
            float(int(row["n_false_rejections"]) > 0) for row in ok_rows
        ]
        mean_fdp = float(np.mean(fdps))
        false_rejection_rate = float(np.mean(false_rejection_indicators))
    else:
        mean_fdp = float("nan")
        false_rejection_rate = float("nan")

    return {
        "mean_fdp": mean_fdp,
        "false_rejection_rate": false_rejection_rate,
        "n_simulations": float(len(materialized_rows)),
        "n_ok": float(len(ok_rows)),
        "n_support_failures": float(len(support_failures)),
        "n_runtime_failures": float(len(runtime_failures)),
    }


def classify_fdr_outcome(
    *,
    layer: FdrLayer,
    mean_fdp: float,
    alpha: float,
    n_ok: int | float = 1,
    n_support_failures: int | float = 0,
) -> str:
    """Classify the diagnostic outcome without hiding unsupported states."""
    if layer == FdrLayer.SELECTED_TREE_INFLATED and int(n_support_failures) > 0:
        return "inflation_support_failure"
    if int(n_ok) == 0 and int(n_support_failures) > 0:
        return "inflation_support_failure"
    if not np.isfinite(float(mean_fdp)):
        return "no_valid_decision_rows"
    if float(mean_fdp) <= float(alpha):
        return (
            "algorithmic_fdr_control"
            if layer == FdrLayer.SYNTHETIC_VALID_P
            else "empirical_control_observed"
        )
    if layer == FdrLayer.SYNTHETIC_VALID_P:
        return "algorithmic_fdr_failure"
    if layer == FdrLayer.FIXED_TREE_WALD:
        return "fixed_tree_calibration_failure"
    if layer == FdrLayer.SELECTED_TREE_WALD:
        return "edge_selected_family_failure"
    if layer == FdrLayer.SELECTED_TREE_INFLATED:
        return "inflation_or_selected_family_failure"
    raise ValueError(f"Unhandled FDR layer: {layer!r}.")


def _balanced_three_level_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")
    tree.add_edge("A", "A1")
    tree.add_edge("A", "A2")
    tree.add_edge("B", "B1")
    tree.add_edge("B", "B2")
    return tree


def _open_edge_annotations(tree: nx.DiGraph) -> pd.DataFrame:
    return init_sibling_annotation_df(
        pd.DataFrame(
            {
                "Child_Parent_Divergence_Significant": pd.Series(
                    {node: node != "root" for node in tree.nodes},
                    dtype=bool,
                )
            }
        )
    )


def _run_synthetic_valid_p_replicate(
    *,
    case_id: str,
    replicate_index: int,
    alpha: float,
    seed: int,
    layer: FdrLayer,
) -> dict[str, object]:
    tree = _balanced_three_level_tree()
    rng = np.random.default_rng(seed + replicate_index)
    parents = ["root", "A", "B"]
    p_values = rng.uniform(0.0, 1.0, size=len(parents))
    results = [(0.0, 1.0, float(value)) for value in p_values]
    annotated = apply_traversal_aligned_sibling_bh_results(
        tree,
        _open_edge_annotations(tree),
        parents,
        results,
        alpha,
    )
    rejected = annotated.loc[parents, "Sibling_BH_Different"].astype(bool).to_numpy()
    corrected = annotated.loc[parents, "Sibling_Divergence_P_Value_Corrected"]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": str(layer),
        "case_id": case_id,
        "replicate_index": int(replicate_index),
        "seed": int(seed + replicate_index),
        "status": "ok",
        "alpha": float(alpha),
        "edge_alpha": float("nan"),
        "n_sibling_candidates": int(len(parents)),
        "n_sibling_fdr_decisions": int(corrected.notna().sum()),
        "n_rejections": int(rejected.sum()),
        "n_false_rejections": int(rejected.sum()),
        "n_support_failures": 0,
        "failure_reason": "",
    }


def _projection_inputs(tree: nx.DiGraph, spectral_context: object) -> tuple[
    dict[str, int],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    return projection_dimensions, parent_projections, parent_eigenvalues


def _run_raw_wald_sibling_fdr(
    *,
    tree: nx.DiGraph,
    edge_annotations: pd.DataFrame,
    spectral_context: object,
    alpha: float,
    feature_space: object,
) -> pd.DataFrame:
    projection_dimensions, parent_projections, parent_eigenvalues = _projection_inputs(
        tree,
        spectral_context,
    )
    annotations_df = init_sibling_annotation_df(edge_annotations)
    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    mark_non_binary_as_skipped(annotations_df, non_binary)
    if not records:
        return annotations_df
    write_record_projection_dimensions(annotations_df, records)
    focal_records = [record for record in records if not record.is_null_like]
    skipped_parents = [record.parent for record in records if record.is_null_like]
    return apply_traversal_aligned_sibling_bh_results(
        tree,
        annotations_df,
        [record.parent for record in focal_records],
        [
            (float(record.stat), float(record.degrees_of_freedom), float(record.p_value))
            for record in focal_records
        ],
        alpha,
        skipped_parents=skipped_parents,
    )


def _run_inflated_sibling_fdr(
    *,
    tree: nx.DiGraph,
    edge_annotations: pd.DataFrame,
    spectral_context: object,
    alpha: float,
    feature_space: object,
) -> pd.DataFrame:
    projection_dimensions, parent_projections, parent_eigenvalues = _projection_inputs(
        tree,
        spectral_context,
    )
    return annotate_sibling_divergence(
        tree,
        edge_annotations,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        significance_level_alpha=alpha,
        feature_space=feature_space,
    )


def _count_sibling_decisions(annotations_df: pd.DataFrame) -> tuple[int, int]:
    corrected = annotations_df["Sibling_Divergence_P_Value_Corrected"]
    decided = corrected.notna()
    rejected = annotations_df.loc[decided, "Sibling_BH_Different"].astype(bool)
    n_rejections = int(rejected.sum())
    return int(decided.sum()), n_rejections


def _case_run_mode(layer: FdrLayer) -> str:
    if layer == FdrLayer.FIXED_TREE_WALD:
        return "fixed_tree"
    if layer in {FdrLayer.SELECTED_TREE_WALD, FdrLayer.SELECTED_TREE_INFLATED}:
        return "selected_tree"
    raise ValueError(f"Layer {layer!r} does not use benchmark case trees.")


def _run_case_replicate(
    *,
    layer: FdrLayer,
    case: dict[str, object],
    replicate_index: int,
    alpha: float,
    edge_alpha: float,
    base_seed: int,
) -> dict[str, object]:
    case_id, source_family, feature_representation, n_samples, n_features, n_categories = (
        _case_contract(case)
    )
    data_seed = int(base_seed + replicate_index * 1009)
    tree_seed = int(base_seed + replicate_index * 917 + 17)
    tree, data, feature_space, selected_tree, fixed_tree = _prepare_tree_for_mode(
        mode=_case_run_mode(layer),
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        tree_seed=tree_seed,
        data_seed=data_seed,
    )
    try:
        edge_annotations, spectral_context = _annotate_edges(
            tree,
            data,
            edge_alpha=edge_alpha,
            feature_space=feature_space,
        )
        if layer in {FdrLayer.FIXED_TREE_WALD, FdrLayer.SELECTED_TREE_WALD}:
            sibling_annotations = _run_raw_wald_sibling_fdr(
                tree=tree,
                edge_annotations=edge_annotations,
                spectral_context=spectral_context,
                alpha=alpha,
                feature_space=feature_space,
            )
        elif layer == FdrLayer.SELECTED_TREE_INFLATED:
            sibling_annotations = _run_inflated_sibling_fdr(
                tree=tree,
                edge_annotations=edge_annotations,
                spectral_context=spectral_context,
                alpha=alpha,
                feature_space=feature_space,
            )
        else:
            raise ValueError(f"Unsupported benchmark FDR layer: {layer!r}.")
    except ValueError as exc:
        message = str(exc)
        status = (
            "support_failure"
            if "selected non-null" in message
            or "empirical-null" in message
            or "calibration" in message
            else "runtime_error"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "layer": str(layer),
            "case_id": case_id,
            "replicate_index": int(replicate_index),
            "seed": int(data_seed),
            "status": status,
            "alpha": float(alpha),
            "edge_alpha": float(edge_alpha),
            "n_sibling_candidates": 0,
            "n_sibling_fdr_decisions": 0,
            "n_rejections": 0,
            "n_false_rejections": 0,
            "n_support_failures": int(status == "support_failure"),
            "failure_reason": message,
            "selected_tree": bool(selected_tree),
            "fixed_tree": bool(fixed_tree),
        }

    n_decisions, n_rejections = _count_sibling_decisions(sibling_annotations)
    n_candidates = int(
        sum(1 for node in tree.nodes if len(list(tree.successors(node))) == 2)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": str(layer),
        "case_id": case_id,
        "replicate_index": int(replicate_index),
        "seed": int(data_seed),
        "status": "ok",
        "alpha": float(alpha),
        "edge_alpha": float(edge_alpha),
        "n_sibling_candidates": n_candidates,
        "n_sibling_fdr_decisions": int(n_decisions),
        "n_rejections": int(n_rejections),
        "n_false_rejections": int(n_rejections),
        "n_support_failures": 0,
        "failure_reason": "",
        "selected_tree": bool(selected_tree),
        "fixed_tree": bool(fixed_tree),
    }


def _replicate_tuple(config: TraversalSiblingFdrConfig, replicate_indices: Sequence[int] | None) -> tuple[int, ...]:
    if replicate_indices is None:
        indices = tuple(range(config.replicates))
    else:
        indices = tuple(int(index) for index in replicate_indices)
    if not indices:
        raise ValueError("At least one replicate index is required.")
    invalid = [index for index in indices if index < 0 or index >= config.replicates]
    if invalid:
        raise ValueError(
            "Replicate indices must satisfy 0 <= index < replicates; "
            f"invalid={invalid!r}, replicates={config.replicates!r}."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Replicate indices must be unique; got {indices!r}.")
    return indices


def run_traversal_sibling_fdr_layer(
    config: TraversalSiblingFdrConfig,
    *,
    replicate_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Run one diagnostic layer and return simulation rows plus summary."""
    indices = _replicate_tuple(config, replicate_indices)
    if config.layer == FdrLayer.SYNTHETIC_VALID_P:
        rows = [
            _run_synthetic_valid_p_replicate(
                case_id=config.case_names[0],
                replicate_index=index,
                alpha=float(config.alpha),
                seed=int(config.base_seed),
                layer=config.layer,
            )
            for index in indices
        ]
    else:
        cases = _select_cases(suite=config.suite, case_names=config.case_names)
        rows = [
            _run_case_replicate(
                layer=config.layer,
                case=case,
                replicate_index=index,
                alpha=float(config.alpha),
                edge_alpha=float(config.edge_alpha),
                base_seed=int(config.base_seed),
            )
            for case in cases
            for index in indices
        ]
    summary = estimate_fdr(rows)
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "layer": str(config.layer),
            "suite": str(config.suite),
            "case_names": ",".join(config.case_names),
            "alpha": float(config.alpha),
            "edge_alpha": float(config.edge_alpha),
            "base_seed": int(config.base_seed),
            "outcome": classify_fdr_outcome(
                layer=config.layer,
                mean_fdp=float(summary["mean_fdp"]),
                alpha=float(config.alpha),
                n_ok=int(summary["n_ok"]),
                n_support_failures=int(summary["n_support_failures"]),
            ),
        }
    )
    return {"simulation_rows": rows, "summary": summary}


def current_git_state() -> dict[str, object]:
    """Return git provenance without hiding command failures."""
    state: dict[str, object] = {
        "build_commit": os.environ.get("KL_TE_GIT_COMMIT", "unknown"),
        "build_branch": os.environ.get("KL_TE_GIT_BRANCH", "unknown"),
        "build_dirty": os.environ.get("KL_TE_GIT_DIRTY", "unknown"),
    }
    commands = {
        "commit": ("git", "rev-parse", "HEAD"),
        "branch": ("git", "branch", "--show-current"),
        "status_short": ("git", "status", "--short"),
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True)
        except OSError as exc:
            state[key] = f"unavailable:{exc}"
            continue
        if completed.returncode != 0:
            state[key] = {
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
        else:
            state[key] = completed.stdout.strip()
    return state


def run_traversal_sibling_fdr_layers(
    configs: Sequence[TraversalSiblingFdrConfig],
    *,
    output_dir: Path,
    replicate_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Run one or more FDR layers and write combined CSV outputs."""
    if not configs:
        raise ValueError("At least one traversal sibling FDR config is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_outputs = [
        run_traversal_sibling_fdr_layer(config, replicate_indices=replicate_indices)
        for config in configs
    ]
    simulation_rows = [
        row
        for output in layer_outputs
        for row in output["simulation_rows"]
    ]
    summary_rows = [output["summary"] for output in layer_outputs]
    pd.DataFrame.from_records(simulation_rows).to_csv(
        output_dir / SIMULATION_OUTPUT_NAME,
        index=False,
    )
    pd.DataFrame.from_records(summary_rows).to_csv(
        output_dir / SUMMARY_OUTPUT_NAME,
        index=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layers": [str(config.layer) for config in configs],
        "suite": configs[0].suite,
        "case_names": sorted({name for config in configs for name in config.case_names}),
        "replicates": int(configs[0].replicates),
        "replicate_indices": (
            list(range(configs[0].replicates))
            if replicate_indices is None
            else [int(index) for index in replicate_indices]
        ),
        "alpha": float(configs[0].alpha),
        "edge_alpha": float(configs[0].edge_alpha),
        "base_seed": int(configs[0].base_seed),
        "n_simulation_rows": int(len(simulation_rows)),
        "n_summary_rows": int(len(summary_rows)),
        "outputs": {
            "simulations": str(output_dir / SIMULATION_OUTPUT_NAME),
            "summary": str(output_dir / SUMMARY_OUTPUT_NAME),
        },
        "git": current_git_state(),
        "note": (
            "Diagnostic traversal-aligned sibling FDR evidence only. Does not "
            "change production FDR, alpha defaults, or calibration behavior."
        ),
    }
    (output_dir / MANIFEST_OUTPUT_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _build_configs(args: argparse.Namespace) -> tuple[TraversalSiblingFdrConfig, ...]:
    layers = parse_fdr_layers(str(args.layers))
    case_names = parse_names(args.case_names)
    if not case_names and layers == (FdrLayer.SYNTHETIC_VALID_P,):
        case_names = ("synthetic_balanced_binary_tree",)
    if not case_names:
        raise ValueError("--case-names is required for benchmark-backed FDR layers.")
    return tuple(
        TraversalSiblingFdrConfig(
            layer=layer,
            suite=str(args.suite),
            case_names=case_names,
            replicates=int(args.replicates),
            alpha=float(args.alpha),
            edge_alpha=float(args.edge_alpha),
            base_seed=int(args.base_seed),
        )
        for layer in layers
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        default=FdrLayer.SYNTHETIC_VALID_P.value,
        help="Comma-separated FDR layers.",
    )
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", default=None)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--base-seed", type=int, default=20260604)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_traversal_sibling_fdr_layers(
        _build_configs(args),
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
