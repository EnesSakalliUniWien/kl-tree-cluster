#!/usr/bin/env python3
"""Grid-search diagnostic for edge and sibling alpha method constants.

This runner evaluates the active KL benchmark path over an explicit grid of
edge and sibling alpha values. It is validation evidence only: it does not
change production defaults and does not add a fallback calibration rule.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.shared.cases import get_default_test_cases, get_test_cases_by_suite
from benchmarks.shared.pipeline import benchmark_cluster_algorithm

SCHEMA_VERSION = "alpha_grid_search/v1"
GENERATED_BY = "benchmarks.validation.alpha_grid_search"
DEFAULT_EDGE_ALPHA_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01)
DEFAULT_SIBLING_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


@dataclass(frozen=True)
class AlphaGridConfig:
    """Runtime contract for one alpha-grid diagnostic run."""

    suite: str
    case_names: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alphas: tuple[float, ...]
    output_dir: Path
    resume: bool
    alpha_pairs: tuple[tuple[float, float], ...] | None = None


def parse_float_grid(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated alpha grid."""
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Alpha grid must contain at least one value.")
    invalid = [value for value in values if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"Alpha values must lie in (0, 1): {invalid!r}")
    return values


def build_alpha_pairs(
    edge_alphas: Sequence[float],
    sibling_alphas: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Return the ordered Cartesian alpha grid."""
    pairs = tuple(
        (float(edge_alpha), float(sibling_alpha))
        for edge_alpha in edge_alphas
        for sibling_alpha in sibling_alphas
    )
    _validate_alpha_pairs(pairs)
    return pairs


def alpha_pair_id(edge_alpha: float, sibling_alpha: float) -> str:
    """Return the canonical file stem for one alpha pair."""
    return f"edge_{edge_alpha:g}__sibling_{sibling_alpha:g}".replace(".", "p")


def _validate_alpha_pairs(alpha_pairs: Sequence[tuple[float, float]]) -> None:
    if not alpha_pairs:
        raise ValueError("At least one alpha pair is required.")
    invalid = [
        (edge_alpha, sibling_alpha)
        for edge_alpha, sibling_alpha in alpha_pairs
        if not (0.0 < float(edge_alpha) < 1.0 and 0.0 < float(sibling_alpha) < 1.0)
    ]
    if invalid:
        raise ValueError(f"Alpha values must lie in (0, 1): {invalid!r}")


def select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict]:
    """Return benchmark cases by suite and optional explicit case-name filter."""
    if suite == "default":
        cases = get_default_test_cases()
    else:
        cases = get_test_cases_by_suite(suite)
    if not case_names:
        return [case.copy() for case in cases]

    requested = tuple(str(name) for name in case_names)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing!r}")
    return [by_name[name].copy() for name in requested]


def summarize_alpha_result(
    result: pd.DataFrame,
    *,
    edge_alpha: float,
    sibling_alpha: float,
    elapsed_sec: float,
) -> dict[str, object]:
    """Summarize one full benchmark result table for one alpha pair."""
    kl_rows = result[result["method"] == "kl"].copy()
    ok_rows = kl_rows[kl_rows["status"] == "ok"].copy()
    skip_rows = kl_rows[kl_rows["status"] != "ok"].copy()

    exact_k = int((ok_rows["found_clusters"] == ok_rows["true_clusters"]).sum())
    under_split = int((ok_rows["found_clusters"] < ok_rows["true_clusters"]).sum())
    over_split = int((ok_rows["found_clusters"] > ok_rows["true_clusters"]).sum())
    skip_reasons = (
        skip_rows["skip_reason"]
        .fillna("")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "n_cases": int(len(kl_rows)),
        "n_ok": int(len(ok_rows)),
        "n_skip": int(len(skip_rows)),
        "exact_k": exact_k,
        "under_split": under_split,
        "over_split": over_split,
        "mean_ari": float(ok_rows["ari"].mean()) if len(ok_rows) else None,
        "median_ari": float(ok_rows["ari"].median()) if len(ok_rows) else None,
        "mean_abs_cluster_count_error": (
            float((ok_rows["found_clusters"] - ok_rows["true_clusters"]).abs().mean())
            if len(ok_rows)
            else None
        ),
        "skip_reasons": skip_reasons,
        "elapsed_sec": round(float(elapsed_sec), 6),
    }


def run_alpha_grid_search(config: AlphaGridConfig) -> dict[str, object]:
    """Run the alpha grid and write per-alpha results plus summary files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(suite=config.suite, case_names=config.case_names)
    summaries: list[dict[str, object]] = []
    alpha_pairs = (
        config.alpha_pairs
        if config.alpha_pairs is not None
        else build_alpha_pairs(config.edge_alphas, config.sibling_alphas)
    )
    _validate_alpha_pairs(alpha_pairs)

    for edge_alpha, sibling_alpha in alpha_pairs:
        combo_id = alpha_pair_id(edge_alpha, sibling_alpha)
        result_path = config.output_dir / f"{combo_id}.csv"
        summary_path = config.output_dir / f"{combo_id}.summary.json"
        if config.resume and result_path.exists() and summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))
            continue

        started = perf_counter()
        result, _ = benchmark_cluster_algorithm(
            test_cases=[case.copy() for case in cases],
            significance_level=sibling_alpha,
            edge_alpha=edge_alpha,
            verbose=False,
            plot_umap=False,
            plot_manifold=False,
            methods=["kl"],
            concat_plots_pdf=False,
        )
        elapsed_sec = perf_counter() - started
        result.insert(0, "grid_edge_alpha", float(edge_alpha))
        result.insert(1, "grid_sibling_alpha", float(sibling_alpha))
        result.to_csv(result_path, index=False)

        summary = summarize_alpha_result(
            result,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
            elapsed_sec=elapsed_sec,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)

        print(
            "alpha-grid "
            f"edge={edge_alpha:g} sibling={sibling_alpha:g} "
            f"ok={summary['n_ok']}/{summary['n_cases']} "
            f"exact_k={summary['exact_k']} "
            f"mean_ari={summary['mean_ari']} "
            f"elapsed={summary['elapsed_sec']}s",
            flush=True,
        )

    summary_df = pd.DataFrame.from_records(summaries).sort_values(
        ["edge_alpha", "sibling_alpha"],
        ignore_index=True,
    )
    summary_df.to_csv(config.output_dir / "alpha_grid_summary.csv", index=False)
    manifest = {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": GENERATED_BY,
        "suite": config.suite,
        "case_names": [str(case["name"]) for case in cases],
        "edge_alphas": list(config.edge_alphas),
        "sibling_alphas": list(config.sibling_alphas),
        "selected_alpha_pairs": [
            {"edge_alpha": edge_alpha, "sibling_alpha": sibling_alpha}
            for edge_alpha, sibling_alpha in alpha_pairs
        ],
        "n_grid_alpha_pairs": int(len(config.edge_alphas) * len(config.sibling_alphas)),
        "n_selected_alpha_pairs": int(len(alpha_pairs)),
        "n_cases": int(len(cases)),
        "output_dir": str(config.output_dir),
        "git": current_git_state(),
        "note": (
            "Diagnostic benchmark grid only. These results compare benchmark "
            "performance across alpha constants; they do not prove selected-tree "
            "Type-I error control."
        ),
    }
    (config.output_dir / "alpha_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {"summary": summary_df, "manifest": manifest}


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


def _parse_case_names(raw_case_names: str | None) -> tuple[str, ...]:
    if raw_case_names is None:
        return ()
    return tuple(item.strip() for item in raw_case_names.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="full",
        help="Benchmark suite: full, binary, categorical, continuous, graph, or default.",
    )
    parser.add_argument(
        "--case-names",
        default=None,
        help="Optional comma-separated case-name subset resolved within the selected suite.",
    )
    parser.add_argument(
        "--edge-alphas",
        default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
        help="Comma-separated edge-alpha grid.",
    )
    parser.add_argument(
        "--sibling-alphas",
        default=",".join(str(value) for value in DEFAULT_SIBLING_ALPHA_GRID),
        help="Comma-separated sibling-alpha grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/alpha_grid"),
        help="Directory for per-alpha CSVs, summaries, and manifest.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute alpha pairs even if matching output files already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AlphaGridConfig(
        suite=str(args.suite),
        case_names=_parse_case_names(args.case_names),
        edge_alphas=parse_float_grid(str(args.edge_alphas)),
        sibling_alphas=parse_float_grid(str(args.sibling_alphas)),
        output_dir=args.output_dir,
        resume=not bool(args.no_resume),
    )
    run_alpha_grid_search(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
