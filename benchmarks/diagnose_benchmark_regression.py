#!/usr/bin/env python3
"""Diagnose benchmark regressions by combining artifact diffs and local replays.

This tool compares two stored benchmark runs, replays changed cases through the
current benchmark pipeline, and classifies each changed case into one of three
buckets:

- current-patch reproducible
- historical-code reproducible
- mixed/unreproducible artifact

It is intentionally focused on the KL family, where the audit CSVs preserve the
node-level sibling statistics needed to explain BH flips and df changes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once
import kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection as record_collection


DEFAULT_BASELINE_RUN = (
    REPO_ROOT / "benchmarks" / "results" / "run_20260328_103732Z"
)
DEFAULT_CANDIDATE_RUN = (
    REPO_ROOT / "benchmarks" / "results" / "run_20260328_113818Z"
)
DEFAULT_HISTORICAL_COMMITS = ["866eeec", "71edd11", "796f1bc"]
DEFAULT_SENTINEL_CASES = [2, 59, 1, 61]


AUDIT_SUFFIX_BY_METHOD = {
    "kl": "kl_stats",
    "kl_complete": "kl_complete_stats",
    "kl_diffusion": "kl_diffusion_stats",
    "kl_single": "kl_single_stats",
}

TRACKED_AUDIT_COLUMNS = [
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_BH_Different",
    "Sibling_Divergence_Skipped",
    "Sibling_Divergence_Invalid",
]

HISTORICAL_REPLAY_CODE = r"""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.util.case_inputs import prepare_case_inputs

case_nums = [int(part) for part in sys.argv[1].split(",") if part]
method = sys.argv[2]
params = {"tree_distance_metric": "hamming", "tree_linkage_method": "average"}
cases = get_default_test_cases()

out = {}
for case_num in case_nums:
    tc = deepcopy(cases[case_num - 1])
    (
        data_t,
        y_t,
        x_original,
        meta,
        distance_condensed,
        distance_matrix,
        precomputed_distance_condensed,
    ) = prepare_case_inputs(tc, [method])
    result = run_clustering_result(
        data_df=data_t,
        method_id=method,
        params=params,
        significance_level=0.05,
        distance_matrix=distance_matrix,
        distance_condensed=distance_condensed,
    )
    if result.status == "ok" and result.labels is not None:
        ari_value = adjusted_rand_score(np.asarray(y_t), np.asarray(result.labels))
        ari = float(ari_value) if not math.isnan(float(ari_value)) else None
        found_clusters = int(result.found_clusters)
    else:
        ari = None
        found_clusters = int(result.found_clusters)
    out[str(case_num)] = {
        "ari": ari,
        "found_clusters": found_clusters,
    }

print(json.dumps(out, sort_keys=True))
"""


@dataclass(frozen=True)
class ReplayResult:
    ari: float | None
    found_clusters: int
    annotations: pd.DataFrame
    source_by_parent: dict[str, str]
    resolved_k_by_parent: dict[str, int | None]

    @property
    def comparable_tuple(self) -> tuple[float | None, int]:
        if self.ari is None:
            return None, self.found_clusters
        return round(self.ari, 12), self.found_clusters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose KL benchmark regressions from stored runs and local replays."
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=DEFAULT_BASELINE_RUN,
        help="Baseline benchmark run directory.",
    )
    parser.add_argument(
        "--candidate-run",
        type=Path,
        default=DEFAULT_CANDIDATE_RUN,
        help="Candidate benchmark run directory.",
    )
    parser.add_argument(
        "--method",
        default="kl",
        choices=sorted(AUDIT_SUFFIX_BY_METHOD),
        help="Benchmark method to diagnose.",
    )
    parser.add_argument(
        "--historical-commits",
        default=",".join(DEFAULT_HISTORICAL_COMMITS),
        help="Comma-separated historical commits to replay for mixed cases.",
    )
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip historical worktree replays.",
    )
    parser.add_argument(
        "--sentinel-cases",
        default=",".join(str(case_num) for case_num in DEFAULT_SENTINEL_CASES),
        help="Comma-separated test_case ids used as sentinel anchors.",
    )
    parser.add_argument(
        "--case-nums",
        default="",
        help="Optional comma-separated subset of test_case ids to analyze.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the diagnosis table.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the diagnosis records.",
    )
    return parser.parse_args()


def _method_label_to_case_file(method: str) -> str:
    try:
        return AUDIT_SUFFIX_BY_METHOD[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported method for audit lookup: {method}") from exc


def _load_full_benchmark_rows(
    baseline_run: Path,
    candidate_run: Path,
    *,
    method: str,
) -> pd.DataFrame:
    baseline_csv = baseline_run / "full_benchmark_comparison.csv"
    candidate_csv = candidate_run / "full_benchmark_comparison.csv"
    baseline_df = pd.read_csv(baseline_csv)
    candidate_df = pd.read_csv(candidate_csv)
    key = ["test_case", "case_id", "method", "params"]
    merged = baseline_df.merge(candidate_df, on=key, suffixes=("_old", "_new"))
    changed_mask = (
        merged["ari_old"].round(12) != merged["ari_new"].round(12)
    ) | (
        merged["found_clusters_old"] != merged["found_clusters_new"]
    ) | (
        merged["status_old"] != merged["status_new"]
    )
    changed = merged[changed_mask & (merged["method"] == method)].copy()
    changed = changed[
        [
            "test_case",
            "case_id",
            "ari_old",
            "found_clusters_old",
            "ari_new",
            "found_clusters_new",
            "status_old",
            "status_new",
        ]
    ].drop_duplicates()
    return changed.sort_values("test_case").reset_index(drop=True)


def _artifact_tuple(ari: float, found_clusters: int) -> tuple[float | None, int]:
    if pd.isna(ari):
        return None, int(found_clusters)
    return round(float(ari), 12), int(found_clusters)


def _load_audit_frame(run_dir: Path, case_num: int, method: str) -> pd.DataFrame:
    suffix = _method_label_to_case_file(method)
    audit_path = run_dir / "audit" / f"case_{int(case_num)}_{suffix}.csv"
    return pd.read_csv(audit_path)


def _coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _same_float(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return round(float(left), 12) == round(float(right), 12)


def _classify_node_change(old_row: pd.Series, new_row: pd.Series) -> str | None:
    old_skip = _coerce_bool(old_row.get("Sibling_Divergence_Skipped"))
    new_skip = _coerce_bool(new_row.get("Sibling_Divergence_Skipped"))
    old_invalid = _coerce_bool(old_row.get("Sibling_Divergence_Invalid"))
    new_invalid = _coerce_bool(new_row.get("Sibling_Divergence_Invalid"))
    if old_skip != new_skip or old_invalid != new_invalid:
        return "availability change"

    old_df = old_row.get("Sibling_Degrees_of_Freedom")
    new_df = new_row.get("Sibling_Degrees_of_Freedom")
    if not _same_float(old_df, new_df):
        return "df reparameterization"

    old_bh = _coerce_bool(old_row.get("Sibling_BH_Different"))
    new_bh = _coerce_bool(new_row.get("Sibling_BH_Different"))
    old_corr = old_row.get("Sibling_Divergence_P_Value_Corrected")
    new_corr = new_row.get("Sibling_Divergence_P_Value_Corrected")
    if old_bh != new_bh or not _same_float(old_corr, new_corr):
        return "boundary flip"
    return None


def _summarize_case_audit_diff(
    baseline_run: Path,
    candidate_run: Path,
    *,
    case_num: int,
    method: str,
) -> dict[str, Any]:
    old_df = _load_audit_frame(baseline_run, case_num, method).set_index("node_id")
    new_df = _load_audit_frame(candidate_run, case_num, method).set_index("node_id")
    merged = old_df.merge(
        new_df,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_old", "_new"),
    )
    mechanism_by_node: dict[str, str] = {}
    for node_id, row in merged.iterrows():
        old_row = pd.Series(
            {col: row.get(f"{col}_old") for col in TRACKED_AUDIT_COLUMNS}
        )
        new_row = pd.Series(
            {col: row.get(f"{col}_new") for col in TRACKED_AUDIT_COLUMNS}
        )
        changed = any(
            not _same_float(old_row[col], new_row[col])
            if col not in {
                "Sibling_BH_Different",
                "Sibling_Divergence_Skipped",
                "Sibling_Divergence_Invalid",
            }
            else _coerce_bool(old_row[col]) != _coerce_bool(new_row[col])
            for col in TRACKED_AUDIT_COLUMNS
        )
        if not changed:
            continue
        mechanism = _classify_node_change(old_row, new_row)
        if mechanism is not None:
            mechanism_by_node[str(node_id)] = mechanism

    mechanism_counts = Counter(mechanism_by_node.values())
    dominant_mechanism = "none"
    if mechanism_counts:
        dominant_mechanism = mechanism_counts.most_common(1)[0][0]
    return {
        "changed_node_count": int(len(mechanism_by_node)),
        "df_reparameterization_count": int(
            mechanism_counts.get("df reparameterization", 0)
        ),
        "boundary_flip_count": int(mechanism_counts.get("boundary flip", 0)),
        "availability_change_count": int(
            mechanism_counts.get("availability change", 0)
        ),
        "dominant_mechanism": dominant_mechanism,
        "changed_nodes": sorted(mechanism_by_node),
        "mechanism_by_node": mechanism_by_node,
    }


def _resolve_case_specs() -> list[dict[str, Any]]:
    return get_default_test_cases()


def _source_from_fallback_decision(
    *,
    spectral_dimension: int | None,
    resolved_k: int | None,
    resolved_projection: np.ndarray | None,
) -> str:
    if spectral_dimension is not None:
        return "spectral"
    if resolved_k is None:
        return "johnson_lindenstrauss_fallback"
    if resolved_projection is not None:
        return "parent_pca_small_k"
    return "unknown"


class CurrentWorkspaceReplayer:
    def __init__(self, method: str) -> None:
        self.method = method
        self.spec = METHOD_SPECS[method]
        self.params = self.spec.param_grid[0]
        self.case_specs = _resolve_case_specs()
        self._case_input_cache: dict[int, tuple[Any, ...]] = {}
        self._result_cache: dict[tuple[int, str], ReplayResult] = {}

    def _prepare_case_inputs(self, case_num: int) -> tuple[Any, ...]:
        cached = self._case_input_cache.get(case_num)
        if cached is not None:
            return cached
        case_spec = deepcopy(self.case_specs[case_num - 1])
        prepared = prepare_case_inputs(case_spec, [self.method])
        self._case_input_cache[case_num] = prepared
        return prepared

    def replay_case(self, case_num: int, *, mode: str) -> ReplayResult:
        cache_key = (case_num, mode)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached

        (
            data_t,
            y_t,
            x_original,
            meta,
            distance_condensed,
            distance_matrix,
            precomputed_distance_condensed,
        ) = self._prepare_case_inputs(case_num)
        case_spec = deepcopy(self.case_specs[case_num - 1])

        original_fallback = record_collection._resolve_parent_pca_fallback
        source_by_parent: dict[str, str] = {}
        resolved_k_by_parent: dict[str, int | None] = {}

        def _old_like_impl(
            parent: str,
            annotations_df: pd.DataFrame,
            *,
            spectral_dimension: int | None,
            sibling_pca_projection: np.ndarray | None,
            sibling_pca_eigenvalues: np.ndarray | None,
        ) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
            return spectral_dimension, sibling_pca_projection, sibling_pca_eigenvalues

        target_impl = original_fallback if mode == "current" else _old_like_impl

        def _instrumented_fallback(
            parent: str,
            annotations_df: pd.DataFrame,
            *,
            spectral_dimension: int | None,
            sibling_pca_projection: np.ndarray | None,
            sibling_pca_eigenvalues: np.ndarray | None,
        ) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
            resolved_k, resolved_projection, resolved_eigenvalues = target_impl(
                parent,
                annotations_df,
                spectral_dimension=spectral_dimension,
                sibling_pca_projection=sibling_pca_projection,
                sibling_pca_eigenvalues=sibling_pca_eigenvalues,
            )
            source_by_parent[parent] = _source_from_fallback_decision(
                spectral_dimension=spectral_dimension,
                resolved_k=resolved_k,
                resolved_projection=resolved_projection,
            )
            resolved_k_by_parent[parent] = None if resolved_k is None else int(resolved_k)
            return resolved_k, resolved_projection, resolved_eigenvalues

        record_collection._resolve_parent_pca_fallback = _instrumented_fallback
        try:
            result_row, computed_result, _ = run_single_method_once(
                method_id=self.method,
                spec=self.spec,
                params=self.params,
                case_idx=case_num,
                case_name=case_spec["name"],
                tc_seed=case_spec.get("seed"),
                significance_level=0.05,
                data_t=data_t,
                y_t=y_t,
                x_original=x_original,
                meta=meta,
                distance_matrix=distance_matrix,
                distance_condensed=distance_condensed,
                precomputed_distance_condensed=precomputed_distance_condensed,
                matrix_audit=False,
            )
        finally:
            record_collection._resolve_parent_pca_fallback = original_fallback

        ari = float(result_row.ari)
        replay = ReplayResult(
            ari=None if math.isnan(ari) else ari,
            found_clusters=int(result_row.found_clusters),
            annotations=computed_result.annotations.copy(),
            source_by_parent=source_by_parent,
            resolved_k_by_parent=resolved_k_by_parent,
        )
        self._result_cache[cache_key] = replay
        return replay


def _find_git_binary() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for historical replays.")
    return git


def _run_historical_replay_for_cases(
    *,
    commit: str,
    case_nums: list[int],
    method: str,
) -> dict[int, tuple[float | None, int]]:
    git_binary = _find_git_binary()
    with tempfile.TemporaryDirectory(prefix="kl-te-regdiag-") as temp_dir:
        worktree_dir = Path(temp_dir) / "worktree"
        subprocess.run(
            [git_binary, "worktree", "add", "--detach", str(worktree_dir), commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        HISTORICAL_REPLAY_CODE,
                        ",".join(str(case_num) for case_num in case_nums),
                        method,
                    ],
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(completed.stdout)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[historical replay] commit {commit} batch failed; "
                    "retrying per-case.",
                    file=sys.stderr,
                )
                if exc.stderr:
                    print(exc.stderr.strip(), file=sys.stderr)
                payload: dict[str, dict[str, Any]] = {}
                for case_num in case_nums:
                    try:
                        completed = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                HISTORICAL_REPLAY_CODE,
                                str(case_num),
                                method,
                            ],
                            cwd=worktree_dir,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as case_exc:
                        print(
                            f"[historical replay] commit {commit} failed for case "
                            f"{case_num}.",
                            file=sys.stderr,
                        )
                        if case_exc.stderr:
                            print(case_exc.stderr.strip(), file=sys.stderr)
                        continue
                    payload.update(json.loads(completed.stdout))
        finally:
            subprocess.run(
                [git_binary, "worktree", "remove", "--force", str(worktree_dir)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

    out: dict[int, tuple[float | None, int]] = {}
    for case_num_text, result in payload.items():
        ari = result.get("ari")
        out[int(case_num_text)] = (
            None if ari is None else round(float(ari), 12),
            int(result["found_clusters"]),
        )
    return out


def _format_result(result: tuple[float | None, int]) -> str:
    ari, clusters = result
    ari_text = "nan" if ari is None else f"{ari:.6f}"
    return f"ari={ari_text};k={clusters}"


def _select_historical_match(
    *,
    case_num: int,
    artifact_new: tuple[float | None, int],
    historical_results: dict[str, dict[int, tuple[float | None, int]]],
) -> tuple[str | None, tuple[float | None, int] | None]:
    for commit, result_by_case in historical_results.items():
        result = result_by_case.get(case_num)
        if result == artifact_new:
            return commit, result
    return None, None


def _source_transition_summary(
    changed_nodes: list[str],
    current_replay: ReplayResult,
    old_like_replay: ReplayResult,
) -> str:
    transitions = Counter()
    for node_id in changed_nodes:
        old_source = old_like_replay.source_by_parent.get(node_id, "missing")
        new_source = current_replay.source_by_parent.get(node_id, "missing")
        transitions[f"{old_source}->{new_source}"] += 1
    if not transitions:
        return "none"
    transition, count = transitions.most_common(1)[0]
    return f"{transition} ({count})"


def _build_diagnosis_rows(
    *,
    changed_cases: pd.DataFrame,
    baseline_run: Path,
    candidate_run: Path,
    method: str,
    current_replayer: CurrentWorkspaceReplayer,
    historical_results: dict[str, dict[int, tuple[float | None, int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in changed_cases.itertuples(index=False):
        case_num = int(record.test_case)
        old_artifact = _artifact_tuple(record.ari_old, record.found_clusters_old)
        new_artifact = _artifact_tuple(record.ari_new, record.found_clusters_new)
        current_replay = current_replayer.replay_case(case_num, mode="current")
        old_like_replay = current_replayer.replay_case(case_num, mode="old_like")
        audit_summary = _summarize_case_audit_diff(
            baseline_run,
            candidate_run,
            case_num=case_num,
            method=method,
        )

        classification = "mixed/unreproducible artifact"
        historical_commit = None
        historical_result = None
        if (
            current_replay.comparable_tuple == new_artifact
            and old_like_replay.comparable_tuple == old_artifact
        ):
            classification = "current-patch reproducible"
        else:
            historical_commit, historical_result = _select_historical_match(
                case_num=case_num,
                artifact_new=new_artifact,
                historical_results=historical_results,
            )
            if historical_commit is not None:
                classification = "historical-code reproducible"

        rows.append(
            {
                "test_case": case_num,
                "case_id": record.case_id,
                "old_artifact": _format_result(old_artifact),
                "new_artifact": _format_result(new_artifact),
                "current_replay": _format_result(current_replay.comparable_tuple),
                "old_like_replay": _format_result(old_like_replay.comparable_tuple),
                "changed_sibling_nodes": audit_summary["changed_node_count"],
                "df_reparameterization_nodes": audit_summary[
                    "df_reparameterization_count"
                ],
                "boundary_flip_nodes": audit_summary["boundary_flip_count"],
                "availability_change_nodes": audit_summary[
                    "availability_change_count"
                ],
                "dominant_mechanism": audit_summary["dominant_mechanism"],
                "projection_source_attribution": _source_transition_summary(
                    audit_summary["changed_nodes"],
                    current_replay,
                    old_like_replay,
                ),
                "classification": classification,
                "historical_commit": historical_commit or "",
                "historical_replay": (
                    _format_result(historical_result) if historical_result else ""
                ),
                "reproducible_from_surviving_diff": (
                    "yes"
                    if classification == "current-patch reproducible"
                    else "no"
                ),
            }
        )
    return rows


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No changed cases found._"
    headers = [
        "test_case",
        "case_id",
        "old_artifact",
        "new_artifact",
        "current_replay",
        "old_like_replay",
        "changed_sibling_nodes",
        "dominant_mechanism",
        "projection_source_attribution",
        "classification",
        "historical_commit",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    changed_cases = _load_full_benchmark_rows(
        args.baseline_run,
        args.candidate_run,
        method=args.method,
    )
    if str(args.case_nums).strip():
        requested_case_nums = {
            int(value)
            for value in str(args.case_nums).split(",")
            if str(value).strip()
        }
        changed_cases = changed_cases[
            changed_cases["test_case"].astype(int).isin(requested_case_nums)
        ].copy()

    sentinel_case_nums = [
        int(value)
        for value in str(args.sentinel_cases).split(",")
        if str(value).strip()
    ]
    historical_commits = [
        value.strip()
        for value in str(args.historical_commits).split(",")
        if value.strip()
    ]

    current_replayer = CurrentWorkspaceReplayer(args.method)
    historical_results: dict[str, dict[int, tuple[float | None, int]]] = {}

    if not args.skip_historical and historical_commits:
        mixed_case_nums = [
            int(row.test_case)
            for row in changed_cases.itertuples(index=False)
        ]
        historical_case_nums = sorted(set(mixed_case_nums) | set(sentinel_case_nums))
        for commit in historical_commits:
            historical_results[commit] = _run_historical_replay_for_cases(
                commit=commit,
                case_nums=historical_case_nums,
                method=args.method,
            )

    rows = _build_diagnosis_rows(
        changed_cases=changed_cases,
        baseline_run=args.baseline_run,
        candidate_run=args.candidate_run,
        method=args.method,
        current_replayer=current_replayer,
        historical_results=historical_results,
    )

    diagnosis_df = pd.DataFrame(rows)
    summary_counts = diagnosis_df["classification"].value_counts().to_dict()

    print(
        f"Compared {args.method} between "
        f"{args.baseline_run.name} and {args.candidate_run.name}."
    )
    print(f"Changed cases: {len(diagnosis_df)}")
    print("Classification counts:")
    for label, count in summary_counts.items():
        print(f"- {label}: {count}")
    print()
    print(_render_markdown_table(rows))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnosis_df.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(rows, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
