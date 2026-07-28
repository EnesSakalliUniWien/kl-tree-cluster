#!/usr/bin/env python3
"""Construct the selected-node sibling-null law and diagnostic examples.

This module is evidence-only. It records the conditional probability law needed
for edge-supported fail-closed cases and summarizes existing experiment outputs.
It does not change production clustering, p-values, alpha defaults, or fallback
behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

SCHEMA_VERSION = "selected_node_sibling_null_law/v1"
GENERATED_BY = "benchmarks.validation.statistics.selected_node_sibling_null_law"

DEFAULT_CASE_DIAGNOSIS_PATH = Path(
    "reports/tree_consensus_topology_difference_diagnosis_20260709/"
    "topology_difference_case_diagnosis.csv"
)
DEFAULT_TRACE_PATH = Path(
    "reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv"
)
DEFAULT_ALPHA_SWEEP_SUMMARY_PATH = Path(
    "reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("reports/selected_node_sibling_null_law_20260709")
DEFAULT_ADAPTIVE_REPLAY_OUTPUT_DIR = Path("reports/selected_node_adaptive_law_replay_20260709")

CONDITIONS_NAME = "selected_node_sibling_null_law_conditions.csv"
EXAMPLES_NAME = "selected_node_sibling_null_law_examples.csv"
OCCURRENCES_NAME = "selected_node_sibling_null_law_occurrences.csv"
REPORT_NAME = "selected_node_sibling_null_law_report.md"
MANIFEST_NAME = "selected_node_sibling_null_law_manifest.json"
ADAPTIVE_DECISIONS_NAME = "selected_node_adaptive_law_decisions.csv"
ADAPTIVE_CELL_SUMMARY_NAME = "selected_node_adaptive_law_cell_summary.csv"
ADAPTIVE_CASE_SUMMARY_NAME = "selected_node_adaptive_law_case_summary.csv"
ADAPTIVE_METHOD_SUMMARY_NAME = "selected_node_adaptive_law_method_summary.csv"
ADAPTIVE_EXTERNAL_AUDIT_NAME = "selected_node_adaptive_law_external_audit.csv"
ADAPTIVE_REPORT_NAME = "selected_node_adaptive_law_replay_report.md"
ADAPTIVE_MANIFEST_NAME = "selected_node_adaptive_law_replay_manifest.json"


@dataclass(frozen=True)
class ExampleSetting:
    """One deterministic projected-Wald reaction example."""

    scenario: str
    n_left: int
    n_right: int
    branch_length_sum: float
    mean_branch_length: float
    projected_components: tuple[float, ...]
    degrees_of_freedom: int
    covariance_spectrum: tuple[float, ...]
    condition_note: str


@dataclass(frozen=True)
class AdaptiveLawPolicy:
    """One diagnostic selected-node replay policy."""

    policy_id: str
    display_name: str
    law_status: str
    base_alpha: float = 0.01
    cap_alpha: float = 0.20
    use_topology_stability: bool = True
    min_child_fraction: float = 0.03
    note: str = ""


ADAPTIVE_LAW_POLICIES: tuple[AdaptiveLawPolicy, ...] = (
    AdaptiveLawPolicy(
        policy_id="strict_selected_law_required",
        display_name="Strict selected conditional law required",
        law_status="requires_exact_selected_node_p_value",
        cap_alpha=0.01,
        note=(
            "Scientifically admissible gate placeholder: fail closed until exact "
            "selected-node conditional p-values or conditional Monte Carlo support exist."
        ),
    ),
    AdaptiveLawPolicy(
        policy_id="adaptive_topology_branch_spending_proxy",
        display_name="Adaptive topology/branch spending proxy",
        law_status="diagnostic_proxy",
        use_topology_stability=True,
        note=(
            "Uses active sibling p-values with edge, diagnostic-covariance, topology "
            "stability, branch-ratio, and child-size spending. Not a production p-value."
        ),
    ),
    AdaptiveLawPolicy(
        policy_id="adaptive_no_stability_ablation",
        display_name="Adaptive no-stability ablation",
        law_status="diagnostic_ablation",
        use_topology_stability=False,
        note=(
            "Ablates topology/branch damping to show what the adaptive evidence channel "
            "would open without selected-topology stability conditioning."
        ),
    ),
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required selected-node-null input is missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _format_float(value: object, digits: int = 4) -> str:
    number = _finite(value)
    if not math.isfinite(number):
        return ""
    if number != 0.0 and abs(number) < 10 ** (-(digits - 1)):
        return f"{number:.2e}"
    return f"{number:.{digits}g}"


def sampling_variance_scale(n_left: float, n_right: float) -> float:
    """Return the current sibling two-sample variance scale."""

    if n_left <= 0.0 or n_right <= 0.0:
        raise ValueError("Sibling sample sizes must be positive.")
    return 1.0 / float(n_left) + 1.0 / float(n_right)


def branch_time_multiplier(branch_length_sum: float, mean_branch_length: float) -> float:
    """Return the current NNLS branch-time variance multiplier."""

    if branch_length_sum < 0.0:
        raise ValueError("branch_length_sum must be non-negative.")
    if branch_length_sum == 0.0:
        return 1.0
    if mean_branch_length <= 0.0:
        raise ValueError("mean_branch_length must be positive when branch length is positive.")
    return 1.0 + float(branch_length_sum) / (2.0 * float(mean_branch_length))


def projected_wald_tail(projected_components: Sequence[float], degrees_of_freedom: int) -> float:
    """Return the fixed-subspace chi-square tail probability."""

    if degrees_of_freedom < 0:
        raise ValueError("degrees_of_freedom must be non-negative.")
    if degrees_of_freedom == 0:
        return 1.0
    statistic = float(np.sum(np.asarray(projected_components, dtype=np.float64) ** 2))
    return float(chi2.sf(statistic, df=degrees_of_freedom))


def build_selected_node_null_conditions() -> pd.DataFrame:
    """Return the mathematical law components as durable CSV rows."""

    rows = [
        {
            "law_component": "fixed_topology_fixed_subspace",
            "conditioning_event": (
                "T, selected node v, child sets A_v/B_v, NNLS branch lengths "
                "ell_A/ell_B, local covariance Sigma_v, parent projection U_v,k, "
                "and projection dimension k are treated as fixed"
            ),
            "null_object": ("Z_v = U_v,k L_v^{-1}(hat_mu_A-hat_mu_B) | conditioning"),
            "reference_law": "Z_v ~ N(0, I_k); W_v = ||Z_v||^2 ~ chi_square(k)",
            "validity_requirement": (
                "The topology, node, projection rows, dimension, covariance chart, "
                "and branch-time scale are fixed or selected from information "
                "independent of the tested sibling contrast."
            ),
            "production_implication": (
                "This is the implemented local fixed-subspace law; it is not by "
                "itself a selected-node law."
            ),
        },
        {
            "law_component": "selected_topology_selected_node",
            "conditioning_event": (
                "E_sel = {adaptive KNN diffusion, topology method, rooting, NNLS "
                "branch lengths, edge-open path, and node v are the observed ones}"
            ),
            "null_object": "P_0(W_v >= w_obs | E_sel, H0_sibling(v))",
            "reference_law": (
                "conditional tail law over the selected hierarchy; generally not "
                "plain chi_square(k)"
            ),
            "validity_requirement": (
                "Either an analytic selected-region law or conditional Monte Carlo "
                "that reruns the selection map and conditions on the selected event."
            ),
            "production_implication": (
                "Required for edge-supported cases before extra alpha can be spent."
            ),
        },
        {
            "law_component": "nnls_branch_length_scale",
            "conditioning_event": (
                "Observed nonnegative NNLS branch lengths ell_A and ell_B and "
                "global mean branch length bar_ell are fixed"
            ),
            "null_object": (
                "Var(hat_mu_A-hat_mu_B | T, ell) = "
                "(1/n_A+1/n_B) * (1+(ell_A+ell_B)/(2 bar_ell)) * Sigma_v"
            ),
            "reference_law": "branch lengths alter the whitening scale, not the chi-square df",
            "validity_requirement": (
                "NNLS lengths must be conditioned on; reusing the same scale after "
                "changing topology is invalid."
            ),
            "production_implication": (
                "Short selected branches sharpen W; long branches absorb contrast "
                "as branch-time noise."
            ),
        },
        {
            "law_component": "local_covariance_eigensystem",
            "conditioning_event": (
                "Local covariance Sigma_v has eigensystem Q Lambda Q^T in the "
                "feature chart used by the sibling contrast"
            ),
            "null_object": (
                "L_v L_v^T = variance_scale * Sigma_v; whitened contrast "
                "L_v^{-1}(hat_mu_A-hat_mu_B)"
            ),
            "reference_law": "standard normal only after valid local whitening",
            "validity_requirement": (
                "Boundary probabilities, dense continuous covariance, and ridge "
                "regularization must preserve a stable positive definite chart."
            ),
            "production_implication": (
                "Covariance errors change the mass of W before topology selection "
                "is even considered."
            ),
        },
        {
            "law_component": "multiplicity_projector",
            "conditioning_event": ("A selected covariance eigenvalue block has multiplicity m > 1"),
            "null_object": (
                "Use the invariant projector P_G onto the whole tied eigenspace, "
                "not an arbitrary ordered eigenvector inside the block"
            ),
            "reference_law": "||P_G Z||^2 ~ chi_square(rank(P_G)) when P_G is fixed",
            "validity_requirement": (
                "Dimension rules may not cut through an indistinguishable "
                "multiplicity block without adding a tie-breaking law."
            ),
            "production_implication": (
                "Open dimensions should be block-stable; otherwise p-values depend "
                "on arbitrary eigenvector orientation."
            ),
        },
        {
            "law_component": "adaptive_dimension_mixture",
            "conditioning_event": (
                "K is selected from edge-derived dimensions or projected-energy "
                "fraction after observing local data"
            ),
            "null_object": "P_0(W_K >= w_obs | E_sel) = sum_k P(W_k >= w_obs, K=k | E_sel)",
            "reference_law": (
                "mixture of selected-dimension tails, or chi_square(k) only after "
                "conditioning on K=k and the selection rule"
            ),
            "validity_requirement": (
                "The accepted dimension count and any open-dimensional boundary "
                "must be part of the conditioning event."
            ),
            "production_implication": (
                "Dimension adaptivity explains why the same Wald mass can move "
                "between significant and non-significant regimes."
            ),
        },
        {
            "law_component": "topology_stability_alpha_spending",
            "conditioning_event": (
                "Accessible topology family M and observed split-support mass "
                "pi_v across methods/resamples are fixed"
            ),
            "null_object": (
                "alpha_v = alpha_global * spend(pi_v) or selected-family "
                "min-p over M with the same sibling statistic"
            ),
            "reference_law": (
                "closed testing / alpha spending over topology family, not a larger global alpha"
            ),
            "validity_requirement": (
                "The spend rule must be predeclared and monotone in stability; "
                "unstable selected splits get little or no alpha."
            ),
            "production_implication": (
                "A branch-supported sibling split still fails closed when its "
                "topology support is weak."
            ),
        },
    ]
    return pd.DataFrame.from_records(rows)


def _example_row(setting: ExampleSetting) -> dict[str, object]:
    components = np.asarray(setting.projected_components, dtype=np.float64)
    statistic = float(np.sum(components[: setting.degrees_of_freedom] ** 2))
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": setting.scenario,
        "n_left": int(setting.n_left),
        "n_right": int(setting.n_right),
        "sampling_variance_scale": sampling_variance_scale(setting.n_left, setting.n_right),
        "branch_length_sum": float(setting.branch_length_sum),
        "mean_branch_length": float(setting.mean_branch_length),
        "branch_time_multiplier": branch_time_multiplier(
            setting.branch_length_sum,
            setting.mean_branch_length,
        ),
        "projected_components": ",".join(f"{value:.6g}" for value in components),
        "degrees_of_freedom": int(setting.degrees_of_freedom),
        "wald_statistic": statistic,
        "chi_square_tail_p_value": projected_wald_tail(
            components[: setting.degrees_of_freedom],
            setting.degrees_of_freedom,
        ),
        "covariance_spectrum": ",".join(f"{value:.6g}" for value in setting.covariance_spectrum),
        "condition_note": setting.condition_note,
    }


def build_wald_reaction_examples() -> pd.DataFrame:
    """Return small examples showing how W reacts under the conditioning variables."""

    signal = 0.35
    balanced_scale = math.sqrt(sampling_variance_scale(80, 80) * branch_time_multiplier(0.20, 0.50))
    imbalanced_scale = math.sqrt(
        sampling_variance_scale(20, 160) * branch_time_multiplier(0.20, 0.50)
    )
    long_branch_scale = math.sqrt(
        sampling_variance_scale(80, 80) * branch_time_multiplier(2.00, 0.50)
    )

    settings = [
        ExampleSetting(
            scenario="balanced_short_branch_same_signal",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(signal / balanced_scale,),
            degrees_of_freedom=1,
            covariance_spectrum=(1.0, 0.7, 0.5),
            condition_note=(
                "Same contrast and fixed k=1; balanced size and short branch leave the largest W."
            ),
        ),
        ExampleSetting(
            scenario="imbalanced_size_same_signal",
            n_left=20,
            n_right=160,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(signal / imbalanced_scale,),
            degrees_of_freedom=1,
            covariance_spectrum=(1.0, 0.7, 0.5),
            condition_note=(
                "Same contrast but larger 1/n_A+1/n_B; W drops because the null variance is wider."
            ),
        ),
        ExampleSetting(
            scenario="long_nnls_branch_same_signal",
            n_left=80,
            n_right=80,
            branch_length_sum=2.00,
            mean_branch_length=0.50,
            projected_components=(signal / long_branch_scale,),
            degrees_of_freedom=1,
            covariance_spectrum=(1.0, 0.7, 0.5),
            condition_note=(
                "Same contrast but longer conditioned branch time; W drops because "
                "branch-time noise absorbs contrast."
            ),
        ),
        ExampleSetting(
            scenario="signal_aligned_with_selected_eigenvector",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(2.8, 0.0),
            degrees_of_freedom=1,
            covariance_spectrum=(4.0, 1.0, 0.5),
            condition_note=(
                "A spiked stable eigensystem keeps aligned contrast inside the "
                "accepted one-dimensional subspace."
            ),
        ),
        ExampleSetting(
            scenario="signal_orthogonal_to_selected_eigenvector",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(0.0, 2.8),
            degrees_of_freedom=1,
            covariance_spectrum=(4.0, 1.0, 0.5),
            condition_note=(
                "The same norm is invisible at k=1 when topology/projection select "
                "the wrong direction."
            ),
        ),
        ExampleSetting(
            scenario="multiplicity_arbitrary_first_vector",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(2.0, 0.0, 0.0),
            degrees_of_freedom=1,
            covariance_spectrum=(1.0, 1.0, 1.0, 0.2),
            condition_note=("A k=1 cut inside a tied eigenspace is not orientation-invariant."),
        ),
        ExampleSetting(
            scenario="multiplicity_rotated_first_vector",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(math.sqrt(2.0), math.sqrt(2.0), 0.0),
            degrees_of_freedom=1,
            covariance_spectrum=(1.0, 1.0, 1.0, 0.2),
            condition_note=(
                "A rotation inside the same tied eigenspace changes k=1 W without "
                "changing the invariant signal norm."
            ),
        ),
        ExampleSetting(
            scenario="multiplicity_full_projector",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(math.sqrt(2.0), math.sqrt(2.0), 0.0),
            degrees_of_freedom=3,
            covariance_spectrum=(1.0, 1.0, 1.0, 0.2),
            condition_note=(
                "Testing the whole tied projector restores orientation invariance "
                "but changes degrees of freedom."
            ),
        ),
        ExampleSetting(
            scenario="adaptive_energy_fraction_k2",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(2.0, 1.5, 0.2),
            degrees_of_freedom=2,
            covariance_spectrum=(2.0, 1.2, 0.4),
            condition_note=(
                "An energy-fraction rule that accepts k=2 changes both W and the "
                "tail df; K must be conditioned on."
            ),
        ),
        ExampleSetting(
            scenario="same_mass_fixed_k3",
            n_left=80,
            n_right=80,
            branch_length_sum=0.20,
            mean_branch_length=0.50,
            projected_components=(2.0, 1.5, 0.2),
            degrees_of_freedom=3,
            covariance_spectrum=(2.0, 1.2, 0.4),
            condition_note=(
                "The same projected mass can become less significant when the "
                "accepted open dimension includes an extra null coordinate."
            ),
        ),
    ]
    return pd.DataFrame.from_records([_example_row(setting) for setting in settings])


def build_occurrence_summary(
    case_diagnosis: pd.DataFrame,
    traversal_trace: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize observed fail-closed occurrences by required null-law bucket."""

    if case_diagnosis.empty:
        return pd.DataFrame(
            columns=[
                "schema_version",
                "loss_bucket",
                "case_count",
                "cases",
                "required_law",
                "active_sibling_p_min",
                "diagnostic_sibling_p_min",
                "active_vs_diagnostic_ratio_median",
                "branch_length_ratio_median",
                "unstable_subtree_count_median",
                "large_unstable_subtree_count_median",
                "interpretation",
            ]
        )
    required = {"case_id", "loss_bucket", "null_hypothesis_change"}
    missing = required.difference(case_diagnosis.columns)
    if missing:
        raise ValueError(f"case_diagnosis is missing required columns: {sorted(missing)!r}")

    rows: list[dict[str, object]] = []
    for loss_bucket, group in case_diagnosis.groupby("loss_bucket", dropna=False):
        active_p = _finite_series(
            group.get("full_min_active_sibling_corrected", pd.Series(dtype=float))
        )
        root_active_p = _finite_series(
            group.get("root_active_sibling_corrected_min", pd.Series(dtype=float))
        )
        diagnostic_p = _finite_series(
            group.get("minimum_diagnostic_sibling_p_value", pd.Series(dtype=float))
        )
        branch_ratio = _finite_series(group.get("root_branch_length_ratio", pd.Series(dtype=float)))
        unstable = _finite_series(group.get("unstable_subtree_count", pd.Series(dtype=float)))
        large_unstable = _finite_series(
            group.get("large_unstable_subtree_count", pd.Series(dtype=float))
        )
        ratio = _finite_series(
            group.get("active_vs_diagnostic_p_value_ratio", pd.Series(dtype=float))
        )
        required_law = _infer_required_law(str(loss_bucket))
        interpretation = _interpret_bucket(str(loss_bucket), group)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "loss_bucket": str(loss_bucket),
                "case_count": int(group["case_id"].nunique()),
                "cases": ",".join(sorted(str(value) for value in group["case_id"].unique())),
                "required_law": required_law,
                "active_sibling_p_min": float(pd.concat([active_p, root_active_p]).min(skipna=True))
                if pd.concat([active_p, root_active_p]).notna().any()
                else math.nan,
                "diagnostic_sibling_p_min": float(diagnostic_p.min(skipna=True))
                if diagnostic_p.notna().any()
                else math.nan,
                "active_vs_diagnostic_ratio_median": float(ratio.median(skipna=True))
                if ratio.notna().any()
                else math.nan,
                "branch_length_ratio_median": float(branch_ratio.median(skipna=True))
                if branch_ratio.notna().any()
                else math.nan,
                "unstable_subtree_count_median": float(unstable.median(skipna=True))
                if unstable.notna().any()
                else math.nan,
                "large_unstable_subtree_count_median": float(large_unstable.median(skipna=True))
                if large_unstable.notna().any()
                else math.nan,
                "interpretation": interpretation,
            }
        )

    occurrence = pd.DataFrame.from_records(rows).sort_values(
        ["required_law", "loss_bucket"],
        kind="mergesort",
    )
    if traversal_trace is not None and not traversal_trace.empty:
        occurrence = _attach_trace_support_counts(occurrence, traversal_trace)
    return occurrence.reset_index(drop=True)


def _infer_required_law(loss_bucket: str) -> str:
    if loss_bucket == "sibling_gate_closed_after_edge_open":
        return "selected_node_sibling_null"
    if loss_bucket == "edge_gate_closed_global":
        return "selected_pipeline_edge_null"
    return "selected_topology_support_law"


def _interpret_bucket(loss_bucket: str, group: pd.DataFrame) -> str:
    if loss_bucket == "sibling_gate_closed_after_edge_open":
        return (
            "Edge support exists, but active sibling calibration remains closed; "
            "condition on topology, NNLS branch time, local covariance, accepted "
            "dimension, and topology stability before spending alpha."
        )
    if loss_bucket == "edge_gate_closed_global":
        return (
            "Sibling law is not reached; selected-pipeline edge calibration is the "
            "first missing object."
        )
    examples = ",".join(sorted(str(value) for value in group["case_id"].head(3)))
    return f"Unsupported selected-topology bucket; inspect cases {examples}."


def _attach_trace_support_counts(
    occurrence: pd.DataFrame,
    traversal_trace: pd.DataFrame,
) -> pd.DataFrame:
    if "case_id" not in traversal_trace.columns:
        return occurrence
    trace = traversal_trace.copy()
    finite_sibling = _finite_series(trace.get("sibling_p_value_corrected", pd.Series()))
    trace["has_active_sibling_p"] = finite_sibling.notna()
    edge_open = trace.get("edge_gate_open", pd.Series(False, index=trace.index))
    trace["edge_gate_open_bool"] = edge_open.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    counts = (
        trace.groupby("case_id", dropna=False)
        .agg(
            trace_rows=("case_id", "size"),
            edge_open_trace_rows=("edge_gate_open_bool", "sum"),
            active_sibling_trace_rows=("has_active_sibling_p", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for row in occurrence.to_dict("records"):
        cases = [case for case in str(row["cases"]).split(",") if case]
        matched = counts[counts["case_id"].isin(cases)]
        row["trace_rows"] = int(matched["trace_rows"].sum()) if not matched.empty else 0
        row["edge_open_trace_rows"] = (
            int(matched["edge_open_trace_rows"].sum()) if not matched.empty else 0
        )
        row["active_sibling_trace_rows"] = (
            int(matched["active_sibling_trace_rows"].sum()) if not matched.empty else 0
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _bool_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _finite_p(value: object) -> float | None:
    result = _finite(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        return None
    return result


def _node_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return str(value)


def _parse_leaf_signature(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raw = str(value).strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (raw,)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return (str(parsed),)


def _child_nodes(row: dict[str, object]) -> tuple[str, str]:
    return _node_text(row.get("left_child")), _node_text(row.get("right_child"))


def _leaf_set(row_by_node: dict[str, dict[str, object]], node: str) -> tuple[str, ...]:
    row = row_by_node.get(node)
    if row is None:
        return (node,) if node else ()
    leaves = _parse_leaf_signature(row.get("descendant_leaf_signature"))
    return leaves if leaves else ((node,) if node else ())


def _child_fraction(row: dict[str, object], row_by_node: dict[str, dict[str, object]]) -> float:
    left_child, right_child = _child_nodes(row)
    parent_size = len(_leaf_set(row_by_node, _node_text(row.get("node_id"))))
    if parent_size <= 0 or not left_child or not right_child:
        return 0.0
    left_size = len(_leaf_set(row_by_node, left_child))
    right_size = len(_leaf_set(row_by_node, right_child))
    return float(min(left_size, right_size)) / float(parent_size)


def _min_child_edge_p(row: dict[str, object]) -> float:
    values = [
        _finite_p(row.get("left_edge_p_value_bh")),
        _finite_p(row.get("right_edge_p_value_bh")),
    ]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else math.nan


def _min_diagnostic_p(row: dict[str, object]) -> float:
    values = [
        _finite_p(row.get("sibling_sparse_p_value")),
        _finite_p(row.get("sibling_dense_p_value")),
        _finite_p(row.get("sibling_fixed_coordinate_bh_p_value")),
        _finite_p(row.get("sibling_fixed_global_p_value")),
    ]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else math.nan


def _root_node(cell_rows: list[dict[str, object]]) -> str:
    if not cell_rows:
        return ""
    ordered = sorted(
        cell_rows,
        key=lambda row: (
            _finite(row.get("depth")),
            _finite(row.get("trace_index")),
            _node_text(row.get("node_id")),
        ),
    )
    return _node_text(ordered[0].get("node_id"))


def _partition_summary(
    *,
    root: str,
    row_by_node: dict[str, dict[str, object]],
    accepted_by_node: dict[str, bool],
) -> dict[str, object]:
    if not root:
        return {"cluster_count": 0, "largest_cluster_fraction": math.nan, "cluster_sizes": ""}

    def walk(node: str) -> list[tuple[str, ...]]:
        row = row_by_node.get(node)
        if row is None:
            return [_leaf_set(row_by_node, node)]
        if accepted_by_node.get(node, False):
            children = [child for child in _child_nodes(row) if child]
            if children:
                clusters: list[tuple[str, ...]] = []
                for child in children:
                    clusters.extend(walk(child))
                return clusters
        return [_leaf_set(row_by_node, node)]

    clusters = walk(root)
    sizes = [len(cluster) for cluster in clusters if cluster]
    total = sum(sizes)
    largest = float(max(sizes)) / float(total) if total > 0 and sizes else math.nan
    return {
        "cluster_count": int(len(sizes)),
        "largest_cluster_fraction": largest,
        "cluster_sizes": ";".join(str(size) for size in sorted(sizes, reverse=True)),
    }


def _case_context(case_diagnosis: pd.DataFrame) -> dict[str, dict[str, object]]:
    if case_diagnosis.empty:
        return {}
    return {
        str(row["case_id"]): row for row in case_diagnosis.to_dict("records") if "case_id" in row
    }


def topology_stability_spend(case_row: Mapping[str, object] | None) -> float:
    """Return a diagnostic topology-stability alpha-spending factor."""

    if not case_row:
        return 0.25
    median_rf = _finite(case_row.get("median_rooted_internal_rf_relative"))
    rf_score = 1.0 if not math.isfinite(median_rf) else max(0.05, 1.0 - min(0.95, median_rf))

    branch_ratio = _finite(case_row.get("root_branch_length_ratio"))
    if not math.isfinite(branch_ratio) or branch_ratio <= 1.0:
        branch_score = 1.0
    else:
        branch_score = 1.0 / (1.0 + math.log10(branch_ratio))

    large_unstable = _finite(case_row.get("large_unstable_subtree_count"))
    if not math.isfinite(large_unstable) or large_unstable <= 0.0:
        subtree_score = 1.0
    else:
        subtree_score = 1.0 / (1.0 + math.log10(1.0 + large_unstable) / 3.0)

    return float(max(0.02, min(1.0, rf_score * branch_score * subtree_score)))


def _adaptive_evidence_multiplier(row: dict[str, object]) -> float:
    edge_p = _min_child_edge_p(row)
    diagnostic_p = _min_diagnostic_p(row)
    if math.isfinite(edge_p) and edge_p <= 1e-12:
        edge_multiplier = 4.0
    elif math.isfinite(edge_p) and edge_p <= 1e-6:
        edge_multiplier = 3.0
    elif math.isfinite(edge_p) and edge_p <= 1e-3:
        edge_multiplier = 2.0
    else:
        edge_multiplier = 1.0

    if math.isfinite(diagnostic_p) and diagnostic_p <= 1e-20:
        diagnostic_multiplier = 20.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-12:
        diagnostic_multiplier = 12.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-6:
        diagnostic_multiplier = 8.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-3:
        diagnostic_multiplier = 4.0
    elif math.isfinite(diagnostic_p) and diagnostic_p <= 1e-2:
        diagnostic_multiplier = 2.0
    else:
        diagnostic_multiplier = 1.0
    return float(edge_multiplier * diagnostic_multiplier)


def adaptive_selected_node_local_alpha(
    row: dict[str, object],
    case_row: Mapping[str, object] | None,
    policy: AdaptiveLawPolicy,
) -> tuple[float, float, float]:
    """Return local alpha, evidence multiplier, and topology spend for a replay row."""

    evidence_multiplier = _adaptive_evidence_multiplier(row)
    topology_spend = topology_stability_spend(case_row) if policy.use_topology_stability else 1.0
    local_alpha = policy.base_alpha * evidence_multiplier * topology_spend
    return float(min(policy.cap_alpha, local_alpha)), evidence_multiplier, topology_spend


def _selected_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace
    required = {"case_id", "run_id", "tree_inference", "node_id", "trace_type"}
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"traversal_trace is missing required columns: {sorted(missing)!r}")
    selected = trace[trace["trace_type"].astype(str).eq("full_edge_traversal_trace")].copy()
    if selected.empty:
        selected = trace.copy()
    selected["depth"] = _finite_series(selected.get("depth", pd.Series(dtype=float))).fillna(0.0)
    selected["trace_index"] = _finite_series(
        selected.get("trace_index", pd.Series(dtype=float))
    ).fillna(0.0)
    return selected


def _evaluate_adaptive_node(
    row: dict[str, object],
    row_by_node: dict[str, dict[str, object]],
    case_row: Mapping[str, object] | None,
    policy: AdaptiveLawPolicy,
) -> tuple[bool, float | None, float, float, float, float, str]:
    p_value = _finite_p(row.get("sibling_p_value_corrected"))
    child_fraction = _child_fraction(row, row_by_node)
    local_alpha, evidence_multiplier, topology_spend = adaptive_selected_node_local_alpha(
        row,
        case_row,
        policy,
    )
    if policy.law_status == "requires_exact_selected_node_p_value":
        return (
            False,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            "missing_exact_selected_node_conditional_p_value",
        )
    if not _bool_value(row.get("edge_gate_open")):
        return (
            False,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            "edge_gate_closed",
        )
    if p_value is None:
        return (
            False,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            "missing_active_sibling_p_value",
        )
    if child_fraction < policy.min_child_fraction:
        return (
            False,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            "child_balance_guard_failed",
        )
    if p_value > local_alpha:
        return (
            False,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            "p_value_above_adaptive_alpha",
        )
    return True, p_value, local_alpha, evidence_multiplier, topology_spend, child_fraction, ""


def replay_adaptive_law_cell(
    cell: pd.DataFrame,
    case_row: Mapping[str, object] | None,
    policy: AdaptiveLawPolicy,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Replay one adaptive selected-node policy on one topology cell."""

    cell_rows = cell.sort_values(["depth", "trace_index"]).to_dict("records")
    row_by_node = {_node_text(row.get("node_id")): row for row in cell_rows}
    root = _root_node(cell_rows)
    queue = [root] if root else []
    accepted_by_node: dict[str, bool] = {}
    decisions: list[dict[str, object]] = []

    while queue:
        node = queue.pop(0)
        row = row_by_node.get(node)
        if row is None:
            continue
        (
            rejected,
            p_value,
            local_alpha,
            evidence_multiplier,
            topology_spend,
            child_fraction,
            blocked_reason,
        ) = _evaluate_adaptive_node(row, row_by_node, case_row, policy)
        accepted_by_node[node] = bool(rejected)
        decisions.append(
            {
                "schema_version": "selected_node_adaptive_law_replay/v1",
                "policy_id": policy.policy_id,
                "display_name": policy.display_name,
                "law_status": policy.law_status,
                "case_id": row.get("case_id"),
                "test_case": row.get("test_case"),
                "tree_inference": row.get("tree_inference"),
                "run_id": row.get("run_id"),
                "node_id": row.get("node_id"),
                "depth": _finite(row.get("depth")),
                "p_value": math.nan if p_value is None else float(p_value),
                "local_alpha": float(local_alpha),
                "evidence_multiplier": float(evidence_multiplier),
                "topology_stability_spend": float(topology_spend),
                "min_child_edge_bh_p_value": _min_child_edge_p(row),
                "min_diagnostic_p_value": _min_diagnostic_p(row),
                "left_branch_length": _finite(row.get("left_branch_length")),
                "right_branch_length": _finite(row.get("right_branch_length")),
                "child_fraction": float(child_fraction),
                "edge_gate_open": _bool_value(row.get("edge_gate_open")),
                "rejected": bool(rejected),
                "blocked_reason": blocked_reason,
            }
        )
        if rejected:
            queue.extend(child for child in _child_nodes(row) if child in row_by_node)

    partition = _partition_summary(
        root=root,
        row_by_node=row_by_node,
        accepted_by_node=accepted_by_node,
    )
    rejected_p = [
        float(row["p_value"])
        for row in decisions
        if bool(row["rejected"]) and math.isfinite(float(row["p_value"]))
    ]
    first = cell_rows[0] if cell_rows else {}
    summary = {
        "schema_version": "selected_node_adaptive_law_replay/v1",
        "policy_id": policy.policy_id,
        "display_name": policy.display_name,
        "law_status": policy.law_status,
        "case_id": first.get("case_id", ""),
        "test_case": first.get("test_case", ""),
        "tree_inference": first.get("tree_inference", ""),
        "run_id": first.get("run_id", ""),
        "trace_nodes": int(len(cell_rows)),
        "tested_nodes": int(len(decisions)),
        "opened_nodes": int(sum(1 for row in decisions if bool(row["rejected"]))),
        "has_split": bool(any(bool(row["rejected"]) for row in decisions)),
        "min_rejected_p_value": min(rejected_p) if rejected_p else math.nan,
        "max_local_alpha": max(
            (float(row["local_alpha"]) for row in decisions),
            default=math.nan,
        ),
        "min_topology_stability_spend": min(
            (float(row["topology_stability_spend"]) for row in decisions),
            default=math.nan,
        ),
        **partition,
    }
    return decisions, summary


def replay_adaptive_law_benchmark(
    trace: pd.DataFrame,
    case_diagnosis: pd.DataFrame,
    *,
    policies: Sequence[AdaptiveLawPolicy] = ADAPTIVE_LAW_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay adaptive selected-node law policies on the benchmark trace."""

    selected = _selected_trace(trace)
    context = _case_context(case_diagnosis)
    decision_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    group_columns = ["case_id", "run_id", "tree_inference"]
    for policy in policies:
        for (case_id, _run_id, _tree_inference), cell in selected.groupby(
            group_columns,
            sort=False,
            dropna=False,
        ):
            decisions, summary = replay_adaptive_law_cell(
                cell,
                context.get(str(case_id)),
                policy,
            )
            decision_rows.extend(decisions)
            cell_rows.append(summary)
    decisions = pd.DataFrame.from_records(decision_rows)
    cell_summary = pd.DataFrame.from_records(cell_rows)
    case_summary = summarize_adaptive_law_cases(cell_summary, case_diagnosis, policies)
    return decisions, cell_summary, case_summary


def _empty_adaptive_case(row: Mapping[str, object], policy: AdaptiveLawPolicy) -> dict[str, object]:
    return {
        "schema_version": "selected_node_adaptive_law_replay/v1",
        "policy_id": policy.policy_id,
        "display_name": policy.display_name,
        "law_status": policy.law_status,
        "case_id": row.get("case_id", ""),
        "test_case": row.get("test_case", ""),
        "case_category": row.get("case_category", ""),
        "loss_bucket": row.get("loss_bucket", ""),
        "true_clusters": row.get("true_clusters", math.nan),
        "trace_cells": 0,
        "split_cells": 0,
        "opened_nodes_total": 0,
        "max_cluster_count": 1,
        "median_cluster_count": 1.0,
        "min_largest_cluster_fraction": 1.0,
        "max_local_alpha": math.nan,
        "min_topology_stability_spend": math.nan,
        "min_rejected_p_value": math.nan,
        "replay_status": "still_edge_blocked_or_no_replay_trace",
    }


def summarize_adaptive_law_cases(
    cell_summary: pd.DataFrame,
    case_diagnosis: pd.DataFrame,
    policies: Sequence[AdaptiveLawPolicy] = ADAPTIVE_LAW_POLICIES,
) -> pd.DataFrame:
    """Return one row per policy and fail-closed case."""

    rows: list[dict[str, object]] = []
    taxonomy = case_diagnosis.to_dict("records")
    for policy in policies:
        policy_cells = (
            cell_summary[cell_summary["policy_id"].eq(policy.policy_id)]
            if not cell_summary.empty
            else pd.DataFrame()
        )
        for case_row in taxonomy:
            case_id = str(case_row.get("case_id", ""))
            cells = (
                policy_cells[policy_cells["case_id"].astype(str).eq(case_id)]
                if not policy_cells.empty
                else pd.DataFrame()
            )
            if cells.empty:
                rows.append(_empty_adaptive_case(case_row, policy))
                continue
            split_cells = int(cells["has_split"].astype(bool).sum())
            cluster_counts = _finite_series(cells["cluster_count"])
            largest = _finite_series(cells["largest_cluster_fraction"])
            local_alpha = _finite_series(cells["max_local_alpha"])
            topology_spend = _finite_series(cells["min_topology_stability_spend"])
            rejected_p = _finite_series(cells["min_rejected_p_value"])
            if policy.law_status == "requires_exact_selected_node_p_value":
                replay_status = "unsupported_missing_exact_selected_node_p_value"
            elif split_cells > 0:
                replay_status = "candidate_split_found"
            elif str(case_row.get("loss_bucket", "")).startswith("edge_gate"):
                replay_status = "still_edge_blocked"
            else:
                replay_status = "no_split_under_policy"
            rows.append(
                {
                    "schema_version": "selected_node_adaptive_law_replay/v1",
                    "policy_id": policy.policy_id,
                    "display_name": policy.display_name,
                    "law_status": policy.law_status,
                    "case_id": case_id,
                    "test_case": case_row.get("test_case", ""),
                    "case_category": case_row.get("case_category", ""),
                    "loss_bucket": case_row.get("loss_bucket", ""),
                    "true_clusters": case_row.get("true_clusters", math.nan),
                    "trace_cells": int(cells["run_id"].nunique()),
                    "split_cells": split_cells,
                    "opened_nodes_total": int(cells["opened_nodes"].sum()),
                    "max_cluster_count": int(cluster_counts.max())
                    if cluster_counts.notna().any()
                    else 0,
                    "median_cluster_count": float(cluster_counts.median())
                    if cluster_counts.notna().any()
                    else math.nan,
                    "min_largest_cluster_fraction": float(largest.min())
                    if largest.notna().any()
                    else math.nan,
                    "max_local_alpha": float(local_alpha.max())
                    if local_alpha.notna().any()
                    else math.nan,
                    "min_topology_stability_spend": float(topology_spend.min())
                    if topology_spend.notna().any()
                    else math.nan,
                    "min_rejected_p_value": float(rejected_p.min())
                    if rejected_p.notna().any()
                    else math.nan,
                    "replay_status": replay_status,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_adaptive_law_methods(case_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one row per adaptive-law replay policy."""

    rows: list[dict[str, object]] = []
    for policy_id, group in case_summary.groupby("policy_id", sort=False):
        split_cases = group[group["split_cells"] > 0]
        sibling_cases = group[group["loss_bucket"].astype(str).str.startswith("sibling_gate")]
        edge_cases = group[group["loss_bucket"].astype(str).str.startswith("edge_gate")]
        rows.append(
            {
                "schema_version": "selected_node_adaptive_law_replay/v1",
                "policy_id": policy_id,
                "display_name": group["display_name"].iloc[0],
                "law_status": group["law_status"].iloc[0],
                "cases": int(group["case_id"].nunique()),
                "sibling_gate_cases": int(sibling_cases["case_id"].nunique()),
                "edge_gate_cases": int(edge_cases["case_id"].nunique()),
                "cases_with_candidate_split": int(split_cases["case_id"].nunique()),
                "sibling_gate_cases_with_split": int(
                    sibling_cases[sibling_cases["split_cells"] > 0]["case_id"].nunique()
                ),
                "edge_gate_cases_with_split": int(
                    edge_cases[edge_cases["split_cells"] > 0]["case_id"].nunique()
                ),
                "total_split_cells": int(group["split_cells"].sum()),
                "total_opened_nodes": int(group["opened_nodes_total"].sum()),
                "max_cluster_count": int(group["max_cluster_count"].max()),
                "median_max_cluster_count": float(group["max_cluster_count"].median()),
                "min_largest_cluster_fraction": float(
                    _finite_series(group["min_largest_cluster_fraction"]).min()
                ),
                "max_local_alpha": float(_finite_series(group["max_local_alpha"]).max()),
                "min_topology_stability_spend": float(
                    _finite_series(group["min_topology_stability_spend"]).min()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_adaptive_external_audit(
    case_summary: pd.DataFrame,
    alpha_sweep_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Join replayed candidate splits to prior rerun metrics where available."""

    rows: list[dict[str, object]] = []
    if case_summary.empty:
        return pd.DataFrame()
    for row in case_summary.to_dict("records"):
        case_id = str(row.get("case_id", ""))
        max_alpha = _finite(row.get("max_local_alpha"))
        sweeps = (
            alpha_sweep_summary[
                alpha_sweep_summary.get("case_id", pd.Series(dtype=str)).astype(str).eq(case_id)
            ].copy()
            if not alpha_sweep_summary.empty and "case_id" in alpha_sweep_summary.columns
            else pd.DataFrame()
        )
        if not bool(_finite(row.get("split_cells")) > 0):
            status = "no_candidate_split"
            chosen = None
        elif sweeps.empty:
            status = "no_alpha_sweep_rerun_available"
            chosen = None
        else:
            sweeps["sibling_alpha"] = _finite_series(sweeps["sibling_alpha"])
            eligible = sweeps[sweeps["sibling_alpha"] >= max_alpha]
            if eligible.empty:
                chosen = sweeps.sort_values("sibling_alpha").tail(1).iloc[0]
                status = "nearest_available_below_required_alpha"
            else:
                chosen = eligible.sort_values("sibling_alpha").head(1).iloc[0]
                status = "nearest_available_rerun"
        rows.append(
            {
                "schema_version": "selected_node_adaptive_law_replay/v1",
                "policy_id": row.get("policy_id", ""),
                "case_id": case_id,
                "replay_status": row.get("replay_status", ""),
                "split_cells": row.get("split_cells", 0),
                "max_local_alpha": max_alpha,
                "external_audit_status": status,
                "nearest_sibling_alpha": math.nan
                if chosen is None
                else _finite(chosen.get("sibling_alpha")),
                "best_ari": math.nan if chosen is None else _finite(chosen.get("best_ari")),
                "best_nmi": math.nan if chosen is None else _finite(chosen.get("best_nmi")),
                "best_macro_f1": math.nan
                if chosen is None
                else _finite(chosen.get("best_macro_f1")),
                "best_found_clusters": math.nan
                if chosen is None
                else _finite(chosen.get("best_found_clusters")),
                "min_largest_cluster_fraction": math.nan
                if chosen is None
                else _finite(chosen.get("min_largest_cluster_fraction")),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_report(
    *,
    conditions: pd.DataFrame,
    examples: pd.DataFrame,
    occurrences: pd.DataFrame,
    case_diagnosis_path: Path,
    traversal_trace_path: Path,
    created_utc: str,
) -> str:
    """Return a Markdown report for the selected-node sibling-null law."""

    edge_supported = occurrences[occurrences["required_law"].eq("selected_node_sibling_null")]
    if edge_supported.empty:
        edge_line = "No edge-supported sibling bucket was present in the supplied inputs."
    else:
        row = edge_supported.iloc[0]
        edge_line = (
            f"`{int(row['case_count'])}` case(s) require the selected-node sibling law; "
            f"minimum active sibling p-value `{_format_float(row['active_sibling_p_min'])}`, "
            f"minimum diagnostic sibling p-value `{_format_float(row['diagnostic_sibling_p_min'])}`, "
            f"median topology branch-ratio `{_format_float(row['branch_length_ratio_median'])}`."
        )

    components = "\n".join(
        f"- `{row.law_component}`: {row.production_implication}"
        for row in conditions.itertuples(index=False)
    )
    example_lines = "\n".join(
        "- `{scenario}`: W `{stat}`, p `{p}`, note: {note}".format(
            scenario=row.scenario,
            stat=_format_float(row.wald_statistic),
            p=_format_float(row.chi_square_tail_p_value),
            note=row.condition_note,
        )
        for row in examples.itertuples(index=False)
    )
    occurrence_lines = "\n".join(
        "- `{loss_bucket}`: `{case_count}` case(s), required law `{law}`, cases `{cases}`.".format(
            loss_bucket=row.loss_bucket,
            case_count=int(row.case_count),
            law=row.required_law,
            cases=row.cases,
        )
        for row in occurrences.itertuples(index=False)
    )
    coverage_lines = "\n".join(
        [
            "- `conditioned selected-node law`: covered by the conditional-law, "
            "fixed-subspace proposition, and selected probability mass sections.",
            "- `topology and topology stability`: covered by `E_sel`, the "
            "`topology_stability_alpha_spending` component, and occurrence split.",
            "- `NNLS branch lengths`: covered by the branch-time whitening scale "
            "and branch-length examples.",
            "- `local covariance`: covered by `Sigma_v`, `L_v`, and the "
            "local-covariance eigensystem component.",
            "- `eigenvalues/eigenvectors/multiplicity/open dimensions`: covered "
            "by the multiplicity-projector and adaptive-dimension-mixture sections.",
            "- `experiment occurrences`: covered by the occurrence table joined "
            "from the topology-difference and traversal-trace reports.",
            "- `smaller examples`: covered by the ten deterministic Wald-reaction rows.",
        ]
    )

    return f"""# Selected-Node Sibling Null Law

Generated UTC: `{created_utc}`

## Status

This report is a mathematical diagnostic and validation target. It does not
promote a production p-value. The fixed-subspace chi-square law remains valid only after conditioning on topology, NNLS branch-time scale, local covariance chart, selected node, projection subspace, and accepted dimension.

{edge_line}

## Conditional Law

Let `T` be the selected rooted topology, `v` the selected binary parent, `A_v`
and `B_v` its child leaf sets, `n_A,n_B` their effective sample sizes, and
`ell_A,ell_B` their NNLS branch lengths. Let `bar_ell` be the mean branch
length. The current whitening scale is

`a_v = (1/n_A + 1/n_B) * (1 + (ell_A + ell_B)/(2 * bar_ell))`.

For local covariance `Sigma_v`, choose `L_v L_v^T = a_v Sigma_v`. With fixed
parent projection `U_{{v,k}}`, the fixed conditional statistic is

`W_v = || U_{{v,k}} L_v^{{-1}}(hat_mu_A - hat_mu_B) ||^2`.

If all selected objects are fixed or independent of the tested contrast, then
`W_v | T,v,k,U,Sigma,ell ~ chi_square(k)`. The selected-node law needed for
edge-supported cases is instead

`p_sel(v) = P_0(W_v >= W_obs | E_sel(T,v,k,U,ell,Sigma,topology_stability), H0_sibling(v))`.

The event `E_sel` includes adaptive KNN diffusion, topology inference method,
rooting, NNLS branch fitting, edge-open path, selected node, accepted dimension,
eigenvalue multiplicity handling, and predeclared topology-stability spending.

## Fixed-Subspace Proposition

Assume `C_v = (T,v,A_v,B_v,ell_A,ell_B,bar_ell,Sigma_v,U_{{v,k}},k)` is fixed.
Under the local sibling null, assume the contrast satisfies

`hat_delta_v = hat_mu_A - hat_mu_B | C_v,H0 ~ N(0, a_v Sigma_v)`.

Let `L_v L_v^T = a_v Sigma_v` and let the rows of `U_{{v,k}}` be orthonormal in
the whitened tangent chart. Then

`Z_v = U_{{v,k}} L_v^{{-1}} hat_delta_v ~ N(0, I_k)`

because `L_v^{{-1}} hat_delta_v ~ N(0,I)` and
`U_{{v,k}} U_{{v,k}}^T = I_k`. Therefore

`W_v = Z_v^T Z_v ~ chi_square(k)`.

This proves the current fixed-subspace reference. It also proves its boundary:
if `C_v` is learned from the same data in a way that depends on
`hat_delta_v`, then the unconditional or selected conditional law is no longer
this plain chi-square law unless the selection event is independent of the
tested whitened contrast.

## Selected Probability Mass

Let `S(X)` be the full construction map from data to adaptive KNN graph,
topology method, rooted topology, NNLS branch lengths, selected node, parent
projection, accepted dimension, and topology-stability state. For the observed
state `s_obs`, define `E_sel = {{X: S(X)=s_obs}}`. The selected p-value is the
conditional mass

`p_sel(v) = integral 1{{W_v(X) >= W_v(X_obs)}} dP_0(X | X in E_sel, H0_sibling(v))`.

Equivalently, it is a conditional Monte Carlo target over null draws that
rerun the same construction and keep only draws that reproduce the selected
state, or an analytic selective-inference target if `E_sel` is written as a
tractable selected region. This is why a larger global alpha is too blunt: it
does not define the conditional mass, it only changes a threshold.

## Literature Anchor

This construction follows the selective-inference principle that the tested
null must be conditioned on the data-dependent selection event. Gao, Bien, and
Witten's [Selective Inference for Hierarchical Clustering](https://arxiv.org/abs/2012.02936)
shows that classical mean-difference tests can be badly anti-conservative when
clusters are selected by hierarchical clustering and replaces them with
cluster-selection-conditioned p-values. Lee, Sun, Sun, and Taylor's
[Exact post-selection inference](https://arxiv.org/abs/1311.6238) gives the
general post-selection template: characterize the law of the tested estimator
conditional on the selection event. Our topology-conditioned sibling law is the
Tree-Break analogue of that template, with `E_sel` containing adaptive KNN,
topology inference, rooting, NNLS branch lengths, selected node, projection,
dimension, and topology-stability state.

## Eigenvalue Multiplicity And Open Dimensions

If `Sigma_v` has an eigenvalue block with multiplicity `m`, individual
eigenvectors inside that block are identifiable only up to an orthogonal
rotation. A statistic that accepts only part of the block can change when the
basis is rotated while the invariant covariance and signal norm are unchanged.
The invariant object is the whole projector `P_G` onto the tied eigenspace, with
`||P_G Z||^2 ~ chi_square(rank(P_G))` under fixed conditioning.

If the accepted dimension `K` is data-selected, the selected law is

`P_0(W_K >= w | E_sel) = sum_k P_0(W_k >= w, K=k | E_sel)`.

A chi-square tail with `df=k` is admissible only after conditioning on the
accepted value `K=k` and on the rule that made `K` observable. This is the
mathematical place where the number of eigenvalues, eigenvector stability,
multiplicity, and open dimensions enter the sibling law.

## Law Components

{components}

## Smaller Examples

{example_lines}

## Observed Occurrences

{occurrence_lines}

## Requirement Coverage

{coverage_lines}

## Interpretation

The overlap/SBM edge-supported failures are not solved by a single larger
global alpha. The missing object is a conditional tail probability for the
selected node. Branch lengths change the variance scale, covariance controls
the mass in each feature direction, eigenvalue multiplicity determines which
subspace is identifiable, accepted dimension changes the degrees of freedom,
and topology stability determines whether a selected split is eligible for
alpha at all.

## Sources

- `{case_diagnosis_path}`
- `{traversal_trace_path}`
- `tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py`
- `tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py`
- `tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/wald_statistic/sibling_divergence_test.py`
- `https://arxiv.org/abs/2012.02936`
- `https://arxiv.org/abs/1311.6238`
"""


def write_selected_node_sibling_null_law_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    case_diagnosis_path: Path = DEFAULT_CASE_DIAGNOSIS_PATH,
    traversal_trace_path: Path = DEFAULT_TRACE_PATH,
    created_utc: str | None = None,
) -> dict[str, Path]:
    """Write the selected-node sibling-null law artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    created = created_utc or _timestamp()

    case_diagnosis = _read_csv(case_diagnosis_path)
    traversal_trace = _read_csv(traversal_trace_path, required=False)

    conditions = build_selected_node_null_conditions()
    examples = build_wald_reaction_examples()
    occurrences = build_occurrence_summary(case_diagnosis, traversal_trace)

    paths = {
        "conditions": output_dir / CONDITIONS_NAME,
        "examples": output_dir / EXAMPLES_NAME,
        "occurrences": output_dir / OCCURRENCES_NAME,
        "report": output_dir / REPORT_NAME,
        "manifest": output_dir / MANIFEST_NAME,
    }
    conditions.to_csv(paths["conditions"], index=False)
    examples.to_csv(paths["examples"], index=False)
    occurrences.to_csv(paths["occurrences"], index=False)
    paths["report"].write_text(
        build_report(
            conditions=conditions,
            examples=examples,
            occurrences=occurrences,
            case_diagnosis_path=case_diagnosis_path,
            traversal_trace_path=traversal_trace_path,
            created_utc=created,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "generated_by": GENERATED_BY,
        "inputs": {
            "case_diagnosis_path": str(case_diagnosis_path),
            "traversal_trace_path": str(traversal_trace_path),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "row_counts": {
            "conditions": int(len(conditions)),
            "examples": int(len(examples)),
            "occurrences": int(len(occurrences)),
        },
        "git": _git_context(),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def build_adaptive_law_replay_report(
    *,
    method_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    external_audit: pd.DataFrame,
    trace_path: Path,
    case_diagnosis_path: Path,
    alpha_sweep_summary_path: Path,
    created_utc: str,
) -> str:
    """Return a Markdown benchmark replay report for adaptive selected-node laws."""

    method_lines = "\n".join(
        "- `{policy}` ({name}): `{status}`, split cases `{split_cases}/{cases}`, "
        "sibling split cases `{sibling_splits}/{sibling_cases}`, total split cells "
        "`{split_cells}`, max local alpha `{alpha}`, min topology spend `{spend}`.".format(
            policy=row.policy_id,
            name=row.display_name,
            status=row.law_status,
            split_cases=int(row.cases_with_candidate_split),
            cases=int(row.cases),
            sibling_splits=int(row.sibling_gate_cases_with_split),
            sibling_cases=int(row.sibling_gate_cases),
            split_cells=int(row.total_split_cells),
            alpha=_format_float(row.max_local_alpha),
            spend=_format_float(row.min_topology_stability_spend),
        )
        for row in method_summary.itertuples(index=False)
    )
    case_lines = "\n".join(
        "- `{policy}` / `{case}`: status `{status}`, split cells `{split_cells}`, "
        "max clusters `{clusters}`, max alpha `{alpha}`.".format(
            policy=row.policy_id,
            case=row.case_id,
            status=row.replay_status,
            split_cells=int(row.split_cells),
            clusters=int(row.max_cluster_count),
            alpha=_format_float(row.max_local_alpha),
        )
        for row in case_summary[case_summary["split_cells"] > 0].itertuples(index=False)
    )
    if not case_lines:
        case_lines = "- No replay policy produced candidate split cells."

    evaluated = external_audit[
        external_audit["external_audit_status"]
        .astype(str)
        .str.contains(
            "rerun|available",
            regex=True,
        )
    ]
    external_lines = "\n".join(
        "- `{policy}` / `{case}` at alpha `{alpha}`: ARI `{ari}`, NMI `{nmi}`, "
        "macro F1 `{f1}`, clusters `{clusters}`, audit `{status}`.".format(
            policy=row.policy_id,
            case=row.case_id,
            alpha=_format_float(row.nearest_sibling_alpha),
            ari=_format_float(row.best_ari),
            nmi=_format_float(row.best_nmi),
            f1=_format_float(row.best_macro_f1),
            clusters=_format_float(row.best_found_clusters),
            status=row.external_audit_status,
        )
        for row in evaluated.itertuples(index=False)
        if int(row.split_cells) > 0
    )
    if not external_lines:
        external_lines = "- No candidate split had matching alpha-sweep external metrics."

    return f"""# Selected-Node Adaptive Law Benchmark Replay

Generated UTC: `{created_utc}`

## Status

This is a benchmark replay, not a production method change. The strict
selected-node law still fails closed because exact selected conditional
p-values or conditional Monte Carlo samples are absent. The adaptive policies
below are diagnostic proxies that test whether the available benchmark traces
support spending more sibling alpha after conditioning on edge evidence,
diagnostic covariance evidence, topology stability, NNLS branch-length
stability, and child-size balance.

## Policy Summary

{method_lines}

## Candidate Split Cases

{case_lines}

## External Metric Audit

{external_lines}

## Interpretation

The strict selected law is scientifically honest but does not rescue benchmark
cases without a selected conditional p-value object. The topology/branch
spending proxy shows what the formal law would need to adjudicate: if topology
stability and branch-length stability suppress local alpha, the method remains
closed; if those penalties are ablated, any newly opened splits are diagnostic
only and must be checked against actual reruns before promotion.

## Sources

- `{trace_path}`
- `{case_diagnosis_path}`
- `{alpha_sweep_summary_path}`
- `benchmarks/validation/statistics/selected_node_sibling_null_law.py`
"""


def write_selected_node_adaptive_law_replay_artifacts(
    *,
    output_dir: Path = DEFAULT_ADAPTIVE_REPLAY_OUTPUT_DIR,
    traversal_trace_path: Path = DEFAULT_TRACE_PATH,
    case_diagnosis_path: Path = DEFAULT_CASE_DIAGNOSIS_PATH,
    alpha_sweep_summary_path: Path = DEFAULT_ALPHA_SWEEP_SUMMARY_PATH,
    created_utc: str | None = None,
) -> dict[str, Path]:
    """Replay adaptive selected-node law policies on benchmark artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    created = created_utc or _timestamp()
    trace = _read_csv(traversal_trace_path)
    case_diagnosis = _read_csv(case_diagnosis_path)
    alpha_sweep = _read_csv(alpha_sweep_summary_path, required=False)

    decisions, cell_summary, case_summary = replay_adaptive_law_benchmark(
        trace,
        case_diagnosis,
    )
    method_summary = summarize_adaptive_law_methods(case_summary)
    external_audit = build_adaptive_external_audit(case_summary, alpha_sweep)

    paths = {
        "decisions": output_dir / ADAPTIVE_DECISIONS_NAME,
        "cell_summary": output_dir / ADAPTIVE_CELL_SUMMARY_NAME,
        "case_summary": output_dir / ADAPTIVE_CASE_SUMMARY_NAME,
        "method_summary": output_dir / ADAPTIVE_METHOD_SUMMARY_NAME,
        "external_audit": output_dir / ADAPTIVE_EXTERNAL_AUDIT_NAME,
        "report": output_dir / ADAPTIVE_REPORT_NAME,
        "manifest": output_dir / ADAPTIVE_MANIFEST_NAME,
    }
    decisions.to_csv(paths["decisions"], index=False)
    cell_summary.to_csv(paths["cell_summary"], index=False)
    case_summary.to_csv(paths["case_summary"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    external_audit.to_csv(paths["external_audit"], index=False)
    paths["report"].write_text(
        build_adaptive_law_replay_report(
            method_summary=method_summary,
            case_summary=case_summary,
            external_audit=external_audit,
            trace_path=traversal_trace_path,
            case_diagnosis_path=case_diagnosis_path,
            alpha_sweep_summary_path=alpha_sweep_summary_path,
            created_utc=created,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "selected_node_adaptive_law_replay/v1",
        "created_utc": created,
        "generated_by": GENERATED_BY,
        "inputs": {
            "traversal_trace_path": str(traversal_trace_path),
            "case_diagnosis_path": str(case_diagnosis_path),
            "alpha_sweep_summary_path": str(alpha_sweep_summary_path),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "row_counts": {
            "decisions": int(len(decisions)),
            "cell_summary": int(len(cell_summary)),
            "case_summary": int(len(case_summary)),
            "method_summary": int(len(method_summary)),
            "external_audit": int(len(external_audit)),
        },
        "policies": [policy.policy_id for policy in ADAPTIVE_LAW_POLICIES],
        "git": _git_context(),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _git_context() -> dict[str, object]:
    def run(args: Sequence[str]) -> str:
        return subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]).splitlines(),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-diagnosis", type=Path, default=DEFAULT_CASE_DIAGNOSIS_PATH)
    parser.add_argument("--traversal-trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alpha-sweep-summary",
        type=Path,
        default=DEFAULT_ALPHA_SWEEP_SUMMARY_PATH,
    )
    parser.add_argument(
        "--adaptive-replay-output-dir",
        type=Path,
        default=DEFAULT_ADAPTIVE_REPLAY_OUTPUT_DIR,
    )
    parser.add_argument(
        "--skip-adaptive-replay",
        action="store_true",
        help="Only write the formal law artifacts, not the benchmark replay.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = write_selected_node_sibling_null_law_artifacts(
        output_dir=args.output_dir,
        case_diagnosis_path=args.case_diagnosis,
        traversal_trace_path=args.traversal_trace,
    )
    print(f"Wrote selected-node sibling-null law artifacts to {args.output_dir}")
    for key, path in sorted(paths.items()):
        print(f"{key}: {path}")
    if not args.skip_adaptive_replay:
        replay_paths = write_selected_node_adaptive_law_replay_artifacts(
            output_dir=args.adaptive_replay_output_dir,
            traversal_trace_path=args.traversal_trace,
            case_diagnosis_path=args.case_diagnosis,
            alpha_sweep_summary_path=args.alpha_sweep_summary,
        )
        print(
            "Wrote selected-node adaptive-law benchmark replay artifacts to "
            f"{args.adaptive_replay_output_dir}"
        )
        for key, path in sorted(replay_paths.items()):
            print(f"adaptive_{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS_NAME",
    "ADAPTIVE_CASE_SUMMARY_NAME",
    "ADAPTIVE_CELL_SUMMARY_NAME",
    "ADAPTIVE_DECISIONS_NAME",
    "ADAPTIVE_EXTERNAL_AUDIT_NAME",
    "ADAPTIVE_LAW_POLICIES",
    "ADAPTIVE_MANIFEST_NAME",
    "ADAPTIVE_METHOD_SUMMARY_NAME",
    "ADAPTIVE_REPORT_NAME",
    "EXAMPLES_NAME",
    "MANIFEST_NAME",
    "OCCURRENCES_NAME",
    "REPORT_NAME",
    "AdaptiveLawPolicy",
    "adaptive_selected_node_local_alpha",
    "build_adaptive_external_audit",
    "build_adaptive_law_replay_report",
    "branch_time_multiplier",
    "build_occurrence_summary",
    "build_report",
    "build_selected_node_null_conditions",
    "build_wald_reaction_examples",
    "projected_wald_tail",
    "replay_adaptive_law_benchmark",
    "replay_adaptive_law_cell",
    "sampling_variance_scale",
    "summarize_adaptive_law_cases",
    "summarize_adaptive_law_methods",
    "topology_stability_spend",
    "write_selected_node_adaptive_law_replay_artifacts",
    "write_selected_node_sibling_null_law_artifacts",
]
