"""Infer trace-level mathematical failure attribution for benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.diagnostics.math_trace.failure_classifier import classify_table
from benchmarks.diagnostics.math_trace.projection_law import audit_projection_law
from benchmarks.diagnostics.math_trace.selected_tail_law import summarize_tail_law
from benchmarks.diagnostics.math_trace.support_thresholds import audit_support_table
from benchmarks.diagnostics.math_trace.trace_schema import validate_node_decision_trace


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    failure_attribution: pd.DataFrame,
    support_audit: pd.DataFrame,
) -> None:
    counts = failure_attribution["primary_failure_label"].value_counts().sort_index()
    case_id = manifest.get("case_id", "unknown")
    lines = [
        "# Math Inference Report",
        "",
        f"- case_id: `{case_id}`",
        f"- n_trace_rows: `{len(failure_attribution)}`",
        "",
        "## Failure Labels",
        "",
    ]
    lines.extend(f"- `{label}`: {int(count)}" for label, count in counts.items())
    lines.extend(
        [
            "",
            "## Support",
            "",
            f"- supported rows: `{int((support_audit['support_status'] == 'supported').sum())}`",
            f"- unsupported rows: `{int((support_audit['support_status'] == 'unsupported').sum())}`",
            "",
            "## Equations",
            "",
            "- `R_u = T_u / (a_u * nu_u)`",
            "- `A_edge = -log(p_min_edge)`",
            "- `rho_u = ||P_k z_u||^2 / ||z_u||^2`",
            "- `tan2(theta_u) = (1 - rho_u) / rho_u`",
            "- `ell_bar = log(max(n_L, n_R) / min(n_L, n_R))`",
            "- `lambda_k_over_mp = lambda_k,u / lambda_+,u`",
        ]
    )
    (output_dir / "math_inference_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def infer_math(
    *,
    output_dir: Path,
    node_decision_trace_csv: Path,
    manifest_json: Path | None = None,
) -> dict[str, Any]:
    """Read trace CSVs, write math-inference artifacts, and return a summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_json)
    trace = pd.read_csv(node_decision_trace_csv)
    validate_node_decision_trace(trace)

    failure_attribution = classify_table(trace)
    support_audit = audit_support_table(trace)
    tail_law = summarize_tail_law(trace)
    projection_audit = audit_projection_law(trace)

    failure_attribution.to_csv(output_dir / "failure_attribution.csv", index=False)
    support_audit.to_csv(output_dir / "support_threshold_audit.csv", index=False)
    tail_law.to_csv(output_dir / "law_fit_selected_tail.csv", index=False)
    projection_audit.to_csv(output_dir / "projection_law_audit.csv", index=False)

    label_counts = {
        str(label): int(count)
        for label, count in failure_attribution["primary_failure_label"]
        .value_counts()
        .sort_index()
        .items()
    }
    summary = {
        "schema_version": "kl_te_math_inference/v1",
        "case_id": manifest.get("case_id"),
        "n_trace_rows": int(len(trace)),
        "failure_label_counts": label_counts,
        "n_supported_rows": int((support_audit["support_status"] == "supported").sum()),
        "n_unsupported_rows": int((support_audit["support_status"] == "unsupported").sum()),
        "outputs": {
            "math_inference_report": "math_inference_report.md",
            "failure_attribution": "failure_attribution.csv",
            "support_threshold_audit": "support_threshold_audit.csv",
            "law_fit_selected_tail": "law_fit_selected_tail.csv",
            "projection_law_audit": "projection_law_audit.csv",
        },
    }
    (output_dir / "math_inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        manifest=manifest,
        failure_attribution=failure_attribution,
        support_audit=support_audit,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--node-decision-trace-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = infer_math(
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        node_decision_trace_csv=args.node_decision_trace_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
