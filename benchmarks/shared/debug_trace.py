import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

AUDIT_CONTRACT_COLUMNS = frozenset(
    {
        "node_id",
        "leaf_count",
        "parent_node",
        "Sibling_Divergence_P_Value",
        "Sibling_BH_Different",
    }
)


def diagnose_benchmark_failures(
    results_csv_path: str, audit_dir: str, output_path: str = "failure_report.md"
):
    """
    Analyzes benchmark results to diagnose underperforming cases (ARI < 0.2).
    Generates a markdown report classifying failures as UNDER-SPLIT or OVER-SPLIT.
    """
    results_path = Path(results_csv_path)
    audit_root = Path(audit_dir)

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        logger.error(f"Failed to read results CSV: {e}")
        return

    if "ari" not in df.columns:
        logger.warning(f"ARI column not found in {df.columns}")
        return

    if "method" in df.columns:
        bad_cases = df[(df["method"] == "kl") & (df["ari"] < 0.2)]
    else:
        bad_cases = df[df["ari"] < 0.2]

    if bad_cases.empty:
        logger.info("No failure cases found (ARI < 0.2).")
        return

    report_lines = [
        "# Benchmark Failure Diagnosis",
        "",
        f"**Source**: `{results_csv_path}`",
        f"**Audit Dir**: `{audit_dir}`",
        "",
        "| Case ID | ARI | Found / True | Mode | Diagnosis |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for _, row in bad_cases.iterrows():
        case_num = row.get("test_case", "N/A")
        case_id = row.get("case_id", "Unknown")
        ari_val = row["ari"]
        true_k = row.get("true_clusters", "?")
        found_k = row.get("found_clusters", "?")

        audit_file = audit_root / f"case_{case_num}_kl_stats.csv"

        diagnosis = analyze_single_case(audit_file)

        report_lines.append(
            f"| {case_id} | {ari_val:.3f} | {found_k} / {true_k} | {diagnosis['mode']} | {diagnosis['reason']} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Failure diagnosis report written to {output_path}")


def analyze_single_case(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"mode": "MISSING", "reason": "Audit log not found"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"mode": "ERROR", "reason": f"Unreadable audit file: {exc}"}

    missing_columns = sorted(AUDIT_CONTRACT_COLUMNS.difference(df.columns))
    if missing_columns:
        return {
            "mode": "ERROR",
            "reason": f"Invalid audit contract; missing columns: {', '.join(missing_columns)}",
        }

    if df.empty:
        return {"mode": "ERROR", "reason": "Invalid audit contract; audit file is empty"}

    leaf_counts = pd.to_numeric(df["leaf_count"], errors="coerce")
    if leaf_counts.notna().sum() == 0:
        return {"mode": "ERROR", "reason": "Invalid audit contract; leaf_count has no numeric values"}

    root_idx = leaf_counts.idxmax()
    root_id = df.loc[root_idx, "node_id"]
    if pd.isna(root_id):
        return {"mode": "ERROR", "reason": "Invalid audit contract; root node_id is missing"}

    # Children
    children = df[df["parent_node"] == root_id]

    root_split_rejected = True
    root_p = float("nan")

    if len(children) > 0:
        sibling_p_values = pd.to_numeric(
            children["Sibling_Divergence_P_Value"], errors="coerce"
        ).dropna()
        if not sibling_p_values.empty:
            root_p = float(sibling_p_values.min())
        if _coerce_bool_series(children["Sibling_BH_Different"]).any():
            root_split_rejected = False

    # Significant splits analysis
    sibling_different = _coerce_bool_series(df["Sibling_BH_Different"])
    sig_splits = df[sibling_different & (df["parent_node"] != root_id)]

    # Classification
    if root_split_rejected:
        return {
            "mode": "**UNDER-SPLIT**",
            "reason": f"Root split rejected (P={root_p:.2e})",
        }
    elif len(sig_splits) > 30:  # Heuristic
        min_p = (
            sig_splits["Sibling_Divergence_P_Value"].min()
            if not sig_splits.empty
            else 0.0
        )
        return {
            "mode": "**OVER-SPLIT**",
            "reason": f"Runaway splitting ({len(sig_splits)} nodes), Min P={min_p:.1e}",
        }
    else:
        return {"mode": "MIXED", "reason": "Root split OK, moderate complexity."}


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})
