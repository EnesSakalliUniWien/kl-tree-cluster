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

    root_row = df.loc[root_idx]
    root_p = _root_sibling_pvalue(root_row)
    root_split_rejected = _root_split_rejected(root_row)

    # Significant splits analysis
    sibling_different = _coerce_bool_series(df["Sibling_BH_Different"])
    sig_splits = df[sibling_different & (df.index != root_idx)]

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


def _root_sibling_pvalue(root_row: pd.Series) -> float:
    for column in (
        "Sibling_Divergence_P_Value_Corrected",
        "Sibling_Divergence_P_Value",
    ):
        if column not in root_row.index:
            continue
        value = pd.to_numeric(pd.Series([root_row[column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return float("nan")


def _root_split_rejected(root_row: pd.Series) -> bool:
    if "Sibling_BH_Different" not in root_row.index or pd.isna(root_row["Sibling_BH_Different"]):
        return True
    return not bool(_coerce_bool_series(pd.Series([root_row["Sibling_BH_Different"]])).iloc[0])
