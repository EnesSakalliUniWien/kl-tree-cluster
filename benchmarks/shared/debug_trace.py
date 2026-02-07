import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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

    # Filter for poor performers
    # Check for 'ari' vs 'ARI' column
    ari_col = "ari" if "ari" in df.columns else "ARI"

    if ari_col not in df.columns:
        logger.warning(f"ARI column not found in {df.columns}")
        return

    keys = df.columns
    method_col = "Method" if "Method" in keys else "method"
    if method_col in keys:
        bad_cases = df[(df[method_col] == "kl") & (df[ari_col] < 0.2)]
    else:
        bad_cases = df[df[ari_col] < 0.2]

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
        case_num = row.get("test_case", row.get("Test", "N/A"))
        case_id = row.get("case_id", row.get("Case_Name", "Unknown"))
        ari_val = row[ari_col]
        true_k = row.get("true_clusters", row.get("True", "?"))
        found_k = row.get("found_clusters", row.get("Found", "?"))

        # Locate audit file
        # Pattern: case_{num}_kl_divergence_stats.csv
        # Note: case_num might be integer
        audit_file = audit_root / f"case_{case_num}_kl_divergence_stats.csv"

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
    except Exception:
        return {"mode": "ERROR", "reason": "Corrupt audit file"}

    if "leaf_count" not in df.columns:
        return {"mode": "ERROR", "reason": "Invalid columns"}

    # Find root
    try:
        root_idx = df["leaf_count"].idxmax()
        root_node = df.loc[root_idx]
        root_id = root_node["node_id"]
    except Exception:
        return {"mode": "ERROR", "reason": "Root finding failed"}

    # Children
    children = df[df["parent_node"] == root_id]

    root_split_rejected = True
    root_p = float("nan")

    if len(children) > 0:
        root_p = children["Sibling_Divergence_P_Value"].min()
        # Determine if split was accepted.
        # Ideally check 'Sibling_BH_Different'.
        # If missing, fallback to raw p-value check (imperfect but useful)
        if "Sibling_BH_Different" in children.columns:
            if children["Sibling_BH_Different"].any():
                root_split_rejected = False
        elif root_p < 0.05:  # Fallback heuristic
            root_split_rejected = False

    # Significant splits analysis
    sig_splits = pd.DataFrame()
    if "Sibling_BH_Different" in df.columns:
        sig_splits = df[
            (df["Sibling_BH_Different"] == True) & (df["parent_node"] != root_id)
        ]

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
