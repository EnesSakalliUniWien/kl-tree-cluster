from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.failure.debug_trace import analyze_single_case


def _write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_analyze_single_case_rejects_missing_sibling_decision_column(tmp_path: Path) -> None:
    audit_path = tmp_path / "case_1_tbs_stats.csv"
    _write_audit(
        audit_path,
        [
            {
                "node_id": "root",
                "leaf_count": 4,
                "parent_node": None,
                "Sibling_Divergence_P_Value": None,
            },
            {
                "node_id": "left",
                "leaf_count": 2,
                "parent_node": "root",
                "Sibling_Divergence_P_Value": 0.001,
            },
            {
                "node_id": "right",
                "leaf_count": 2,
                "parent_node": "root",
                "Sibling_Divergence_P_Value": 0.001,
            },
        ],
    )

    diagnosis = analyze_single_case(audit_path)

    assert diagnosis == {
        "mode": "ERROR",
        "reason": "Invalid audit contract; missing columns: Sibling_BH_Different",
    }


def test_analyze_single_case_uses_sibling_decision_column(tmp_path: Path) -> None:
    audit_path = tmp_path / "case_1_tbs_stats.csv"
    _write_audit(
        audit_path,
        [
            {
                "node_id": "root",
                "leaf_count": 4,
                "parent_node": None,
                "Sibling_Divergence_P_Value": 0.35,
                "Sibling_BH_Different": False,
            },
            {
                "node_id": "left",
                "leaf_count": 2,
                "parent_node": "root",
                "Sibling_Divergence_P_Value": 0.001,
                "Sibling_BH_Different": True,
            },
            {
                "node_id": "right",
                "leaf_count": 2,
                "parent_node": "root",
                "Sibling_Divergence_P_Value": 0.001,
                "Sibling_BH_Different": True,
            },
        ],
    )

    diagnosis = analyze_single_case(audit_path)

    assert diagnosis == {
        "mode": "**UNDER-SPLIT**",
        "reason": "Root split rejected (P=3.50e-01)",
    }
