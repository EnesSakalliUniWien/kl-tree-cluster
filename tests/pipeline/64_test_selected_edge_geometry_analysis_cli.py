from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.analysis.selected_edge_geometry_analysis import (
    analyze_selected_edge_geometry,
)


def test_analyze_selected_edge_geometry_writes_ranked_models(tmp_path: Path) -> None:
    edge_rows = pd.DataFrame(
        {
            "mode": ["selected_tree", "selected_tree", "fixed_tree", "fixed_tree"],
            "edge_rejected": [1, 0, 0, 0],
            "edge_raw_p": [0.0001, 0.2, 0.4, 0.8],
            "edge_bh_action": [4.0, 0.7, 0.4, 0.1],
            "edge_statistic_margin": [3.0, -0.5, -1.0, -2.0],
            "selected_eigenvalue_over_mp": [2.0, 1.1, 0.9, 0.8],
            "tree_balance": [0.5, 0.6, 0.5, 0.7],
            "sample_ratio": [0.5, 0.5, 0.5, 0.5],
            "path_length_from_root": [1, 1, 1, 1],
        }
    )
    edge_path = tmp_path / "edges.csv"
    edge_rows.to_csv(edge_path, index=False)
    output_path = tmp_path / "analysis.csv"

    result = analyze_selected_edge_geometry(edge_path=edge_path, output_path=output_path)

    assert output_path.exists()
    assert "model_name" in result.columns
    assert "score" in result.columns
    assert set(result["analysis_role"]) == {"diagnostic_candidate_law_search"}
