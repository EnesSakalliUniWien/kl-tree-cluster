from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = PROJECT_ROOT / "applications/scrna/plots/tree_plot_helpers.R"


def test_tree_plot_helpers_preserve_tree_and_branch_length_contract() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed.")

    program = f"""
source({json.dumps(str(HELPER_PATH))})
stopifnot(identical(readable_branch_lengths(c(0, NA, -1)), rep(1, 3)))
edges <- data.frame(
  parent = c("N0", "N0", "N1", "N1"),
  child = c("L0", "N1", "L1", "L2"),
  branch_length = c(1, 4, 9, 16)
)
phy <- edge_table_to_phylo(edges)
stopifnot(identical(phy$tip.label, c("L0", "L1", "L2")))
stopifnot(all(is.finite(phy$edge.length)))
descendants <- descendant_tip_indices(phy)
root_id <- length(phy$tip.label) + match("N0", phy$node.label)
stopifnot(setequal(descendants[[as.character(root_id)]], seq_along(phy$tip.label)))
tip_data <- data.frame(
  label = phy$tip.label,
  cluster_id = c("C1", "C2", "C2")
)
metadata <- build_cluster_tree_metadata(phy, tip_data, descendants)
stopifnot(as.character(metadata$branch_cluster[[root_id]]) == "shared ancestors")
palette <- cluster_palette(c("C2", "C1"))
stopifnot(identical(names(palette), c("C1", "C2")))
tree_plot <- list(
  data = data.frame(
    node = seq_len(length(phy$tip.label) + phy$Nnode),
    x = seq_len(length(phy$tip.label) + phy$Nnode),
    y = 0
  )
)
root_rows <- cluster_root_rows(phy, tip_data, tree_plot)
stopifnot(setequal(root_rows$cluster_id, c("C1", "C2")))
audit <- audit_cluster_rows("method", "edges.csv", phy, tip_data, descendants)
stopifnot(all(audit$exact_clade))
"""

    subprocess.run(
        [rscript, "--vanilla", "-e", program],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_all_scrna_tree_renderers_use_the_shared_tree_module() -> None:
    for name in (
        "goncalves_progenitor_trees_ggtree.R",
        "pancreas_cluster_radial_trees_ggtree.R",
        "pancreas_radial_trees_ggtree.R",
        "pancreas_umap_tree_combo_ggtree.R",
    ):
        source = (HELPER_PATH.parent / name).read_text(encoding="utf-8")
        assert 'source(file.path(dirname(script_path), "tree_plot_helpers.R")' in source
        assert "edge_table_to_phylo <- function" not in source
