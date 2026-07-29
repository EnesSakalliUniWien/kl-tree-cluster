#!/usr/bin/env Rscript
# Dataset-specific pancreas radial-tree renderer.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) {
  stop("Unable to resolve script path from Rscript command arguments.")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
source(file.path(dirname(script_path), "tree_plot_helpers.R"), local = TRUE)
context <- scrna_tree_plot_context(
  script_path,
  "pancreas_scrna_cluster_benchmark_20260623",
  c("ape", "ggplot2", "ggtree")
)
output_dir <- context$output_dir

method_map <- pancreas_tree_method_map("edge_gate")

missing_edge_csvs <- method_map$edge_csv[!file.exists(file.path(output_dir, method_map$edge_csv))]
if (length(missing_edge_csvs)) {
  stop("Missing current TBS tree edge CSV file(s): ", paste(missing_edge_csvs, collapse = ", "))
}

edge_state <- function(edges) {
  significant <- as.logical(edges$Child_Parent_Divergence_Significant)
  tested <- as.logical(edges$Child_Parent_Divergence_Tested)
  blocked <- as.logical(edges$Child_Parent_Divergence_Ancestor_Blocked)
  ifelse(
    significant,
    "edge significant",
    ifelse(blocked, "ancestor blocked", ifelse(tested, "tested closed", "not tested"))
  )
}

node_metadata <- function(phy, edges) {
  node_lookup <- c(
    stats::setNames(seq_along(phy$tip.label), phy$tip.label),
    stats::setNames(length(phy$tip.label) + seq_along(phy$node.label), phy$node.label)
  )
  data.frame(
    node = unname(node_lookup[as.character(edges$child)]),
    edge_state = factor(
      edge_state(edges),
      levels = c("edge significant", "tested closed", "ancestor blocked", "not tested")
    )
  )
}

plot_one_tree <- function(edge_csv, title) {
  edge_path <- file.path(output_dir, edge_csv)
  edges <- read.csv(edge_path, stringsAsFactors = FALSE, check.names = FALSE)
  phy <- edge_table_to_phylo(edges)
  meta <- node_metadata(phy, edges)

  plot_base <- tools::file_path_sans_ext(basename(edge_csv))
  plot_base <- sub("_tree_edges$", "_radial_tree_ggtree", plot_base)
  png_path <- file.path(output_dir, paste0(plot_base, ".png"))
  pdf_path <- file.path(output_dir, paste0(plot_base, ".pdf"))

  p <- ggtree(phy, layout = "circular") %<+% meta +
    geom_tree(aes(color = edge_state), size = 0.08) +
    geom_tippoint(size = 0.10, alpha = 0.20, color = "#111827") +
    scale_color_manual(
      values = c(
        "edge significant" = "#2563eb",
        "tested closed" = "#f97316",
        "ancestor blocked" = "#9ca3af",
        "not tested" = "#d1d5db"
      ),
      drop = FALSE,
      name = "Edge gate"
    ) +
    ggtitle(paste0(title, "\nwhole radial tree, ", length(phy$tip.label), " leaves; display branch lengths sqrt-capped")) +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      plot.margin = margin(6, 6, 6, 6)
    )

  ggsave(png_path, p, width = 15, height = 15, dpi = 320, bg = "white", limitsize = FALSE)
  ggsave(pdf_path, p, width = 15, height = 15, bg = "white", limitsize = FALSE)

  data.frame(
    edge_csv = edge_path,
    png = png_path,
    pdf = pdf_path,
    tips = length(phy$tip.label),
    internal_nodes = phy$Nnode,
    edges = nrow(edges),
    stringsAsFactors = FALSE
  )
}

rows <- do.call(rbind, Map(plot_one_tree, method_map$edge_csv, method_map$title))
write.csv(
  rows,
  file.path(output_dir, "tbs_radial_tree_ggtree_outputs.csv"),
  row.names = FALSE
)
print(rows)
