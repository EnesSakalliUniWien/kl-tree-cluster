#!/usr/bin/env Rscript
# Dataset-specific pancreas cluster-tree renderer.

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

assignments <- read.csv(
  file.path(output_dir, "method_assignments.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

method_map <- pancreas_tree_method_map("final_clusters")

plot_one_method <- function(method_key, edge_csv, title) {
  edges <- read.csv(file.path(output_dir, edge_csv), stringsAsFactors = FALSE, check.names = FALSE)
  phy <- edge_table_to_phylo(edges)

  tip_data <- pancreas_tip_data(phy, assignments, method_key)
  cluster_sizes <- sort(table(tip_data$cluster_id), decreasing = TRUE)
  full_palette <- cluster_palette(names(cluster_sizes))
  tree_palette <- c(full_palette, "shared ancestors" = "#eeeeee")
  tip_data$cluster_id <- factor(tip_data$cluster_id, levels = names(full_palette))
  n_clusters <- length(unique(tip_data$cluster_id))
  plot_base <- sub("_tree_edges[.]csv$", "_clusters_radial_tree_ggtree", edge_csv)
  png_path <- file.path(output_dir, paste0(plot_base, ".png"))
  pdf_path <- file.path(output_dir, paste0(plot_base, ".pdf"))

  tip_desc <- descendant_tip_indices(phy)
  audit_data <- audit_cluster_rows(method_key, edge_csv, phy, tip_data, tip_desc)
  base_tree <- ggtree(phy, layout = "circular")
  tree_metadata <- build_cluster_tree_metadata(phy, tip_data, tip_desc)
  root_data <- cluster_root_rows(phy, tip_data, base_tree)
  root_data$cluster_label <- ifelse(
    root_data$cluster_size >= 25,
    paste0(root_data$cluster_id, " n=", root_data$cluster_size),
    ""
  )
  tip_plot_data <- merge(
    data.frame(
      node = seq_along(phy$tip.label),
      label = phy$tip.label,
      cluster_id = tip_data$cluster_id,
      stringsAsFactors = FALSE
    ),
    base_tree$data[, c("node", "x", "y")],
    by = "node",
    all.x = TRUE
  )

  p <- base_tree %<+% tree_metadata +
    geom_tree(aes(color = branch_cluster), linewidth = 0.24, alpha = 0.98) +
    geom_point(
      data = tip_plot_data,
      aes(x = x, y = y, color = cluster_id),
      inherit.aes = FALSE,
      size = 0.18,
      alpha = 0.78,
      stroke = 0
    ) +
    geom_point(
      data = root_data,
      aes(x = x, y = y, size = cluster_size),
      inherit.aes = FALSE,
      shape = 21,
      fill = "#ffffff",
      color = "#111827",
      stroke = 0.20,
      alpha = 0.85
    ) +
    geom_text2(
      data = root_data[root_data$cluster_label != "", ],
      aes(x = x, y = y, subset = cluster_label != "", label = cluster_label),
      size = 1.7,
      hjust = -0.05,
      color = "#111827",
      inherit.aes = FALSE
    ) +
    scale_color_manual(values = tree_palette, guide = "none") +
    ggtitle(paste0(title, "\n", n_clusters, " final clusters; display branch lengths sqrt-capped")) +
    guides(color = "none", size = guide_legend(title = "Cluster size")) +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      plot.margin = margin(10, 10, 10, 10)
    )

  ggsave(png_path, p, width = 16, height = 16, dpi = 320, bg = "white", limitsize = FALSE)
  ggsave(pdf_path, p, width = 16, height = 16, bg = "white", limitsize = FALSE)

  cluster_table <- aggregate(
    cell_id ~ cluster_id + celltype,
    data = tip_data,
    FUN = length
  )
  names(cluster_table)[names(cluster_table) == "cell_id"] <- "n_cells"
  write.csv(
    cluster_table,
    file.path(output_dir, paste0(sub("_tree_edges[.]csv$", "_cluster_celltype_counts", edge_csv), ".csv")),
    row.names = FALSE
  )

  data.frame(
    method_key = method_key,
    png = png_path,
    pdf = pdf_path,
    tips = length(phy$tip.label),
    internal_nodes = phy$Nnode,
    edges = nrow(edges),
    clusters = n_clusters,
    exact_clade_clusters = sum(audit_data$exact_clade),
    non_exact_clade_clusters = sum(!audit_data$exact_clade),
    audit_csv = file.path(output_dir, "tbs_cluster_radial_tree_highlighting_audit.csv"),
    audit_data = I(list(audit_data)),
    stringsAsFactors = FALSE
  )
}

rows_with_audit <- do.call(
  rbind,
  Map(plot_one_method, method_map$key, method_map$edge_csv, method_map$title)
)
audit_rows <- do.call(rbind, rows_with_audit$audit_data)
rows <- rows_with_audit[, setdiff(names(rows_with_audit), "audit_data")]
write.csv(
  audit_rows,
  file.path(output_dir, "tbs_cluster_radial_tree_highlighting_audit.csv"),
  row.names = FALSE
)
write.csv(
  rows,
  file.path(output_dir, "tbs_cluster_radial_tree_ggtree_outputs.csv"),
  row.names = FALSE
)
print(rows)
