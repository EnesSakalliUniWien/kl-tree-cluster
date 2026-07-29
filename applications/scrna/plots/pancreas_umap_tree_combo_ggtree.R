#!/usr/bin/env Rscript
# Dataset-specific pancreas UMAP/tree renderer.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) {
  stop("Unable to resolve script path from Rscript command arguments.")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
source(file.path(dirname(script_path), "tree_plot_helpers.R"), local = TRUE)
context <- scrna_tree_plot_context(
  script_path,
  "pancreas_scrna_cluster_benchmark_20260623",
  c("ape", "ggplot2", "ggtree", "patchwork")
)
trailing_args <- context$trailing_args
output_dir <- context$output_dir
plot_title <- scrna_arg_value(
  trailing_args,
  "--title",
  default = "Pancreas scRNA TBS clusters: UMAP and full radial trees"
)

assignments <- read.csv(
  file.path(output_dir, "method_assignments.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

range_with_padding <- function(values, padding_fraction = 0.025) {
  limits <- range(values, finite = TRUE)
  padding <- diff(limits) * padding_fraction
  limits + c(-padding, padding)
}

umap_x_limits <- range_with_padding(assignments$umap1)
umap_y_limits <- range_with_padding(assignments$umap2)

method_map <- pancreas_tree_method_map("short")
method_map$label <- method_map$title

min_large_cluster_size <- 50

plot_one_row <- function(method_key, edge_csv, label) {
  edges <- read.csv(file.path(output_dir, edge_csv), stringsAsFactors = FALSE, check.names = FALSE)
  phy <- edge_table_to_phylo(edges)

  tip_data <- pancreas_tip_data(phy, assignments, method_key)
  cluster_sizes <- sort(table(tip_data$cluster_id), decreasing = TRUE)
  large_cluster_ids <- names(cluster_sizes[cluster_sizes >= min_large_cluster_size])
  full_palette <- cluster_palette(names(cluster_sizes))
  tree_palette <- c(
    full_palette,
    "shared ancestors" = "#eeeeee"
  )
  tip_data$cluster_id <- factor(tip_data$cluster_id, levels = names(full_palette))

  umap_data <- data.frame(
    umap1 = assignments$umap1,
    umap2 = assignments$umap2,
    cluster_id = factor(paste0("C", assignments[[method_key]]), levels = names(full_palette)),
    stringsAsFactors = FALSE
  )
  n_clusters <- length(cluster_sizes)
  n_large_clusters <- length(large_cluster_ids)

  label_data <- aggregate(
    cbind(umap1, umap2) ~ cluster_id,
    data = umap_data[as.character(umap_data$cluster_id) %in% large_cluster_ids, ],
    FUN = median
  )
  label_data$n_cells <- as.integer(cluster_sizes[as.character(label_data$cluster_id)])

  umap_plot <- ggplot(umap_data, aes(umap1, umap2, color = cluster_id)) +
    geom_point(size = 1.75, alpha = 0.96, stroke = 0) +
    geom_label(
      data = label_data,
      aes(label = paste0(cluster_id, "\n", n_cells)),
      size = 3.05,
      linewidth = 0.22,
      alpha = 0.88,
      color = "#111827",
      fill = "white",
      label.padding = unit(0.12, "lines"),
      show.legend = FALSE
    ) +
    scale_color_manual(values = full_palette, guide = "none") +
    coord_equal(
      xlim = umap_x_limits,
      ylim = umap_y_limits,
      expand = FALSE,
      clip = "off"
    ) +
    labs(
      title = paste0(label, " UMAP"),
      subtitle = paste0(
        n_clusters,
        " total clusters; ",
        n_large_clusters,
        " clusters >= ",
        min_large_cluster_size,
        " cells labeled"
      ),
      x = NULL,
      y = NULL
    ) +
    theme_void(base_size = 9) +
    theme(
      panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.22),
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10.5),
      plot.margin = margin(3, 4, 3, 4)
    )

  tree_base <- ggtree(phy, layout = "circular")
  tip_desc <- descendant_tip_indices(phy)
  tree_metadata <- build_cluster_tree_metadata(phy, tip_data, tip_desc)
  root_data <- cluster_root_rows(phy, tip_data, tree_base)
  root_data$cluster_label <- ifelse(
    root_data$cluster_size >= min_large_cluster_size,
    paste0(root_data$cluster_id, " n=", root_data$cluster_size),
    ""
  )
  audit_data <- audit_cluster_rows(method_key, edge_csv, phy, tip_data, tip_desc)

  tree_plot <- tree_base %<+% tree_metadata +
    geom_tree(aes(color = branch_cluster), linewidth = 0.44, alpha = 0.98) +
    geom_point(
      data = root_data,
      aes(x = x, y = y, size = cluster_size),
      inherit.aes = FALSE,
      shape = 21,
      fill = "white",
      color = "#111827",
      stroke = 0.28,
      alpha = 0.92
    ) +
    geom_text2(
      data = root_data[root_data$cluster_label != "", ],
      aes(x = x, y = y, subset = cluster_label != "", label = cluster_label),
      inherit.aes = FALSE,
      size = 2.15,
      hjust = -0.06,
      color = "#111827"
    ) +
    scale_color_manual(values = tree_palette, guide = "none") +
    scale_size_area(max_size = 3.0) +
    guides(size = guide_legend(title = "Cluster size")) +
    labs(
      title = paste0(label, " full radial tree"),
      subtitle = paste0(
        "Colored branches = final cluster subtrees; pale grey = shared ancestors; display branch lengths sqrt-capped"
      )
    ) +
    theme(
      plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 9.5, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 7),
      plot.margin = margin(5, 5, 5, 5)
    )

  list(
    plot = umap_plot + tree_plot + plot_layout(widths = c(0.92, 1.08)),
    audit_data = audit_data
  )
}

row_results <- Map(plot_one_row, method_map$key, method_map$edge_csv, method_map$label)
rows <- lapply(row_results, `[[`, "plot")
audit_rows <- do.call(rbind, lapply(row_results, `[[`, "audit_data"))
write.csv(
  audit_rows,
  file.path(output_dir, "tbs_umap_tree_highlighting_audit.csv"),
  row.names = FALSE
)

combo <- wrap_plots(rows, ncol = 1) +
  plot_annotation(
    title = plot_title,
    subtitle = "Each row uses the same colors in UMAP and tree for every final TBS cluster. Labels are shown only for larger clusters; shared tree ancestors are grey.",
    theme = theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, hjust = 0.5)
    )
  )

png_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png")
pdf_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.pdf")

ggsave(png_path, combo, width = 24, height = 60, dpi = 300, bg = "white", limitsize = FALSE)
ggsave(pdf_path, combo, width = 24, height = 60, bg = "white", limitsize = FALSE)

manifest <- data.frame(
  png = png_path,
  pdf = pdf_path,
  audit_csv = file.path(output_dir, "tbs_umap_tree_highlighting_audit.csv"),
  exact_clade_clusters = sum(audit_rows$exact_clade),
  non_exact_clade_clusters = sum(!audit_rows$exact_clade),
  rows = nrow(method_map),
  tips_per_tree = nrow(assignments),
  color_policy = "all final clusters colored on UMAP and terminal tree subtrees; labels only for clusters >= 50 cells; shared ancestors grey",
  umap_scaling = "tight global UMAP limits, no axis chrome, larger points, balanced UMAP/tree width",
  branch_length_display = "sqrt transform with 99th-percentile cap and small positive floor; raw edge lengths remain in tree_edges CSV files",
  stringsAsFactors = FALSE
)
write.csv(
  manifest,
  file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree_outputs.csv"),
  row.names = FALSE
)
print(manifest)
