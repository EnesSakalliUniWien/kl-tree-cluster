#!/usr/bin/env Rscript
# Dataset-specific pancreas UMAP/tree renderer.

suppressPackageStartupMessages({
  library(ape)
  library(ggplot2)
  library(ggtree)
  library(patchwork)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) {
  stop("Unable to resolve script path from Rscript command arguments.")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
project_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."))
default_output_dir <- file.path(
  project_root,
  "raw",
  "assets",
  "benchmark-results",
  "pancreas_scrna_cluster_benchmark_20260623"
)
trailing_args <- commandArgs(trailingOnly = TRUE)
output_dir_arg <- grep("^--output-dir=", trailing_args, value = TRUE)
output_dir <- if (length(output_dir_arg)) {
  normalizePath(sub("^--output-dir=", "", output_dir_arg[[1]]), mustWork = FALSE)
} else {
  default_output_dir
}
title_arg <- grep("^--title=", trailing_args, value = TRUE)
plot_title <- if (length(title_arg)) {
  sub("^--title=", "", title_arg[[1]])
} else {
  "Pancreas scRNA TBS clusters: UMAP and full radial trees"
}

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

method_map <- data.frame(
  key = c(
    "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001"
  ),
  edge_csv = c(
    "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv"
  ),
  label = c(
    "Topology-only TBS",
    "Recomputed NNLS branch-time TBS",
    "Raw-linkage branch-time diagnostic TBS",
    "Adaptive diffusion topology TBS",
    "Adaptive diffusion recomputed NNLS branch-time TBS",
    "Adaptive diffusion raw-linkage branch-time diagnostic TBS"
  ),
  stringsAsFactors = FALSE
)

min_large_cluster_size <- 50

readable_branch_lengths <- function(branch_lengths) {
  values <- as.numeric(branch_lengths)
  values[!is.finite(values) | values < 0] <- 0
  if (!any(values > 0)) {
    return(rep(1, length(values)))
  }

  transformed <- sqrt(values)
  positive <- transformed[transformed > 0]
  floor_value <- stats::median(positive, na.rm = TRUE) * 0.05
  if (!is.finite(floor_value) || floor_value <= 0) {
    floor_value <- min(positive, na.rm = TRUE)
  }
  transformed <- pmax(transformed, floor_value)

  cap_value <- stats::quantile(transformed, probs = 0.99, names = FALSE, na.rm = TRUE)
  if (is.finite(cap_value) && cap_value > 0) {
    transformed <- pmin(transformed, cap_value)
  }
  transformed
}

edge_table_to_phylo <- function(edges) {
  parents <- unique(as.character(edges$parent))
  children <- unique(as.character(edges$child))
  root <- setdiff(parents, children)
  if (length(root) != 1L) {
    stop("Expected exactly one root; found ", length(root))
  }

  tips <- sort(setdiff(children, parents))
  tip_order <- order(as.integer(sub("^L", "", tips)))
  tips <- tips[tip_order]
  internal_nodes <- c(root, sort(setdiff(parents, root)))
  tip_ids <- stats::setNames(seq_along(tips), tips)
  internal_ids <- stats::setNames(length(tips) + seq_along(internal_nodes), internal_nodes)
  node_ids <- c(tip_ids, internal_ids)

  branch_lengths <- readable_branch_lengths(edges$branch_length)

  phy <- list(
    edge = cbind(
      unname(node_ids[as.character(edges$parent)]),
      unname(node_ids[as.character(edges$child)])
    ),
    tip.label = tips,
    Nnode = length(internal_nodes),
    node.label = internal_nodes,
    edge.length = branch_lengths
  )
  class(phy) <- "phylo"
  ape::reorder.phylo(phy, order = "cladewise")
}

descendant_tip_indices <- function(phy) {
  children <- split(phy$edge[, 2], phy$edge[, 1])
  cache <- new.env(parent = emptyenv())

  visit <- function(node) {
    key <- as.character(node)
    if (exists(key, cache, inherits = FALSE)) {
      return(get(key, cache, inherits = FALSE))
    }
    if (node <= length(phy$tip.label)) {
      tips <- node
    } else {
      tips <- unlist(lapply(children[[key]], visit), use.names = FALSE)
    }
    assign(key, tips, cache)
    tips
  }

  nodes <- seq_len(length(phy$tip.label) + phy$Nnode)
  stats::setNames(lapply(nodes, visit), nodes)
}

build_tree_metadata <- function(phy, tip_data) {
  node_count <- length(phy$tip.label) + phy$Nnode
  cluster_by_tip <- stats::setNames(as.character(tip_data$cluster_id), tip_data$label)
  tip_desc <- descendant_tip_indices(phy)

  node_cluster <- rep("shared ancestors", node_count)
  names(node_cluster) <- as.character(seq_len(node_count))
  for (node in seq_len(node_count)) {
    tip_labels <- phy$tip.label[tip_desc[[as.character(node)]]]
    clusters <- unique(cluster_by_tip[tip_labels])
    if (length(clusters) == 1L) {
      node_cluster[[as.character(node)]] <- clusters
    }
  }

  data.frame(
    node = seq_len(node_count),
    branch_cluster = factor(node_cluster),
    stringsAsFactors = FALSE
  )
}

cluster_root_rows <- function(phy, tip_data, tree_plot) {
  rows <- list()
  split_tips <- split(tip_data$label, tip_data$cluster_id)
  for (cluster_id in names(split_tips)) {
    leaves <- split_tips[[cluster_id]]
    node <- if (length(leaves) == 1L) {
      match(leaves[[1]], phy$tip.label)
    } else {
      ape::getMRCA(phy, leaves)
    }
    rows[[length(rows) + 1L]] <- data.frame(
      node = node,
      cluster_id = cluster_id,
      cluster_size = length(leaves),
      stringsAsFactors = FALSE
    )
  }
  root_data <- do.call(rbind, rows)
  root_data <- merge(
    root_data,
    tree_plot$data[, c("node", "x", "y")],
    by = "node",
    all.x = TRUE
  )
  root_data$cluster_label <- ifelse(
    root_data$cluster_size >= min_large_cluster_size,
    paste0(root_data$cluster_id, " n=", root_data$cluster_size),
    ""
  )
  root_data
}

audit_cluster_rows <- function(method_key, edge_csv, phy, tip_data, tip_desc) {
  rows <- list()
  split_tips <- split(tip_data$label, tip_data$cluster_id)
  for (cluster_id in names(split_tips)) {
    leaves <- split_tips[[cluster_id]]
    node <- if (length(leaves) == 1L) {
      match(leaves[[1]], phy$tip.label)
    } else {
      ape::getMRCA(phy, leaves)
    }
    descendant_labels <- phy$tip.label[tip_desc[[as.character(node)]]]
    extra_tips <- setdiff(descendant_labels, leaves)
    missing_tips <- setdiff(leaves, descendant_labels)
    rows[[length(rows) + 1L]] <- data.frame(
      method_key = method_key,
      edge_csv = edge_csv,
      cluster_id = as.character(cluster_id),
      cluster_size = length(leaves),
      mrca_node = node,
      mrca_descendant_tip_count = length(descendant_labels),
      exact_clade = length(extra_tips) == 0L && length(missing_tips) == 0L,
      extra_tip_count = length(extra_tips),
      missing_tip_count = length(missing_tips),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

cluster_palette <- function(cluster_ids) {
  ids <- sort(
    unique(as.character(cluster_ids)),
    method = "radix"
  )
  cluster_numbers <- suppressWarnings(as.integer(sub("^C", "", ids)))
  ids <- ids[order(ifelse(is.na(cluster_numbers), Inf, cluster_numbers), ids)]
  hues <- (15 + (seq_along(ids) - 1L) * 137.508) %% 360
  chroma <- rep(c(90, 80, 70, 86), length.out = length(ids))
  luminance <- rep(c(42, 50, 36, 56), length.out = length(ids))
  colors <- grDevices::hcl(h = hues, c = chroma, l = luminance, fixup = TRUE)
  stats::setNames(colors, ids)
}

plot_one_row <- function(method_key, edge_csv, label) {
  edges <- read.csv(file.path(output_dir, edge_csv), stringsAsFactors = FALSE, check.names = FALSE)
  phy <- edge_table_to_phylo(edges)

  leaf_indices <- as.integer(sub("^L", "", phy$tip.label)) + 1L
  tip_data <- data.frame(
    label = phy$tip.label,
    cell_id = assignments$cell_id[leaf_indices],
    celltype = assignments$celltype[leaf_indices],
    cluster_id = paste0("C", assignments[[method_key]][leaf_indices]),
    stringsAsFactors = FALSE
  )
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
  tree_metadata <- build_tree_metadata(phy, tip_data)
  root_data <- cluster_root_rows(phy, tip_data, tree_base)
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
all_clusters_png_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.png")
all_clusters_pdf_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.pdf")
legacy_png_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_ggtree.png")
legacy_pdf_path <- file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_ggtree.pdf")

ggsave(png_path, combo, width = 24, height = 60, dpi = 300, bg = "white", limitsize = FALSE)
ggsave(pdf_path, combo, width = 24, height = 60, bg = "white", limitsize = FALSE)
file.copy(png_path, all_clusters_png_path, overwrite = TRUE)
file.copy(pdf_path, all_clusters_pdf_path, overwrite = TRUE)
file.copy(png_path, legacy_png_path, overwrite = TRUE)
file.copy(pdf_path, legacy_pdf_path, overwrite = TRUE)

manifest <- data.frame(
  png = png_path,
  pdf = pdf_path,
  all_clusters_png = all_clusters_png_path,
  all_clusters_pdf = all_clusters_pdf_path,
  legacy_png = legacy_png_path,
  legacy_pdf = legacy_pdf_path,
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
  file.path(output_dir, "tbs_umap_cluster_radial_tree_combo_ggtree_outputs.csv"),
  row.names = FALSE
)
print(manifest)
