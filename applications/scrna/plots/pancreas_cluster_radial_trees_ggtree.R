#!/usr/bin/env Rscript
# Dataset-specific pancreas cluster-tree renderer.

suppressPackageStartupMessages({
  library(ape)
  library(ggplot2)
  library(ggtree)
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

assignments <- read.csv(
  file.path(output_dir, "method_assignments.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

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
  title = c(
    "TBS topology-only final clusters",
    "TBS recomputed NNLS branch-time final clusters",
    "TBS raw-linkage branch-time diagnostic final clusters",
    "TBS adaptive diffusion topology final clusters",
    "TBS adaptive diffusion recomputed NNLS branch-time final clusters",
    "TBS adaptive diffusion raw-linkage branch-time diagnostic final clusters"
  ),
  stringsAsFactors = FALSE
)

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

build_tree_metadata <- function(phy, tip_data, tip_desc) {
  node_count <- length(phy$tip.label) + phy$Nnode
  cluster_by_tip <- stats::setNames(as.character(tip_data$cluster_id), tip_data$label)

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

cluster_root_rows <- function(phy, tip_data, tree_plot) {
  rows <- list()
  split_tips <- split(tip_data$label, tip_data$cluster_id)
  for (cluster_id in names(split_tips)) {
    leaves <- split_tips[[cluster_id]]
    if (length(leaves) == 1L) {
      node <- match(leaves[[1]], phy$tip.label)
    } else {
      node <- ape::getMRCA(phy, leaves)
    }
    rows[[length(rows) + 1L]] <- data.frame(
      node = node,
      cluster_id = cluster_id,
      cluster_size = length(leaves),
      stringsAsFactors = FALSE
    )
  }
  root_data <- do.call(rbind, rows)
  merge(
    root_data,
    tree_plot$data[, c("node", "x", "y")],
    by = "node",
    all.x = TRUE
  )
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

plot_one_method <- function(method_key, edge_csv, title) {
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
  tree_metadata <- build_tree_metadata(phy, tip_data, tip_desc)
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
