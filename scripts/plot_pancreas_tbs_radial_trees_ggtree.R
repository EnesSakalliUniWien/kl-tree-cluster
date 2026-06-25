#!/usr/bin/env Rscript

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
project_root <- normalizePath(file.path(dirname(script_path), ".."))
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

method_map <- data.frame(
  edge_csv = c(
    "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv",
    "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv"
  ),
  title = c(
    "TBS topology-only edge gate",
    "TBS recomputed NNLS branch-time edge gate",
    "TBS raw-linkage branch-time diagnostic edge gate",
    "TBS adaptive diffusion topology edge gate",
    "TBS adaptive diffusion recomputed NNLS branch-time edge gate",
    "TBS adaptive diffusion raw-linkage branch-time diagnostic edge gate"
  ),
  stringsAsFactors = FALSE
)

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
