# Shared tree-construction helpers for scRNA ggtree renderers.

scrna_project_root <- function(script_path) {
  normalizePath(file.path(dirname(script_path), "..", "..", ".."))
}

scrna_arg_value <- function(args, name, default = NULL, normalize = FALSE, numeric = FALSE) {
  arg <- grep(paste0("^", name, "="), args, value = TRUE)
  value <- if (length(arg)) {
    sub(paste0("^", name, "="), "", arg[[1]])
  } else {
    default
  }
  if (is.null(value)) {
    return(NULL)
  }
  if (normalize) {
    value <- normalizePath(value, mustWork = FALSE)
  }
  if (numeric) {
    value <- as.numeric(value)
  }
  value
}

scrna_tree_plot_context <- function(script_path, benchmark_dir, packages) {
  suppressPackageStartupMessages({
    for (package in packages) {
      library(package, character.only = TRUE)
    }
  })
  project_root <- scrna_project_root(script_path)
  default_output_dir <- file.path(
    project_root,
    "raw",
    "assets",
    "benchmark-results",
    benchmark_dir
  )
  trailing_args <- commandArgs(trailingOnly = TRUE)
  list(
    project_root = project_root,
    trailing_args = trailing_args,
    output_dir = scrna_arg_value(
      trailing_args,
      "--output-dir",
      default = default_output_dir,
      normalize = TRUE
    )
  )
}

pancreas_tree_method_map <- function(title_style = c("short", "final_clusters", "edge_gate")) {
  title_style <- match.arg(title_style)
  titles <- list(
    short = c(
      "Topology-only TBS",
      "Recomputed NNLS branch-time TBS",
      "Raw-linkage branch-time diagnostic TBS",
      "Adaptive diffusion topology TBS",
      "Adaptive diffusion recomputed NNLS branch-time TBS",
      "Adaptive diffusion raw-linkage branch-time diagnostic TBS"
    ),
    final_clusters = c(
      "TBS topology-only final clusters",
      "TBS recomputed NNLS branch-time final clusters",
      "TBS raw-linkage branch-time diagnostic final clusters",
      "TBS adaptive diffusion topology final clusters",
      "TBS adaptive diffusion recomputed NNLS branch-time final clusters",
      "TBS adaptive diffusion raw-linkage branch-time diagnostic final clusters"
    ),
    edge_gate = c(
      "TBS topology-only edge gate",
      "TBS recomputed NNLS branch-time edge gate",
      "TBS raw-linkage branch-time diagnostic edge gate",
      "TBS adaptive diffusion topology edge gate",
      "TBS adaptive diffusion recomputed NNLS branch-time edge gate",
      "TBS adaptive diffusion raw-linkage branch-time diagnostic edge gate"
    )
  )
  keys <- c(
    "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
    "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001"
  )
  data.frame(
    key = keys,
    edge_csv = paste0(keys, "_tree_edges.csv"),
    title = titles[[title_style]],
    stringsAsFactors = FALSE
  )
}

pancreas_tip_data <- function(phy, assignments, method_key) {
  leaf_indices <- as.integer(sub("^L", "", phy$tip.label)) + 1L
  data.frame(
    label = phy$tip.label,
    cell_id = assignments$cell_id[leaf_indices],
    celltype = assignments$celltype[leaf_indices],
    cluster_id = paste0("C", assignments[[method_key]][leaf_indices]),
    stringsAsFactors = FALSE
  )
}

umap_panel_theme <- function(legend_position = "none", legend_title = TRUE) {
  theme_void(base_size = 13) +
    theme(
      plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      legend.position = legend_position,
      legend.title = if (legend_title) element_text(size = 10) else element_blank(),
      legend.text = element_text(size = 9),
      plot.margin = margin(4, 4, 4, 4),
      panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.26)
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

  phy <- list(
    edge = cbind(
      unname(node_ids[as.character(edges$parent)]),
      unname(node_ids[as.character(edges$child)])
    ),
    tip.label = tips,
    Nnode = length(internal_nodes),
    node.label = internal_nodes,
    edge.length = readable_branch_lengths(edges$branch_length)
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

build_cluster_tree_metadata <- function(
  phy,
  tip_data,
  tip_desc = descendant_tip_indices(phy)
) {
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
  ids <- unique(as.character(cluster_ids))
  cluster_numbers <- suppressWarnings(as.integer(sub("^C", "", ids)))
  ids <- ids[order(ifelse(is.na(cluster_numbers), Inf, cluster_numbers), ids)]
  hues <- (15 + (seq_along(ids) - 1L) * 137.508) %% 360
  chroma <- rep(c(90, 80, 70, 86), length.out = length(ids))
  luminance <- rep(c(42, 50, 36, 56), length.out = length(ids))
  colors <- grDevices::hcl(h = hues, c = chroma, l = luminance, fixup = TRUE)
  stats::setNames(colors, ids)
}

cluster_mrca_node <- function(phy, leaves) {
  if (length(leaves) == 1L) {
    match(leaves[[1]], phy$tip.label)
  } else {
    ape::getMRCA(phy, leaves)
  }
}

cluster_root_rows <- function(phy, tip_data, tree_plot) {
  split_tips <- split(tip_data$label, tip_data$cluster_id)
  rows <- lapply(names(split_tips), function(cluster_id) {
    leaves <- split_tips[[cluster_id]]
    data.frame(
      node = cluster_mrca_node(phy, leaves),
      cluster_id = cluster_id,
      cluster_size = length(leaves),
      stringsAsFactors = FALSE
    )
  })
  root_data <- do.call(rbind, rows)
  merge(
    root_data,
    tree_plot$data[, c("node", "x", "y")],
    by = "node",
    all.x = TRUE
  )
}

audit_cluster_rows <- function(method_key, edge_csv, phy, tip_data, tip_desc) {
  split_tips <- split(tip_data$label, tip_data$cluster_id)
  rows <- lapply(names(split_tips), function(cluster_id) {
    leaves <- split_tips[[cluster_id]]
    node <- cluster_mrca_node(phy, leaves)
    descendant_labels <- phy$tip.label[tip_desc[[as.character(node)]]]
    extra_tips <- setdiff(descendant_labels, leaves)
    missing_tips <- setdiff(leaves, descendant_labels)
    data.frame(
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
  })
  do.call(rbind, rows)
}
