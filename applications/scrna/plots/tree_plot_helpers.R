# Shared tree-construction helpers for scRNA ggtree renderers.

scrna_project_root <- function(script_path) {
  normalizePath(file.path(dirname(script_path), "..", "..", ".."))
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
