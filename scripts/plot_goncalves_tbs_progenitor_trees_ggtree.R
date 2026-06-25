#!/usr/bin/env Rscript

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
project_root <- normalizePath(file.path(dirname(script_path), ".."))
default_output_dir <- file.path(
  project_root,
  "raw",
  "assets",
  "benchmark-results",
  "goncalves_fetal_pancreas_progenitor_benchmark_20260624"
)
trailing_args <- commandArgs(trailingOnly = TRUE)
output_dir_arg <- grep("^--output-dir=", trailing_args, value = TRUE)
output_dir <- if (length(output_dir_arg)) {
  normalizePath(sub("^--output-dir=", "", output_dir_arg[[1]]), mustWork = FALSE)
} else {
  default_output_dir
}
pdf_width_arg <- grep("^--pdf-width=", trailing_args, value = TRUE)
pdf_width <- if (length(pdf_width_arg)) {
  as.numeric(sub("^--pdf-width=", "", pdf_width_arg[[1]]))
} else {
  30
}
pdf_height_arg <- grep("^--pdf-height=", trailing_args, value = TRUE)
pdf_height <- if (length(pdf_height_arg)) {
  as.numeric(sub("^--pdf-height=", "", pdf_height_arg[[1]]))
} else {
  42
}
page_width_arg <- grep("^--page-width=", trailing_args, value = TRUE)
page_width <- if (length(page_width_arg)) {
  as.numeric(sub("^--page-width=", "", page_width_arg[[1]]))
} else {
  30
}
page_height_arg <- grep("^--page-height=", trailing_args, value = TRUE)
page_height <- if (length(page_height_arg)) {
  as.numeric(sub("^--page-height=", "", page_height_arg[[1]]))
} else {
  16
}
png_dpi_arg <- grep("^--png-dpi=", trailing_args, value = TRUE)
png_dpi <- if (length(png_dpi_arg)) {
  as.numeric(sub("^--png-dpi=", "", png_dpi_arg[[1]]))
} else {
  180
}
method_key <- "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001"
edge_csv <- paste0(method_key, "_tree_edges.csv")
large_cluster_label_min <- 50

assignments <- read.csv(
  file.path(output_dir, "method_assignments.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
edges <- read.csv(file.path(output_dir, edge_csv), stringsAsFactors = FALSE, check.names = FALSE)
inner_nodes <- read.csv(
  file.path(output_dir, "goncalves_tbs_inner_node_progenitor_signature_scores.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
meetings <- read.csv(
  file.path(output_dir, "goncalves_tbs_monophyletic_meeting_progenitor_signature_scores.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
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

edge_table_to_phylo <- function(edge_table) {
  parents <- unique(as.character(edge_table$parent))
  children <- unique(as.character(edge_table$child))
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
      unname(node_ids[as.character(edge_table$parent)]),
      unname(node_ids[as.character(edge_table$child)])
    ),
    tip.label = tips,
    Nnode = length(internal_nodes),
    node.label = internal_nodes,
    edge.length = readable_branch_lengths(edge_table$branch_length)
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

build_cluster_tree_metadata <- function(phy, tip_data) {
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
    root_data$cluster_size >= large_cluster_label_min,
    paste0(root_data$cluster_id, " n=", root_data$cluster_size),
    ""
  )
  root_data
}

state_class <- function(interpretation) {
  ifelse(
    grepl("tip-progenitor", interpretation),
    "tip progenitor",
    ifelse(
      grepl("proliferating", interpretation),
      "proliferating progenitor",
      ifelse(
        grepl("mixed fetal progenitor", interpretation),
        "mixed fetal progenitor",
        ifelse(
          grepl("endocrine", interpretation),
          "endocrine",
          ifelse(grepl("partial progenitor", interpretation), "partial progenitor", "other")
        )
      )
    )
  )
}

phy <- edge_table_to_phylo(edges)
node_lookup <- c(
  stats::setNames(seq_along(phy$tip.label), phy$tip.label),
  stats::setNames(length(phy$tip.label) + seq_along(phy$node.label), phy$node.label)
)

tip_indices <- as.integer(sub("^L", "", phy$tip.label)) + 1L
tip_meta <- data.frame(
  label = phy$tip.label,
  node = unname(node_lookup[phy$tip.label]),
  dominant_population = assignments$celltype[tip_indices],
  progenitor_population_fraction = ifelse(
    assignments$celltype[tip_indices] %in% c("trunk", "tip", "proliferating"),
    1,
    0
  ),
  n_cells = 1,
  state_class = ifelse(
    assignments$celltype[tip_indices] %in% c("trunk", "tip", "proliferating"),
    paste(assignments$celltype[tip_indices], "cell"),
    assignments$celltype[tip_indices]
  ),
  stringsAsFactors = FALSE
)
internal_meta <- data.frame(
  label = inner_nodes$label,
  node = unname(node_lookup[inner_nodes$label]),
  dominant_population = inner_nodes$dominant_population,
  progenitor_population_fraction = inner_nodes$progenitor_population_fraction,
  n_cells = inner_nodes$n_cells,
  state_class = state_class(inner_nodes$progenitor_interpretation),
  progenitor_interpretation = inner_nodes$progenitor_interpretation,
  stringsAsFactors = FALSE
)
node_meta <- rbind(
  tip_meta[, c("label", "node", "dominant_population", "progenitor_population_fraction", "n_cells", "state_class")],
  internal_meta[, c("label", "node", "dominant_population", "progenitor_population_fraction", "n_cells", "state_class")]
)
node_meta$dominant_population <- factor(
  node_meta$dominant_population,
  levels = c("trunk", "tip", "proliferating", "endocrine", "mesenchyme", "blood", "neurons", "unknown")
)
node_meta$state_class <- factor(
  node_meta$state_class,
  levels = c(
    "tip progenitor",
    "proliferating progenitor",
    "mixed fetal progenitor",
    "partial progenitor",
    "endocrine",
    "trunk cell",
    "tip cell",
    "proliferating cell",
    "mesenchyme",
    "blood",
    "neurons",
    "unknown",
    "other"
  )
)

key_nodes <- unique(c(
  "N2851",
  "N2903",
  "N2758",
  "N2876",
  "N2910",
  "N2907",
  "N2872"
))
key_node_labels <- data.frame(
  label = key_nodes,
  node = unname(node_lookup[key_nodes]),
  node_label = c(
    "N2851\ntip n=31",
    "N2903\ntrunk/prolif n=54",
    "N2758\ntip/trunk n=70",
    "N2876\nC1+C2 n=27",
    "N2910\nC1-C3 n=47",
    "N2907\nC22+mes n=713",
    "N2872\nendocrine n=19"
  ),
  stringsAsFactors = FALSE
)

base_tree <- ggtree(phy, layout = "circular")
label_xy <- merge(
  key_node_labels,
  base_tree$data[, c("node", "x", "y")],
  by = "node",
  all.x = TRUE
)

population_colors <- c(
  trunk = "#1f78b4",
  tip = "#33a02c",
  proliferating = "#e31a1c",
  endocrine = "#6a3d9a",
  mesenchyme = "#b15928",
  blood = "#ff7f00",
  neurons = "#a6cee3",
  unknown = "#9ca3af"
)
state_colors <- c(
  "tip progenitor" = "#33a02c",
  "proliferating progenitor" = "#e31a1c",
  "mixed fetal progenitor" = "#2563eb",
  "partial progenitor" = "#7c3aed",
  endocrine = "#6a3d9a",
  "trunk cell" = "#1f78b4",
  "tip cell" = "#33a02c",
  "proliferating cell" = "#e31a1c",
  mesenchyme = "#b15928",
  blood = "#ff7f00",
  neurons = "#a6cee3",
  unknown = "#9ca3af",
  other = "#d1d5db"
)

label_layer <- list(
  geom_point(
    data = label_xy,
    aes(x = x, y = y),
    inherit.aes = FALSE,
    shape = 21,
    size = 2.4,
    fill = "white",
    color = "#111827",
    stroke = 0.35
  ),
  geom_text2(
    data = label_xy,
    aes(x = x, y = y, label = node_label),
    inherit.aes = FALSE,
    size = 2.15,
    hjust = -0.05,
    color = "#111827"
  )
)

range_with_padding <- function(values, padding_fraction = 0.025) {
  limits <- range(values, finite = TRUE)
  padding <- diff(limits) * padding_fraction
  limits + c(-padding, padding)
}

umap_x_limits <- range_with_padding(assignments$umap1)
umap_y_limits <- range_with_padding(assignments$umap2)
umap_data <- data.frame(
  cell_id = assignments$cell_id,
  umap1 = assignments$umap1,
  umap2 = assignments$umap2,
  dominant_population = factor(
    assignments$celltype,
    levels = c("trunk", "tip", "proliferating", "endocrine", "mesenchyme", "blood", "neurons", "unknown")
  ),
  progenitor_population_fraction = ifelse(
    assignments$celltype %in% c("trunk", "tip", "proliferating"),
    1,
    0
  ),
  state_class = factor(
    ifelse(
      assignments$celltype %in% c("trunk", "tip", "proliferating"),
      paste(assignments$celltype, "cell"),
      assignments$celltype
    ),
    levels = levels(node_meta$state_class)
  ),
  stringsAsFactors = FALSE
)

cluster_ids <- paste0("C", assignments[[method_key]])
cluster_colors <- cluster_palette(cluster_ids)
cluster_sizes <- table(factor(cluster_ids, levels = names(cluster_colors)))
large_cluster_ids <- names(cluster_sizes[cluster_sizes >= large_cluster_label_min])
umap_cluster_data <- data.frame(
  umap1 = assignments$umap1,
  umap2 = assignments$umap2,
  cluster_id = factor(cluster_ids, levels = names(cluster_colors)),
  stringsAsFactors = FALSE
)
cluster_label_data <- aggregate(
  cbind(umap1, umap2) ~ cluster_id,
  data = umap_cluster_data[as.character(umap_cluster_data$cluster_id) %in% large_cluster_ids, ],
  FUN = median
)
cluster_label_data$n_cells <- as.integer(cluster_sizes[as.character(cluster_label_data$cluster_id)])

tip_cluster_data <- data.frame(
  label = phy$tip.label,
  cluster_id = factor(
    paste0("C", assignments[[method_key]][tip_indices]),
    levels = names(cluster_colors)
  ),
  stringsAsFactors = FALSE
)
cluster_tree_base <- ggtree(phy, layout = "circular")
cluster_tree_metadata <- build_cluster_tree_metadata(phy, tip_cluster_data)
cluster_root_data <- cluster_root_rows(phy, tip_cluster_data, cluster_tree_base)
cluster_tree_colors <- c(cluster_colors, "shared ancestors" = "#eeeeee")

population_label_data <- aggregate(
  cbind(umap1, umap2) ~ dominant_population,
  data = umap_data,
  FUN = median
)
population_label_data$n_cells <- as.integer(table(umap_data$dominant_population)[
  as.character(population_label_data$dominant_population)
])

umap_point_size <- 2.9
umap_label_size <- 4.8

umap_cluster <- ggplot(umap_cluster_data, aes(umap1, umap2, color = cluster_id)) +
  geom_point(size = 2.45, alpha = 0.94, stroke = 0) +
  geom_label(
    data = cluster_label_data,
    aes(label = paste0(cluster_id, "\n", n_cells)),
    size = 3.9,
    linewidth = 0.25,
    alpha = 0.9,
    color = "#111827",
    fill = "white",
    label.padding = unit(0.14, "lines"),
    show.legend = FALSE
  ) +
  scale_color_manual(values = cluster_colors, guide = "none") +
  coord_equal(xlim = umap_x_limits, ylim = umap_y_limits, expand = FALSE) +
  labs(
    title = "UMAP by TBS cluster",
    subtitle = paste0(
      length(cluster_sizes),
      " clusters from adaptive-diffusion topology TBS; labels shown for clusters >= ",
      large_cluster_label_min,
      " cells"
    ),
    x = NULL,
    y = NULL
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(4, 4, 4, 4),
    panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.26)
  )

umap_population <- ggplot(umap_data, aes(umap1, umap2, color = dominant_population)) +
  geom_point(size = umap_point_size, alpha = 0.94, stroke = 0) +
  geom_label(
    data = population_label_data,
    aes(label = paste0(dominant_population, "\n", n_cells)),
    size = umap_label_size,
    linewidth = 0.25,
    alpha = 0.9,
    color = "#111827",
    fill = "white",
    label.padding = unit(0.16, "lines"),
    show.legend = FALSE
  ) +
  scale_color_manual(values = population_colors, na.value = "#d1d5db", drop = FALSE) +
  coord_equal(xlim = umap_x_limits, ylim = umap_y_limits, expand = FALSE) +
  labs(
    title = "UMAP by fetal population",
    subtitle = "Same cell labels used for terminal tree tips",
    x = NULL,
    y = NULL
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(4, 4, 4, 4),
    panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.26)
  )

umap_fraction <- ggplot(umap_data, aes(umap1, umap2, color = progenitor_population_fraction)) +
  geom_point(size = umap_point_size, alpha = 0.94, stroke = 0) +
  scale_color_viridis_c(
    limits = c(0, 1),
    option = "plasma",
    name = "Cell is trunk/tip/prolif"
  ) +
  coord_equal(xlim = umap_x_limits, ylim = umap_y_limits, expand = FALSE) +
  labs(
    title = "UMAP by progenitor-label membership",
    subtitle = "Tree color is descendant fraction; cell-level UMAP is 0/1 membership",
    x = NULL,
    y = NULL
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    plot.margin = margin(4, 4, 4, 4),
    panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.26)
  )

umap_state <- ggplot(umap_data, aes(umap1, umap2, color = state_class)) +
  geom_point(size = umap_point_size, alpha = 0.94, stroke = 0) +
  scale_color_manual(values = state_colors, na.value = "#d1d5db", drop = FALSE) +
  coord_equal(xlim = umap_x_limits, ylim = umap_y_limits, expand = FALSE) +
  labs(
    title = "UMAP by progenitor interpretation",
    subtitle = "Terminal cells use population-derived state classes",
    x = NULL,
    y = NULL
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.margin = margin(4, 4, 4, 4),
    panel.border = element_rect(color = "#d1d5db", fill = NA, linewidth = 0.26)
  )

cluster_tree <- cluster_tree_base %<+% cluster_tree_metadata +
  geom_tree(aes(color = branch_cluster), linewidth = 0.35, alpha = 0.96) +
  geom_tippoint(aes(color = branch_cluster), size = 0.38, alpha = 0.72) +
  geom_point(
    data = cluster_root_data,
    aes(x = x, y = y, size = cluster_size),
    inherit.aes = FALSE,
    shape = 21,
    fill = "white",
    color = "#111827",
    stroke = 0.28,
    alpha = 0.92
  ) +
  geom_text2(
    data = cluster_root_data[cluster_root_data$cluster_label != "", ],
    aes(x = x, y = y, subset = cluster_label != "", label = cluster_label),
    inherit.aes = FALSE,
    size = 2.05,
    hjust = -0.06,
    color = "#111827"
  ) +
  scale_color_manual(values = cluster_tree_colors, guide = "none") +
  scale_size_area(max_size = 3.0) +
  guides(size = guide_legend(title = "Cluster size")) +
  ggtitle(
    "Goncalves adaptive-diffusion TBS tree by final cluster",
    "Colored branches are exact final cluster subtrees; pale grey marks shared ancestors"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
  )

population_tree <- ggtree(phy, layout = "circular") %<+% node_meta +
  geom_tree(aes(color = dominant_population), linewidth = 0.35, alpha = 0.96) +
  geom_tippoint(aes(color = dominant_population), size = 0.38, alpha = 0.72) +
  label_layer +
  scale_color_manual(values = population_colors, na.value = "#d1d5db", drop = FALSE) +
  ggtitle(
    "Goncalves adaptive-diffusion TBS tree by dominant fetal population",
    "Whole tree; labels mark progenitor/endocrine nodes from the signature analysis"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8)
  )

fraction_tree <- ggtree(phy, layout = "circular") %<+% node_meta +
  geom_tree(aes(color = progenitor_population_fraction), linewidth = 0.35, alpha = 0.96) +
  geom_tippoint(aes(color = progenitor_population_fraction), size = 0.38, alpha = 0.72) +
  label_layer +
  scale_color_viridis_c(
    limits = c(0, 1),
    option = "plasma",
    name = "Trunk/tip/prolif fraction"
  ) +
  ggtitle(
    "Goncalves adaptive-diffusion TBS tree by progenitor fraction",
    "Continuous color is fraction of descendant cells labeled trunk, tip, or proliferating"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
  )

state_tree <- ggtree(phy, layout = "circular") %<+% node_meta +
  geom_tree(aes(color = state_class), linewidth = 0.35, alpha = 0.96) +
  geom_tippoint(aes(color = state_class), size = 0.38, alpha = 0.72) +
  label_layer +
  scale_color_manual(values = state_colors, na.value = "#d1d5db", drop = FALSE) +
  ggtitle(
    "Goncalves adaptive-diffusion TBS tree by progenitor interpretation",
    "Internal branches use marker/population interpretation; terminal cells use their population label"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7)
  )

plots <- list(
  population = population_tree,
  progenitor_fraction = fraction_tree,
  progenitor_state = state_tree
)
paths <- list()
for (name in names(plots)) {
  png_path <- file.path(output_dir, paste0("goncalves_tbs_", name, "_radial_tree_ggtree.png"))
  pdf_path <- file.path(output_dir, paste0("goncalves_tbs_", name, "_radial_tree_ggtree.pdf"))
  ggsave(png_path, plots[[name]], width = 15, height = 15, dpi = 320, bg = "white", limitsize = FALSE)
  ggsave(pdf_path, plots[[name]], width = 15, height = 15, bg = "white", limitsize = FALSE)
  paths[[length(paths) + 1L]] <- data.frame(plot = name, png = png_path, pdf = pdf_path)
}

panel <- (population_tree | fraction_tree) / state_tree +
  plot_annotation(
    title = "Goncalves fetal pancreas progenitor trees",
    subtitle = "Adaptive-diffusion TBS whole radial tree with progenitor-relevant nodes labeled",
    theme = theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5)
    )
  )
panel_png <- file.path(output_dir, "goncalves_tbs_progenitor_tree_panel_ggtree.png")
panel_pdf <- file.path(output_dir, "goncalves_tbs_progenitor_tree_panel_ggtree.pdf")
ggsave(panel_png, panel, width = 28, height = 28, dpi = 300, bg = "white", limitsize = FALSE)
ggsave(panel_pdf, panel, width = 28, height = 28, bg = "white", limitsize = FALSE)

cluster_tree_png <- file.path(output_dir, "goncalves_tbs_cluster_radial_tree_ggtree.png")
cluster_tree_pdf <- file.path(output_dir, "goncalves_tbs_cluster_radial_tree_ggtree.pdf")
ggsave(cluster_tree_png, cluster_tree, width = 15, height = 15, dpi = 320, bg = "white", limitsize = FALSE)
ggsave(cluster_tree_pdf, cluster_tree, width = 15, height = 15, bg = "white", limitsize = FALSE)

relation_umap_grid <- (
  (umap_cluster | umap_population) /
    (umap_fraction | umap_state)
) +
  plot_annotation(
    title = "Goncalves cells: TBS clusters versus progenitor context",
    subtitle = "All four UMAPs use identical coordinates; only the coloring changes",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, hjust = 0.5)
    )
  )
relation_umap_grid_png <- file.path(output_dir, "goncalves_tbs_relation_umap_grid_ggtree.png")
relation_umap_grid_pdf <- file.path(output_dir, "goncalves_tbs_relation_umap_grid_ggtree.pdf")
ggsave(relation_umap_grid_png, relation_umap_grid, width = 22, height = 18, dpi = 240, bg = "white", limitsize = FALSE)
ggsave(relation_umap_grid_pdf, relation_umap_grid, width = 22, height = 18, bg = "white", limitsize = FALSE)

relation_tree_grid <- (
  (cluster_tree | population_tree) /
    (fraction_tree | state_tree)
) +
  plot_annotation(
    title = "Goncalves adaptive-diffusion TBS tree: cluster result versus progenitor context",
    subtitle = "All four panels use the same TBS tree; only the branch coloring changes",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, hjust = 0.5)
    )
  )
relation_tree_grid_png <- file.path(output_dir, "goncalves_tbs_relation_tree_grid_ggtree.png")
relation_tree_grid_pdf <- file.path(output_dir, "goncalves_tbs_relation_tree_grid_ggtree.pdf")
ggsave(relation_tree_grid_png, relation_tree_grid, width = 22, height = 22, dpi = 240, bg = "white", limitsize = FALSE)
ggsave(relation_tree_grid_pdf, relation_tree_grid, width = 22, height = 22, bg = "white", limitsize = FALSE)

umap_tree_panel <- (
  (umap_population | population_tree) /
    (umap_fraction | fraction_tree) /
    (umap_state | state_tree)
) +
  plot_layout(widths = c(1.08, 0.92), guides = "collect") +
  plot_annotation(
    title = "Goncalves fetal pancreas progenitor UMAPs and full TBS trees",
    subtitle = "Each row pairs one UMAP with the matching whole radial adaptive-diffusion TBS tree using the same coloring rule",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, hjust = 0.5)
    )
  )
umap_tree_png <- file.path(output_dir, "goncalves_tbs_progenitor_umap_tree_panel_wide_ggtree.png")
umap_tree_pdf <- file.path(output_dir, "goncalves_tbs_progenitor_umap_tree_panel_wide_ggtree.pdf")
ggsave(umap_tree_png, umap_tree_panel, width = pdf_width, height = pdf_height, dpi = png_dpi, bg = "white", limitsize = FALSE)
ggsave(umap_tree_pdf, umap_tree_panel, width = pdf_width, height = pdf_height, bg = "white", limitsize = FALSE)

cluster_page_plot <- (umap_cluster | cluster_tree) +
  plot_layout(widths = c(1.08, 0.92), guides = "collect") +
  plot_annotation(
    title = "Goncalves TBS clustering result on the adaptive-diffusion tree",
    subtitle = "Same cells on UMAP and same full tree; this is the clustering result used for the progenitor analysis below",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 13, hjust = 0.5)
    )
  )
cluster_page_png <- file.path(output_dir, "goncalves_tbs_cluster_umap_tree_page_ggtree.png")
ggsave(cluster_page_png, cluster_page_plot, width = page_width, height = page_height, dpi = png_dpi, bg = "white", limitsize = FALSE)

umap_tree_page_plots <- list(
  population = (umap_population | population_tree) +
    plot_layout(widths = c(1.08, 0.92), guides = "collect") +
    plot_annotation(
      title = "Goncalves fetal population UMAP and full TBS tree",
      subtitle = "Cell UMAP and matching whole radial adaptive-diffusion TBS tree",
      theme = theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5)
      )
    ),
  progenitor_fraction = (umap_fraction | fraction_tree) +
    plot_layout(widths = c(1.08, 0.92), guides = "collect") +
    plot_annotation(
      title = "Goncalves progenitor-fraction UMAP and full TBS tree",
      subtitle = "Cell-level trunk/tip/proliferating membership beside descendant fraction on the whole tree",
      theme = theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5)
      )
    ),
  progenitor_state = (umap_state | state_tree) +
    plot_layout(widths = c(1.08, 0.92), guides = "collect") +
    plot_annotation(
      title = "Goncalves progenitor-state UMAP and full TBS tree",
      subtitle = "Terminal cells and internal tree branches shown with progenitor-state interpretation",
      theme = theme(
        plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5)
      )
    )
)

page_rows <- list()
for (name in names(umap_tree_page_plots)) {
  page_png <- file.path(output_dir, paste0("goncalves_tbs_", name, "_umap_tree_page_ggtree.png"))
  ggsave(page_png, umap_tree_page_plots[[name]], width = page_width, height = page_height, dpi = png_dpi, bg = "white", limitsize = FALSE)
  page_rows[[length(page_rows) + 1L]] <- data.frame(
    plot = paste0("umap_tree_page_", name),
    png = page_png,
    pdf = NA_character_,
    stringsAsFactors = FALSE
  )
}
umap_tree_pages_pdf <- file.path(output_dir, "goncalves_tbs_progenitor_umap_tree_pages_ggtree.pdf")
grDevices::pdf(umap_tree_pages_pdf, width = page_width, height = page_height, bg = "white", onefile = TRUE, paper = "special")
for (name in names(umap_tree_page_plots)) {
  print(umap_tree_page_plots[[name]])
}
invisible(grDevices::dev.off())

relation_pages_pdf <- file.path(output_dir, "goncalves_tbs_relation_umap_tree_pages_ggtree.pdf")
relation_page_plots <- c(list(cluster = cluster_page_plot), umap_tree_page_plots)
grDevices::pdf(relation_pages_pdf, width = page_width, height = page_height, bg = "white", onefile = TRUE, paper = "special")
for (name in names(relation_page_plots)) {
  print(relation_page_plots[[name]])
}
invisible(grDevices::dev.off())

manifest <- do.call(rbind, paths)
manifest <- rbind(
  manifest,
  data.frame(plot = "cluster_tree", png = cluster_tree_png, pdf = cluster_tree_pdf),
  data.frame(plot = "tree_panel", png = panel_png, pdf = panel_pdf),
  data.frame(plot = "relation_umap_grid", png = relation_umap_grid_png, pdf = relation_umap_grid_pdf),
  data.frame(plot = "relation_tree_grid", png = relation_tree_grid_png, pdf = relation_tree_grid_pdf),
  data.frame(plot = "umap_tree_panel_wide", png = umap_tree_png, pdf = umap_tree_pdf),
  data.frame(plot = "umap_tree_page_cluster", png = cluster_page_png, pdf = NA_character_),
  do.call(rbind, page_rows),
  data.frame(plot = "umap_tree_pages", png = NA_character_, pdf = umap_tree_pages_pdf),
  data.frame(plot = "relation_umap_tree_pages", png = NA_character_, pdf = relation_pages_pdf)
)
manifest$tips <- length(phy$tip.label)
manifest$internal_nodes <- phy$Nnode
manifest$edge_csv <- file.path(output_dir, edge_csv)
manifest$labeled_nodes <- paste(key_nodes, collapse = ",")
manifest$pdf_width <- c(
  rep(15, length(paths)),
  15,
  28,
  22,
  22,
  pdf_width,
  page_width,
  rep(page_width, length(page_rows) + 2L)
)
manifest$pdf_height <- c(
  rep(15, length(paths)),
  15,
  28,
  18,
  22,
  pdf_height,
  page_height,
  rep(page_height, length(page_rows) + 2L)
)
manifest$png_dpi <- c(
  rep(320, length(paths)),
  320,
  300,
  240,
  240,
  png_dpi,
  png_dpi,
  rep(png_dpi, length(page_rows)),
  NA,
  NA
)
write.csv(
  manifest,
  file.path(output_dir, "goncalves_tbs_progenitor_tree_outputs.csv"),
  row.names = FALSE
)
print(manifest)
