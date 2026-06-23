# tfidf / adaptive_modes_11_21

Run ID: `current__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_11_21`
Status: `ok`
Assignment source: `accepted_tbs`
Specificity-aware rank: `2`
Linkage leaves: `150`
Assigned genes: `150`
Clusters exported: `5`
Gene memberships exported: `150`

Files:
- `cluster_roster.csv`: one row per cluster with complete member-gene lists and local GO annotations.
- `cluster_annotations.csv`: cluster-level annotation fields without the long member-gene cell.
- `gene_membership.csv`: one row per gene in this subspace with its cluster annotation context.
- `accepted_tbs_cluster_assignments.csv`: accepted final TBS assignments when the TBS gate completed.
- `diagnostic_linkage_cluster_assignments.csv`: diagnostic linkage-cut assignments when the TBS gate failed.
- `radial_tree_clusters.png`: radial hierarchy colored by the recorded cluster id when linkage and assignments are available.
- `radial_tree_clusters_compact.png`: compact radial hierarchy used inside the annotation PDF.
- `full_space_embedding_clusters.png`: full feature-matrix PCA coordinates colored by this subspace's cluster ids.
- `full_space_embedding_cluster_coordinates.csv`: full-space PCA coordinates joined to this subspace's cluster ids.
- `subspace_embedding_clusters.png`: regenerated subspace-coordinate embedding colored by this subspace's cluster ids.
- `subspace_embedding_cluster_coordinates.csv`: subspace coordinates joined to this subspace's cluster ids.
- `tree_distance_embedding_clusters.png`: tree-distance MDS embedding from the saved linkage matrix colored by cluster id.
- `tree_distance_embedding_cluster_coordinates.csv`: tree-distance MDS coordinates joined to cluster ids.
- `source_artifacts/`: copied source tables and plots from the canonical subspace experiment.
- `source_artifacts_manifest.csv`: copied-source inventory.
- `subspace_status.csv`: one-row status record.

Failure status:

`nan`
