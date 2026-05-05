"""Real-data benchmark cases loaded from pre-existing files."""

REAL_DATA_CASES = {
    "real_data": [
        {
            "name": "feature_matrix_go_terms",
            "generator": "preloaded",
            "file_path": "data/feature_matrices/feature_matrix.tsv",
            "sep": "\t",
            "n_clusters": None,  # no ground truth
        },
    ],
}
