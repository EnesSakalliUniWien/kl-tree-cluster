"""Geometry helpers for benchmark case recipes."""

from __future__ import annotations

from collections.abc import Mapping


def case_recipe_geometry(case: Mapping[str, object]) -> tuple[int, int]:
    """Return generated sample/feature counts from the canonical case recipe."""
    generator = str(case["generator"])
    if generator == "sbm":
        n_nodes = sum(int(size) for size in case["sizes"])
        return n_nodes, n_nodes
    if generator == "phylogenetic":
        return int(case["n_taxa"]) * int(case["samples_per_taxon"]), int(
            case["n_features"]
        ) * int(case["n_categories"])
    if generator == "temporal_evolution":
        return int(case["n_time_points"]) * int(case["samples_per_time"]), int(
            case["n_features"]
        ) * int(case["n_categories"])
    if generator in {"categorical", "blobs_quantile"}:
        return int(case["n_samples"]), int(case["n_features"]) * int(case["n_categories"])
    if generator == "binary":
        return int(case["n_samples"]), int(case["n_features"]) + int(case["noise_features"])
    if generator in {
        "blobs",
        "blobs_continuous",
        "dimensional_gaussian",
        "dimensional_gaussian_continuous",
        "gaussian_outliers",
        "gaussian_outliers_continuous",
    }:
        return int(case["n_samples"]), int(case["n_features"])
    raise ValueError(f"Unknown benchmark case generator {generator!r}.")


def case_recipe_cluster_count(case: Mapping[str, object]) -> int:
    """Return the true cluster count declared by a benchmark case recipe."""
    if str(case["generator"]) == "sbm":
        return len(case["sizes"])
    if "n_clusters" in case:
        return int(case["n_clusters"])
    if "n_taxa" in case:
        return int(case["n_taxa"])
    raise ValueError(f"Case {case['name']!r} does not declare a cluster count.")


__all__ = [
    "case_recipe_cluster_count",
    "case_recipe_geometry",
]
