#!/usr/bin/env python3
"""Validate coherence and perturbation stability of one cosine subspace split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the TF-IDF cosine eigen-subspace components 2-5 split using "
            "GO enrichment coherence and perturbation stability."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Blob TSV matrix.")
    parser.add_argument("--assignments", type=Path, required=True, help="Reference split CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-local", type=float, default=float(config.EDGE_ALPHA))
    parser.add_argument("--sibling-alpha", type=float, default=float(config.SIBLING_ALPHA))
    parser.add_argument("--feature-runs", type=int, default=8)
    parser.add_argument("--gene-runs", type=int, default=8)
    parser.add_argument("--feature-fractions", nargs="+", type=float, default=[0.8, 0.6])
    parser.add_argument("--gene-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data = data.apply(pd.to_numeric, errors="raise").astype(float)
    data = data.loc[:, data.sum(axis=0) > 0]
    if (data.sum(axis=1) == 0).any():
        zero = data.index[data.sum(axis=1) == 0].tolist()[:10]
        raise ValueError(f"Rows with no GO terms: {zero}")
    return data


def load_assignments(path: Path, index: pd.Index) -> pd.Series:
    assignments = pd.read_csv(path)
    if "Unnamed: 0" in assignments.columns:
        assignments = assignments.rename(columns={"Unnamed: 0": "gene"})
    if "gene" not in assignments.columns:
        assignments = assignments.rename(columns={assignments.columns[0]: "gene"})
    if "cluster_id" not in assignments.columns:
        candidates = [c for c in assignments.columns if "cluster" in c.lower()]
        if not candidates:
            raise ValueError(f"No cluster column in {path}")
        assignments = assignments.rename(columns={candidates[0]: "cluster_id"})
    assignments["gene"] = assignments["gene"].astype(str)
    return assignments.set_index("gene").loc[index, "cluster_id"].astype(int)


def tfidf_cosine_components_2_5(data: pd.DataFrame) -> np.ndarray:
    values = data.to_numpy(dtype=float)
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True).fit_transform(values).toarray()
    row_normed = normalize(tfidf, norm="l2", axis=1)
    operator = row_normed @ row_normed.T
    eigvals, eigvecs = np.linalg.eigh(operator)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    if len(eigvals) < 5:
        raise ValueError(f"Need at least 5 positive eigenvalues, got {len(eigvals)}")
    coords = eigvecs[:, 1:5] * np.sqrt(np.maximum(eigvals[1:5], 0.0))
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def run_subspace_split(
    data: pd.DataFrame,
    *,
    alpha_local: float,
    sibling_alpha: float,
) -> pd.Series:
    coords = tfidf_cosine_components_2_5(data)
    distances = pdist(coords, metric="euclidean")
    if not np.isfinite(distances).all() or np.allclose(distances, 0):
        raise ValueError("Degenerate subspace distances")
    tree = PosetTree.from_linkage(linkage(distances, method="average"), leaf_names=data.index.tolist())
    decomposition = tree.decompose(
        leaf_data=data,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
    )
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    return assignments["cluster_id"].astype(int)


def mean_within_cosine(data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(
        data.to_numpy(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(labels.unique()):
        idx = np.where(labels.to_numpy() == cluster_id)[0]
        size = len(idx)
        if size < 2:
            mean_sim = np.nan
        else:
            gram = (tfidf[idx] @ tfidf[idx].T).toarray()
            mean_sim = float((gram.sum() - np.trace(gram)) / (size * (size - 1)))
        rows.append({"cluster_id": int(cluster_id), "cluster_size": size, "mean_within_tfidf_cosine": mean_sim})
    return pd.DataFrame(rows)


def enrichment_coherence(data: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = data.columns
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        cluster_size = int(mask.sum())
        rest_size = int((~mask).sum())
        cluster = data.loc[mask]
        rest = data.loc[~mask]
        a = cluster.sum(axis=0).astype(int)
        c = rest.sum(axis=0).astype(int)
        b = cluster_size - a
        d = rest_size - c
        p_values: list[float] = []
        odds_ratios: list[float] = []
        for term in terms:
            odds, p_value = fisher_exact(
                [[int(a[term]), int(b[term])], [int(c[term]), int(d[term])]],
                alternative="greater",
            )
            odds_ratios.append(float(odds))
            p_values.append(float(p_value))
        q_values = multipletests(p_values, method="fdr_bh")[1]
        cluster_prev = a / cluster_size
        rest_prev = c / max(rest_size, 1)
        frame = pd.DataFrame(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "term": terms,
                "cluster_prevalence": cluster_prev.to_numpy(),
                "rest_prevalence": rest_prev.to_numpy(),
                "prevalence_delta": (cluster_prev - rest_prev).to_numpy(),
                "odds_ratio": odds_ratios,
                "p_value": p_values,
                "q_value": q_values,
                "cluster_count": a.to_numpy(),
                "rest_count": c.to_numpy(),
            }
        ).sort_values(["q_value", "prevalence_delta"], ascending=[True, False])
        rows.append(frame.head(100))
        sig = frame[frame["q_value"] < 0.05]
        top = frame.iloc[0]
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "n_significant_go_terms_q05": int(len(sig)),
                "min_q_value": float(top["q_value"]),
                "top_term": str(top["term"]),
                "top_term_cluster_prevalence": float(top["cluster_prevalence"]),
                "top_term_rest_prevalence": float(top["rest_prevalence"]),
                "top_term_prevalence_delta": float(top["prevalence_delta"]),
                "coherent_by_rule": bool(
                    cluster_size >= 3
                    and len(sig) >= 3
                    and float(top["q_value"]) < 0.05
                    and float(top["prevalence_delta"]) >= 0.25
                ),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def best_cluster_jaccards(
    reference: pd.Series,
    candidate: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aligned = reference.index.intersection(candidate.index)
    ref = reference.loc[aligned]
    cand = candidate.loc[aligned]
    for cluster_id in sorted(reference.unique()):
        full_ref_members = set(reference.index[reference == cluster_id])
        available_ref_members = set(ref.index[ref == cluster_id])
        if not available_ref_members:
            rows.append(
                {
                    "reference_cluster_id": int(cluster_id),
                    "reference_cluster_size": len(full_ref_members),
                    "available_reference_members": 0,
                    "best_jaccard": np.nan,
                    "best_candidate_cluster": np.nan,
                }
            )
            continue
        best_jaccard = -1.0
        best_candidate = None
        for candidate_id in sorted(cand.unique()):
            cand_members = set(cand.index[cand == candidate_id])
            union = available_ref_members | cand_members
            jaccard = len(available_ref_members & cand_members) / len(union) if union else np.nan
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_candidate = candidate_id
        rows.append(
            {
                "reference_cluster_id": int(cluster_id),
                "reference_cluster_size": len(full_ref_members),
                "available_reference_members": len(available_ref_members),
                "best_jaccard": float(best_jaccard),
                "best_candidate_cluster": int(best_candidate) if best_candidate is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_perturbations(
    data: pd.DataFrame,
    reference: pd.Series,
    *,
    out: Path,
    alpha_local: float,
    sibling_alpha: float,
    feature_runs: int,
    gene_runs: int,
    feature_fractions: list[float],
    gene_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    run_rows: list[dict[str, object]] = []
    jaccard_frames: list[pd.DataFrame] = []
    assignment_dir = out / "perturbation_assignments"
    assignment_dir.mkdir(exist_ok=True)

    run_specs: list[tuple[str, float, int]] = []
    for fraction in feature_fractions:
        for i in range(feature_runs):
            run_specs.append(("feature_subsample", fraction, i))
    for i in range(gene_runs):
        run_specs.append(("gene_subsample", gene_fraction, i))

    for run_number, (kind, fraction, replicate) in enumerate(run_specs, start=1):
        run_id = f"{kind}_{fraction:.2f}_rep{replicate:02d}"
        try:
            if kind == "feature_subsample":
                n_features = max(10, int(round(data.shape[1] * fraction)))
                cols = rng.choice(data.columns.to_numpy(), size=n_features, replace=False)
                perturbed = data.loc[:, cols]
                # Rarely, a gene can lose all active terms after feature subsampling.
                if (perturbed.sum(axis=1) == 0).any():
                    lost = int((perturbed.sum(axis=1) == 0).sum())
                    raise ValueError(f"{lost} genes lost all active terms")
            else:
                n_genes = max(20, int(round(data.shape[0] * fraction)))
                genes = rng.choice(data.index.to_numpy(), size=n_genes, replace=False)
                perturbed = data.loc[genes].copy()

            labels = run_subspace_split(
                perturbed,
                alpha_local=alpha_local,
                sibling_alpha=sibling_alpha,
            )
            labels.to_frame("cluster_id").to_csv(assignment_dir / f"{run_id}_assignments.csv")

            shared = reference.index.intersection(labels.index)
            ari = adjusted_rand_score(reference.loc[shared], labels.loc[shared])
            nmi = normalized_mutual_info_score(reference.loc[shared], labels.loc[shared])
            sizes = labels.value_counts()
            run_rows.append(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "fraction": float(fraction),
                    "replicate": int(replicate),
                    "status": "ok",
                    "n_genes": int(perturbed.shape[0]),
                    "n_features": int(perturbed.shape[1]),
                    "n_clusters": int(labels.nunique()),
                    "largest_cluster": int(sizes.max()),
                    "largest_cluster_fraction": float(sizes.max() / len(labels)),
                    "ari_vs_reference": float(ari),
                    "nmi_vs_reference": float(nmi),
                }
            )
            jacc = best_cluster_jaccards(reference, labels)
            jacc.insert(0, "run_id", run_id)
            jacc.insert(1, "kind", kind)
            jacc.insert(2, "fraction", float(fraction))
            jaccard_frames.append(jacc)
        except Exception as exc:
            run_rows.append(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "fraction": float(fraction),
                    "replicate": int(replicate),
                    "status": "failed",
                    "error": repr(exc),
                }
            )
        print(f"[{run_number}/{len(run_specs)}] {run_id}: {run_rows[-1]['status']}", flush=True)

    run_df = pd.DataFrame(run_rows)
    jaccard_df = pd.concat(jaccard_frames, ignore_index=True) if jaccard_frames else pd.DataFrame()
    return run_df, jaccard_df


def write_plots(out: Path, coherence: pd.DataFrame, stability: pd.DataFrame, jaccard: pd.DataFrame) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)

    coh = coherence.copy()
    coh["neg_log10_min_q"] = -np.log10(coh["min_q_value"].clip(lower=1e-300))
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=coh, x="cluster_id", y="n_significant_go_terms_q05", hue="coherent_by_rule", dodge=False)
    plt.xlabel("20-cluster split cluster ID")
    plt.ylabel("Significant GO terms (FDR q < 0.05)")
    plt.title("GO coherence by cluster")
    plt.tight_layout()
    plt.savefig(plot_dir / "go_coherence_significant_terms_by_cluster.png", dpi=180)
    plt.close()

    ok = stability[stability["status"].eq("ok")].copy()
    if not ok.empty:
        plt.figure(figsize=(8.5, 5))
        sns.boxplot(data=ok, x="kind", y="ari_vs_reference", hue="fraction")
        sns.stripplot(data=ok, x="kind", y="ari_vs_reference", color="black", alpha=0.45, dodge=True)
        plt.xlabel("Perturbation")
        plt.ylabel("ARI vs reference 20-cluster split")
        plt.title("Perturbation stability")
        plt.tight_layout()
        plt.savefig(plot_dir / "perturbation_stability_ari.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8.5, 5))
        sns.boxplot(data=ok, x="kind", y="largest_cluster_fraction", hue="fraction")
        sns.stripplot(data=ok, x="kind", y="largest_cluster_fraction", color="black", alpha=0.45, dodge=True)
        plt.xlabel("Perturbation")
        plt.ylabel("Largest cluster fraction after rerun")
        plt.title("Core size under perturbation")
        plt.tight_layout()
        plt.savefig(plot_dir / "perturbation_largest_core_fraction.png", dpi=180)
        plt.close()

    if not jaccard.empty:
        jsummary = (
            jaccard.groupby("reference_cluster_id", as_index=False)
            .agg(
                reference_cluster_size=("reference_cluster_size", "first"),
                mean_best_jaccard=("best_jaccard", "mean"),
                median_best_jaccard=("best_jaccard", "median"),
            )
            .sort_values("reference_cluster_id")
        )
        plt.figure(figsize=(11, 5.5))
        sns.barplot(data=jsummary, x="reference_cluster_id", y="mean_best_jaccard", color="#4C78A8")
        plt.xlabel("Reference cluster ID")
        plt.ylabel("Mean best-match Jaccard under perturbation")
        plt.title("Cluster-level stability")
        plt.tight_layout()
        plt.savefig(plot_dir / "cluster_level_stability_jaccard.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    started = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    data = load_matrix(args.input)
    labels = load_assignments(args.assignments, data.index)

    enrichment, coherence = enrichment_coherence(data, labels)
    within = mean_within_cosine(data, labels)
    coherence = coherence.merge(within, on=["cluster_id", "cluster_size"], how="left")

    enrichment.to_csv(out / "go_enrichment_top100_per_cluster.csv", index=False)
    coherence.to_csv(out / "cluster_biological_coherence_summary.csv", index=False)

    stability, jaccard = run_perturbations(
        data,
        labels,
        out=out,
        alpha_local=args.alpha_local,
        sibling_alpha=args.sibling_alpha,
        feature_runs=args.feature_runs,
        gene_runs=args.gene_runs,
        feature_fractions=args.feature_fractions,
        gene_fraction=args.gene_fraction,
        random_seed=args.random_seed,
    )
    stability.to_csv(out / "perturbation_stability_summary.csv", index=False)
    jaccard.to_csv(out / "perturbation_cluster_jaccard.csv", index=False)

    write_plots(out, coherence, stability, jaccard)

    ok = stability[stability["status"].eq("ok")]
    coherent_count = int(coherence["coherent_by_rule"].sum())
    non_singleton = int((coherence["cluster_size"] >= 3).sum())
    stable_by_kind = (
        ok.groupby(["kind", "fraction"], as_index=False)
        .agg(
            successful_runs=("run_id", "count"),
            median_ari=("ari_vs_reference", "median"),
            median_nmi=("nmi_vs_reference", "median"),
            median_largest_cluster_fraction=("largest_cluster_fraction", "median"),
            median_n_clusters=("n_clusters", "median"),
        )
        if not ok.empty
        else pd.DataFrame()
    )
    stable_by_kind.to_csv(out / "perturbation_stability_by_kind.csv", index=False)

    if not jaccard.empty:
        cluster_stability = (
            jaccard.groupby("reference_cluster_id", as_index=False)
            .agg(
                reference_cluster_size=("reference_cluster_size", "first"),
                mean_best_jaccard=("best_jaccard", "mean"),
                median_best_jaccard=("best_jaccard", "median"),
            )
            .sort_values(["mean_best_jaccard", "reference_cluster_size"], ascending=[False, False])
        )
    else:
        cluster_stability = pd.DataFrame()
    cluster_stability.to_csv(out / "cluster_level_stability_summary.csv", index=False)

    readme = [
        "# Cosine Subspace Split Validation",
        "",
        "Reference split: TF-IDF cosine components 2-5 tree with the project split method.",
        "",
        f"Genes: `{data.shape[0]}`",
        f"Non-empty deduplicated GO terms: `{data.shape[1]}`",
        f"Reference clusters: `{labels.nunique()}`",
        "",
        "## Biological Coherence",
        "",
        f"Clusters passing the coherence rule: `{coherent_count}` of `{len(coherence)}`.",
        f"Non-singleton/non-tiny clusters (size >= 3): `{non_singleton}`.",
        "",
        "Coherence rule: cluster size >= 3, at least 3 GO terms with FDR q < 0.05,",
        "top GO-term q < 0.05, and top prevalence delta >= 0.25.",
        "",
        "## Perturbation Stability",
        "",
        stable_by_kind.to_string(index=False) if not stable_by_kind.empty else "No successful perturbation runs.",
        "",
        "## Most Stable Reference Clusters",
        "",
        cluster_stability.head(10).to_string(index=False) if not cluster_stability.empty else "No cluster-level stability results.",
        "",
        f"Runtime seconds: `{time.time() - started:.2f}`",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme))

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            manifest.append(f"{path.relative_to(out).as_posix()}\t{path.stat().st_size} bytes")
    (out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
