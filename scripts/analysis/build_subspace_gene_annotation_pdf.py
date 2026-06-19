#!/usr/bin/env python3
"""Build systematic subspace PDFs with cluster-level gene annotations."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

GO_ID_RE = re.compile(r"(GO:\d{7})")


@dataclass
class ClusterAnnotation:
    run_id: str
    display_rank: int
    weighting: str
    block_name: str
    assignment_source: str
    cluster_id: int
    cluster_size: int
    coherent_by_rule: bool
    n_significant_terms_q05: int
    min_q_value: float
    top_term: str
    top_go_id: str
    top_term_prevalence_delta: float
    representative_genes: list[str]
    representative_gene_proteins: list[str]
    local_top_terms: list[str]
    local_top_go_ids: list[str]
    representative_gene_annotations: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--top-subspaces", type=int, default=0, help="0 means all completed subspaces.")
    parser.add_argument("--top-clusters-per-subspace", type=int, default=8)
    parser.add_argument("--genes-per-cluster", type=int, default=8)
    parser.add_argument("--terms-per-cluster", type=int, default=4)
    parser.add_argument("--life-science-term-limit", type=int, default=80)
    parser.add_argument("--life-science-gene-limit", type=int, default=60)
    parser.add_argument("--skip-life-science-lookups", action="store_true")
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def subspace_dir_name(row: pd.Series) -> str:
    rank_value = row.get("_rank", row.get("specificity_aware_rank", row.get("display_rank", -1)))
    try:
        rank = int(float(rank_value))
    except (TypeError, ValueError):
        rank = -1
    status = str(row.get("status", ""))
    if rank >= 0:
        prefix = f"rank{rank:02d}"
    elif status and status != "ok":
        try:
            block_id = int(float(row.get("block_id", 0)))
        except (TypeError, ValueError):
            block_id = 0
        prefix = f"failed{block_id:02d}"
    else:
        prefix = "rankNA"
    return f"{prefix}_{safe_name(row.get('weighting', ''))}_{safe_name(row.get('block_name', ''))}".strip("_")


def attach_organized_plot_paths(ranking: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ranking = ranking.copy()
    radial_paths: list[str] = []
    full_space_paths: list[str] = []
    subspace_paths: list[str] = []
    for _, row in ranking.iterrows():
        subspace_dir = output_dir / "subspaces" / subspace_dir_name(row)
        compact_path = subspace_dir / "radial_tree_clusters_compact.png"
        full_path = subspace_dir / "radial_tree_clusters.png"
        if compact_path.exists():
            radial_paths.append(str(compact_path))
        elif full_path.exists():
            radial_paths.append(str(full_path))
        else:
            radial_paths.append("")
        full_space_path = subspace_dir / "full_space_embedding_clusters.png"
        full_space_paths.append(str(full_space_path) if full_space_path.exists() else "")
        generated_subspace_path = subspace_dir / "subspace_embedding_clusters.png"
        subspace_paths.append(str(generated_subspace_path) if generated_subspace_path.exists() else "")
    ranking["radial_tree_clusters_png"] = radial_paths
    ranking["full_space_embedding_clusters_png"] = full_space_paths
    ranking["organized_subspace_embedding_clusters_png"] = subspace_paths
    return ranking


def attach_roster_status(ranking: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    status_path = output_dir / "subspace_cluster_status.csv"
    if not status_path.exists():
        return ranking
    status = pd.read_csv(status_path)
    keep = [
        "run_id",
        "assignment_source",
        "n_genes",
        "n_unique_genes",
        "linkage_leaves",
        "n_clusters",
        "organized_subspace_dir",
        "radial_tree_clusters_png",
        "radial_tree_clusters_compact_png",
        "full_space_embedding_clusters_png",
        "subspace_embedding_clusters_png",
        "subspace_embedding_cluster_coordinates",
        "tree_distance_embedding_clusters_png",
        "tree_distance_embedding_cluster_coordinates",
    ]
    keep = [col for col in keep if col in status.columns]
    merged = ranking.drop(
        columns=[col for col in keep if col != "run_id" and col in ranking.columns],
        errors="ignore",
    ).merge(status[keep], on="run_id", how="left")
    if "subspace_embedding_clusters_png" in merged.columns:
        merged["organized_subspace_embedding_clusters_png"] = merged["subspace_embedding_clusters_png"].fillna("")
    return merged


def dataset_slug(input_path: Path, label: str | None = None) -> str:
    raw = label or input_path.stem
    if raw.startswith("feature_matrix_"):
        raw = raw[len("feature_matrix_") :]
    return safe_name(raw).strip("_").lower() or "feature_matrix"


def parse_go_term(value: object) -> tuple[str, str]:
    text = str(value)
    match = GO_ID_RE.search(text)
    go_id = match.group(1) if match else ""
    term = text.replace(f"({go_id})", "").strip() if go_id else text
    return term, go_id


def truncate(value: object, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: max(width - 3, 0)] + "..."


def wrap(value: object, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width)) or ""


def load_binary_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)
    data = data.apply(pd.to_numeric, errors="raise")
    values = data.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{path} contains non-binary values.")
    return data.astype(int)


def resolve_path(value: object, experiment_dir: Path) -> Path:
    path = Path(str(value))
    if path.exists() or path.is_absolute():
        return path
    return experiment_dir / path


def load_artifact_index(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "artifact_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    rank_col = "specificity_aware_rank" if "specificity_aware_rank" in frame.columns else "display_rank"
    frame["_rank"] = pd.to_numeric(frame[rank_col], errors="coerce")
    if "run_id" not in frame.columns:
        if "method_run_id" in frame.columns:
            frame["run_id"] = frame["method_run_id"].astype(str)
        else:
            frame["run_id"] = (
                "current__adaptive_diffusion_cosine_subspace__"
                + frame["weighting"].astype(str)
                + "__"
                + frame["block_name"].astype(str)
            )
    return frame.sort_values(["_rank", "weighting", "block_name"], na_position="last").reset_index(drop=True)


def local_top_terms_for_cluster(
    data: pd.DataFrame,
    cluster_genes: list[str],
    *,
    top_n: int,
) -> pd.DataFrame:
    cluster = data.loc[cluster_genes]
    rest = data.drop(index=cluster_genes)
    cluster_prev = cluster.mean(axis=0)
    rest_prev = rest.mean(axis=0) if len(rest) else pd.Series(0.0, index=data.columns)
    delta = cluster_prev - rest_prev
    frame = pd.DataFrame(
        {
            "column": data.columns,
            "cluster_prevalence": cluster_prev.to_numpy(),
            "rest_prevalence": rest_prev.to_numpy(),
            "prevalence_delta": delta.to_numpy(),
            "cluster_support": cluster.sum(axis=0).to_numpy(dtype=int),
        }
    )
    parsed = [parse_go_term(col) for col in frame["column"]]
    frame["go_term"] = [term for term, _go_id in parsed]
    frame["go_id"] = [go_id for _term, go_id in parsed]
    return frame.sort_values(
        ["prevalence_delta", "cluster_support", "cluster_prevalence"],
        ascending=[False, False, False],
    ).head(top_n)


def representative_genes_for_cluster(
    data: pd.DataFrame,
    cluster_genes: list[str],
    top_term_columns: list[str],
    *,
    top_n: int,
) -> list[str]:
    cluster = data.loc[cluster_genes]
    if top_term_columns:
        term_score = cluster[top_term_columns].sum(axis=1)
    else:
        term_score = pd.Series(0, index=cluster.index)
    burden = cluster.sum(axis=1)
    score = pd.DataFrame({"term_score": term_score, "active_terms": burden})
    return (
        score.sort_values(["term_score", "active_terms"], ascending=[False, False])
        .head(top_n)
        .index.astype(str)
        .tolist()
    )


def gene_annotation_blurbs(data: pd.DataFrame, genes: Iterable[str], term_columns: list[str]) -> list[str]:
    blurbs: list[str] = []
    for gene in genes:
        active = [col for col in term_columns if int(data.loc[gene, col]) == 1]
        if not active:
            active = data.loc[gene].sort_values(ascending=False).loc[lambda s: s > 0].index[:3].tolist()
        terms = [parse_go_term(col)[0] for col in active[:3]]
        blurbs.append(f"{gene}: " + "; ".join(terms))
    return blurbs


def call_rest(payload: dict[str, Any], *, timeout_sec: int = 45) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "scripts/rest_request.py"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout_sec + 5,
        check=False,
    )
    if not completed.stdout.strip():
        return {"ok": False, "error": {"code": "empty_output", "message": completed.stderr}}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": {"code": "bad_json", "message": str(exc)}}


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"quickgo": {}, "uniprot": {}, "warnings": []}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def quickgo_lookup(go_ids: list[str], cache: dict[str, Any], limit: int) -> None:
    missing = [go_id for go_id in go_ids[:limit] if go_id and go_id not in cache["quickgo"]]
    for ids in chunked(missing, 20):
        payload = {
            "base_url": "https://www.ebi.ac.uk/QuickGO/services",
            "path": "ontology/go/terms/" + ",".join(ids),
            "headers": {"Accept": "application/json"},
            "record_path": "results",
            "max_items": 20,
            "max_depth": 5,
            "timeout_sec": 30,
        }
        response = call_rest(payload, timeout_sec=35)
        if not response.get("ok"):
            cache["warnings"].append({"quickgo": ids, "response": response})
            continue
        for record in response.get("records", []):
            go_id = str(record.get("id", ""))
            definition = record.get("definition")
            if isinstance(definition, dict):
                definition_text = str(definition.get("text", ""))
            else:
                definition_text = str(definition or "")
            cache["quickgo"][go_id] = {
                "name": record.get("name", ""),
                "aspect": record.get("aspect", ""),
                "definition": definition_text,
            }


def uniprot_lookup(genes: list[str], cache: dict[str, Any], limit: int) -> None:
    missing = [
        gene
        for gene in genes[:limit]
        if gene
        and (
            gene not in cache["uniprot"]
            or not (
                cache["uniprot"].get(gene, {}).get("accession")
                or cache["uniprot"].get(gene, {}).get("protein_name")
            )
        )
    ]
    for gene in missing:
        payload = {
            "base_url": "https://rest.uniprot.org",
            "path": "uniprotkb/search",
            "params": {
                "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                "fields": "accession,id,gene_names,protein_name,organism_name",
                "size": 1,
                "format": "json",
            },
            "record_path": "results",
            "max_items": 20,
            "max_depth": 5,
            "timeout_sec": 30,
        }
        response = call_rest(payload, timeout_sec=35)
        if not response.get("ok") or not response.get("records"):
            payload["params"]["query"] = f"gene:{gene} AND organism_id:9606 AND reviewed:true"
            response = call_rest(payload, timeout_sec=35)
        if not response.get("ok"):
            cache["warnings"].append({"uniprot": gene, "response": response})
            cache["uniprot"][gene] = {"accession": "", "protein_name": "", "gene_names": ""}
            continue
        record = (response.get("records") or [{}])[0]
        genes_field = record.get("genes") or []
        gene_names: list[str] = []
        if isinstance(genes_field, list):
            for item in genes_field:
                if isinstance(item, dict):
                    primary = item.get("geneName")
                    if isinstance(primary, dict) and primary.get("value"):
                        gene_names.append(str(primary["value"]))
        protein = record.get("proteinDescription") or {}
        protein_name = ""
        if isinstance(protein, dict):
            recommended = protein.get("recommendedName") or {}
            if isinstance(recommended, dict):
                full_name = recommended.get("fullName") or {}
                if isinstance(full_name, dict):
                    protein_name = str(full_name.get("value", ""))
        cache["uniprot"][gene] = {
            "accession": record.get("primaryAccession", ""),
            "entry": record.get("uniProtkbId", ""),
            "protein_name": protein_name,
            "gene_names": ";".join(gene_names),
        }


def go_label(go_id: str, fallback: str, cache: dict[str, Any]) -> str:
    record = cache.get("quickgo", {}).get(go_id) or {}
    name = record.get("name") or fallback
    definition = record.get("definition") or ""
    if definition:
        return f"{name} ({go_id}): {truncate(definition, 140)}"
    return f"{name} ({go_id})" if go_id else fallback


def protein_label(gene: str, cache: dict[str, Any]) -> str:
    record = cache.get("uniprot", {}).get(gene) or {}
    protein = record.get("protein_name") or ""
    accession = record.get("accession") or ""
    if protein and accession:
        return f"{gene} [{accession}] {protein}"
    if protein:
        return f"{gene}: {protein}"
    return gene


def _image_panel(ax: plt.Axes, path: object, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left")
    if path is None or pd.isna(path) or not Path(str(path)).exists():
        ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=9)
        return
    image = plt.imread(str(path))
    ax.imshow(image)


def _first_existing_image_path(*paths: object) -> str:
    for path in paths:
        if path is None or pd.isna(path):
            continue
        candidate = Path(str(path))
        if candidate.exists():
            return str(candidate)
    return ""


def subspace_label(row: pd.Series) -> str:
    rank = row.get("_rank", row.get("display_rank", np.nan))
    rank_text = f"rank {int(rank)}" if pd.notna(rank) else "unranked"
    source = row.get("assignment_source", "")
    source_text = f" [{source}]" if source else ""
    return f"{rank_text} {row.get('weighting')} / {row.get('block_name')}{source_text}"


def build_cluster_annotations(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    experiment_dir: Path,
    top_clusters: int,
    genes_per_cluster: int,
    terms_per_cluster: int,
) -> list[ClusterAnnotation]:
    annotations: list[ClusterAnnotation] = []
    for row in ranking.itertuples(index=False):
        assignments_path = resolve_path(getattr(row, "cluster_assignments", ""), experiment_dir)
        coherence_path = resolve_path(getattr(row, "cluster_coherence", ""), experiment_dir)
        if not assignments_path.exists() or not coherence_path.exists():
            continue
        assignments = pd.read_csv(assignments_path)
        assignments["gene"] = assignments["gene"].astype(str)
        coherence = pd.read_csv(coherence_path)
        coherence = coherence.sort_values(
            ["coherent_by_rule", "top_term_prevalence_delta", "cluster_size"],
            ascending=[False, False, False],
        ).head(top_clusters)
        labels = assignments.set_index("gene")["cluster_id"].astype(int)
        for cluster in coherence.itertuples(index=False):
            cluster_id = int(cluster.cluster_id)
            cluster_genes = labels.index[labels.eq(cluster_id)].tolist()
            local_terms = local_top_terms_for_cluster(
                data,
                cluster_genes,
                top_n=terms_per_cluster,
            )
            top_columns = local_terms["column"].tolist()
            genes = representative_genes_for_cluster(
                data,
                cluster_genes,
                top_columns,
                top_n=genes_per_cluster,
            )
            top_term, top_go_id = parse_go_term(cluster.top_term)
            annotations.append(
                ClusterAnnotation(
                    run_id=str(row.run_id),
                    display_rank=int(getattr(row, "display_rank", getattr(row, "_rank", 0))),
                    weighting=str(row.weighting),
                    block_name=str(row.block_name),
                    assignment_source="accepted_kl",
                    cluster_id=cluster_id,
                    cluster_size=int(cluster.cluster_size),
                    coherent_by_rule=bool(cluster.coherent_by_rule),
                    n_significant_terms_q05=int(cluster.n_significant_terms_q05),
                    min_q_value=float(cluster.min_q_value),
                    top_term=top_term,
                    top_go_id=top_go_id,
                    top_term_prevalence_delta=float(cluster.top_term_prevalence_delta),
                    representative_genes=genes,
                    representative_gene_proteins=[],
                    local_top_terms=local_terms["go_term"].astype(str).tolist(),
                    local_top_go_ids=local_terms["go_id"].astype(str).tolist(),
                    representative_gene_annotations=gene_annotation_blurbs(data, genes, top_columns),
                )
            )
    return annotations


def split_semicolon(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item for item in str(value).split(";") if item]


def split_annotations(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(" | ") if item.strip()]


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_cluster_annotations_from_roster(
    roster_path: Path,
    *,
    top_clusters: int,
) -> list[ClusterAnnotation]:
    if not roster_path.exists():
        return []
    frame = pd.read_csv(roster_path)
    if frame.empty:
        return []
    frame["_coherent_sort"] = frame.get("coherent_by_rule", False).astype(str).str.lower().eq("true")
    frame["_delta_sort"] = pd.to_numeric(frame.get("top_term_prevalence_delta"), errors="coerce")
    frame["_size_sort"] = pd.to_numeric(frame.get("cluster_size"), errors="coerce")
    frame = frame.sort_values(
        ["run_id", "_coherent_sort", "_delta_sort", "_size_sort"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    selected = frame.groupby("run_id", sort=False).head(top_clusters)
    annotations: list[ClusterAnnotation] = []
    for row in selected.to_dict(orient="records"):
        annotations.append(
            ClusterAnnotation(
                run_id=str(row.get("run_id", "")),
                display_rank=int(float(row.get("display_rank", row.get("specificity_aware_rank", -1))))
                if pd.notna(row.get("display_rank", row.get("specificity_aware_rank", np.nan)))
                else -1,
                weighting=str(row.get("weighting", "")),
                block_name=str(row.get("block_name", "")),
                assignment_source=str(row.get("assignment_source", "")),
                cluster_id=to_int(row.get("cluster_id", 0)),
                cluster_size=to_int(row.get("cluster_size", 0)),
                coherent_by_rule=str(row.get("coherent_by_rule", "")).lower() == "true",
                n_significant_terms_q05=to_int(row.get("n_significant_terms_q05", 0)),
                min_q_value=to_float(row.get("min_q_value", math.nan)),
                top_term=str(row.get("top_term", "")) if pd.notna(row.get("top_term", "")) else "",
                top_go_id=str(row.get("top_go_id", "")) if pd.notna(row.get("top_go_id", "")) else "",
                top_term_prevalence_delta=to_float(row.get("top_term_prevalence_delta", math.nan)),
                representative_genes=split_semicolon(row.get("representative_genes", "")),
                representative_gene_proteins=split_semicolon(row.get("representative_gene_proteins", "")),
                local_top_terms=split_semicolon(row.get("local_top_terms", "")),
                local_top_go_ids=split_semicolon(row.get("local_top_go_ids", "")),
                representative_gene_annotations=split_annotations(row.get("representative_gene_annotations", "")),
            )
        )
    return annotations


def axis_go_ids(ranking: pd.DataFrame, *, experiment_dir: Path, per_subspace: int = 12) -> list[str]:
    ids: list[str] = []
    for row in ranking.itertuples(index=False):
        path = resolve_path(getattr(row, "axis_top_terms", ""), experiment_dir)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "selection" in frame.columns:
            frame = frame[frame["selection"].eq("top_absolute")]
        for go_id in frame.sort_values("abs_loading", ascending=False).get("go_id", []).head(per_subspace):
            if isinstance(go_id, str) and go_id.startswith("GO:"):
                ids.append(go_id)
    return ids


def write_csv_outputs(
    output_dir: Path,
    annotations: list[ClusterAnnotation],
    cache: dict[str, Any],
) -> None:
    rows = []
    for annotation in annotations:
        proteins = [protein_label(gene, cache) for gene in annotation.representative_genes]
        annotation.representative_gene_proteins = proteins
        rows.append(
            {
                "run_id": annotation.run_id,
                "display_rank": annotation.display_rank,
                "weighting": annotation.weighting,
                "block_name": annotation.block_name,
                "assignment_source": annotation.assignment_source,
                "cluster_id": annotation.cluster_id,
                "cluster_size": annotation.cluster_size,
                "coherent_by_rule": annotation.coherent_by_rule,
                "n_significant_terms_q05": annotation.n_significant_terms_q05,
                "min_q_value": annotation.min_q_value,
                "top_term": annotation.top_term,
                "top_go_id": annotation.top_go_id,
                "quickgo_top_term": go_label(annotation.top_go_id, annotation.top_term, cache),
                "top_term_prevalence_delta": annotation.top_term_prevalence_delta,
                "representative_genes": ";".join(annotation.representative_genes),
                "representative_gene_proteins": ";".join(proteins),
                "local_top_terms": ";".join(annotation.local_top_terms),
                "local_top_go_ids": ";".join(annotation.local_top_go_ids),
                "representative_gene_annotations": " | ".join(annotation.representative_gene_annotations),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "subspace_cluster_gene_annotation_summary.csv", index=False)
    quickgo_rows = [
        {"go_id": go_id, **record} for go_id, record in sorted(cache.get("quickgo", {}).items())
    ]
    pd.DataFrame(quickgo_rows).to_csv(output_dir / "quickgo_term_interpretations.csv", index=False)
    uniprot_rows = [
        {"gene": gene, **record} for gene, record in sorted(cache.get("uniprot", {}).items())
    ]
    pd.DataFrame(uniprot_rows).to_csv(output_dir / "uniprot_representative_gene_interpretations.csv", index=False)


def draw_subspace_plot_page(
    pdf: PdfPages,
    row: pd.Series,
    cache: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(18, 11), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1, 1])
    ax_tree = fig.add_subplot(grid[:, 0])
    ax_subspace = fig.add_subplot(grid[0, 1])
    ax_full = fig.add_subplot(grid[0, 2])
    ax_diffusion = fig.add_subplot(grid[1, 1])
    ax_terms = fig.add_subplot(grid[1, 2])

    _image_panel(
        ax_tree,
        _first_existing_image_path(
            row.get("radial_tree_clusters_png"),
            row.get("tree_subtree_clusters_png"),
            row.get("tree_dendrogram_png"),
        ),
        "Radial tree clusters",
    )
    _image_panel(
        ax_subspace,
        _first_existing_image_path(
            row.get("organized_subspace_embedding_clusters_png"),
            row.get("subspace_embedding_annotated_terms_png"),
            row.get("subspace_embedding_png"),
        ),
        "Subspace embedding by cluster",
    )
    _image_panel(
        ax_full,
        _first_existing_image_path(row.get("full_space_embedding_clusters_png")),
        "Full feature-space embedding by cluster",
    )
    _image_panel(
        ax_diffusion,
        _first_existing_image_path(
            row.get("tree_distance_embedding_clusters_png"),
            row.get("adaptive_diffusion_embedding_annotated_terms_png"),
            row.get("adaptive_diffusion_embedding_png"),
        ),
        "Adaptive/tree-distance embedding by cluster",
    )
    _image_panel(
        ax_terms,
        _first_existing_image_path(row.get("axis_terms_combined_png")),
        "Axis GO-term loadings",
    )

    fig.suptitle(
        f"Subspace plots: {subspace_label(row)}",
        fontsize=15,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_subspace_annotation_page(
    pdf: PdfPages,
    row: pd.Series,
    annotations: list[ClusterAnnotation],
    cache: dict[str, Any],
) -> None:
    subset = [ann for ann in annotations if ann.run_id == row["run_id"]]
    fig, ax_text = plt.subplots(figsize=(18, 11), layout="constrained")

    metric_lines = [
        f"Subspace: {subspace_label(row)}",
        f"Status: {row.get('status', '')}",
        f"Assignment source: {row.get('assignment_source', '')}",
        f"Leaves / assigned genes: {row.get('linkage_leaves', 'NA')} / {row.get('n_unique_genes', 'NA')}",
        f"Clusters: {int(row.get('n_clusters', 0)) if pd.notna(row.get('n_clusters')) else 'NA'}",
        f"Quality tier: {row.get('quality_tier_label', '')}",
        f"Specificity score: {float(row.get('specificity_score', math.nan)):.3f}"
        if pd.notna(row.get("specificity_score", math.nan))
        else "Specificity score: NA",
        f"GO-BIC/gene: {float(row.get('go_bic_active_per_gene', math.nan)):.2f}"
        if pd.notna(row.get("go_bic_active_per_gene", math.nan))
        else "GO-BIC/gene: NA",
        "",
        "Top cluster gene annotations:",
    ]
    cluster_blocks = []
    for ann in subset:
        top_gene_labels = [protein_label(gene, cache) for gene in ann.representative_genes[:5]]
        top_term_label = go_label(ann.top_go_id, ann.top_term, cache)
        block = [
            f"C{ann.cluster_id} n={ann.cluster_size} coherent={ann.coherent_by_rule} delta={ann.top_term_prevalence_delta:.3f}",
            f"Top term: {truncate(top_term_label, 145)}",
            "Genes: " + "; ".join(truncate(label, 42) for label in top_gene_labels),
            "Gene GO examples: " + " | ".join(truncate(text, 52) for text in ann.representative_gene_annotations[:3]),
        ]
        cluster_blocks.append("\n".join(block))
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(metric_lines) + "\n\n" + "\n\n".join(cluster_blocks),
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        linespacing=1.2,
        transform=ax_text.transAxes,
    )
    fig.suptitle(
        f"Subspace annotations: {subspace_label(row)}",
        fontsize=14,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf(
    output_path: Path,
    *,
    ranking: pd.DataFrame,
    annotations: list[ClusterAnnotation],
    cache: dict[str, Any],
) -> None:
    with PdfPages(output_path) as pdf:
        for _, row in ranking.iterrows():
            draw_subspace_plot_page(pdf, row, cache)
            draw_subspace_annotation_page(pdf, row, annotations, cache)


def main() -> None:
    args = parse_args()
    dataset = dataset_slug(args.feature_matrix, args.dataset_label)
    output_dir = args.output_dir or args.experiment_dir / "systematic_subspace_gene_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_binary_matrix(args.feature_matrix)
    ranking = load_artifact_index(args.experiment_dir)
    if args.top_subspaces > 0:
        ranking = ranking.head(args.top_subspaces).copy()
    ranking = attach_roster_status(ranking, output_dir)
    ranking = attach_organized_plot_paths(ranking, output_dir)

    annotations = load_cluster_annotations_from_roster(
        output_dir / "subspace_cluster_roster.csv",
        top_clusters=args.top_clusters_per_subspace,
    )
    if not annotations:
        annotations = build_cluster_annotations(
            data,
            ranking,
            experiment_dir=args.experiment_dir,
            top_clusters=args.top_clusters_per_subspace,
            genes_per_cluster=args.genes_per_cluster,
            terms_per_cluster=args.terms_per_cluster,
        )
    go_ids = []
    for annotation in annotations:
        go_ids.extend(annotation.local_top_go_ids)
        if annotation.top_go_id:
            go_ids.append(annotation.top_go_id)
    go_ids.extend(axis_go_ids(ranking, experiment_dir=args.experiment_dir))
    go_ids = list(dict.fromkeys(go_id for go_id in go_ids if go_id.startswith("GO:")))
    genes = list(dict.fromkeys(gene for ann in annotations for gene in ann.representative_genes))

    cache_path = output_dir / "life_science_lookup_cache.json"
    cache = load_cache(cache_path)
    if args.skip_life_science_lookups:
        cache["warnings"].append({"lookups": "skipped by user"})
    else:
        quickgo_lookup(go_ids, cache, args.life_science_term_limit)
        uniprot_lookup(genes, cache, args.life_science_gene_limit)
    save_cache(cache_path, cache)

    write_csv_outputs(output_dir, annotations, cache)
    pdf_path = output_dir / f"{dataset}_systematic_subspace_gene_annotation_report.pdf"
    write_pdf(
        pdf_path,
        ranking=ranking,
        annotations=annotations,
        cache=cache,
    )
    readme = [
        "# Systematic Subspace Gene Annotation Report",
        "",
        f"Dataset: `{dataset}`",
        f"Feature matrix: `{args.feature_matrix}`",
        f"Experiment directory: `{args.experiment_dir}`",
        "",
        "Files:",
        f"- `{pdf_path.name}`",
        "  - each accepted KL or diagnostic linkage-cut subspace has one plot page followed by one annotation page",
        "- `subspace_cluster_gene_annotation_summary.csv`",
        "- `subspace_cluster_roster.csv`",
        "- `subspace_gene_membership_long.csv`",
        "- `subspace_cluster_status.csv`",
        "- `subspaces/`: one directory per accepted or diagnostic subspace with roster, annotations, source artifacts, radial tree, full-space embedding, subspace embedding, and tree-distance embedding when available",
        "- `quickgo_term_interpretations.csv`",
        "- `uniprot_representative_gene_interpretations.csv`",
        "- `life_science_lookup_cache.json`",
        "",
        "`assignment_source=accepted_kl` rows are final accepted KL assignments; `assignment_source=diagnostic_linkage_cut` rows are diagnostic linkage-tree cuts for failed gates.",
        "",
        "Life-science interpretation uses QuickGO term records and UniProt reviewed human gene/protein lookups through `scripts/rest_request.py`.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "clusters": len(annotations)}, indent=2))


if __name__ == "__main__":
    main()
