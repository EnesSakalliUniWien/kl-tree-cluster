"""Shared data and naming contracts for endotype application entry points."""

import csv
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

_MYGENE_QUERY_ENDPOINT = "https://mygene.info/v3/query"


def safe_name(value: object) -> str:
    """Return a filesystem-safe representation without changing case."""

    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def matrix_slug(input_path: Path, dataset_label: str | None = None) -> str:
    """Return the stable application slug for an input feature matrix."""

    raw = dataset_label or input_path.stem
    if raw.startswith("feature_matrix_"):
        raw = raw[len("feature_matrix_") :]
    return safe_name(raw).strip("_").lower() or "feature_matrix"


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load a numeric feature matrix and enforce cosine-space row support."""

    data = pd.read_csv(path, sep="\t", index_col=0)
    zero_columns = data.columns[(data.sum(axis=0) == 0).to_numpy()]
    if len(zero_columns):
        data = data.drop(columns=zero_columns)
    zero_rows = data.index[(data.sum(axis=1) == 0).to_numpy()]
    if len(zero_rows):
        raise ValueError(
            f"Rows with zero feature mass cannot enter cosine analysis: {list(zero_rows[:10])!r}"
        )
    return data.astype(float)


def parse_reference_endotypes(path: Path) -> dict[str, dict[str, object]]:
    """Index a Julia reference-endotype table by Entrez gene ID."""

    entrez_to_endotype: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            first_cell = row[0].strip()
            if first_cell.startswith("#") or first_cell in {"SUM", "AVERAGE"}:
                continue
            if len(row) < 10:
                continue
            cluster_id_text = row[4].strip()
            if not cluster_id_text.isdigit():
                continue
            endotype = {
                "reference_cluster_id": int(cluster_id_text),
                "reference_cluster_color": row[5].strip() or "#808080",
                "reference_cluster_rank": row[6].strip(),
                "reference_cluster_name": row[7].strip() or f"Cluster {cluster_id_text}",
                "reference_cluster_type": row[8].strip() or "cluster",
            }
            for gene_id in (cell.strip() for cell in row[9:] if cell.strip()):
                entrez_to_endotype[gene_id] = endotype
    return entrez_to_endotype


def map_symbols_to_entrez(
    symbols: list[str],
    *,
    batch_size: int = 200,
) -> dict[str, str | None]:
    """Resolve human gene symbols to at most one Entrez ID per symbol."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mapped: dict[str, str | None] = {}
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        body = urlencode(
            {
                "q": ",".join(batch),
                "scopes": "symbol",
                "fields": "symbol,entrezgene,taxid",
                "species": "human",
                "size": 1,
            }
        ).encode("utf-8")
        request = Request(
            _MYGENE_QUERY_ENDPOINT,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Failed to query mygene.info for gene symbol -> Entrez mapping."
            ) from exc

        records = [payload] if isinstance(payload, dict) else payload
        for record in records:
            query = record.get("query")
            if not query:
                continue
            entrez = record.get("entrezgene") or record.get("_id")
            mapped[str(query)] = None if entrez is None else str(entrez)
        for symbol in batch:
            mapped.setdefault(symbol, None)
    return mapped
