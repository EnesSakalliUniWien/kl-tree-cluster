from pathlib import Path

from tbs_repo_audit.field_lineage import build_field_lineage


def test_field_lineage_classifies_schema_dead_and_reused_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "ROW_COLUMNS = ('kept_schema',)\n"
        "\n"
        "def build():\n"
        "    row = {}\n"
        "    row['dead_field'] = 1\n"
        "    row['reused_field'] = 2\n"
        "    return row['reused_field']\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["dead_field"]["reuse"] == "no_reader_dead_candidate"
    assert fields["reused_field"]["reuse"] == "same_function_reader"
    assert fields["kept_schema"]["reuse"] == "schema_only_export_candidate"
    assert fields["dead_field"]["writer_scopes"] == [
        "tree_break_selection/result.py::build"
    ]


def test_field_lineage_traces_pandas_loc_columns_and_field_helper_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def annotation_bool(df, node, column):\n"
        "    return bool(df.loc[node, column])\n"
        "\n"
        "def build(df, node):\n"
        "    df.loc[node, 'loc_field'] = 1\n"
        "    return annotation_bool(df, node, 'loc_field')\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["loc_field"]["reuse"] == "same_function_reader"
    assert fields["loc_field"]["operations"] == {"read": 1, "write": 1}


def test_field_lineage_traces_benchmark_annotation_helper_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    production = tmp_path / "tree_break_selection/result.py"
    production.parent.mkdir(parents=True)
    production.write_text(
        "def build(df, node):\n"
        "    df.loc[node, 'exported_field'] = 1.0\n",
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmarks/diagnostics/panel.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(
        "def _annotation_float(df, node, column):\n"
        "    return float(df.loc[node, column])\n"
        "\n"
        "def consume(df, node):\n"
        "    return _annotation_float(df, node, 'exported_field')\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["exported_field"]["reuse"] == "cross_function_reader"
    assert fields["exported_field"]["operations"] == {"read": 1, "write": 1}
