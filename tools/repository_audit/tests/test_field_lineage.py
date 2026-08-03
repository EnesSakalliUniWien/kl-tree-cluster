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
    assert fields["dead_field"]["cleanup_classification"] == "dead_write_candidate"
    assert fields["reused_field"]["reuse"] == "same_function_reader"
    assert fields["reused_field"]["cleanup_classification"] == "used_field"
    assert fields["kept_schema"]["reuse"] == "schema_only_export_candidate"
    assert fields["kept_schema"]["cleanup_classification"] == "output_schema_column"
    assert fields["dead_field"]["writer_scopes"] == ["tree_break_selection/result.py::build"]


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
        "def build(df, node):\n    df.loc[node, 'exported_field'] = 1.0\n",
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


def test_field_lineage_excludes_graph_env_and_config_keys_from_dead_writes(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import os\n"
        "\n"
        "def build(tree, graph, config):\n"
        "    os.environ['TBS_CACHE_DIR'] = '/tmp/cache'\n"
        "    tree.graph['branch_length_optimization'] = {'method': 'nnls'}\n"
        "    graph.es['weight'] = [1.0]\n"
        "    config['output_dir'] = 'reports'\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["TBS_CACHE_DIR"]["cleanup_classification"] == "environment_key"
    assert fields["branch_length_optimization"]["cleanup_classification"] == "graph_key"
    assert fields["weight"]["cleanup_classification"] == "graph_key"
    assert fields["output_dir"]["cleanup_classification"] == "configuration_key"


def test_field_lineage_marks_schema_columns_apart_from_dead_writes(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/panel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "OUTPUT_COLUMNS = ('reported_metric',)\n"
        "\n"
        "def build():\n"
        "    row = {}\n"
        "    row['reported_metric'] = 1.0\n"
        "    row['dead_metric'] = 2.0\n"
        "    return row\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["reported_metric"]["cleanup_classification"] == "output_schema_column"
    assert fields["dead_metric"]["cleanup_classification"] == "dead_write_candidate"


def test_field_lineage_marks_returned_dict_keys_as_output_schema(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def counters():\n"
        "    out = {'live_split_count': 0}\n"
        "    out['live_split_count'] += 1\n"
        "    return out\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["live_split_count"]["cleanup_classification"] == "output_schema_column"


def test_field_lineage_marks_pandas_exported_columns_as_output_schema(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/panel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def write_panel(table, output_path):\n"
        "    table['reported_metric'] = 1.0\n"
        "    table.to_csv(output_path, index=False)\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["reported_metric"]["cleanup_classification"] == "output_schema_column"


def test_field_lineage_traces_dictionary_iteration_as_field_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/panel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def signature():\n"
        "    parts = {}\n"
        "    parts['tie_cell_presence'] = 'present'\n"
        "    return '|'.join(f'{key}={value}' for key, value in parts.items())\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["tie_cell_presence"]["reuse"] == "same_function_reader"
    assert fields["tie_cell_presence"]["cleanup_classification"] == "used_field"


def test_field_lineage_traces_direct_container_iteration_as_field_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/panel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def keys():\n"
        "    parts = {}\n"
        "    parts['section_order'] = 1\n"
        "    return [key for key in parts]\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["section_order"]["reuse"] == "same_function_reader"
    assert fields["section_order"]["cleanup_classification"] == "used_field"


def test_field_lineage_classifies_dataframe_attrs_as_output_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/pipeline.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def finish(results):\n"
        "    results.attrs['plot_generation_status'] = 'not_requested'\n"
        "    return results\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["plot_generation_status"]["cleanup_classification"] == ("output_schema_column")


def test_field_lineage_classifies_anndata_uns_as_output_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "applications/scrna/pipeline.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def prepare(adata):\n"
        "    adata.uns['input_expression_min'] = 0.0\n"
        "    adata.write_h5ad('prepared.h5ad')\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["input_expression_min"]["cleanup_classification"] == ("output_schema_column")


def test_field_lineage_classifies_matplotlib_rcparams_as_configuration(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/plots.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def configure(mpl):\n    mpl.rcParams['pdf.fonttype'] = 42\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["pdf.fonttype"]["cleanup_classification"] == "configuration_key"


def test_field_lineage_traces_expanded_keyword_arguments_as_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "benchmarks/adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def load(client, url):\n"
        "    kwargs = {}\n"
        "    kwargs['url'] = url\n"
        "    return client.load_dataset(**kwargs)\n",
        encoding="utf-8",
    )

    lineage = build_field_lineage(tmp_path)
    fields = {field["key"]: field for field in lineage["fields"]}

    assert fields["url"]["reuse"] == "same_function_reader"
    assert fields["url"]["cleanup_classification"] == "used_field"
