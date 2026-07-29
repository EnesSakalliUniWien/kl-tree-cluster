import json
from pathlib import Path

from tbs_repo_audit.inventory import build_inventory, write_inventory


def test_inventory_classifies_test_only_calibration_and_diagnostic_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    module = tmp_path / "benchmarks/diagnostics/calibration/panel.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'def build():\n    result = {}\n    result["diagnostic_score"] = 1\n    return result\n',
        encoding="utf-8",
    )
    test = tmp_path / "tests/validation/calibration/test_panel.py"
    test.parent.mkdir(parents=True)
    test.write_text(
        "from benchmarks.diagnostics.calibration.panel import build\n"
        "\n"
        "def test_panel():\n"
        '    assert build()["diagnostic_score"] == 1\n',
        encoding="utf-8",
    )

    inventory = build_inventory(tmp_path)

    calibration = inventory["calibration"]
    assert calibration["module_count"] == 1
    assert calibration["test_only_count"] == 1
    field = next(
        item for item in inventory["fields"]["keys"] if item["key"] == "diagnostic_score"
    )
    assert field["diagnostic_only"] is True
    assert field["write_only"] is False


def test_inventory_marks_unresolved_unimported_calibration_module(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    module = tmp_path / "benchmarks/diagnostics/calibration/orphan.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    calibration = inventory["calibration"]
    assert calibration["statically_unimported_count"] == 1
    assert calibration["unresolved_count"] == 1
    assert calibration["modules"][0]["disposition"] == "unresolved_unimported"


def test_inventory_does_not_count_package_initializers_as_studies(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    package = tmp_path / "benchmarks/diagnostics/calibration"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "panel.py").write_text("VALUE = 1\n", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    assert inventory["calibration"]["module_count"] == 1


def test_inventory_recognizes_documented_standalone_runner(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    runner = tmp_path / "benchmarks/diagnostics/calibration/run_panel.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        'def main():\n    return 0\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
        encoding="utf-8",
    )
    wiki = tmp_path / "wiki/panel.md"
    wiki.parent.mkdir()
    wiki.write_text("Run `run_panel.py` to reproduce this study.\n", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    panel = inventory["calibration"]["modules"][0]
    assert panel["has_main_guard"] is True
    assert panel["documented_by"] == ["wiki/panel.md"]
    assert panel["disposition"] == "standalone_documented"


def test_inventory_matches_hyphenated_documentation_name(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    module = tmp_path / "benchmarks/diagnostics/calibration/same_z_study.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    wiki = tmp_path / "wiki/study.md"
    wiki.parent.mkdir()
    wiki.write_text("The same-z study records this result.\n", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    assert inventory["calibration"]["modules"][0]["documented_by"] == ["wiki/study.md"]


def test_inventory_distinguishes_production_importer(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    module = tmp_path / "benchmarks/diagnostics/calibration/shared.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    production = tmp_path / "tree_break_selection/use_panel.py"
    production.parent.mkdir(parents=True)
    production.write_text(
        "from benchmarks.diagnostics.calibration import shared\n",
        encoding="utf-8",
    )

    inventory = build_inventory(tmp_path)

    panel = inventory["calibration"]["modules"][0]
    assert panel["test_only"] is False
    assert panel["non_test_importers"] == ["tree_break_selection.use_panel"]


def test_inventory_marks_field_without_reads_as_write_only(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "tree_break_selection/result.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'def result():\n    value = {}\n    value["unused_metadata"] = 1\n    return value\n',
        encoding="utf-8",
    )

    inventory = build_inventory(tmp_path)

    field = next(
        item for item in inventory["fields"]["keys"] if item["key"] == "unused_metadata"
    )
    assert field["write_only"] is True
    assert field["diagnostic_only"] is False


def test_inventory_records_parse_errors_without_aborting(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "scripts/broken.py"
    source.parent.mkdir(parents=True)
    source.write_text("def broken(:\n", encoding="utf-8")

    inventory = build_inventory(tmp_path)

    assert inventory["parse_errors"][0]["path"] == "scripts/broken.py"


def test_write_inventory_creates_parent_and_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "reports/audit.json"

    write_inventory({"schema_version": 1}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert not output.with_suffix(".json.tmp").exists()
