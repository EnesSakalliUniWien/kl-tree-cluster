import json
from pathlib import Path

from tbs_repo_audit.duplicates import (
    CloneEndpoint,
    build_duplicate_report,
    classify_duplicate,
    duplicate_report_markdown,
)


def test_classify_r_bootstrap_duplicate_as_structural() -> None:
    first = CloneEndpoint(
        path="a.R",
        full_path="applications/scrna/plots/a.R",
        start=4,
        end=15,
    )
    second = CloneEndpoint(
        path="b.R",
        full_path="applications/scrna/plots/b.R",
        start=4,
        end=15,
    )

    classification, risk, recommendation = classify_duplicate(first, second)

    assert classification == "bootstrap_structural"
    assert risk == "low"
    assert "dispatcher" in recommendation


def test_classify_analysis_duplicate_as_methodological() -> None:
    first = CloneEndpoint(
        path="a.py",
        full_path="applications/scrna/analysis/a.py",
        start=10,
        end=20,
    )
    second = CloneEndpoint(
        path="b.py",
        full_path="applications/scrna/analysis/b.py",
        start=30,
        end=40,
    )

    classification, risk, recommendation = classify_duplicate(first, second)

    assert classification == "analysis_methodological"
    assert risk == "medium"
    assert "semantics" in recommendation


def test_duplicate_report_markdown_is_reader_facing() -> None:
    report = {
        "scope": ["applications"],
        "summary": {
            "clone_group_count": 1,
            "duplicated_lines": 10,
            "duplicated_percent": 0.5,
            "classification_counts": {"plot_report": 1},
        },
        "groups": [
            {
                "lines": 10,
                "classification": "plot_report",
                "risk": "low",
                "recommendation": "Extract shared report helper.",
                "first": {
                    "full_path": "applications/a.py",
                    "start": 1,
                    "end": 10,
                },
                "second": {
                    "full_path": "applications/b.py",
                    "start": 20,
                    "end": 30,
                },
            }
        ],
    }

    markdown = duplicate_report_markdown(report)

    assert "# Duplicate Cleanup Report" in markdown
    assert "plot_report" in markdown
    assert "applications/a.py:1-10" in markdown


def test_build_duplicate_report_runs_jscpd_and_classifies(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".git").mkdir()
    app = tmp_path / "applications/scrna/analysis"
    app.mkdir(parents=True)
    (app / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (app / "b.py").write_text("VALUE = 2\n", encoding="utf-8")

    def fake_which(name: str) -> str:
        assert name == "jscpd"
        return "/usr/bin/jscpd"

    def fake_run(command, cwd, check, capture_output, text):
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "jscpd-report.json").write_text(
            json.dumps(
                {
                    "statistics": {
                        "total": {
                            "duplicatedLines": 10,
                            "percentage": 0.5,
                        }
                    },
                    "duplicates": [
                        {
                            "lines": 10,
                            "tokens": 80,
                            "firstFile": {
                                "name": "scrna/analysis/a.py",
                                "start": 1,
                                "end": 10,
                            },
                            "secondFile": {
                                "name": "scrna/analysis/b.py",
                                "start": 20,
                                "end": 30,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("tbs_repo_audit.duplicates.shutil.which", fake_which)
    monkeypatch.setattr("tbs_repo_audit.duplicates.subprocess.run", fake_run)

    report = build_duplicate_report(tmp_path, [Path("applications")])

    assert report["summary"]["clone_group_count"] == 1
    assert report["groups"][0]["classification"] == "analysis_methodological"
    assert report["groups"][0]["first"]["full_path"] == "applications/scrna/analysis/a.py"
