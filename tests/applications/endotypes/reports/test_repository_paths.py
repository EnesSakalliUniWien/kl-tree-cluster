"""Repository-path contracts for relocated endotype report commands."""

from pathlib import Path

from applications.endotypes.reports.export_subspace_cluster_rosters import REPO_ROOT


def test_roster_export_resolves_repository_root() -> None:
    assert REPO_ROOT == Path(__file__).resolve().parents[4]
