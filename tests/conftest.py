import os
import sys
from dataclasses import dataclass

# Ensure the project root is on sys.path so tests can import packages like
# `tree` and `misc` when running directly from the repository.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add the project root (one level above the package) so absolute imports
# like ``import kl_clustering_analysis`` work without installing the package.
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Prevent pytest from collecting functions whose names start with ``test_``
# from the *source* package when they are re-exported into a test module's
# namespace via ``from ... import test_node_pair_divergence`` etc.
collect_ignore_glob = ["**/kl_clustering_analysis/**"]


@dataclass
class _ProgressState:
    enabled: bool = False
    total: int = 0
    completed: int = 0
    config: object | None = None


_PROGRESS = _ProgressState()


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    parser.addoption(
        "--progress",
        action="store_true",
        default=False,
        help="Show running test count (completed/total).",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    _PROGRESS.enabled = bool(config.getoption("--progress")) or bool(
        os.environ.get("PYTEST_PROGRESS")
    )
    _PROGRESS.total = 0
    _PROGRESS.completed = 0
    _PROGRESS.config = config


def pytest_collection_modifyitems(session, config, items):  # type: ignore[no-untyped-def]
    if not _PROGRESS.enabled:
        return
    _PROGRESS.total = len(items)


def _write_progress() -> None:
    if not _PROGRESS.enabled or _PROGRESS.total <= 0 or _PROGRESS.config is None:
        return
    terminal_reporter = _PROGRESS.config.pluginmanager.getplugin("terminalreporter")
    msg = f"\r[pytest] {_PROGRESS.completed}/{_PROGRESS.total} tests"
    if terminal_reporter is not None:
        terminal_reporter.write(msg)
    else:
        sys.stderr.write(msg)
        sys.stderr.flush()


def pytest_runtest_logreport(report):  # type: ignore[no-untyped-def]
    if not _PROGRESS.enabled:
        return
    # Count a test as "completed" once it has a final outcome:
    # - normal tests: at call phase
    # - skipped in setup: at setup phase
    if report.when == "call" or (report.when == "setup" and report.outcome == "skipped"):
        _PROGRESS.completed += 1
        _write_progress()


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    if not _PROGRESS.enabled:
        return
    terminal_reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write("\n")
    else:
        sys.stderr.write("\n")
