from benchmarks.shared.config import DEFAULT_METHODS
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.method_selection import resolve_methods_from_env


def test_full_runner_default_method_selection_matches_shared_defaults(monkeypatch):
    monkeypatch.delenv("KL_TE_METHODS", raising=False)

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)


def test_full_runner_all_keyword_expands_to_shared_defaults(monkeypatch):
    monkeypatch.setenv("KL_TE_METHODS", "all")

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)
