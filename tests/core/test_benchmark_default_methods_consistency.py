import pytest
from benchmarks.shared.config import DEFAULT_METHODS
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.method_selection import (
    resolve_methods_from_env,
    resolve_selected_methods_and_param_sets,
)


def test_full_runner_default_method_selection_matches_shared_defaults(monkeypatch):
    monkeypatch.delenv("TBS_METHODS", raising=False)

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)


def test_full_runner_all_keyword_expands_to_shared_defaults(monkeypatch):
    monkeypatch.setenv("TBS_METHODS", "all")

    resolved = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )

    assert resolved == list(DEFAULT_METHODS)


def test_explicit_empty_method_param_grid_is_invalid():
    with pytest.raises(ValueError, match="empty parameter grid"):
        resolve_selected_methods_and_param_sets(
            methods=["tbs"],
            method_params={"tbs": []},
            default_methods=DEFAULT_METHODS,
            method_specs=METHOD_SPECS,
        )
