from __future__ import annotations

import pytest
from benchmarks.shared.env import get_env_bool, get_env_int


def test_get_env_bool_rejects_unknown_boolean_token(monkeypatch) -> None:
    monkeypatch.setenv("TBS_TEST_BOOL", "maybe")

    with pytest.raises(ValueError, match="must be a boolean flag"):
        get_env_bool("TBS_TEST_BOOL")


def test_get_env_bool_accepts_explicit_false_tokens(monkeypatch) -> None:
    monkeypatch.setenv("TBS_TEST_BOOL", "off")

    assert get_env_bool("TBS_TEST_BOOL", default=True) is False


def test_get_env_int_rejects_malformed_integer(monkeypatch) -> None:
    monkeypatch.setenv("TBS_TEST_INT", "1.5")

    with pytest.raises(ValueError, match="must be an integer"):
        get_env_int("TBS_TEST_INT", 1)
