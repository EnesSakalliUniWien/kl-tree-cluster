"""Unit tests for strict boolean extraction from DataFrame columns."""

import pandas as pd
import pytest

from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict


def test_extract_bool_column_dict_normalizes_false_string_correctly():
    df = pd.DataFrame({"flag": ["False", "true"]}, index=["A", "B"])

    result = extract_bool_column_dict(df, "flag")

    assert result == {"A": False, "B": True}


def test_extract_bool_column_dict_raises_on_null_by_default():
    df = pd.DataFrame({"flag": [True, None]}, index=["A", "B"])

    with pytest.raises(ValueError, match="contains missing values"):
        extract_bool_column_dict(df, "flag")


def test_extract_bool_column_dict_accepts_mixed_strict_boolean_tokens():
    df = pd.DataFrame(
        {
            "flag": [
                True,
                False,
                1,
                0,
                1.0,
                0.0,
                "TRUE",
                " false ",
                "1",
                " 0 ",
            ]
        },
        index=list("ABCDEFGHIJ"),
    )

    result = extract_bool_column_dict(df, "flag")

    assert result == {
        "A": True,
        "B": False,
        "C": True,
        "D": False,
        "E": True,
        "F": False,
        "G": True,
        "H": False,
        "I": True,
        "J": False,
    }


@pytest.mark.parametrize("bad_value", ["maybe", 2, -1, 0.5, ""])
def test_extract_bool_column_dict_rejects_invalid_values(bad_value):
    df = pd.DataFrame({"flag": [True, bad_value]}, index=["A", "B"])

    with pytest.raises(ValueError, match="non-boolean values"):
        extract_bool_column_dict(df, "flag")


@pytest.mark.parametrize("bad_value", ["t", "f", "yes", "no", "y", "n"])
def test_extract_bool_column_dict_rejects_legacy_tokens(bad_value):
    df = pd.DataFrame({"flag": [True, bad_value]}, index=["A", "B"])

    with pytest.raises(ValueError, match="non-boolean values"):
        extract_bool_column_dict(df, "flag")


def test_extract_bool_column_dict_null_policy_false():
    df = pd.DataFrame({"flag": [True, None]}, index=["A", "B"])

    result = extract_bool_column_dict(df, "flag", null_policy="false")

    assert result == {"A": True, "B": False}


def test_extract_bool_column_dict_null_policy_true():
    df = pd.DataFrame({"flag": [False, None]}, index=["A", "B"])

    result = extract_bool_column_dict(df, "flag", null_policy="true")

    assert result == {"A": False, "B": True}
