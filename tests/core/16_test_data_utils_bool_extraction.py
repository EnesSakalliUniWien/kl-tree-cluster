"""Unit tests for boolean extraction from DataFrame columns."""

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict


def test_extract_bool_column_dict_basic():
    df = pd.DataFrame({"flag": [True, False, True]}, index=["A", "B", "C"])

    result = extract_bool_column_dict(df, "flag")

    assert result == {"A": True, "B": False, "C": True}


def test_extract_bool_column_dict_numpy_bool():
    df = pd.DataFrame({"flag": np.array([True, False], dtype=bool)}, index=["A", "B"])

    result = extract_bool_column_dict(df, "flag")

    assert result == {"A": True, "B": False}
    assert all(isinstance(v, bool) for v in result.values())


def test_extract_bool_column_dict_raises_on_null():
    df = pd.DataFrame({"flag": [True, None]}, index=["A", "B"])

    with pytest.raises(ValueError, match="contains missing values"):
        extract_bool_column_dict(df, "flag")


def test_extract_bool_column_dict_raises_on_missing_column():
    df = pd.DataFrame({"other": [True]}, index=["A"])

    with pytest.raises(KeyError, match="Missing required column"):
        extract_bool_column_dict(df, "flag")


def test_extract_bool_column_dict_raises_on_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Empty DataFrame"):
        extract_bool_column_dict(df, "flag")


def test_extract_bool_column_dict_raises_on_non_dataframe():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        extract_bool_column_dict({"flag": [True]}, "flag")
