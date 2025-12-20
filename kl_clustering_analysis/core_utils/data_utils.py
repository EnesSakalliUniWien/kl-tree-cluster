from __future__ import annotations
import pandas as pd


def extract_bool_column_dict(
    df: object, column_name: str, default: bool = False
) -> dict[str, bool]:
    """Extract a boolean column from DataFrame as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame or similar
        DataFrame containing the column.
    column_name : str
        Name of the column to extract.
    default : bool
        Default value for missing/NaN entries.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping index to boolean values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for {column_name!r} extraction.")
    if df.empty:
        raise ValueError(f"Empty DataFrame; missing required column {column_name!r}.")
    if column_name not in df.columns:
        raise KeyError(f"Missing required column {column_name!r} in dataframe.")

    series = df[column_name]
    if series.isna().any():
        missing = series[series.isna()].index.tolist()
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(
            f"Column {column_name!r} contains missing values for nodes: {preview}."
        )
    return series.astype(bool).to_dict()
