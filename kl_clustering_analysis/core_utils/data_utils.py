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
    if not isinstance(df, pd.DataFrame) or df.empty or column_name not in df.columns:
        return {}
    return df[column_name].fillna(default).astype(bool).to_dict()
