# data_tools/eda.py

import pandas as pd


def basic_overview(df: pd.DataFrame) -> dict:
    """
    Return a basic overview: shape, dtypes, missing counts.
    """
    overview = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
    }
    return overview


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return describe() for numeric columns.
    """
    return df.describe()


def value_counts_for_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return value counts for a single column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in dataframe")
    return df[column].value_counts()
