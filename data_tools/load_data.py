# data_tools/load_data.py

import pandas as pd


def load_fintech_csv(path: str) -> pd.DataFrame:
    """
    Load the fintech churn dataset from a CSV path.
    """
    df = pd.read_csv(path)
    return df


def clean_and_split_id_target(df: pd.DataFrame, target_col: str = "Exited"):
    """
    Drop obvious ID columns and split into X (features) and y (target).
    """
    id_cols = ["RowNumber", "CustomerId", "Surname"]
    for col in id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
