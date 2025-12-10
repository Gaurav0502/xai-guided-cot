from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from scripts.configs import Dataset as DatasetConfig

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_loan(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=["loan_id"])

    df = df.dropna()
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    print(f"[LOAN] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    categorical_cols = [' education', ' self_employed']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.replace({True: 1, False: 0})
    df = df.replace({' Approved': 1, ' Rejected': 0})
    df = df.astype('int64')
    return df

def preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after

    print(f"[Wine] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    return df


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    # Drop irrelevant or high-cardinality columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode Sex values
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Encode Embarked values
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_map)

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after

    print(f"[Titanic] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    return df


def load_tabular_dataset(
    cfg: DatasetConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Load CSV, apply dataset-specific preprocessing, split into train/test
    df = pd.read_csv(cfg.path)

    # Apply dataset-specific preprocessing
    if cfg.preprocess_fn is not None:
        df = cfg.preprocess_fn(df)

    if cfg.target_col == "":
        raise ValueError(f"target_col is empty for dataset {cfg.name} - please set it.")

    print(f"\n[{cfg.name}] Columns after preprocessing:")
    print(list(df.columns))

    # Separate features and target
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
