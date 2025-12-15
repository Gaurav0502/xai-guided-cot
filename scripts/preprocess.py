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


def preprocess_mushroom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace("?", pd.NA)
    # Target column mapping (edible=0, poisonous=1)
    df["class"] = df["class"].map({"e": 0, "p": 1})

    # Feature mappings (letter codes to integers)
    feature_mappings = {
        "cap-shape": {"b": 0, "c": 1, "x": 2, "f": 3, "k": 4, "s": 5},
        "cap-surface": {"f": 0, "g": 1, "y": 2, "s": 3},
        "cap-color": {
            "n": 0,
            "b": 1,
            "c": 2,
            "g": 3,
            "r": 4,
            "p": 5,
            "u": 6,
            "e": 7,
            "w": 8,
            "y": 9,
        },
        "bruises": {"t": 1, "f": 0},
        "odor": {
            "a": 0,
            "l": 1,
            "c": 2,
            "y": 3,
            "f": 4,
            "m": 5,
            "n": 6,
            "p": 7,
            "s": 8,
        },
        "gill-attachment": {"a": 0, "d": 1, "f": 2, "n": 3},
        "gill-spacing": {"c": 0, "w": 1, "d": 2},
        "gill-size": {"b": 0, "n": 1},
        "gill-color": {
            "k": 0,
            "n": 1,
            "b": 2,
            "h": 3,
            "g": 4,
            "r": 5,
            "o": 6,
            "p": 7,
            "u": 8,
            "e": 9,
            "w": 10,
            "y": 11,
        },
        "stalk-shape": {"e": 0, "t": 1},
        "stalk-root": {"b": 0, "c": 1, "u": 2, "e": 3, "z": 4, "r": 5},
        "stalk-surface-above-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-surface-below-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-color-above-ring": {
            "n": 0,
            "b": 1,
            "c": 2,
            "g": 3,
            "o": 4,
            "p": 5,
            "e": 6,
            "w": 7,
            "y": 8,
        },
        "stalk-color-below-ring": {
            "n": 0,
            "b": 1,
            "c": 2,
            "g": 3,
            "o": 4,
            "p": 5,
            "e": 6,
            "w": 7,
            "y": 8,
        },
        "veil-type": {"p": 0, "u": 1},
        "veil-color": {"n": 0, "o": 1, "w": 2, "y": 3},
        "ring-number": {"n": 0, "o": 1, "t": 2},
        "ring-type": {"c": 0, "e": 1, "f": 2, "l": 3, "n": 4, "p": 5, "s": 6, "z": 7},
        "spore-print-color": {
            "k": 0,
            "n": 1,
            "b": 2,
            "h": 3,
            "r": 4,
            "o": 5,
            "u": 6,
            "w": 7,
            "y": 8,
        },
        "population": {"a": 0, "c": 1, "n": 2, "s": 3, "v": 4, "y": 5},
        "habitat": {"g": 0, "l": 1, "m": 2, "p": 3, "u": 4, "w": 5, "d": 6},
    }

    # Apply mappings to each feature column
    for col, mapping in feature_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after

    print(f"[Mushroom] Dropped {dropped} rows due to NaNs (kept {after} rows).")

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
