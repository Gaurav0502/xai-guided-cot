from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_col: str
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    test_size: float = 0.2
    random_state: int = 42


def preprocess_world_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only pollutant values (X) + AQI category (Y)
    keep_cols = [
        "AQI Category",
        "CO AQI Value",
        "Ozone AQI Value",
        "NO2 AQI Value",
        "PM2.5 AQI Value",
    ]
    df = df[keep_cols].copy()

    df.rename(columns={"AQI Category": "AQI_Category"}, inplace=True)
    aqi_map = {
        "Good": 0,
        "Moderate": 1,
        "Unhealthy for Sensitive Groups": 2,
        "Unhealthy": 3,
        "Very Unhealthy": 4,
    }
    df["AQI_Category"] = df["AQI_Category"].map(aqi_map)

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after

    print(f"[AQI] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    return df


def preprocess_wine_quality(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID column
    df = df.drop(columns=["Id"])
    unique_scores = sorted(df["quality"].unique())
    mapping = {score: idx for idx, score in enumerate(unique_scores)}
    df["quality"] = df["quality"].map(mapping)
    print("[Wine] Mapped quality scores:", mapping)

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
