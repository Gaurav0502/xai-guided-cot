from typing import List, Dict, Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from preprocess import (
    DatasetConfig,
    load_tabular_dataset,
    preprocess_world_air_quality,
    preprocess_wine_quality,
    preprocess_titanic,
)


def build_tree_estimator() -> XGBClassifier:
    # Return an XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,
        min_child_weight=3,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    return model


def train_tree_baseline(cfg: DatasetConfig) -> Dict[str, Any]:
    """
    Train and evaluate a tree-based baseline model for one dataset.
    Returns a dict with:
      - config
      - model
      - metrics
      - y_test, y_pred
    """
    print(f"\n=== Dataset: {cfg.name} ===")

    # Load & split
    X_train, X_test, y_train, y_test = load_tabular_dataset(cfg)

    # Model
    model = build_tree_estimator()

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["f1_macro"] = f1_score(y_test, y_pred, average="macro")

    print("Metrics:", metrics)

    return {
        "config": cfg,
        "model": model,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_all_tree_baselines(configs: List[DatasetConfig]) -> pd.DataFrame:
    # Run baselines for all configs and summarize metrics
    rows = []
    for cfg in configs:
        result = train_tree_baseline(cfg)
        row = {"dataset": cfg.name}
        row.update(result["metrics"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary


def main():
    world_aqi_cfg = DatasetConfig(
        name="world_air_quality",
        path="data/AQI.csv",
        target_col="AQI_Category",
        preprocess_fn=preprocess_world_air_quality,
    )

    wine_cfg = DatasetConfig(
        name="wine_quality",
        path="data/WineQT.csv",
        target_col="quality",
        preprocess_fn=preprocess_wine_quality,
    )

    titanic_cfg = DatasetConfig(
        name="titanic",
        path="data/Titanic.csv",
        target_col="Survived",
        preprocess_fn=preprocess_titanic,
    )

    configs = [world_aqi_cfg, wine_cfg, titanic_cfg]

    summary = run_all_tree_baselines(configs)
    output_path = "tree_baseline_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved summary metrics to: {output_path}")


if __name__ == "__main__":
    main()
