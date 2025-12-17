# modules used for data handling
import argparse
import pandas as pd
import numpy as np
import json
import random

# modules used for model training
import xgboost as xgb

# user-defined modules

## configurations
from scripts.configs import (Dataset, Model, COT)

## experiment setup
from scripts.constants import (titanic_dataset, diabetes_dataset,
                               loan_dataset, mushroom_dataset,
                               reasoning_gen_model, objective_judge_model,
                               cot_model)

TUNE_CONFIG_FILE = "data/tune_config/xgb.json"

## pipeline
from scripts.pipeline import Pipeline

# modules used for typing hinting
from typing import Callable

def mask_dataset_config(dataset: Dataset, preprocess_fn: Callable) -> Dataset:

    # metadata
    dataset.name = "unknown"
    dataset.labels = {0: "Class0", 1: "Class1"}

    # new shap values path
    new_shap_vals_path = "data/shap_values/masked_shap.csv"
    df = pd.read_csv(dataset.path)
    df = dataset.preprocess_fn(df)
    cols = list(df.columns)
    masked_cols = ["x" + str(i) for i in range(len(cols))]
    col_mask_map = dict(zip(cols, masked_cols))
    shap_file = pd.read_csv(dataset.shap_vals_path)
    shap_file.rename(columns=col_mask_map, inplace=True)
    shap_file.to_csv(new_shap_vals_path, index=False) 
    dataset.shap_vals_path = new_shap_vals_path

    # update target column
    dataset.target_col = col_mask_map[dataset.target_col]

    # new preprocess function
    def mask_dataset(df: pd.DataFrame) -> pd.DataFrame:
        df = preprocess_fn(df)
        cols = list(df.columns)
        masked_cols = ["x" + str(i) for i in range(len(cols))]
        col_mask_map = dict(zip(cols, masked_cols))
        df_masked = df.rename(columns=col_mask_map)
        return df_masked

    dataset.preprocess_fn = mask_dataset
    return dataset

def main(dataset_name: str, masked: bool = False):

    # dataset mapping
    if dataset_name == "titanic":
        dataset = titanic_dataset
    elif dataset_name == "diabetes":
        dataset = diabetes_dataset
    elif dataset_name == "loan":
        dataset = loan_dataset
    elif dataset_name == "mushroom":
        dataset = mushroom_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    if masked:
        dataset = mask_dataset_config(dataset, 
                                      preprocess_fn=dataset.preprocess_fn)

    # initialize pipeline
    pipeline = Pipeline(
        dataset=dataset,
        explanable_model=xgb.XGBClassifier(),
        tune_config_file=TUNE_CONFIG_FILE,
        reasoning_gen_model=reasoning_gen_model,
        objective_judge_model=objective_judge_model,
        cot_model=cot_model
    )

    # run pipeline
    pipeline.run(
        baseline=True,
        objective_judge=True,
        cot_ablation=True,
        masked=masked
    )

    json.dump(
        pipeline.results, 
        open(f"data/metrics/{dataset_name}_{'masked' if masked else 'unmasked'}.json", "w"),
        indent=4
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI-guided CoT pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["titanic", "diabetes", "loan", "mushroom"],
        help="Dataset name to use"
    )
    parser.add_argument(
        "--masked",
        action="store_true",
        help="Use masked dataset (default: False)"
    )
    
    args = parser.parse_args()
    main(dataset_name=args.dataset, 
         masked=args.masked)