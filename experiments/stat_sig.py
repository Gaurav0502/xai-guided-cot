# modules used 
# for file handling
import json
import sys
import os
from pathlib import Path

# add parent directory to 
# path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# modules used for data handling
import pandas as pd
import numpy as np

# module used for statistical testing
from statsmodels.stats.contingency_tables import mcnemar

# module used for type hinting
from typing import Callable, Optional

# user-defined modules

## postprocessing functions
from scripts.postprocess import (
    parse_baseline_llm_results,
    parse_zero_shot_cot_llm_results,
    parse_cot_llm_results
)

## dataset configuration
from scripts.configs import Dataset


# statistical significance testing
# function
def mcnemar_test(
        dataset: Dataset,
        model1_predictions_path: str,
        model2_predictions_path: str,
        model1_name: str,
        model2_name: str,
        model1_parse_fn: Callable,
        model2_parse_fn: Callable,
        alpha: float = 0.05
    ) -> None:
    """
    Performs McNemar's test and prints whether the difference is statistically significant.
    
    Args:
        dataset: Dataset configuration object
        model1_predictions_path: Path to model1 predictions JSONL file
        model2_predictions_path: Path to model2 predictions JSONL file
        model1_name: Name of model1 for reporting
        model2_name: Name of model2 for reporting
        model1_parse_fn: Function to parse model1 predictions (default: parse_baseline_llm_results)
        model2_parse_fn: Function to parse model2 predictions (default: parse_baseline_llm_results)
        alpha: Significance level (default: 0.05)
    """
    
    # load predictions
    model1_pred = model1_parse_fn(model1_predictions_path)
    model2_pred = model2_parse_fn(model2_predictions_path)
    
    # load dataset and 
    # get true labels
    df = pd.read_csv(dataset.path)
    df = dataset.preprocess_fn(df)
    dataset_config = json.load(open(dataset.config_file_path, 'r'))
    
    # get test indices and 
    # true labels
    test_indices = dataset_config["test_data_idx"]
    true_labels = df.loc[test_indices, dataset.target_col].tolist()
    
    # create contingency 
    # table
    common_indices = set(model1_pred.keys()) & set(model2_pred.keys())
    a = b = c = d = 0
    
    # iterate over 
    # common indices
    for idx in common_indices:
        true_label = true_labels[test_indices.index(idx)]
        pred1_correct = (model1_pred[idx] == true_label)
        pred2_correct = (model2_pred[idx] == true_label)
        
        if pred1_correct and pred2_correct:
            a += 1
        elif pred1_correct and not pred2_correct:
            b += 1
        elif not pred1_correct and pred2_correct:
            c += 1
        else:
            d += 1
    
    # create contingency 
    # table array
    contingency_table = np.array([[a, b], [c, d]])
    
    # perform McNemar's test
    result = mcnemar(contingency_table, 
                     exact=False, 
                     correction=True)
    
    # print result
    if result.pvalue < alpha:
        print(f"Difference between {model1_name} and {model2_name} "
              f"is statistically significant (p={result.pvalue:.4f})")
    else:
        print(f"Difference between {model1_name} and {model2_name} "
              f"is not statistically significant (p={result.pvalue:.4f})")