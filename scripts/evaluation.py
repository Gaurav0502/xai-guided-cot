# modules used for data handling
import json
import pandas as pd
import numpy as np

# modules used for evaluation
from sklearn.metrics import (f1_score, 
                             accuracy_score)

# user-defined modules

## dataset configuration
from scripts.configs import Dataset

## llm output postprocessing
from scripts.postprocess import (parse_baseline_llm_results,
                                 parse_zero_shot_cot_llm_results,
                                 parse_cot_llm_results)

## constants
from scripts.constants import PROMPTING_STRATEGIES

# model evaluation 
# class
class Evaluator:

    # initialization
    def __init__(
            self, 
            prompting_strategy: str, 
            dataset: Dataset, 
            results_jsonl_path: str, 
            postprocess_fn: callable
    ) -> None:
        """
        Initializes the Evaluator with strategy, dataset, results path, and postprocessing function.

        Args:
            prompting_strategy (str): The prompting strategy to evaluate.
            dataset (Dataset): Dataset configuration object.
            results_jsonl_path (str): Path to the JSONL file with LLM results.
            postprocess_fn (callable): Function to postprocess LLM results.

        Returns:
            None
    """

        # inputs

        ## prompting strategy
        self.prompting_strategy = prompting_strategy
        if prompting_strategy not in PROMPTING_STRATEGIES:
            raise ValueError(
                    f"Prompting strategy '{prompting_strategy}' not recognized. "
                    f"Must be one of: {PROMPTING_STRATEGIES}."
                )

        ## dataset and
        ## config
        self.dataset = dataset
        self.dataset_config = json.load(open(self.dataset.config_file_path, 'r'))
        
        ## llm results
        self.results_jsonl_path = results_jsonl_path
        self.results = postprocess_fn(self.results_jsonl_path)

        # outputs
        self.y_true = []
        self.y_pred = []

        self.metrics = {"xgboost": None, prompting_strategy: None}

    # loads results
    # and map with true labels
    def __load_results(self) -> None:
        """
        Loads and preprocesses the dataset, and maps LLM results to true labels.

        Args:
            None

        Returns:
            None.
            Use `y_true` and `y_pred` attributes for true and predicted labels.
    """

        self.df = pd.read_csv(self.dataset.path)
        self.df = self.dataset.preprocess_fn(self.df)

        for i in self.results:
            idx = i
            y_true = self.df.loc[idx, self.dataset.target_col]
            self.y_true.append(y_true)

        self.y_pred = list(map(int, self.results.values()))
    
    # evaluates
    def evaluate(self) -> None:
        """
        Evaluates predictions from both XGBoost and the prompting strategy, computing accuracy and macro F1.

        Args:
            None

        Returns:
            None.
            Use the `metrics` attribute to access evaluation results.
    """

        self.__load_results()

        test_pred_xgboost = self.dataset_config["test_predictions"]
        y_true_xgboost = self.df.loc[self.dataset_config["test_data_idx"], self.dataset.target_col].tolist()
        self.metrics['xgboost'] = {"accuracy": round(accuracy_score(y_true_xgboost, 
                                                              test_pred_xgboost), 3),
                                   "macro_f1_score": round(f1_score(y_true_xgboost, test_pred_xgboost, 
                                                              average='macro'), 3)}
        
        self.metrics[self.prompting_strategy] = {"macro_f1_score": round(f1_score(self.y_true, self.y_pred, 
                                                                                  average='macro'), 3),
                                                 "accuracy": round(accuracy_score(self.y_true, 
                                                                                  self.y_pred), 3)}