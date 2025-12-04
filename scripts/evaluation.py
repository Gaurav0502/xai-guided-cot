# modules used for data handling
import json
import pandas as pd
import numpy as np

# modules used for evaluation
from sklearn.metrics import f1_score, log_loss

# user-defined modules
from scripts.configs import Dataset
from scripts.postprocess import parse_cot_llm_results

class Evaluator:

    def __init__(self, prompting_strategy: str, dataset: Dataset, 
                 results_jsonl_path: str, postprocess_fn: callable):

        # inputs
        self.prompting_strategy = prompting_strategy

        self.dataset = dataset
        self.dataset_config = json.load(open(self.dataset.config_file_path, 'r'))

        self.results_jsonl_path = results_jsonl_path
        self.results = postprocess_fn(self.results_jsonl_path)
        assert type(self.results) == dict, \
            "Postprocess function must return a dictionary of results with index as key and predicted label as value."

        # outputs
        self.y_true = []
        self.y_pred = []

        assert prompting_strategy in ["xai-guided-cot", "zero-shot-prompting", "zero-shot-cot"], \
            f"Prompting strategy {prompting_strategy} not recognized. must be one of 'xai-guided-cot', 'zero-shot-prompting', 'zero-shot-cot'."
        self.metrics = {"xgboost": None, prompting_strategy: None}

    def __load_results(self):

        self.df = pd.read_csv(self.dataset.path)
        self.df = self.dataset.preprocess_fn(self.df)

        for i in self.results:
            idx = i
            y_true = self.df.loc[idx, self.dataset.target_col]
            self.y_true.append(y_true)

        self.y_pred = list(map(int, self.results.values()))
    
    def evaluate(self):

        self.__load_results()

        test_pred_xgboost = self.dataset_config["test_predictions"]
        y_true_xgboost = self.df.loc[self.dataset_config["test_data_idx"], self.dataset.target_col].tolist()
        self.metrics['xgboost'] = {"log_loss": log_loss(y_true_xgboost, test_pred_xgboost),
                                   "macro_f1_score": f1_score(y_true_xgboost, test_pred_xgboost, 
                                                              average='macro')}
        
        self.metrics[self.prompting_strategy] = {"macro_f1_score": f1_score(self.y_true, self.y_pred, average='macro'),
                                                 "log_loss": log_loss(self.y_true, self.y_pred)}