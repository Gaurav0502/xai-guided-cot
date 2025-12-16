# module used for data handling
import pandas as pd
import numpy as np
import json
from scripts.sanitize_wandb_config import sanitize_wandb_config
from scripts.configs import Dataset

# module used for model training
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# module used for hyperparameter tuning
import wandb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# module used for handling secrets
import dotenv
import os
dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")

# module used for explainability
import shap

# to suppress wandb info logs
os.environ["WANDB_SILENT"] = "true"

class ExplainableModel:
    def __init__(self, dataset: Dataset, estimator: xgb.XGBClassifier | DecisionTreeClassifier):

        # model
        self.model = estimator

        # dataset
        self.dataset_path = dataset.path
        if dataset.preprocess_fn:
            self.preprocess = dataset.preprocess_fn
        else:
            raise ValueError("Preprocess function must be provided in Dataset.")
        
        self.df = self.preprocess(pd.read_csv(self.dataset_path))
        self.target_column = dataset.target_col

        self.SHAP_DATA_DIR = dataset.shap_vals_path
        self.DATASET_CONFIG_DIR = dataset.config_file_path

        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # wandb
        self.project_name = WANDB_PROJECT_NAME

        # explanation attributes
        self.feature_importances = None
        self.shap_values = None

    def __train_sweep(self, config=None):

        with wandb.init(config=config, project=self.project_name) as run:
            config = wandb.config

            self.model.set_params(**config)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            acc, f2_scores = [], []

            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                self.model.fit(X_tr, y_tr)
                y_pred = self.model.predict(X_val)

                acc.append(accuracy_score(y_val, y_pred))
                f2_scores.append(f1_score(y_val, y_pred, 
                                          average="macro"))
            
            results = {
                "accuracy": np.mean(acc),
                "accuracy_std": np.std(acc),
                "f1_macro": np.mean(f2_scores),
                "f1_macro_std": np.std(f2_scores)
            }

            run.log(results)

    def __tune(self, params_grid_file: str):
        
        sweep_config = json.load(open(params_grid_file, 'r'))
        
        self.sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        wandb.agent(self.sweep_id, function=self.__train_sweep, 
                    count=sweep_config.get("count", 10))
    
    def __fetch_best_params(self):
        api = wandb.Api()
        sweep = api.sweep(f"{self.project_name}/{self.sweep_id}")
        best_run = sweep.best_run()
        cfg = best_run.config
        if not isinstance(cfg, dict):
            raise TypeError(f"best_run.config is {type(cfg).__name__}, expected dict")
        return cfg
    
    def __train_model(self):
        best_params = self.__fetch_best_params()
        best_params = sanitize_wandb_config(best_params)
        self.model.set_params(**best_params)
        self.model.fit(self.X_train, self.y_train)
        self.train_pred = self.model.predict(self.X_train).tolist()
        self.test_pred = self.model.predict(self.X_test).tolist()
    
    def __log(self):

        feature_imps = dict(zip(self.X_train.columns, 
                                self.feature_importances.tolist()))

        df_shap = pd.DataFrame(
            data=self.shap_values,
            columns=list(self.X_train.columns)
        )

        df_shap.index = self.X_train.index
        df_shap.index.name = "idx"
        df_shap.to_csv(self.SHAP_DATA_DIR, index=True)

        log = {
            "dataset_path": self.dataset_path,
            "shap_values_path": self.SHAP_DATA_DIR,
            "feature_importances": feature_imps,
            "train_data_idx": self.X_train.index.tolist(),
            "test_data_idx": self.X_test.index.tolist(),
            "train_predictions": self.train_pred,
            "test_predictions": self.test_pred
        }
        
        with open(self.DATASET_CONFIG_DIR, 'w') as f:
            json.dump(log, f, indent=4)
        
        print(f"[XAI-MODEL] Logged explanation data to {self.DATASET_CONFIG_DIR}")

    def explain(self, params_grid_file: str):
        
        self.__tune(params_grid_file=params_grid_file)
        print("[XAI-MODEL] Completed hyperparameter tuning.")
        self.__train_model()
        print("[XAI-MODEL] Trained model with best hyperparameters.")
        self.feature_importances = self.model.feature_importances_

        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X_train)
        self.__log()
        print("[XAI-MODEL] Explanation process completed.")