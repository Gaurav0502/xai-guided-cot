import os
from dotenv import load_dotenv
load_dotenv()

# for tuning the tree-based
# model.
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")

# gcp related constants
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCATION = os.getenv("LOCATION")

# llm api keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


# set of supported explainable models
SUPPORTED_EXPLAINABLE_MODELS = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "AdaBoostClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
}

# maximum clusters
# budget for 
# diverse examples
MAX_CLUSTERS_BUDGET = 10

# prompting strategies
PROMPTING_STRATEGIES = ["xai-guided-cot", "zero-shot-prompting", 
                        "zero-shot-cot"]

# global sleep time
SLEEP_TIME = 60


# expriment setup
from scripts.configs import Dataset, Model
from scripts.preprocess import (preprocess_titanic,
                                preprocess_diabetes,
                                preprocess_loan,
                                preprocess_mushroom)

titanic_dataset = Dataset(
    name="titanic",
    path="data/datasets/titanic.csv",
    config_file_path="data/dataset_config/titanic_config.json",
    shap_vals_path="data/shap_values/titanic_shap.csv",
    preprocess_fn=preprocess_titanic,
    target_col="Survived",
    labels={0: "Did not survive", 1: "Survived"}
)

diabetes_dataset = Dataset(
    name="diabetes",
    path="data/datasets/diabetes.csv",
    config_file_path="data/dataset_config/diabetes_config.json",
    shap_vals_path="data/shap_values/diabetes_shap.csv",
    preprocess_fn=preprocess_diabetes,
    target_col="Outcome",
    labels={0: "No", 1: "Yes"}
)

loan_dataset = Dataset(
    name="loan",
    path="data/datasets/loan.csv",
    config_file_path="data/dataset_config/loan_config.json",
    shap_vals_path="data/shap_values/loan_shap.csv",
    preprocess_fn=preprocess_loan,
    target_col="Loan_Status",
    labels={0: "Not Approved", 1: "Approved"}
)

mushroom_dataset = Dataset(
    name="mushroom",
    path="data/datasets/mushroom.csv",
    config_file_path="data/dataset_config/mushroom_config.json",
    shap_vals_path="data/shap_values/mushroom_shap.csv",
    preprocess_fn=preprocess_mushroom,
    target_col="Class",
    labels={0: "Poisonous", 1: "Edible"}
)

reasoning_gen_model = Model(
    provider="together",
    name="deepseek-ai/DeepSeek-R1",
    temperature=0.6,
    max_tokens=4096
)

objective_judge_model = Model(
    provider="anthropic",
    name="claude-haiku-4-5",
    temperature=0.6,
    max_tokens=4096
)

cot_model = Model(
    provider="google",
    name="gemini-2.5-flash",
    temperature=0.0,
    max_tokens=4096
)