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

# valid LLM providers and models
VALID_PROVIDERS = ["google", "anthropic", "together"]
VALID_MODELS = {
    "google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "anthropic": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", 
                  "claude-sonnet-4-5", "claude-haiku-4-5"],
    "together": ["deepseek-ai/DeepSeek-R1"]
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
