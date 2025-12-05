
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

VALID_PROVIDERS = ["google", "anthropic", "together"]
VALID_MODELS = {
    "google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "anthropic": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", 
                  "claude-sonnet-4-5", "claude-haiku-4-5"],
    "together": ["deepseek-ai/DeepSeek-R1"]
}