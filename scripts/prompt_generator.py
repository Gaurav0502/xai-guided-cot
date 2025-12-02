from scripts.configs import Dataset

def zero_shot_prompt_generator(dataset: Dataset, example: str, labels: list[str]) -> str:

    return f"""
            You are a classifier for the tabular dataset '{dataset.name}'.
            Each example has features and a target label called '{dataset.target_col}'.
            Given the following feature values for one example, predict the label.
            Return EXACTLY one of the following labels (just the value, no extra words):
            {", ".join(map(str, labels))}

            Here are the feature values:
            {example}

            Question: What is the predicted value of '{dataset.target_col}' for this example?

            Note: Answer with exactly one of the allowed label values, nothing else.
    """

def reasoning_generator_prompt(dataset: Dataset, prediction: str, ground_truth: str, 
                               ft_raw_vals: str, shap_vals: str, feature_importances: str) -> str:
    
    return f"""
            ROLE:
            You are an expert in machine learning explanability and interpretability.
            You have been given feature values and SHAP values for an example from the tabular dataset '{dataset.name}'.
            The target label for this dataset is '{dataset.target_col}'.
            Your task is to provide a detailed reasoning process that leads to a prediction for the target label.
            Use the SHAP values to understand the contribution of each feature to the prediction.

            INSTRUCTIONS:
            1. Determine whether the model's prediction is correct by comparing it to the ground truth label.
            2. Understand the features and shap values provided.
            3. Provide a detailed reasoning process that explains how the feature values and their SHAP values lead to the model's prediction.
            4. Your final output should be a single paragraph of reasoning that culminates in the prediction in <REASONING></REASONING> tags.


            DATA:
            Predicted Label: {prediction}
            Ground Truth Label: {ground_truth}

            Here are the feature values for the example:
            {ft_raw_vals}

            Here are the SHAP values for the example:
            {shap_vals}

            Here are the overall feature importances for the model:
            {feature_importances}

            Based on the above information, provide your reasoning and prediction.

            NOTE: YOUR FINAL RESULT MUST BE ONE PARAGRAPH OF REASONING THAT LEADS TO PREDICTION WITHIN <REASONING></REASONING> TAGS.
    """
