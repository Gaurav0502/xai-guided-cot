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

def objective_judge_prompt_generator(
    dataset: Dataset,
    prediction: float,
    ground_truth: float,
    ft_raw_vals: dict,
    shap_vals: dict,
    feature_importances: dict,
    reason: str
) -> str:

    return f"""
ROLE:
You are an expert, objective judge for the tabular dataset '{dataset.name}'.
Your role is to assess the faithfulness and quality of the model's reasoning against the provided data.
Each example has features and a target label called '{dataset.target_col}'.

--- INPUT DATA ---
Here are the **Feature Values** for this specific example:
{ft_raw_vals}

Here are the **SHAP Values** (the source of truth for feature contribution):
{shap_vals}

Here are the overall **Feature Importances** (Global context):
{feature_importances}

The model predicted: {prediction}
The ground truth label is: {ground_truth}

Here is the **Model's Reasoning** for its prediction:
{reason}

--- EVALUATION RUBRICS ---
You must perform a detailed analysis.

### METRICS TO SCORE:
**Score on a 1.00 to 5.00 scale using only 0.25 increments (e.g., 4.00, 4.25, 4.50, 4.75, 5.00), where 5.00 is best.**

1. **Faithfulness:** How accurately does the reasoning reflect the **sign** (direction) and **relative magnitude** of the SHAP values? Score 1.00 if the reasoning contradicts the major SHAP values.
2. **Consistency:** Does the reasoning focus on the features with the **highest absolute SHAP values** (most influential features) and prioritize them correctly?
3. **Coherence:** Is the reasoning grammatically sound, well-structured, and easy for a non-expert to understand?

--- OUTPUT FORMAT ---
You must first output the full **Chain of Thought (CoT)**, explaining your scoring decisions. Immediately after your CoT, provide the structured JSON metrics block. **DO NOT generate any text, commentary, or markdown fences after the final closing brace of the JSON object.**

[Your detailed Chain of Thought goes here, incorporating all justification for the scores...]

Evaluation:
{{
    "metrics": {{
        "faithfulness": [1.00 to 5.00 in 0.25 increments],
        "consistency": [1.00 to 5.00 in 0.25 increments],
        "coherence": [1.00 to 5.00 in 0.25 increments]
    }}
}}

THE JSON OBJECT MUST BE PRECEDED BY "EVALUATION:" VERBATIM. NO EXTRA TEXT BEYOND THE JSON OBJECT.

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
