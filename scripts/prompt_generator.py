from scripts.configs import Dataset
import pandas as pd

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
def zero_shot_cot_prompt_generator(dataset: Dataset, example: str, labels: list[str]) -> str:

    return f"""
            You are a classifier for the tabular dataset '{dataset.name}'.
            Each example has features and a target label called '{dataset.target_col}'.
            Given the following feature values for one example, predict the label.
            Return EXACTLY one of the following labels:
            {", ".join(map(str, labels))}

            Here are the feature values:
            {example}

            Question: What is the predicted value of '{dataset.target_col}' for this example?

            Think step by step.

            OUTPUT FORMAT:

            Your reasoning process should be detailed and step-by-step, culminating in the final predicted label for the test example.

            FINAL PREDICTION: [YOUR FINAL PREDICTION LABEL]

            NOTE:
            - THE FINAL PREDICTION MUST BE PRECEDED BY "FINAL PREDICTION:" VERBATIM. NO EXTRA TEXT BEYOND THE LABEL.
            - YOUR REASONING MUST BE WITHIN A WORD LIMIT OF 500 WORDS.
            - IF YOU ARE NOT CONFIDENT, PLEASE PROVIDE YOUR BEST GUESS FROM THE ALLOWED LABELS (ONLY A NUMBER NOT TEXT HERE!).
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
You must give your output in following format:

CHAIN OF THOUGHT:

FAITHFULNESS ANALYSIS: <YOUR DETAILED ANALYSIS HERE IN ONE PARAGRAPH>
CONSISTENCY ANALYSIS: <YOUR DETAILED ANALYSIS HERE IN ONE PARAGRAPH>
COHERENCE ANALYSIS: <YOUR DETAILED ANALYSIS HERE IN ONE PARAGRAPH

EVALUATION:
{{
    "metrics": {{
        "faithfulness": [1.00 to 5.00 in 0.25 increments],
        "consistency": [1.00 to 5.00 in 0.25 increments],
        "coherence": [1.00 to 5.00 in 0.25 increments]
    }}
}}

NOTE:
- YOUR OUTPUT MUST END WITH THE JSON OBJECT AS SPECIFIED.
- THE JSON OBJECT MUST BE PRECEDED BY "EVALUATION:" VERBATIM. NO EXTRA TEXT BEYOND THE JSON OBJECT.
- YOUR THOUGHT PROCESS MUST BE WITHIN A WORD LIMIT OF 800 WORDS.
- ALL SCORES MUST BE PROVIDED INSIDE THE JSON OBJECT ONLY. FOLLOW THE OUTPUT FORMAT STRICTLY. 

"""

def reasoning_prompt_generator(dataset: Dataset, prediction: str, ground_truth: str, 
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

            OUTPUT FORMAT:

            <think>YOUR DETAILED REASONING PROCESS HERE</think>
            <REASONING>YOUR FINAL REASONING PARAGRAPH HERE</REASONING>

            NOTE: 
            1. YOUR FINAL RESULT MUST BE ONE PARAGRAPH OF REASONING THAT LEADS TO PREDICTION WITHIN <REASONING></REASONING> TAGS.
            2. STICK TO ONLY ENGLISH LANGUAGE. NO OTHER LANGUAGES MUST BE USED!
            3. AFTER THE CLOSING </REASONING> TAG, NO ADDITIONAL TEXT OR MARKDOWN FENCES MUST BE PROVIDED.
            4. YOUR THINKING MUST BE WITHIN A WORD LIMIT OF 500 WORDS.
            5. DO NOT MIX UP THE TAGS. STICK TO OUTPUT FORMAT STRICTLY.

    """
def cot_prompt_generator(
    dataset: Dataset,
    examples: pd.DataFrame,
    example_predictions: list,
    feature_importances: dict,
    example_shap_values: pd.DataFrame,
    reasoning: dict,
    test_example: dict
) -> str:

    example_strs = []
    for i, row in examples.iterrows():
        ex_idx = i
        ex_features = row.to_dict()
        ex_prediction = example_predictions[i]
        ex_shap_values = example_shap_values.loc[ex_idx].to_dict()
        ex_reasoning = reasoning[ex_idx]

        example_str = f"""
        Example {i+1}:
        Feature Values: {ex_features}
        SHAP Values: {ex_shap_values}
        Model Prediction: {ex_prediction}
        Model Reasoning: {ex_reasoning}
        """
        example_strs.append(example_str)

    examples_block = "\n".join(example_strs)

    return f"""
    CONTEXT:
    You are a classifier for the tabular dataset '{dataset.name}'.
    Each example has features and a target label called '{dataset.target_col}'.
    Given the following training examples with their feature values, SHAP values, model predictions, and reasonings, predict the label for the test example.
    The possible labels are:
    {dataset.labels}

    TRAINING EXAMPLES:

    Global Feature Importances: {feature_importances}

    {examples_block}

    TEST EXAMPLE:
    Feature Values: 
    {test_example}

    Question: What is the predicted value of '{dataset.target_col}' for this test example?

    Think step by step, considering how the SHAP values influenced the model's predictions in the training examples.

    OUTPUT FORMAT:

    Your reasoning process should be detailed and step-by-step, culminating in the final predicted label for the test example.

    FINAL PREDICTION: [YOUR FINAL PREDICTION LABEL]

    NOTE:
    THE FINAL PREDICTION MUST BE PRECEDED BY "FINAL PREDICTION:" VERBATIM. NO EXTRA TEXT BEYOND THE LABEL.
    YOUR REASONING MUST BE WITHIN A WORD LIMIT OF 500 WORDS.
    """