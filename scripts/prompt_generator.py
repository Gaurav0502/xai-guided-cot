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

            DATA:
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

            EVALUATION RUBRICS:
            You must perform a detailed analysis and report your findings in the specified XML format.

            METRICS (Use a 1-5 scale, where 5 is best):
            1. **Faithfulness (1-5):** How accurately does the reasoning reflect the **sign** (direction) and **relative magnitude** of the SHAP values? Score 1 if the reasoning contradicts the major SHAP values.
            2. **Consistency (1-5):** Does the reasoning focus on the features with the **highest absolute SHAP values** (most influential features)?
            3. **Coherence (1-5):** Is the reasoning grammatically sound, well-structured, and easy for a non-expert to understand?

            OUTPUT FORMAT:
            You must strictly adhere to the following XML format. Do not include any text outside of the XML tags.

            <thought>
            </thought>
            <evaluation>
                <metrics>
                    <faithfulness>[1-5]</faithfulness>
                    <consistency>[1-5]</consistency>
                    <coherence>[1-5]</coherence>
                </metrics>
            </evaluation>
    """