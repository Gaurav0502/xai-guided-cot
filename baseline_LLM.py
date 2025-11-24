import os
from typing import Any, Dict, List, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from google import genai
from google.genai.types import HttpOptions

from preprocess import (
    DatasetConfig,
    load_tabular_dataset,
    preprocess_titanic,
    preprocess_wine_quality,
    preprocess_world_air_quality,
)

PromptBuilder = Callable[[DatasetConfig, pd.Series, List[Any]], str]

def create_vertex_llm_client() -> genai.Client:
    # Use Google Vertex AI GenAI client

    # Set these environment variables in your shell before running:
    # export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT_ID
    # export GOOGLE_CLOUD_LOCATION=global
    # export GOOGLE_GENAI_USE_VERTEXAI=True

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")

    client = genai.Client(
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )
    return client


def build_zero_shot_prompt(
    cfg: DatasetConfig,
    row: pd.Series,
    label_values: List[Any],
) -> str:
    """
    Build a simple zero-shot classification prompt for a single tabular row.
    We provide the dataset name, target column, and allowed label values.
    """
    prompt_lines = [
        f"You are a classifier for the tabular dataset '{cfg.name}'.",
        f"Each example has features and a target label called '{cfg.target_col}'.",
        "",
        "Given the following feature values for one example, predict the label.",
        "Return EXACTLY one of the following labels (just the value, no extra words):",
        ", ".join(map(str, label_values)),
        "",
        "Here are the feature values:",
    ]

    # In this function we append column names and values explicitly
    for col, val in row.items():
        prompt_lines.append(f"- {col}: {val}")

    prompt_lines.append("")
    prompt_lines.append(
        f"Question: What is the predicted value of '{cfg.target_col}' for this example?"
    )
    prompt_lines.append(
        "Answer with exactly one of the allowed label values, nothing else."
    )

    return "\n".join(prompt_lines)


def build_zero_shot_prompt_anonymized(
    cfg: DatasetConfig,
    row: pd.Series,
    label_values: List[Any]
) -> str:
    """
    Build a zero-shot classification prompt where feature names are anonymized.
    Still tells the model what the target label is called and the allowed values.
    """
    feature_prefix = "x"
    # Create synthetic feature names x1, x2 ...
    feature_names = [f"{feature_prefix}{i+1}" for i in range(len(row))]
    feature_values = list(row.values)

    prompt_lines = [
        "You are a classifier for a tabular prediction task.",
        "Each example has several numeric features and a target label.",
        "",
        "Given the following feature values for one example, predict the label.",
        "Return EXACTLY one of the following labels (just the value, no extra words):",
        ", ".join(map(str, label_values)),
        "",
        "Here are the feature values:",
    ]

    for name, val in zip(feature_names, feature_values):
        prompt_lines.append(f"- {name}: {val}")

    prompt_lines.append("")
    prompt_lines.append("Question: What is the predicted label for this example?")
    prompt_lines.append(
        "Answer with exactly one of the allowed label values, nothing else."
    )

    return "\n".join(prompt_lines)


def parse_llm_label(raw_text: str, label_values: List[Any]) -> Any:
    """
    Try to parse the LLM output into one of the known label values.
    """
    text = raw_text.strip()
    # Exact match first
    for lab in label_values:
        if text == str(lab):
            return lab
    print(f"Warning: LLM output '{text}' did not match any label exactly.")
    # Otherwise, try token-based match
    tokens = text.replace(",", " ").split()
    for lab in label_values:
        if str(lab) in tokens:
            return lab

    # Fallback: return the first label (couldn't parse)
    print(f"Warning: Fallback to first label value '{label_values[0]}'.")
    return label_values[0]


def llm_zero_shot_evaluate_dataset(
    client: genai.Client,
    cfg: DatasetConfig,
    model_name: str,
    max_samples: int,
    prompt_builder: PromptBuilder,
    prompt_mode: str,
) -> Dict[str, Any]:
    """
    Run zero-shot classification with a Gemini model on a subset of the test data
    for a given DatasetConfig and a given prompt-building strategy.
    """
    X_train, X_test, y_train, y_test = load_tabular_dataset(cfg)

    # Use labels present in training data as the allowed label set
    label_values = sorted(pd.Series(y_train).unique().tolist())

    # Subsample test rows to control cost
    n = min(max_samples, len(X_test))
    X_sample = X_test.sample(n=n, random_state=cfg.random_state)
    y_sample = y_test.loc[X_sample.index]

    y_pred_llm: List[Any] = []

    for idx, row in X_sample.iterrows():
        prompt = prompt_builder(cfg, row, label_values)
        # print(f"\n[LLM Prompt for index {idx} | mode={prompt_mode}]:\n{prompt}\n")

        # Call Gemini on Vertex AI
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        raw_answer = response.text 
        # print(f"[LLM Raw Answer | mode={prompt_mode}]: {raw_answer}")
        pred_label = parse_llm_label(raw_answer, label_values)
        y_pred_llm.append(pred_label)

    # Compute metrics
    acc = accuracy_score(y_sample, y_pred_llm)
    f1 = f1_score(y_sample, y_pred_llm, average="macro")

    return {
        "dataset": cfg.name,
        "prompt_mode": prompt_mode,
        "num_eval_samples": n,
        "accuracy": acc,
        "f1_macro": f1,
        "y_true": y_sample,
        "y_pred": np.array(y_pred_llm),
    }


def run_all_llm_zero_shot(
    client: genai.Client,
    configs: List[DatasetConfig],
    model_name: str = "gemini-2.5-flash",
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Run zero-shot LLM baselines on all datasets for both prompt modes
    (real feature names vs anonymized) and summarize metrics.
    """
    rows = []

    modes: List[tuple[str, PromptBuilder]] = [
        ("with_names", build_zero_shot_prompt),
        ("anonymized", build_zero_shot_prompt_anonymized),
    ]

    for cfg in configs:
        for mode_name, builder in modes:
            print(f"\n=== LLM zero-shot: {cfg.name} | mode={mode_name} ===")
            result = llm_zero_shot_evaluate_dataset(
                client=client,
                cfg=cfg,
                model_name=model_name,
                max_samples=max_samples,
                prompt_builder=builder,
                prompt_mode=mode_name,
            )
            print(
                f"{cfg.name} [{mode_name}]: "
                f"accuracy={result['accuracy']:.3f}, "
                f"f1_macro={result['f1_macro']:.3f} "
                f"on {result['num_eval_samples']} samples"
            )
            rows.append(
                {
                    "dataset": cfg.name,
                    "prompt_mode": mode_name,
                    "num_eval_samples": result["num_eval_samples"],
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                }
            )

    return pd.DataFrame(rows)


def main_llm_zero_shot():
    # Reuse the same DatasetConfig definitions as in tree baseline
    world_aqi_cfg = DatasetConfig(
        name="world_air_quality",
        path="data/AQI.csv",
        target_col="AQI_Category",
        preprocess_fn=preprocess_world_air_quality,
    )

    wine_cfg = DatasetConfig(
        name="wine_quality",
        path="data/WineQT.csv",
        target_col="quality",
        preprocess_fn=preprocess_wine_quality,
    )

    titanic_cfg = DatasetConfig(
        name="titanic",
        path="data/Titanic.csv",
        target_col="Survived",
        preprocess_fn=preprocess_titanic,
    )

    configs = [world_aqi_cfg, wine_cfg, titanic_cfg]

    client = create_vertex_llm_client()

    summary_llm = run_all_llm_zero_shot(
        client=client,
        configs=configs,
        model_name="gemini-2.5-flash",
        max_samples=50,
    )

    summary_llm.to_csv("llm_zero_shot_summary.csv", index=False)
    print("\nSaved LLM zero-shot summary to llm_zero_shot_summary.csv")


if __name__ == "__main__":
    main_llm_zero_shot()
