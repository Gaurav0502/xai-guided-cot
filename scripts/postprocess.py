import json
import pandas as pd
import re
from typing import Tuple


def parse_baseline_llm_results(
    results_jsonl_path: str, config_file_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse baseline LLM results from zero_shot baseline batch inference output.

    Args:
        results_jsonl_path: Path to the JSONL file containing batch results
        config_file_path: Path to the dataset config JSON file containing ground truth
    """
    with open(config_file_path, "r") as f:
        config = json.load(f)

    test_data_idx = config["test_data_idx"]
    test_predictions = config["test_predictions"]

    unmasked_results = []
    masked_results = []

    with open(results_jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            response = json.loads(line.strip())

            # Parse the key to get masked/unmasked and batch ID
            key = response["key"]

            if "unmasked" in key:
                is_masked = False
                batch_id = int(key.split("baseline_unmasked_batch-")[1])
            elif "masked" in key:
                is_masked = True
                batch_id = int(key.split("baseline_masked_batch-")[1])
            else:
                print(f"Warning: Unknown key format: {key}")
                continue

            # Extract response details
            candidates = response.get("response", {}).get("candidates", [])

            if candidates:
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "UNKNOWN")

                # Extract the prediction text
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                if parts:
                    prediction_text = parts[0].get("text", "").strip()
                else:
                    prediction_text = ""
            else:
                finish_reason = "NO_CANDIDATES"
                prediction_text = ""

            # Check if completed normally
            if finish_reason == "STOP":
                completed = 1
            else:
                completed = 0

            # Parse prediction as numeric
            try:
                prediction = int(float(prediction_text)) if prediction_text else None
            except ValueError:
                prediction = prediction_text  # Keep as string, should not happen

            # Get ground truth
            ground_truth = test_predictions[batch_id] if batch_id < len(test_predictions) else None
            test_idx = test_data_idx[batch_id] if batch_id < len(test_data_idx) else None

            # Check correctness
            correct = (
                (prediction == ground_truth)
                if prediction is not None and ground_truth is not None
                else None
            )

            result = {
                "batch_id": batch_id,
                "test_idx": test_idx,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
                "finish_reason": finish_reason,
                "completed": completed,
                "raw_output": prediction_text,
            }

            if is_masked:
                masked_results.append(result)
            else:
                unmasked_results.append(result)

    unmasked_df = pd.DataFrame(unmasked_results)
    masked_df = pd.DataFrame(masked_results)

    if not unmasked_df.empty:
        unmasked_df = unmasked_df.sort_values("batch_id").reset_index(drop=True)

    if not masked_df.empty:
        masked_df = masked_df.sort_values("batch_id").reset_index(drop=True)

    return unmasked_df, masked_df


def summarize_baseline_results(
    unmasked_df: pd.DataFrame, masked_df: pd.DataFrame
) -> dict:
    """
    Generate summary statistics for baseline results.
    """
    summary = {}

    for name, df in [("unmasked", unmasked_df), ("masked", masked_df)]:
        if df.empty:
            summary[name] = {"count": 0}
            continue

        total = len(df)
        completed = df["completed"].sum()
        correct = df["correct"].sum() if "correct" in df.columns else 0

        summary[name] = {
            "total": total,
            "completed": completed,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "accuracy_of_completed": correct / completed if completed > 0 else 0,
        }

    return summary


def parse_reasoning_llm_results(results_jsonl_path: str) -> dict:
    
    reasoning = dict()
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            response = json.loads(line.strip())
            row_id = int(response["custom_id"].split("-")[1])
            llm_output = response["response"]["body"]["choices"][0]["message"]["content"]
            reasoning[row_id] = str(llm_output.split("<REASONING>")[1].replace("\n</REASONING>", "").strip())
    
    return reasoning

def parse_objective_judge_results(results_jsonl_path: str) -> dict:

    evaluations = dict()
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            response = json.loads(line.strip())
            row_id = int(response["request_id"].split("-")[1])

            llm_output = response["evaluation"]
            metrics = json.loads(llm_output.split("EVALUATION:")[1])["metrics"]
            evaluations[row_id] = metrics
    
    return evaluations

def parse_cot_llm_results(results_jsonl_path: str) -> dict:
    
    predictions = dict()
    interrupted_requests = 0
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            response = json.loads(line.strip())
            row_id = int(response["key"].split("-")[1].split("_")[0])

            llm_output = ""
            if "avgLogprobs" in response["response"]["candidates"][0]:
                llm_output = response["response"]["candidates"][0]["content"]["parts"][0]["text"]
                predictions[row_id] = llm_output.split("FINAL PREDICTION:")[-1].strip()
                predictions[row_id] = re.search(r'\d+', predictions[row_id]).group() 
            else:
                interrupted_requests += 1

    print(f"{interrupted_requests} requests were interrupted due to token limit and are ignored for evaluation.")
    return predictions
