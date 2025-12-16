# modules used for data handling
import pandas as pd
import numpy as np
import json

# modules used for postprocessing
# llm output text
import re

# modules used for type hinting
from typing import Dict, Any

# function to parse zero-shot
# prompting baseline results
def parse_baseline_llm_results(
        results_jsonl_path: str
    ) -> Dict[int, int]:
    """
    Parses zero-shot baseline LLM results from a JSONL file and extracts numeric predictions.

    Args:
        results_jsonl_path (str): Path to the JSONL file with LLM results.

    Returns:
        Dict[int, int]: Mapping from row index to predicted class label.
    """
    
    # count interrupted requests
    interrupted_requests = 0

    # extract predictions
    predictions = dict()
    with open(results_jsonl_path, "r") as f:

        # process jsonl
        # line by line
        for line in f:

            # get response
            response = json.loads(line.strip())

            # extract row_id for mapping
            row_id = int(response["key"].split("-")[-1])

            # extract llm output
            llm_output = ""

            # verify if request stopped normally
            # and not interrupted due to token limit
            if response["response"]["candidates"][0]["finishReason"] == "STOP":

                # extract llm output text
                llm_output = response["response"]["candidates"][0]["content"]["parts"][0]["text"]
                prediction = llm_output.strip()
                predictions[row_id] = prediction

                # keep only numeric part
                num = re.search(r'\d+', predictions[row_id])
                if num != None:
                    # get class label
                    predictions[row_id] = int(num.group())
                else:
                    # if no numeric part found
                    # consider prediction as neither
                    # positive nor negative
                    predictions[row_id] = -1
            else:
                interrupted_requests += 1

    print(f"[POSTPROCESS] {interrupted_requests} requests were interrupted due to token limit and are ignored for evaluation.")
    
    return predictions

# function to parse reasoning output
def parse_reasoning_llm_results(results_jsonl_path: str) -> Dict[int, str]:
    """
    Parses reasoning outputs from a JSONL file and extracts the reasoning text for each example.

    Args:
        results_jsonl_path (str): Path to the JSONL file with LLM reasoning results.

    Returns:
        Dict[int, str]: Mapping from row index to extracted reasoning string.
    """

    # extract reasoning
    reasoning = dict()
    with open(results_jsonl_path, 'r') as f:

        # process jsonl
        # line by line
        for line in f:

            # get response
            response = json.loads(line.strip())

            # extract row_id for mapping
            row_id = int(response["custom_id"].split("-")[1])

            # extract llm text output
            llm_output = response["response"]["body"]["choices"][0]["message"]["content"]

            try:

                reasoning[row_id] = str(llm_output.split('</think>')[1].split('<REASONING>')[1].replace('</REASONING>', '').strip())
            
            except Exception:
                # if LLM does not follow the output format
                # and there are parsing issues, ignore
                # and continue
                continue
    
    return reasoning

# function to parse LLM-as-a-Judge results
def parse_objective_judge_results(results_jsonl_path: str) -> dict:
    """
    Parses LLM-as-a-Judge results from a JSONL file and extracts evaluation metrics for each example.

    Args:
        results_jsonl_path (str): Path to the JSONL file with judge results.

    Returns:
        dict: Mapping from row index to evaluation metrics dictionary.
    """

    # extract evaluations
    evaluations = dict()
    with open(results_jsonl_path, 'r') as f:

        # process jsonl
        # line by line
        for line in f:

            # get response
            response = json.loads(line.strip())

            # extract row_id for mapping
            row_id = int(response["request_id"].split("-")[1])

            # extract llm text output
            llm_output = response["evaluation"]

            try:

                # parse evaluation metrics
                metrics = json.loads(llm_output.split("EVALUATION:")[1])["metrics"]
                evaluations[row_id] = metrics

            except Exception:
                # if LLM does not follow the output format
                # and there are parsing issues, ignore
                # and continue
                continue
    
    return evaluations

# function to parse zero-shot
# chain-of-thought prompting baseline results
def parse_zero_shot_cot_llm_results(results_jsonl_path: str) -> Dict[int, int]:
    """
    Parses zero-shot chain-of-thought LLM results from a JSONL file and extracts numeric predictions.

    Args:
        results_jsonl_path (str): Path to the JSONL file with LLM results.

    Returns:
        Dict[int, int]: Mapping from row index to predicted class label.
    """

    # count interrupted requests
    interrupted_requests = 0

    # extract predictions
    predictions = dict()
    with open(results_jsonl_path, 'r') as f:

        # process jsonl
        # line by line
        for line in f:

            # get response
            response = json.loads(line.strip())

            # extract row_id for mapping
            row_id = int(response["key"].split("-")[-1])

            # extract llm output
            llm_output = ""

            # verify if request stopped normally
            # and not interrupted due to token limit
            if response["response"]["candidates"][0]["finishReason"] == "STOP":

                # extract  llm output text
                llm_output = response["response"]["candidates"][0]["content"]["parts"][0]["text"]

                try:

                    # extract final prediction
                    predictions[row_id] = llm_output.split("FINAL PREDICTION:")[-1].strip()

                except Exception:
                    # if LLM does not follow the output format
                    # and there are parsing issues, ignore
                    # and continue
                    continue

                # keep only numeric part
                num = re.search(r'\d+', predictions[row_id])
                if num != None:
                    # get class label
                    predictions[row_id] = int(num.group())
                else:
                    # if no numeric part found
                    # consider prediction as neither
                    # positive nor negative
                    predictions[row_id] = -1
            else:
                interrupted_requests += 1

    print(f"[POSTPROCESS] {interrupted_requests} requests were interrupted due to token limit and are ignored for evaluation.")
    return predictions

# helper function to compute mode
def compute_mode(lst: list[int]) -> int:
    """
    Computes the mode (most frequent value) of a list of integers.

    Args:
        lst (list[int]): List of integer values.

    Returns:
        int: The mode of the list.
    """

    # get frequency counts
    vals, counts = np.unique(lst, return_counts=True)

    # apply argmax to get mode
    mode = vals[np.argmax(counts)]
    return int(mode)

# function to parse 
# CoT LLM results with multiple agents
def parse_cot_llm_results(results_jsonl_path: str) -> Dict[int, int]:
    """
    Parses CoT LLM results with multiple agents from a JSONL file, aggregates predictions by majority vote.

    Args:
        results_jsonl_path (str): Path to the JSONL file with LLM results.

    Returns:
        Dict[int, int]: Mapping from row index to aggregated predicted class label.
    """
    
    # count interrupted requests
    interrupted_requests = 0

    # extract predictions
    predictions = dict()
    with open(results_jsonl_path, 'r') as f:

        # process jsonl
        # line by line
        for line in f:

            # get response
            response = json.loads(line.strip())

            # extract row_id and agent_id for mapping
            row_id = int(response["key"].split("-")[1].split("_")[0])
            agent_id = int(response["key"].split("-")[-1])

            llm_output = ""

            # verify if request stopped normally
            # and not interrupted due to token limit
            if response["response"]["candidates"][0]["finishReason"] == "STOP":

                # extract llm output text
                llm_output = response["response"]["candidates"][0]["content"]["parts"][0]["text"]

                try:

                    # extract final prediction
                    prediction = llm_output.split("FINAL PREDICTION:")[-1].strip()

                except Exception:
                    # if LLM does not follow the output format
                    # and there are parsing issues, ignore
                    # and continue
                    continue

                # keep only numeric part
                num = re.search(r'\d+', prediction)
                if num != None:
                    # get class label
                    prediction = int(num.group())
                else:
                    # if no numeric part found
                    # consider prediction as neither
                    # positive nor negative
                    prediction = -1

                # aggregate predictions
                if row_id not in predictions:
                    predictions[row_id] = [prediction]
                else:
                    predictions[row_id].append(prediction)
            else:
                interrupted_requests += 1
        
        # take majority vote
        for row_id in predictions:

            predictions[row_id] = compute_mode(predictions[row_id])

    print(f"[POSTPROCESS] {interrupted_requests} requests were interrupted due to token limit and are ignored for evaluation.")
    return predictions
