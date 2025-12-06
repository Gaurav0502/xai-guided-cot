import json
import pandas as pd
import re


def parse_baseline_llm_results(results_jsonl_path: str) -> pd.DataFrame:
    pass

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
            
