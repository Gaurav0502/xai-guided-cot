import json
import pandas as pd


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
    

            
