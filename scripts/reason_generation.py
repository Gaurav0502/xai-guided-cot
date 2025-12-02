# module for data handling 
from xmlrpc import client
import pandas as pd
import numpy as np
import json
from toon_format import encode, decode

# module for prompting
from together import Together
from scripts.diverse_examples import get_diverse_examples

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model

# module for env variables
import os
import dotenv
dotenv.load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class ReasonGenerator:
    def __init__(self, dataset: Dataset, model: Model, prompt_gen_fn: callable):

        # inputs
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        self.model = model

        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.output_file = f"data/batches/{self.dataset.name}_reasoning_batches.jsonl"
        self.responses = []

    def __create_batch(self, request_id: str, prompt: str):

        return {
            "custom_id": request_id,
            "body": {
                "model": self.model.name,
                "messages": [
                    {"role": "user",
                    "content": prompt}
                ],
            "max_tokens": self.model.max_tokens,
            "temperature": self.model.temperature
            }
        }

    def create_batch_prompts(self):
        
        unique_row_idx = get_diverse_examples(self.dataset.shap_vals_path)

        feature_imps = self.dataset_config["feature_importances"]
        feature_imps = encode(feature_imps)

        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        train_df = pd.DataFrame.from_dict(zip(train_idx, train_preds),
                                         columns=["idx", "prediction"])
        train_df.set_index("idx", inplace=True)

        df = pd.read_csv(self.dataset.path)
        df = df.loc[unique_row_idx]
        train_df = train_df.loc[unique_row_idx]

        df_shap = pd.read_csv(self.dataset.shap_vals_path)
        df_shap = df_shap.loc[unique_row_idx]
        self.batches = []
        batch_id = 0

        for idx, row in df.iterrows():

            example_raw_features = encode(row.to_dict())
            example_shap_values = encode(df_shap.loc[idx].to_dict())
            prediction = train_df.loc[idx, "prediction"]
            ground_truth = row[self.dataset.target_col]

            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                prediction=prediction,
                ground_truth=ground_truth,
                ft_raw_vals=example_raw_features,
                shap_vals=example_shap_values,
                feature_importances=feature_imps
            )

            request_id = f"reasoning_batch-{batch_id}"

            batch = self.__create_batch(
                request_id=request_id,
                prompt=prompt
            )
            self.batches.append(batch)
            batch_id += 1
    
    def save_batches_as_jsonl(self):

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    def submit_batches(self):

        client = Together(api_key=TOGETHER_API_KEY)

        file_resp = client.files.upload(file=self.output_file, 
                                        purpose="batch-api")
        
        batch = client.batches.create_batch(file_id=file_resp.id,
                                            endpoint="/v1/chat/completions" )
        batch_status = None

        while batch.status not in ["COMPLETED", "FAILED"]:
            time.sleep(10)
            batch_status = client.batches.get_batch(batch.id)
            print(f"Current Status: {batch_status.status}")
        
        if batch.status == "COMPLETED":
            destination_file_name = f"data/batch_outputs/{self.dataset.name}_reasoning_predictions.jsonl"

            client.files.retrieve_content(id=batch_status.output_file_id, 
                                          output=destination_file_name)
    
        






