# module for data handling 
import pandas as pd
import numpy as np
import json
from toon_format import encode, decode

# module for prompting
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model

# module for env variables
import os
import dotenv
dotenv.load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

class ObjectiveJudge:
    def __init__(self, dataset: Dataset, model: Model, prompt_gen_fn: callable):

        # inputs
        self.dataset = dataset
        self.model = model
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))
        
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = []
        self.evals = []
        self.batch_id = None
        self.filepath = f"data/batch_outputs/{self.dataset.name}_objective_judge_evaluations.jsonl"
    
    def __create_batch(self, request_id: str, prompt: str) -> Request:

        return Request(
            custom_id=request_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model.name,
                max_tokens=self.model.max_tokens,
                temperature=self.model.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        )

    def create_batch_prompts(self, reasoning: dict):

        unique_row_idx = list(reasoning.keys())

        feature_imps = self.dataset_config["feature_importances"]
        feature_imps = encode(feature_imps)

        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        train_df = pd.DataFrame.from_dict(dict(zip(train_idx, train_preds)),
                                         columns=["prediction"],
                                         orient="index")
        
        df = pd.read_csv(self.dataset.path)
        df = self.dataset.preprocess_fn(df)
        df = df.loc[unique_row_idx]
        train_df = train_df.loc[unique_row_idx]

        df_shap = pd.read_csv(self.dataset.shap_vals_path)
        df_shap.set_index("idx", inplace=True)
        df_shap = df_shap.loc[unique_row_idx]
        self.batches = []
        batch_id = 0

        for idx, row in df.iterrows():

            example_raw_features = encode(row.to_dict())
            example_shap_values = encode(df_shap.loc[idx].to_dict())
            prediction = train_df.loc[idx, "prediction"]
            ground_truth = row[self.dataset.target_col]
            reason = reasoning[idx]

            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                prediction=prediction,
                ground_truth=ground_truth,
                ft_raw_vals=example_raw_features,
                shap_vals=example_shap_values,
                feature_importances=feature_imps,
                reason=reason
            )

            request_id = f"judge_batch-{idx}"

            batch = self.__create_batch(
                request_id=request_id,
                prompt=prompt
            )
            self.batches.append(batch)
            batch_id += 1
        
    def __retrieve_batch_results(self, client: anthropic.Anthropic):

        results = client.messages.batches.results(
            self.batch_id
        )

        result_types = {"succeeded":0, "errored":0, "expired":0}

        for result in results:
            match result.result.type:
               case "succeeded":
                    result_types["succeeded"] += 1
                    result_dict = {
                        "request_id": result.custom_id,
                        "evaluation": result.result.message.content[0].text
                    }
                    self.evals.append(result_dict)
               case "errored":
                    result_types["errored"] += 1
               case "expired":
                    result_types["expired"] += 1
        
        print("Batch result types:", result_types)

    def __save_to_jsonl(self):

        with open(self.filepath, "w") as f:
            for eval in self.evals:
                f.write(json.dumps(eval) + "\n")
        print(f"Saved evaluations to {self.filepath}")

    def submit_batch(self):

        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        batch_info = client.messages.batches.create(
            requests=self.batches
        )

        print("Submitted batch with id:", batch_info.id)
        self.batch_id = batch_info.id

        message_batch = None
        while True:
            message_batch = client.messages.batches.retrieve(
                self.batch_id
            )
            if message_batch.processing_status == "ended":
                break

            print(f"Batch {self.batch_id} is still processing...")
            time.sleep(60)

        print(f"Batch {self.batch_id} has completed processing.")

        self.__retrieve_batch_results(client)
        self.__save_to_jsonl()
