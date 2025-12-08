# module for data handling 
import pandas as pd
import numpy as np
import json
from toon_format import encode, decode
import random
import math

# module for prompting
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model, COT

# module for env variables
import os
import dotenv
dotenv.load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCATION = "us-east4"

class ICLClassifier:
    def __init__(self, dataset: Dataset, model: Model, cot: COT, prompt_gen_fn: callable):

        # inputs
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        self.model = model

        self.cot = cot
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = None
        self.output_file = f"data/batches/{self.dataset.name}_icl_batches.jsonl"
        self.gcp_uri = None
        self.base_output_dir = f"gs://{BUCKET_NAME}/batch_outputs/gemini"
        self.destination_file_name = None
        self.job = None
    
    def __create_batch(self, request_id: str, prompt: str):

        return {
            "key": request_id,
            "request": {
                "contents": [
                    {
                        "role": "user", 
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.model.temperature,
                    "maxOutputTokens": self.model.max_tokens,
                    "thinkingConfig": {
                        "thinkingBudget": self.cot.thinking_budget
                    }
                }
            }
        }
    
    def __prepare_example_sets(self):

        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        idx2pred = dict(zip(train_idx, train_preds))

        diverse_examples = list(self.cot.reasoning.keys())
        all_examples = [ (idx, idx2pred[idx]) for idx in diverse_examples ]
        
        random.shuffle(all_examples)

        n = self.cot.num_examples_per_agent
        N = len(all_examples)
        self.example_sets = [all_examples[i:i+n] for i in range(0, len(all_examples), n)]
    
    def create_batch_prompts(self):
        
        self.__prepare_example_sets()

        feature_imps = self.dataset_config["feature_importances"]

        test_idx = self.dataset_config["test_data_idx"]
        df = pd.read_csv(self.dataset.path)
        df = self.dataset.preprocess_fn(df)
        test_df = df.drop(self.dataset.target_col, axis=1, inplace=False).loc[test_idx]

        df_shap = pd.read_csv(self.dataset.shap_vals_path)
        df_shap.set_index("idx", inplace=True)
        self.batches = []

        agent_id = 0
        for example_set in self.example_sets:

            train_idx_subset, train_preds_subset = list(map(list, zip(*example_set)))

            train_df = df.loc[train_idx_subset]
            train_preds = dict(zip(train_idx_subset, train_preds_subset))

            df_shap_train = df_shap.loc[train_idx_subset]

            for idx, row in test_df.iterrows():

                test_sample = encode(row.to_dict())

                request_id = f"cot_batch-{idx}_agent-{agent_id}"

                prompt = self.prompt_gen_fn(
                    dataset=self.dataset,
                    examples=train_df,
                    example_predictions=train_preds,
                    feature_importances=feature_imps,
                    example_shap_values=df_shap_train,
                    reasoning=self.cot.reasoning,
                    test_example=test_sample
                )

                batch = self.__create_batch(request_id=request_id, 
                                            prompt=prompt)
                self.batches.append(batch)
            agent_id += 1

    def save_batches_as_jsonl(self):

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    def upload_batches_to_gcs(self):

        destination_blob_name = f"batch_inputs/gemini/{self.dataset.name}_icl_batches.jsonl"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(self.output_file)

        self.gcp_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

        print(f"[GCS CLIENT] File {self.output_file} uploaded to {destination_blob_name}")
    
    def submit_batch_inference_job(self):

        OUTPUT_DIR = f"{self.base_output_dir}/{self.dataset.name}_cot_{int(time.time())}/"

        client = genai.Client(vertexai=True, 
                              project=PROJECT_ID, 
                              location=LOCATION)

        job = self.job = client.batches.create(
            model=self.model.name,
            src=self.gcp_uri,
            config=CreateBatchJobConfig(
                dest=OUTPUT_DIR
            ),
        )

        print(f"[ICL CLASSIFIER] Submitted Job: {job.name}")
        print(f"[ICL CLASSIFIER] Output base dir: {OUTPUT_DIR}")
    
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        while job.state not in completed_states:
            time.sleep(60)
            job = client.batches.get(name=job.name)
            print(f"[ICL CLASSIFIER] {job.name} state: {job.state}")

        print(f"[ICL CLASSIFIER] Final state: {job.state}")

    def download_job_outputs_from_gcs(self):

        bucket_prefix = f"gs://{BUCKET_NAME}/"
        BUCKET_LOCATION=self.base_output_dir.split(bucket_prefix)[1]

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)

        most_recent_time = datetime.min.replace(tzinfo=timezone.utc)
        file_to_download = None
        for i in bucket.list_blobs(prefix=BUCKET_LOCATION):

            if not i.name.endswith("predictions.jsonl"):
                    continue
                
            if f"{self.dataset.name}_cot" in i.name:
                if i.updated > most_recent_time:
                    most_recent_time = i.updated
                    file_to_download = i.name

        if not file_to_download == None:
            self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_cot_predictions.jsonl"
            blob = bucket.blob(file_to_download)
            blob.download_to_filename(self.destination_file_name)
            print(f"[GCS CLIENT] Downloaded {file_to_download} to {self.destination_file_name}.")
        else:
            raise ValueError("No predictions.jsonl file found in GCS location.")