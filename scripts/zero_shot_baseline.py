# module for data handling 
import pandas as pd
import numpy as np
import json
from toon_format import encode, decode

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

class ZeroShotBaseline:
    def __init__(self, dataset: Dataset, model: Model,
                 prompt_gen_fn: callable, cot_flag: bool = False, 
                 cot: COT = None):

        # inputs
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        self.model = model

        self.cot_flag = cot_flag
        self.id = "zero-shot" if not cot_flag else "zero-shot-cot"
        self.cot = cot

        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = None
        self.output_file = f"data/batches/{self.dataset.name}_{self.id}_baseline_batches.jsonl"
        self.gcp_uri = None
        self.base_output_dir = f"gs://{BUCKET_NAME}/batch_outputs/gemini"
        self.job = None
        self.destination_file_name = None

    def __create_batch(self, request_id: str, prompt: str):

        thinking_budget = 0
        if self.cot_flag and self.cot is not None:
            thinking_budget = self.cot.thinking_budget

        return {
            "key": request_id,
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.model.temperature,
                    "maxOutputTokens": self.model.max_tokens,
                    "thinkingConfig": { 
                        "thinkingBudget": thinking_budget
                    },
                },
            },
        }

    def create_batch_prompts(self):

        # load the dataset
        df = pd.read_csv(self.dataset_config["dataset_path"])
        df = self.dataset.preprocess_fn(df)

        # get the original data splits
        train_df = df.loc[self.dataset_config["train_data_idx"]]
        test_df = df.loc[self.dataset_config["test_data_idx"]]
        labels = train_df[self.dataset.target_col].unique().tolist()
        test_df = test_df.drop(columns=[self.dataset.target_col], axis=1)

        self.batches = []
        batch_id = 0

        # creation of batches
        for idx, row in test_df.iterrows():

            request_id = f"baseline_{self.id}_batch-{idx}"
            example_features = encode(row.to_dict())
            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                example=example_features,
                labels=labels
            )
            batch = self.__create_batch(request_id, prompt)
            self.batches.append(batch)
            batch_id += 1
    
    def save_batches_as_jsonl(self):

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    def upload_batches_to_gcs(self):

        destination_blob_name = f"batch_inputs/gemini/{self.dataset.name}_{self.id}_baseline_batches.jsonl"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(self.output_file)

        self.gcp_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

        print(f"File {self.output_file} uploaded to {destination_blob_name}")
    
    def submit_batch_inference_job(self):

        OUTPUT_DIR = f"{self.base_output_dir}/{self.dataset.name}_{self.id}_{int(time.time())}/"

        client = genai.Client(vertexai=True, 
                              project=PROJECT_ID, 
                              location=LOCATION)

        job = self.job = client.batches.create(
            model=self.model.name,
            src=self.gcp_uri,
            config=CreateBatchJobConfig(dest=OUTPUT_DIR),
        )

        print(f"Submitted Job: {job.name}")
        print(f"Output base dir: {self.base_output_dir}")
    
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        while job.state not in completed_states:
            time.sleep(30)
            job = client.batches.get(name=job.name)
            print(f"{job.name} state: {job.state}")

        print(f"Final state: {job.state}")
    
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

            if f"{self.dataset.name}_{self.id}" in i.name:
                if i.updated > most_recent_time:
                    most_recent_time = i.updated
                    file_to_download = i.name

        if not file_to_download == None:
            self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_{self.id}_baseline_predictions.jsonl"
            blob = bucket.blob(file_to_download)
            blob.download_to_filename(self.destination_file_name)
            print(f"Downloaded {file_to_download} to {self.destination_file_name}.")
        else:
            raise ValueError("No predictions.jsonl file found in GCS location.")


