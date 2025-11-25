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
from scripts.configs import Dataset, Model

# module for env variables
import os
import dotenv
dotenv.load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCATION = "us-east4"

class ZeroShotBaseline:
    def __init__(self, dataset: Dataset, model: Model, prompt_gen_fn: callable):

        # inputs
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        self.model = model

        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = None
        self.output_file = f"data/batches/{self.dataset.name}_baseline_batches.jsonl"
        self.gcp_uri = None
        self.base_output_dir = f"gs://{BUCKET_NAME}/batch_outputs/gemini"
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
                    "maxTokens": self.model.max_tokens
                }
            }
        }

    def create_batch_prompts(self):

        # load the dataset
        df = pd.read_csv(self.dataset_config["dataset_path"])
        df = self.dataset.preprocess_fn(df)

        # get the original data splits
        train_df = df.loc[self.dataset_config["train_data_idx"]]
        test_df = df.loc[self.dataset_config["test_data_idx"]]
        labels = train_df[self.dataset_config["target_col"]].unique().tolist()
        test_df = test_df.drop(columns=[self.dataset_config["target_col"]], axis=1)

        # create test dataset that is masked
        cols = list(test_df.columns)
        masked_cols = ["x" + str(i) for i in range(len(cols))]
        col_mask_map = dict(zip(cols, masked_cols))
        test_df_masked = test_df.rename(columns=col_mask_map)

        self.batches = []
        batch_id = 0

        # creation of batches
        for idx, row in test_df.iterrows():

            # unmasked row
            request_id_unmasked = f"baseline_unmasked_batch-{batch_id}"
            example_features_unmasked = encode(row.to_dict())
            prompt_unmasked = self.prompt_gen_fn(
                dataset=self.dataset,
                example=example_features_unmasked,
                labels=labels
            )
            batch_unmasked = self.__create_batch(request_id_unmasked, prompt_unmasked)
            self.batches.append(batch_unmasked)

            # masked row
            row_masked = test_df_masked.loc[idx]
            request_id_masked = f"baseline_masked_batch-{batch_id}"
            example_features_masked = encode(row_masked.to_dict())
            prompt_masked = self.prompt_gen_fn(
                dataset=self.dataset,
                example=example_features_masked,
                labels=labels
            )

            batch_masked = self.__create_batch(request_id_masked, prompt_masked)
            self.batches.append(batch_masked) 
            batch_id += 1
    
    def save_batches_as_jsonl(self):

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    def upload_batches_to_gcs(self):

        destination_blob_name = f"batch_inputs/gemini/{self.dataset.name}_baseline_batches.jsonl"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(self.output_file)

        self.gcp_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

        print(f"File {self.output_file} uploaded to {destination_blob_name}")
    
    def submit_batch_inference_job(self):

        client = genai.Client(vertexai=True, 
                              project=PROJECT_ID, 
                              location=LOCATION)

        job = self.job = client.batches.create(
            model=self.model.name,
            src=self.gcp_uri,
            config=CreateBatchJobConfig(dest=self.base_output_dir),
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
                
            if "pH" in i.name:
                if i.updated > most_recent_time:
                    most_recent_time = i.updated
                    file_to_download = i.name

        if not file_to_download == None:
            destination_file_name = f"data/batch_outputs/{self.dataset.name}_baseline_predictions.jsonl"
            blob = bucket.blob(file_to_download)
            blob.download_to_filename(destination_file_name)
            print(f"Downloaded {file_to_download} to {destination_file_name}.")
        else:
            raise ValueError("No predictions.jsonl file found in GCS location.")


