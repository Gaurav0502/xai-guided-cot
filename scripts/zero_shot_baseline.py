# module for data handling 
import pandas as pd
import numpy as np
import json
from toon_format import encode

# module for prompting
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model, COT

# env variable
from scripts.constants import (BUCKET_NAME, 
                               PROJECT_ID, 
                               LOCATION,
                               SLEEP_TIME)

# module used for type hinting
from typing import Dict, Any, Callable

# zero-shot baseline
# class
class ZeroShotBaseline:

    # initialization
    def __init__(
            self, dataset: Dataset, 
            model: Model,
            prompt_gen_fn: Callable, 
            cot_flag: bool = False, 
            cot: COT = None
        ) -> None:
        """
        Initializes the ZeroShotBaseline with dataset, model, prompt generator, and optional CoT configuration.

        Args:
            dataset (Dataset): Dataset configuration object.
            model (Model): Model configuration object.
            prompt_gen_fn (callable): Function to generate prompts.
            cot_flag (bool, optional): Whether to use chain-of-thought prompting.
            cot (COT, optional): Chain-of-thought configuration object.

        Returns:
            None
        """

        # inputs

        ## dataset and
        ## configuration
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        ## model
        self.model = model

        ## cot configuration
        self.cot_flag = cot_flag
        self.id = "zero-shot" if not cot_flag else "zero-shot-cot"
        self.cot = cot

        ## prompt generation 
        ## function
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = None
        self.output_file = f"data/batches/{self.dataset.name}_{self.id}_baseline_batches.jsonl"
        self.gcp_uri = None
        self.base_output_dir = f"gs://{BUCKET_NAME}/batch_outputs/gemini"
        self.job = None
        self.destination_file_name = None

    # creates a batch for 
    # a single test example
    def __create_batch(
            self, 
            request_id: str, 
            prompt: str
        ) -> Dict[str, Any]:
        """
        Creates a batch dictionary for a single test example.

        Args:
            request_id (str): Unique identifier for the batch request.
            prompt (str): Prompt to be sent to the model.

        Returns:
            Dict[str, Any]: Batch dictionary formatted for API submission.
        """

        # set thinking budget
        # based on type of 
        # prompting
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

    # creates batch prompts
    # for all test examples
    def create_batch_prompts(self):
        """
        Generates batch prompts for all test examples using the prompt generation function.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Use `batches` attribute to access generated batches.
        """

        # load the dataset
        df = pd.read_csv(self.dataset_config["dataset_path"])
        df = self.dataset.preprocess_fn(df)

        # get the original data splits
        train_df = df.loc[self.dataset_config["train_data_idx"]]
        test_df = df.loc[self.dataset_config["test_data_idx"]]
        labels = train_df[self.dataset.target_col].unique().tolist()
        test_df = test_df.drop(columns=[self.dataset.target_col], axis=1)

        # create batches
        self.batches = []

        for idx, row in test_df.iterrows():
            
            # create a meaningful
            # request_id
            request_id = f"baseline_{self.id}_batch-{idx}"

            # generate prompt
            example_features = encode(row.to_dict())
            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                example=example_features,
                labels=labels
            )

            # create batch
            batch = self.__create_batch(request_id, prompt)
            self.batches.append(batch)
    
    # saves batches locally
    def save_batches_as_jsonl(self):
        """
        Saves all generated batches to disk in JSONL format.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Batches are saved to the file specified by `output_file` attribute.
        """

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    # upload batches to
    # gcp bucket
    def upload_batches_to_gcs(self):
        """
        Uploads the batch JSONL file to a Google Cloud Storage bucket.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            File gets uploaded to GCP at the location specified by `gcp_uri` attribute.
        """

        # set the location
        # in the gcs bucket
        destination_blob_name = f"batch_inputs/gemini/{self.dataset.name}_{self.id}_baseline_batches.jsonl"
        
        # upload file
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)

        try:
            blob.upload_from_filename(self.output_file)
        except Exception:
            print(
                "[GCS CLIENT] Error uploading batch file to GCS. "
                "This is not a mandatory component and can be retried later."
            )
            return

        self.gcp_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

        print(f"[GCS CLIENT] File {self.output_file} uploaded to {destination_blob_name}")
    
    # submits batch inference
    # job to vertex ai
    def submit_batch_inference_job(self):
        """
        Submits a batch inference job to Vertex AI and monitors its status.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Batch job is submitted to Vertex AI and monitored until completion.
        """

        # set the output
        # directory
        OUTPUT_DIR = f"{self.base_output_dir}/{self.dataset.name}_{self.id}_{int(time.time())}/"

        # create the client
        client = genai.Client(vertexai=True, 
                              project=PROJECT_ID, 
                              location=LOCATION)

        try:
            # submit the job
            job = self.job = client.batches.create(
                model=self.model.name,
                src=self.gcp_uri,
                config=CreateBatchJobConfig(dest=OUTPUT_DIR),
            )
        except Exception:
            print(
                f"[{self.id.upper()}] Error uploading batch file to GCS. "
                "This is not a mandatory component and can be retried later."
            )
            return

        print(f"[{self.id.upper()}] Submitted Job: {job.name}")
        print(f"[{self.id.upper()}] Output base dir: {self.base_output_dir}")
        
        # monitor job status
        # synchronously
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        while job.state not in completed_states:
            time.sleep(SLEEP_TIME)
            job = client.batches.get(name=job.name)
            print(f"[{self.id.upper()}] {job.name} state: {job.state}")

        print(f"[{self.id.upper()}] Final state: {job.state}")
    
    # downloads job outputs
    # from gcs bucket
    def download_job_outputs_from_gcs(self):
        """
        Downloads the most recent predictions JSONL file from Google Cloud Storage.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Output file is downloaded locally to the path specified by `destination_file_name` attribute.

        Raises:
            ValueError: If no predictions.jsonl file is found in the GCS location.
        """

        # set the gcs location
        bucket_prefix = f"gs://{BUCKET_NAME}/"
        BUCKET_LOCATION=self.base_output_dir.split(bucket_prefix)[1]

        # create gcs client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)

        # find the most recent
        # file for the dataset
        most_recent_time = datetime.min.replace(tzinfo=timezone.utc)
        file_to_download = None
        for i in bucket.list_blobs(prefix=BUCKET_LOCATION):

            if not i.name.endswith("predictions.jsonl"):
                continue

            if f"{self.dataset.name}_{self.id}" in i.name:
                if i.updated > most_recent_time:
                    most_recent_time = i.updated
                    file_to_download = i.name

        # download the file
        if not file_to_download == None:
            self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_{self.id}_baseline_predictions.jsonl"
            blob = bucket.blob(file_to_download)
            blob.download_to_filename(self.destination_file_name)
            print(f"[GCS CLIENT] Downloaded {file_to_download} to {self.destination_file_name}.")
        else:
            raise ValueError("No predictions.jsonl file found in GCS location.")


