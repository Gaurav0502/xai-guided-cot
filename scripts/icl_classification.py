# module for data handling 
import pandas as pd
import numpy as np
import json
import random

# module for prompting
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState
from toon_format import encode

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model, COT

# env variables
from scripts.constants import (BUCKET_NAME,
                               PROJECT_ID,
                               LOCATION,
                               SLEEP_TIME)

# module used for type hinting
from typing import Dict, Any, Callable

# icl classification 
# class
class ICLClassifier:
    def __init__(
            self, 
            dataset: Dataset, 
            model: Model, 
            cot: COT, 
            prompt_gen_fn: Callable
        ) -> None:
        """
        Initializes the ICLClassifier with dataset, model, COT configuration, and prompt generation function.

        Args:
            dataset (Dataset): Dataset configuration object.
            model (Model): Model configuration object.
            cot (COT): Chain-of-thought configuration object.
            prompt_gen_fn (callable): Function to generate prompts for ICL.

        Returns:
            None
        """

        # inputs

        ## dataset
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        ## model config
        self.model = model

        ## cot config
        self.cot = cot

        ## prompt generation 
        ## function
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = None
        self.output_file = f"data/batches/{self.dataset.name}_icl_batches.jsonl"
        self.gcp_uri = None
        self.base_output_dir = f"gs://{BUCKET_NAME}/batch_outputs/gemini"
        self.destination_file_name = None
        self.job = None
    
    # create batch for a 
    # single test sample
    def __create_batch(
            self, 
            request_id: str, 
            prompt: str
        ) -> Dict[str, Any]:
        """
        Creates a batch dictionary for a single test sample.

        Args:
            request_id (str): Unique identifier for the batch request.
            prompt (str): Prompt to be sent to the model.

        Returns:
            Dict[str, Any]: Batch dictionary formatted for API submission.
        """

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
    
    # prepares example sets
    def __prepare_example_sets(self):
        """
        Prepares and shuffles example sets for in-context learning agents.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Use `example_sets` attribute to access prepared sets.
        """

        # map train idx to
        # predictions
        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        idx2pred = dict(zip(train_idx, train_preds))

        # get diverse examples
        diverse_examples = list(self.cot.reasoning.keys())
        all_examples = [ (idx, idx2pred[idx]) for idx in diverse_examples ]
        
        # shuffle examples
        random.shuffle(all_examples)

        # split into sets
        n = self.cot.num_examples_per_agent
        self.example_sets = [all_examples[i:i+n] for i in range(0, len(all_examples), n)]
    
    # create batch prompts
    # for all test samples
    def create_batch_prompts(self):
        """
        Generates batch prompts for all test samples using diverse examples and encodes features.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Use the `batches` attribute to access generated batches.
        
        Raises:
            KeyError: If the SHAP values CSV does not contain an 'idx' column.
        """
        
        # prepare example sets
        self.__prepare_example_sets()

        # get feature importances
        feature_imps = self.dataset_config["feature_importances"]

        # read test data
        test_idx = self.dataset_config["test_data_idx"]
        df = pd.read_csv(self.dataset.path)
        df = self.dataset.preprocess_fn(df)
        test_df = df.drop(self.dataset.target_col, axis=1, inplace=False).loc[test_idx]

        # read shap values
        # read shap values
        df_shap = pd.read_csv(self.dataset.shap_vals_path)
        try:
            df_shap.set_index("idx", inplace=True)
        except KeyError:
            raise KeyError(
                "SHAP values CSV must contain an 'idx' column "
                "for mapping to original indices."
            )

        # create batches
        self.batches = []
        agent_id = 0

        # iterate over example sets
        for example_set in self.example_sets:

            # unpack example set
            train_idx_subset, train_preds_subset = list(map(list, zip(*example_set)))

            # get train df and
            # corresponding preds
            train_df = df.loc[train_idx_subset]
            train_preds = dict(zip(train_idx_subset, train_preds_subset))

            # get shap values for
            # train examples
            df_shap_train = df_shap.loc[train_idx_subset]

            # iterate over test samples
            for idx, row in test_df.iterrows():

                # encode test sample
                test_sample = encode(row.to_dict())

                # create meaningful 
                # request_id
                request_id = f"cot_batch-{idx}_agent-{agent_id}"

                # generate prompt
                prompt = self.prompt_gen_fn(
                    dataset=self.dataset,
                    examples=train_df,
                    example_predictions=train_preds,
                    feature_importances=feature_imps,
                    example_shap_values=df_shap_train,
                    reasoning=self.cot.reasoning,
                    test_example=test_sample
                )

                # create batch
                batch = self.__create_batch(request_id=request_id, 
                                            prompt=prompt)
                
                # collect batch
                self.batches.append(batch)
            agent_id += 1

    # save batches 
    # locally
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
    # GCP bucket
    def upload_batches_to_gcs(self):
        """
        Uploads the batch JSONL file to a Google Cloud Storage bucket.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Uploaded file gets stored in GCP at the location specified by `gcp_uri` attribute.
        
        Raises:
            AssertionError: If there is an error during file upload to GCS.
        """

        # create destination
        # file name
        destination_blob_name = f"batch_inputs/gemini/{self.dataset.name}_icl_batches.jsonl"

        # upload to gcs
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)

        try:
            blob.upload_from_filename(self.output_file)
        except Exception as e:
            print("[GCS CLIENT] Error uploading batch file to GCS.")
            raise AssertionError("ICL Classifier is a mandatory component. Kindly debug the issue and retry:", str(e))

        self.gcp_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

        print(f"[GCS CLIENT] File {self.output_file} uploaded to {destination_blob_name}")
    
    # submit batch inference
    # job to vertex ai
    def submit_batch_inference_job(self):
        """
        Submits a batch inference job to Vertex AI and monitors its status.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Batch job is submitted and monitored until completion.
        
        Raises:
            AssertionError: If there is an error submitting the batch job.
        """

        # create output dir
        OUTPUT_DIR = f"{self.base_output_dir}/{self.dataset.name}_cot_{int(time.time())}/"

        # create genai client
        client = genai.Client(vertexai=True, 
                              project=PROJECT_ID, 
                              location=LOCATION)

        try:
            # submit batch job
            job = self.job = client.batches.create(
                model=self.model.name,
                src=self.gcp_uri,
                config=CreateBatchJobConfig(
                    dest=OUTPUT_DIR
                ),
            )
        except Exception as e:
            print("[ICL CLASSIFIER] Error submitting batch job to Vertex AI.")
            raise AssertionError("ICL Classifier is a mandatory component. "
                                 "Kindly debug the issue and retry:", str(e))

        print(f"[ICL CLASSIFIER] Submitted Job: {job.name}")
        print(f"[ICL CLASSIFIER] Output base dir: {OUTPUT_DIR}")
    
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
            print(f"[ICL CLASSIFIER] {job.name} state: {job.state}")

        print(f"[ICL CLASSIFIER] Final state: {job.state}")
        if not job.state == JobState.JOB_STATE_SUCCEEDED:
            raise AssertionError("ICL Classifier is a mandatory component."
                                 "Kindly debug the issue and retry. ")

    # download job outputs
    # from GCS
    def download_job_outputs_from_gcs(self):
        """
        Downloads the most recent predictions JSONL file from Google Cloud Storage.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Use the `destination_file_name` attribute to access the downloaded file path.

        Raises:
            ValueError: If no predictions.jsonl file is found in the GCS location.
        """

        # set bucket in location
        bucket_prefix = f"gs://{BUCKET_NAME}/"
        BUCKET_LOCATION=self.base_output_dir.split(bucket_prefix)[1]

        # initialize gcs client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)

        # find the most recent file
        # for the current dataset
        most_recent_time = datetime.min.replace(tzinfo=timezone.utc)
        file_to_download = None
        for i in bucket.list_blobs(prefix=BUCKET_LOCATION):

            if not i.name.endswith("predictions.jsonl"):
                    continue
                
            if f"{self.dataset.name}_cot" in i.name:
                if i.updated > most_recent_time:
                    most_recent_time = i.updated
                    file_to_download = i.name

        # download the file
        if not file_to_download == None:
            self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_cot_predictions.jsonl"
            blob = bucket.blob(file_to_download)
            blob.download_to_filename(self.destination_file_name)
            print(f"[GCS CLIENT] Downloaded {file_to_download} to {self.destination_file_name}.")
        else:
            raise ValueError("No predictions.jsonl file found in GCS location.")