# module for data handling
import pandas as pd
import numpy as np
import json

# module for prompting
from together import Together
from toon_format import encode
from scripts.diverse_examples import get_diverse_examples

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model

# env variable
from scripts.constants import (TOGETHER_API_KEY,
                                SLEEP_TIME)

# modules for typing
from typing import Dict, Any, Callable

# reason generation class
class ReasonGenerator:

    # initialization
    def __init__(
            self, 
            dataset: Dataset, 
            model: Model, 
            prompt_gen_fn: Callable
        ) -> None:
        """
        Initializes the ReasonGenerator with dataset, model, and prompt generation function.

        Args:
            dataset (Dataset): Dataset configuration object.
            model (Model): Model configuration object.
            prompt_gen_fn (callable): Function to generate prompts for reasoning.

        Returns:
            None
        """

        # inputs

        ## dataset
        self.dataset = dataset
        self.dataset_config = json.load(open(dataset.config_file_path, "r"))

        ## model
        self.model = model

        ## prompt generation 
        ## function
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.output_file = f"data/batches/{self.dataset.name}_reasoning_batches.jsonl"
        self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_reasoning_predictions.jsonl"
        self.responses = []

    # creates batch for a test sample
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

    # creates batch prompts 
    # for all test samples
    def create_batch_prompts(self) -> None:
        """
        Generates batch prompts for all selected test samples using diverse examples and encodes features.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Use the `batches` attribute to access generated batches.
        
        Raises:
            KeyError: If the SHAP values CSV does not contain an 'idx' column.
        """

        # get diverse examples
        unique_row_idx = get_diverse_examples(self.dataset.shap_vals_path)

        # format feature importances
        feature_imps = self.dataset_config["feature_importances"]
        feature_imps = encode(feature_imps)

        # read train data and
        # predictions on it
        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        train_df = pd.DataFrame.from_dict(dict(zip(train_idx, train_preds)),
                                         columns=["prediction"],
                                         orient="index")

        # read test data
        df = pd.read_csv(self.dataset.path)
        df = self.dataset.preprocess_fn(df)
        df = df.loc[unique_row_idx]
        train_df = train_df.loc[unique_row_idx]

        # read shap values
        df_shap = pd.read_csv(self.dataset.shap_vals_path)
        try:
            df_shap.set_index("idx", inplace=True)
            df_shap = df_shap.loc[unique_row_idx]
        except KeyError:
            raise KeyError(
                "SHAP values CSV must contain an 'idx' column "
                "for mapping to original indices."
            )

        # create batches
        self.batches = []

        for idx, row in df.iterrows():
            
            # encode feature values
            # and shap values
            example_raw_features = encode(row.to_dict())
            example_shap_values = encode(df_shap.loc[idx].to_dict())

            # get prediction and ground truth
            prediction = train_df.loc[idx, "prediction"]
            ground_truth = row[self.dataset.target_col]

            # generate prompt
            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                prediction=prediction,
                ground_truth=ground_truth,
                ft_raw_vals=example_raw_features,
                shap_vals=example_shap_values,
                feature_importances=feature_imps
            )

            # create a meaningful 
            # request_id
            request_id = f"reasoning_batch-{int(idx)}"

            # create batch
            batch = self.__create_batch(
                request_id=request_id,
                prompt=prompt
            )

            # collect batch
            self.batches.append(batch)
    
    # saves batches
    # to disk as jsonl
    def save_batches_as_jsonl(self) -> None:
        """
        Saves all generated batches to disk in JSONL format.

        Args:
            None
            All inputs are class attributes.

        Returns:
            None.
            Batches are saved to the file specified by `output_file` attribute.
        """

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n") 
    
    # submits batches
    # to Together API
    def submit_batches(self) -> None:
        """
        Submits the batch jobs to the Together API, monitors their status, and downloads outputs upon completion.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Output is saved to the file specified by `destination_file_name` attribute.

        Raises:
            AssertionError: If the batch inference job fails.
        """

        # initialize Together client
        client = Together(api_key=TOGETHER_API_KEY)

        # upload batch file
        try:
            file_resp = client.files.upload(file=self.output_file, 
                                            purpose="batch-api")
        except Exception:
            raise
        
        # submits batch job
        batch = client.batches.create_batch(file_id=file_resp.id,
                                            endpoint="/v1/chat/completions" )
        batch_status = batch

        # synchronous monitoring 
        # of batch status
        while True:
            time.sleep(SLEEP_TIME)
            batch_status = client.batches.get_batch(batch.id)
            print(f"[REASON GENERATION] Current Status: {batch_status.status}")
            if "COMPLETED" in batch_status.status:
                print("[REASON GENERATION] Batch completed successfully.")
                break
            elif "FAILED" in batch_status.status or "CANCELLED" in batch_status.status:
                print(f"[REASON GENERATION] Batch failed with status: {batch_status.status}")
                break

        # automatically download outputs
        if "COMPLETED" in batch_status.status:

            client.files.retrieve_content(id=batch_status.output_file_id, 
                                          output=self.destination_file_name)
            
            print(f"[REASON GENERATION] Batch outputs downloaded to {self.destination_file_name}")
    
        else:
            raise AssertionError("Reason generation batch inference job is required to be successful. However, it has failed.")
        





