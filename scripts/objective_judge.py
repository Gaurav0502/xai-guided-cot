# module for data handling 
import pandas as pd
import numpy as np
import json
from toon_format import encode

# module for prompting
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# module for time handling
import time
from datetime import datetime, timezone

# user defined modules
from scripts.configs import Dataset, Model

# env variables
from scripts.constants import (CLAUDE_API_KEY,
                                SLEEP_TIME)

# module used for type hinting
from typing import Callable

# objective judge class
class ObjectiveJudge:

    # initialization
    def __init__(
            self, 
            dataset: Dataset, 
            model: Model, 
            prompt_gen_fn: Callable
        ) -> None:
        """
        Initializes the ObjectiveJudge with dataset, model, and prompt generation function.

        Args:
            dataset (Dataset): Dataset configuration object.
            model (Model): Model configuration object.
            prompt_gen_fn (callable): Function to generate prompts for evaluation.

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
        
        ## prompt generation 
        ## function
        self.prompt_gen_fn = prompt_gen_fn

        # outputs
        self.batches = []
        self.evals = []
        self.batch_id = None
        self.destination_file_name = f"data/batch_outputs/{self.dataset.name}_objective_judge_evaluations.jsonl"
    
    # generate a batch for a single test sample
    def __create_batch(
            self, 
            request_id: str, 
            prompt: str
        ) -> Request:
        """
        Creates a batch request for a single test sample.

        Args:
            request_id (str): Unique identifier for the batch request.
            prompt (str): Prompt to be sent to the model.

        Returns:
            Request: Batch request object for the Anthropic API.
        """

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

    # create batch prompts for 
    # all test samples
    def create_batch_prompts(
            self, 
            reasoning: dict
        ) -> None:
        """
        Generates batch prompts for all test samples using feature values, SHAP values, and model reasoning.

        Args:
            reasoning (dict): Dictionary mapping sample indices to model reasoning.

        Returns:
            None.
            Use `batches` attribute to access the generated batch requests.
        """

        # get row indices
        unique_row_idx = list(reasoning.keys())

        # read feature importances
        feature_imps = self.dataset_config["feature_importances"]
        feature_imps = encode(feature_imps)

        # read training predictions
        train_idx = self.dataset_config["train_data_idx"]
        train_preds = self.dataset_config["train_predictions"]
        train_df = pd.DataFrame.from_dict(dict(zip(train_idx, train_preds)),
                                         columns=["prediction"],
                                         orient="index")
        
        # read train dataset
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

            # encode features and
            # shap values
            example_raw_features = encode(row.to_dict())
            example_shap_values = encode(df_shap.loc[idx].to_dict())

            # get prediction,
            # ground truth, and
            # reason
            prediction = train_df.loc[idx, "prediction"]
            ground_truth = row[self.dataset.target_col]
            reason = reasoning[idx]

            # generate prompt
            prompt = self.prompt_gen_fn(
                dataset=self.dataset,
                prediction=prediction,
                ground_truth=ground_truth,
                ft_raw_vals=example_raw_features,
                shap_vals=example_shap_values,
                feature_importances=feature_imps,
                reason=reason
            )

            # create a meaningful 
            # request_id
            request_id = f"judge_batch-{idx}"

            # create batch
            batch = self.__create_batch(
                request_id=request_id,
                prompt=prompt
            )

            # collect batch
            self.batches.append(batch)
        
    # retrieve batch results
    def __retrieve_batch_results(
            self, 
            client: anthropic.Anthropic
        ) -> None:
        """
        Retrieves and processes batch results from the Anthropic API.

        Args:
            client (anthropic.Anthropic): Anthropic API client instance.

        Returns:
            None.
            Use `evals` attribute to access the processed evaluations.
        """

        # fetch results
        results = client.messages.batches.results(
            self.batch_id
        )
        
        # process results
        result_types = {"succeeded":0, "errored":0, "expired":0}

        # retrieve evaluations
        # and record final
        # status
        for result in results:

            # process based on
            # final status
            match result.result.type:
               
               # successful cases
               case "succeeded":
                    result_types["succeeded"] += 1

                    # record evaluation
                    result_dict = {
                        "request_id": result.custom_id,
                        "evaluation": result.result.message.content[0].text
                    }
                    self.evals.append(result_dict)

               # unexpected cases
               case "errored":
                    result_types["errored"] += 1
               case "expired":
                    result_types["expired"] += 1
        
        print("[OBJECTIVE JUDGE] Batch result types:", result_types)

    # save evaluations 
    # to jsonl
    def __save_to_jsonl(self):
        """
        Saves all evaluation results to a JSONL file (on disk).

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Evaluations are saved to the file specified by `destination_file_name` attribute.
        """

        with open(self.destination_file_name, "w") as f:
            for eval in self.evals:
                f.write(json.dumps(eval) + "\n")

        print(f"[OBJECTIVE JUDGE] Saved evaluations to {self.destination_file_name}")

    # submit batch to
    # Anthropic API
    def submit_batch(self):
        """
        Submits the batch to the Anthropic API, monitors processing, retrieves results, and saves evaluations.

        Args:
            None.
            All inputs are class attributes.

        Returns:
            None.
            Evaluations are saved to the file specified by `destination_file_name` attribute.
        """

        # submit batch
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        try:
            batch_info = client.messages.batches.create(
                requests=self.batches
            )
        except Exception:
            print(
                "[OBJECTIVE JUDGE] Error submitting batch to Anthropic API. "
                "This component is optional and can be retried later."
            )
            return

        print("[OBJECTIVE JUDGE] Submitted batch with id:", batch_info.id)
        self.batch_id = batch_info.id

        # monitor batch status
        # synchronously
        message_batch = None
        while True:

            # retrieve batch status
            message_batch = client.messages.batches.retrieve(
                self.batch_id
            )

            if message_batch.processing_status == "ended":
                break

            print(f"[OBJECTIVE JUDGE] Batch {self.batch_id} is still processing...")
            time.sleep(SLEEP_TIME)

        print(f"[OBJECTIVE JUDGE] Batch {self.batch_id} has completed processing.")

        # retrieve results
        # automatically
        self.__retrieve_batch_results(client)
        self.__save_to_jsonl()
