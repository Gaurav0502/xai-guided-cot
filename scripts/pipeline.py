# modules used for data handling
import pandas as pd
import numpy as np
import json
import random

# modules used for model training
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# user-defined modules

## configurations
from scripts.configs import (Dataset, Model, COT)

## prompt generation functions
from scripts.prompt_generator import (zero_shot_prompt_generator, 
                                      zero_shot_cot_prompt_generator,
                                      reasoning_prompt_generator,
                                      objective_judge_prompt_generator,
                                      cot_prompt_generator)

## postprocessing functions
from scripts.postprocess import (parse_baseline_llm_results, 
                                 parse_reasoning_llm_results,
                                 parse_objective_judge_results,
                                 parse_zero_shot_cot_llm_results,
                                 parse_cot_llm_results)

## main pipeline components
from scripts.explanable_tree_model import ExplainableModel
from scripts.zero_shot_baseline import ZeroShotBaseline
from scripts.reason_generation import ReasonGenerator
from scripts.objective_judge import ObjectiveJudge
from scripts.icl_classification import ICLClassifier
from scripts.evaluation import Evaluator

# modules used for validation
# and error handling
from typing import Any, Dict
from scripts.constants import SUPPORTED_EXPLAINABLE_MODELS

# env variables
from scripts.constants import (
    WANDB_API_KEY, WANDB_PROJECT_NAME, PROJECT_ID,
    BUCKET_NAME, LOCATION, TOGETHER_API_KEY, 
    CLAUDE_API_KEY
)

# main genai
# pipeline class
class Pipeline:

    # initialization
    def __init__(
            self, 
            dataset: Dataset,
            explanable_model: Any,
            tune_config_file: str,
            reasoning_gen_model: Model,
            objective_judge_model: Model,
            cot_model: Model
        ) -> None:
        """
        Initializes the Pipeline with dataset, models, and configuration.

        Args:
            dataset (Dataset): Dataset configuration object.
            explanable_model (Any): Explainable model instance.
            tune_config_file (str): Path to tuning configuration file.
            reasoning_gen_model (Model): Model for reasoning generation.
            objective_judge_model (Model): Model for objective judging.
            cot_model (Model): Model for chain-of-thought classification.

        Returns:
            None
        """
        
        # input
        ## dataset
        self.dataset = dataset

        ## xai model
        self.explanable_model = explanable_model
        
        ## wandb sweep config
        self.tune_config_file = tune_config_file

        ## model configs
        self.reasoning_gen_model = reasoning_gen_model
        self.objective_judge_model = objective_judge_model
        self.cot_model = cot_model

        self.__validate_inputs()

        # output
        self.results = {}
    
    # input validation
    def __validate_inputs(self) -> None:
        """
        Validates the types and supported classes of input arguments.

        Args:
            None

        Returns:
            None

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If the explainable model class is unsupported.
        """

        # type checks
        if not isinstance(self.dataset, Dataset):
            raise TypeError(
               "dataset must be a scripts.configs.Dataset instance\n"
                f"Got: {type(self.dataset).__name__}"
            )
        if not isinstance(self.reasoning_gen_model, Model):
            raise TypeError(
               "reasoning_gen_model must be a scripts.configs.Model instance\n"
                f"Got: {type(self.reasoning_gen_model).__name__}"
            )
        if not isinstance(self.objective_judge_model, Model):
            raise TypeError(
               "objective_judge_model must be a scripts.configs.Model instance\n"
                f"Got: {type(self.objective_judge_model).__name__}"
            )
        if not isinstance(self.cot_model, Model):
            raise TypeError(
               "cot_model must be a scripts.configs.Model instance\n"
                f"Got: {type(self.cot_model).__name__}"
            )
        
        # explainable model check
        cls_name = self.explanable_model.__class__.__name__
        if cls_name not in SUPPORTED_EXPLAINABLE_MODELS:
           allowed = ", ".join(sorted(SUPPORTED_EXPLAINABLE_MODELS))
           raise ValueError(
               f"Unsupported model class: {cls_name}\n"
                f"Allowed: {allowed}"
            )
    
    # baseline evaluation
    def _run_baseline_eval(
            self, baseline_obj, 
            postprocess_fn: callable
        ) -> Dict[str, Dict[str, float]]:
        """
        Runs baseline evaluation and computes metrics.

        Args:
            baseline_obj: Baseline object for evaluation.
            postprocess_fn (callable): Function to postprocess results.

        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics.
        """

        baseline_obj.create_batch_prompts()
        baseline_obj.save_batches_as_jsonl()
        baseline_obj.upload_batches_to_gcs()
        baseline_obj.submit_batch_inference_job()
        baseline_obj.download_job_outputs_from_gcs()

        strategy = "zero-shot-cot" if baseline_obj.cot_flag else "zero-shot-prompting"
        eval = Evaluator(
            prompting_strategy=strategy,
            dataset=self.dataset,
            results_jsonl_path=baseline_obj.destination_file_name,
            postprocess_fn=postprocess_fn
        )
        eval.evaluate()
        return eval.metrics

    # baseline computations
    def compute_baselines(self, 
            cot_ablation: bool = False
        ) -> None:
        """
        Computes baseline and ablation metrics for the pipeline.

        Args:
            cot_ablation (bool): Whether to run chain-of-thought ablation.

        Returns:
            None.
            Access the results dictionary using the `results` attribute.
        """

        # zero-shot prompting baseline
        baseline = ZeroShotBaseline(
            dataset=self.dataset, 
            model=self.cot_model, 
            prompt_gen_fn=zero_shot_prompt_generator
        )

        # run evaluation
        self.results["zero_shot_baseline"] = self._run_baseline_eval(
            baseline, postprocess_fn=parse_baseline_llm_results
        )

        # zero-shot cot ablation
        if cot_ablation:

            ## cot config
            cot_config = COT(
                num_examples_per_agent=10,
                reasoning={},
                thinking_budget=1000
            )

            ## zero-shot cot baseline
            cot_ablation = ZeroShotBaseline(
                dataset=self.dataset, 
                model=self.cot_model, 
                prompt_gen_fn=zero_shot_cot_prompt_generator,
                cot_flag=True,
                cot=cot_config
            )

            ## run evaluation
            self.results["zero_shot_cot_ablation"] = self._run_baseline_eval(
                cot_ablation, postprocess_fn=parse_zero_shot_cot_llm_results
            )

    # main pipeline
    def run(self, 
            baseline: bool = False,
            objective_judge: bool = False,
            cot_ablation: bool = False,
            masked: bool = False
        ) -> None:
        """
        Executes the main pipeline workflow including training, reasoning, judging, and evaluation.

        Args:
            baseline (bool): Whether to compute baseline metrics.
            objective_judge (bool): Whether to run objective judge evaluation.
            cot_ablation (bool): Whether to run chain-of-thought ablation.
            masked (bool): Whether to skip explainable model training.

        Returns:
            None.
            Access the results dictionary using the `results` attribute.
        """

        if not masked:
            # xai model training
            # and tuning
            xai_model = ExplainableModel(
                dataset=self.dataset,
                estimator=self.explanable_model
            )

            xai_model.explain(params_grid_file=self.tune_config_file)

        # baseline computations
        if baseline:
            self.compute_baselines(cot_ablation)
            print("[PIPELINE] Baseline metrics computed.")
        
        # reasoning generation component

        ## configure reason 
        ## generator
        reason_generator = ReasonGenerator(
            dataset=self.dataset,
            model=self.reasoning_gen_model,
            prompt_gen_fn=reasoning_prompt_generator
        )

        ## generate reasoning
        reason_generator.create_batch_prompts()
        reason_generator.save_batches_as_jsonl()
        reason_generator.submit_batches()
        reasoning = parse_reasoning_llm_results(
            results_jsonl_path=reason_generator.destination_file_name
        )
        print("[PIPELINE] Reasoning generation completed.")

        ## validate presence
        ## of reasoning
        if len(reasoning) == 0:
            raise AssertionError(
                "Reasoning outputs are required. This likely means the reasoning LLM did not follow the expected output format, causing postprocessing to fail. "
                "Please check the LLM's raw outputs and ensure they match the required format for parsing."
            )
        self.results["reasoning"] = reasoning

        # llm as judge
        if objective_judge:

            ## configure objective 
            ## judge
            judge = ObjectiveJudge(
                dataset=self.dataset,
                model=self.objective_judge_model,
                prompt_gen_fn=objective_judge_prompt_generator
            )

            ## judge the reasoning
            judge.create_batch_prompts(reasoning=reasoning)
            judge.submit_batch()
            self.results["llm_as_judge"] = parse_objective_judge_results(
                results_jsonl_path=judge.destination_file_name
            )
            print("[PIPELINE] Objective judge evaluation completed.")
        
        # cot config
        cot_config = COT(
            num_examples_per_agent=10,
            reasoning=reasoning,
            thinking_budget=1000
        )

        # icl classification with cot

        ## configure icl classifier
        icl_classifier = ICLClassifier(
            dataset=self.dataset,
            model=self.cot_model,
            cot=cot_config,
            prompt_gen_fn=cot_prompt_generator
        )

        ## generate predictions
        icl_classifier.create_batch_prompts()
        icl_classifier.save_batches_as_jsonl()
        icl_classifier.upload_batches_to_gcs()
        icl_classifier.submit_batch_inference_job()
        icl_classifier.download_job_outputs_from_gcs()
        print("[PIPELINE] ICL classification with COT completed.")

        # evaluation
        eval = Evaluator(
            prompting_strategy="xai-guided-cot",
            dataset=self.dataset,
            results_jsonl_path=icl_classifier.destination_file_name,
            postprocess_fn=parse_cot_llm_results
        )
        eval.evaluate()
        self.results["cot"] = eval.metrics
        print("[PIPELINE] Evaluation of COT predictions completed.")
        print("[PIPELINE] Pipeline run completed.")



        
