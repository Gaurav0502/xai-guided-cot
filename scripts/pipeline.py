# modules used for data handling
from typing import Any
from scripts.constants import SUPPORTED_EXPLAINABLE_MODELS
import pandas as pd
import numpy as np
import json
import random

# modules used for model training
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# user-defined modules
from scripts.configs import (Dataset, Model, COT)
from scripts.postprocess import (parse_baseline_llm_results, 
                                 parse_reasoning_llm_results,
                                 parse_objective_judge_results,
                                 parse_zero_shot_cot_llm_results,
                                 parse_cot_llm_results)
from scripts.prompt_generator import (zero_shot_prompt_generator, 
                                      zero_shot_cot_prompt_generator,
                                      reasoning_generator_prompt,
                                      objective_judge_prompt_generator)

from scripts.explanable_tree_model import ExplainableModel
from scripts.zero_shot_baseline import ZeroShotBaseline
from scripts.diverse_examples import get_diverse_examples
from scripts.reason_generation import ReasonGenerator
from scripts.objective_judge import ObjectiveJudge
from scripts.icl_classification import ICLClassifier
from scripts.evaluation import Evaluator

# modules used for env variables
import os
from dotenv import load_dotenv
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

class Pipeline:

    def __init__(
        self, 
        dataset: Dataset,
        explanable_model: Any,
        tune_config_file: str,
        reasoning_gen_model: Model,
        objective_judge_model: Model,
        cot_model: Model
    ):
        # input validations
        if not isinstance(dataset, Dataset):
            raise TypeError(
               "dataset must be a scripts.configs.Dataset instance\n"
                f"Got: {type(dataset).__name__}"
            )
        if not isinstance(reasoning_gen_model, Model):
            raise TypeError(
               "reasoning_gen_model must be a scripts.configs.Model instance\n"
                f"Got: {type(reasoning_gen_model).__name__}"
            )
        if not isinstance(objective_judge_model, Model):
            raise TypeError(
               "objective_judge_model must be a scripts.configs.Model instance\n"
                f"Got: {type(objective_judge_model).__name__}"
            )
        if not isinstance(cot_model, Model):
            raise TypeError(
               "cot_model must be a scripts.configs.Model instance\n"
                f"Got: {type(cot_model).__name__}"
            )
        
        self.dataset = dataset
        self.explanable_model = explanable_model
        cls_name = self.explanable_model.__class__.__name__
        if cls_name not in SUPPORTED_EXPLAINABLE_MODELS:
           allowed = ", ".join(sorted(SUPPORTED_EXPLAINABLE_MODELS))
           raise ValueError(
               f"Unsupported model class: {cls_name}\n"
                f"Allowed: {allowed}"
            )
        
        self.tune_config_file = tune_config_file

        self.reasoning_gen_model = reasoning_gen_model
        self.objective_judge_model = objective_judge_model
        self.cot_model = cot_model
    
    def _run_baseline_eval(
            self, baseline_obj, predictions_key, 
            metrics_key, postprocess_fn: callable):

        baseline_obj.create_batch_prompts()
        baseline_obj.save_batches_as_jsonl()
        baseline_obj.upload_batches_to_gcs()
        baseline_obj.submit_batch_inference_job()
        baseline_obj.download_job_outputs_from_gcs()

        strategy = "zero_shot_cot" if baseline_obj.cot_flag else "zero_shot_baseline"
        eval = Evaluator(
            prompting_strategy=strategy,
            dataset=self.dataset,
            results_jsonl_path=baseline_obj.destination_file_name,
            postprocess_fn=postprocess_fn
        )
        eval.evaluate()
        return {
            predictions_key: eval.results,
            metrics_key: eval.metrics
        }

    def compute_baselines(self, cot_ablation: bool = False):

        results = {}

        baseline = ZeroShotBaseline(
            dataset=self.dataset, 
            model=self.cot_model, 
            prompt_gen_fn=zero_shot_cot_prompt_generator
        )
        results.update(self._run_baseline_eval(
            baseline, "baseline_predictions", 
            "baseline_metrics",
            postprocess_fn=parse_baseline_llm_results
        ))

        if cot_ablation:
            cot_config = COT(
                num_examples_per_agent=10,
                reasoning={},
                thinking_budget=1000
            )
            cot_ablation = ZeroShotBaseline(
                dataset=self.dataset, 
                model=self.cot_model, 
                prompt_gen_fn=zero_shot_cot_prompt_generator,
                cot_flag=True,
                cot=cot_config
            )
            results.update(self._run_baseline_eval(
                cot_ablation, "baseline_ablation_predictions", 
                "baseline_ablation_metrics",
                postprocess_fn=parse_zero_shot_cot_llm_results
            ))

        self.results = self.results | results

    def run(self, 
            baseline: bool = False,
            objective_judge: bool = False):
        
        results = {}

        # xai model training
        # and tuning
        xai_model = ExplainableModel(
            dataset=self.dataset,
            estimator=self.explanable_model
        )
        xai_model.explain(params_grid_file=self.tune_config_file)

        # baseline computations
        if baseline:
            self.compute_baselines()
        
        # reasoning generation
        reason_generator = ReasonGenerator(
            dataset=self.dataset,
            model=self.reasoning_gen_model,
            prompt_gen_fn=reasoning_generator_prompt
        )
        reason_generator.create_batch_prompts()
        reason_generator.save_batches_as_jsonl()
        reason_generator.submit_batches()
        reasoning = parse_reasoning_llm_results(
            results_jsonl_path=reason_generator.destination_file_name
        )
        results["reasoning"] = reasoning

        # llm as judge
        if objective_judge:
            judge = ObjectiveJudge(
                dataset=self.dataset,
                model=self.objective_judge_model,
                prompt_gen_fn=objective_judge_prompt_generator
            )
            judge.create_batch_prompts(reasoning=reasoning)
            judge.submit_batch()
            results["llm_as_judge"] = parse_objective_judge_results(
                results_jsonl_path=judge.destination_file_name
            )
        
        # cot
        cot_config = COT(
            num_examples_per_agent=10,
            reasoning=reasoning,
            thinking_budget=1000
        )
        icl_classifier = ICLClassifier(
            dataset=self.dataset,
            model=self.cot_model,
            cot=cot_config,
            prompt_gen_fn=reasoning_generator_prompt
        )
        icl_classifier.create_batch_prompts(reasoning=reasoning)
        icl_classifier.submit_batch()
        icl_classifier.save_batches_as_jsonl()
        icl_classifier.submit_batch_inference_job()
        icl_classifier.download_job_outputs_from_gcs()

        # evaluation
        eval = Evaluator(
            prompting_strategy="xai-guided-cot",
            dataset=self.dataset,
            results_jsonl_path=icl_classifier.destination_file_name,
            postprocess_fn=parse_cot_llm_results
        )
        eval.evaluate()
        results["cot_predictions"] = eval.results
        results["cot_metrics"] = eval.metrics

        self.results = self.results | results

        return results



        
