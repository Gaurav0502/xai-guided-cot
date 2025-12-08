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
                                      reasoning_prompt_generator,
                                      objective_judge_prompt_generator,
                                      cot_prompt_generator)

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
        self.results = {}
    
    def _run_baseline_eval(
            self, baseline_obj, postprocess_fn: callable):

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

    def compute_baselines(self, cot_ablation: bool = False):

        baseline = ZeroShotBaseline(
            dataset=self.dataset, 
            model=self.cot_model, 
            prompt_gen_fn=zero_shot_prompt_generator
        )
        self.results["zero_shot_baseline"] = self._run_baseline_eval(
            baseline, postprocess_fn=parse_baseline_llm_results
        )

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
            self.results["zero_shot_cot_ablation"] = self._run_baseline_eval(
                cot_ablation, postprocess_fn=parse_zero_shot_cot_llm_results
            )

    def run(self, 
            baseline: bool = False,
            objective_judge: bool = False,
            cot_ablation: bool = False):

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
        
        # reasoning generation
        reason_generator = ReasonGenerator(
            dataset=self.dataset,
            model=self.reasoning_gen_model,
            prompt_gen_fn=reasoning_prompt_generator
        )
        reason_generator.create_batch_prompts()
        reason_generator.save_batches_as_jsonl()
        reason_generator.submit_batches()
        reasoning = parse_reasoning_llm_results(
            results_jsonl_path=reason_generator.destination_file_name
        )
        self.results["reasoning"] = reasoning
        print("[PIPELINE] Reasoning generation completed.")

        # llm as judge
        if objective_judge:
            judge = ObjectiveJudge(
                dataset=self.dataset,
                model=self.objective_judge_model,
                prompt_gen_fn=objective_judge_prompt_generator
            )
            judge.create_batch_prompts(reasoning=reasoning)
            judge.submit_batch()
            self.results["llm_as_judge"] = parse_objective_judge_results(
                results_jsonl_path=judge.destination_file_name
            )
            print("[PIPELINE] Objective judge evaluation completed.")
        
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
            prompt_gen_fn=cot_prompt_generator
        )
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



        
