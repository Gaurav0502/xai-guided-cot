import json
import os
from dotenv import load_dotenv

from xgboost import XGBClassifier

# custom modules
from scripts.pipeline import Pipeline
from scripts.configs import Dataset, Model
from scripts.preprocess import preprocess_titanic

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
PROJECT_NAME = os.getenv("PROJECT_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")

def get_default_models():
    reasoning_gen_model = Model(
        provider="together",
        name="deepseek-ai/DeepSeek-R1",
        temperature=0.6,
        max_tokens=4096,
    )
    objective_judge_model = Model(
        provider="anthropic", name="claude-haiku-4-5", temperature=0.6, max_tokens=4096
    )
    cot_model = Model(
        provider="google", name="gemini-2.5-flash", temperature=0.0, max_tokens=4096
    )
    return reasoning_gen_model, objective_judge_model, cot_model


def get_titanic_dataset():
    return Dataset(
        name="titanic",
        path="data/datasets/titanic_small.csv",
        config_file_path="data/dataset_config/titanic_config.json",
        shap_vals_path="data/shap_values/titanic_shap.csv",
        preprocess_fn=preprocess_titanic,
        target_col="Survived",
        labels={0: "Did not survive", 1: "Survived"},
    )


def run_experiment(
    output_path: str,
    dataset: Dataset = None,
    baseline: bool = False,
    objective_judge: bool = False,
    cot_ablation: bool = False,
    num_examples_per_agent: int = 10,
    random_sample_count: int = None,
    thinking_budget: int = 1000,
):
    """
    Run the pipeline with specified parameters and save results.

    Args:
        output_path: Path to save the results JSON
        dataset: Dataset config
        baseline: Run zero-shot baseline
        objective_judge: Run reasoning judge evaluation
        cot_ablation: Run zero-shot CoT ablation for baseline
        num_examples_per_agent: Examples per agent (-1 for single agent)
        random_sample_count: If set, use random sampling instead of diverse selection
        thinking_budget: Thinking budget for icl_classifier
    """
    if dataset is None:
        dataset = get_titanic_dataset()

    reasoning_gen_model, objective_judge_model, cot_model = get_default_models()

    # Create pipeline
    pipeline = Pipeline(
        dataset=dataset,
        explanable_model=XGBClassifier(),
        tune_config_file="data/tune_config/xgb.json",
        reasoning_gen_model=reasoning_gen_model,
        objective_judge_model=objective_judge_model,
        cot_model=cot_model,
    )

    # Run
    pipeline.run(
        baseline=baseline,
        objective_judge=objective_judge,
        cot_ablation=cot_ablation,
        num_examples_per_agent=num_examples_per_agent,
        random_sample_count=random_sample_count,
        thinking_budget=thinking_budget,
    )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # Filter out non-serializable reasoning dict for cleaner output
        results_to_save = {
            k: v for k, v in pipeline.results.items() if k != "reasoning"
        }
        results_to_save["config"] = {
            "num_examples_per_agent": num_examples_per_agent,
            "random_sample_count": random_sample_count,
            "thinking_budget": thinking_budget,
        }
        json.dump(results_to_save, f, indent=2)

    print(f"[EXPERIMENT] Results saved to {output_path}")
    return pipeline.results

