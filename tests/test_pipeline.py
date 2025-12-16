# modules used for file handling
import os

# modules used for model training
from xgboost import XGBClassifier

# modules used for testing
import pytest

# modules to be tested
from scripts.configs import Dataset, Model, COT
from scripts.preprocess import preprocess_titanic
from scripts.explanable_tree_model import ExplainableModel
from scripts.zero_shot_baseline import ZeroShotBaseline
from scripts.reason_generation import ReasonGenerator
from scripts.objective_judge import ObjectiveJudge
from scripts.icl_classification import ICLClassifier
from scripts.evaluation import Evaluator
from scripts.prompt_generator import (
    zero_shot_prompt_generator,
    zero_shot_cot_prompt_generator,
    reasoning_prompt_generator,
    objective_judge_prompt_generator,
    cot_prompt_generator
)
from scripts.postprocess import (
    parse_baseline_llm_results,
    parse_reasoning_llm_results,
    parse_objective_judge_results,
    parse_zero_shot_cot_llm_results,
    parse_cot_llm_results
)

# modules used for type hints
from typing import Dict, Any

# check for API
# credentials
@pytest.mark.skipif(
    not all([
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("BUCKET_NAME"),
        os.getenv("TOGETHER_API_KEY"),
        os.getenv("CLAUDE_API_KEY")
    ]),
    reason="API credentials not configured"
)

# full pipeline execution 
# tests
class TestPipelineTitanic:
    """Test full Pipeline execution on titanic_small.csv with real APIs"""
    
    # titanic dataset fixture
    @pytest.fixture(scope="class")
    def titanic_dataset(self) -> Dataset:
        """Create Dataset object for titanic_small.csv"""

        # dataset config
        dataset_path = "data/datasets/titanic_small.csv"
        config_path = "data/dataset_config/titanic_config.json"
        shap_path = "data/shap_values/titanic_shap.csv"
        
        # check if dataset file exists
        if not os.path.exists(dataset_path):
            pytest.skip(f"Dataset file not found: {dataset_path}")
        
        # return dataset 
        # configuration
        return Dataset(
            name="titanic",
            path=dataset_path,
            config_file_path=config_path,
            shap_vals_path=shap_path,
            preprocess_fn=preprocess_titanic,
            target_col="Survived",
            labels={0: "Did not survive", 1: "Survived"}
        )
    
    @pytest.fixture(scope="class")
    def pipeline_configs(self) -> Dict[str, Any]:
        """Create pipeline configuration matching tests.ipynb"""
        return {
            "explanable_model": XGBClassifier(),
            "tune_config_file": "data/tune_config/xgb.json",
            "reasoning_gen_model": Model(
                provider="together",
                name="deepseek-ai/DeepSeek-R1",
                temperature=0.6,
                max_tokens=4096
            ),
            "objective_judge_model": Model(
                provider="anthropic",
                name="claude-haiku-4-5",
                temperature=0.6,
                max_tokens=4096
            ),
            "cot_model": Model(
                provider="google",
                name="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=4096
            )
        }
    
    # results dictionary fixture
    @pytest.fixture(scope="class")
    def results(self) -> Dict[str, Any]:
        """Shared results dictionary for all test steps"""
        return {}
    
    # xai model training and tuning 
    # test
    def test_step_1_xai_model_training(
            self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any]
        ) -> None:
        """Step 1: XAI Model Training and Tuning"""

        # create explainable model
        xai_model = ExplainableModel(
            dataset=titanic_dataset,
            estimator=pipeline_configs["explanable_model"]
        )
        xai_model.explain(params_grid_file=pipeline_configs["tune_config_file"])
        
        assert xai_model.model is not None
        assert xai_model.feature_importances is not None
        assert xai_model.shap_values is not None
        assert os.path.exists(titanic_dataset.config_file_path)
        assert os.path.exists(titanic_dataset.shap_vals_path)
    
    # zero-shot prompting baseline test
    def test_step_2_zero_shot_baseline(
            self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any], 
            results: Dict[str, Any]
        ) -> None:
        """Step 2: Zero-shot Prompting Baseline"""

        # create zero-shot baseline
        baseline = ZeroShotBaseline(
            dataset=titanic_dataset,
            model=pipeline_configs["cot_model"],
            prompt_gen_fn=zero_shot_prompt_generator
        )
        baseline.create_batch_prompts()
        baseline.save_batches_as_jsonl()
        baseline.upload_batches_to_gcs()
        baseline.submit_batch_inference_job()
        baseline.download_job_outputs_from_gcs()
        
        # evaluate baseline
        eval_baseline = Evaluator(
            prompting_strategy="zero-shot-prompting",
            dataset=titanic_dataset,
            results_jsonl_path=baseline.destination_file_name,
            postprocess_fn=parse_baseline_llm_results
        )
        eval_baseline.evaluate()
        results["zero_shot_baseline"] = eval_baseline.metrics
        
        assert "xgboost" in results["zero_shot_baseline"]
        assert "zero-shot-prompting" in results["zero_shot_baseline"]
        assert "accuracy" in results["zero_shot_baseline"]["xgboost"]
        assert "macro_f1_score" in results["zero_shot_baseline"]["xgboost"]
        assert "accuracy" in results["zero_shot_baseline"]["zero-shot-prompting"]
        assert "macro_f1_score" in results["zero_shot_baseline"]["zero-shot-prompting"]
    
    # zero-shot cot ablation test   
    def test_step_3_zero_shot_cot_ablation(
            self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any], 
            results: Dict[str, Any]
        ) -> None:
        """Step 3: Zero-shot CoT Ablation"""

        # create cot config ablation
        cot_config_ablation = COT(
            num_examples_per_agent=10,
            reasoning={},
            thinking_budget=1000
        )
        
        # create zero-shot cot baseline
        cot_ablation = ZeroShotBaseline(
            dataset=titanic_dataset,
            model=pipeline_configs["cot_model"],
            prompt_gen_fn=zero_shot_cot_prompt_generator,
            cot_flag=True,
            cot=cot_config_ablation
        )
        cot_ablation.create_batch_prompts()
        cot_ablation.save_batches_as_jsonl()
        cot_ablation.upload_batches_to_gcs()
        cot_ablation.submit_batch_inference_job()
        cot_ablation.download_job_outputs_from_gcs()
        
        # evaluate cot ablation
        eval_cot_ablation = Evaluator(
            prompting_strategy="zero-shot-cot",
            dataset=titanic_dataset,
            results_jsonl_path=cot_ablation.destination_file_name,
            postprocess_fn=parse_zero_shot_cot_llm_results
        )
        eval_cot_ablation.evaluate()
        results["zero_shot_cot_ablation"] = eval_cot_ablation.metrics
        
        assert "xgboost" in results["zero_shot_cot_ablation"]
        assert "zero-shot-cot" in results["zero_shot_cot_ablation"]
        assert "accuracy" in results["zero_shot_cot_ablation"]["xgboost"]
        assert "macro_f1_score" in results["zero_shot_cot_ablation"]["xgboost"]
        assert "accuracy" in results["zero_shot_cot_ablation"]["zero-shot-cot"]
        assert "macro_f1_score" in results["zero_shot_cot_ablation"]["zero-shot-cot"]
    
    # reasoning generation test
    def test_step_4_reasoning_generation(
            self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any], 
            results: Dict[str, Any]
        ) -> None:
        """Step 4: Reasoning Generation"""

        # create reasoning generator
        reason_generator = ReasonGenerator(
            dataset=titanic_dataset,
            model=pipeline_configs["reasoning_gen_model"],
            prompt_gen_fn=reasoning_prompt_generator
        )
        reason_generator.create_batch_prompts()
        reason_generator.save_batches_as_jsonl()
        reason_generator.submit_batches()
        reasoning = parse_reasoning_llm_results(
            results_jsonl_path=reason_generator.destination_file_name
        )
        
        results["reasoning"] = reasoning
        
        assert isinstance(reasoning, dict)
        assert len(reasoning) > 0
    
    # objective judge evaluation test   
    def test_step_5_objective_judge_evaluation(self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any], 
            results: Dict[str, Any]
        ) -> None:
        """Step 5: Objective Judge Evaluation"""

        # create objective judge
        reasoning = results["reasoning"]
        
        # evaluate objective judge
        judge = ObjectiveJudge(
            dataset=titanic_dataset,
            model=pipeline_configs["objective_judge_model"],
            prompt_gen_fn=objective_judge_prompt_generator
        )
        judge.create_batch_prompts(reasoning=reasoning)
        judge.submit_batch()
        judge_results = parse_objective_judge_results(
            results_jsonl_path=judge.destination_file_name
        )
        results["llm_as_judge"] = judge_results
        
        # validate judge results
        assert isinstance(judge_results, dict)
        assert len(judge_results) > 0
        for idx, metrics in judge_results.items():
            assert "faithfulness" in metrics
            assert "consistency" in metrics
            assert "coherence" in metrics
            assert 1.0 <= metrics["faithfulness"] <= 5.0
            assert 1.0 <= metrics["consistency"] <= 5.0
            assert 1.0 <= metrics["coherence"] <= 5.0

    # cot configuration test
    def test_step_6_cot_configuration(
            self, 
            results: Dict[str, Any]
        ) -> None:
        """Step 6: COT Configuration"""

        # get reasoning from results
        reasoning = results["reasoning"]
        
        # validate cot config
        cot_config = COT(
            num_examples_per_agent=10,
            reasoning=reasoning,
            thinking_budget=1000
        )
        
        assert cot_config.num_examples_per_agent == 10
        assert cot_config.reasoning == reasoning
        assert cot_config.thinking_budget == 1000
        
        results["cot_config"] = cot_config
    
    # icl classification with cot test  
    def test_step_7_icl_classification_with_cot(
            self, 
            titanic_dataset: Dataset, 
            pipeline_configs: Dict[str, Any], 
            results: Dict[str, Any]
        ) -> None:
        """Step 7: ICL Classification with COT"""

        # create icl classifier
        cot_config = results["cot_config"]
        
        icl_classifier = ICLClassifier(
            dataset=titanic_dataset,
            model=pipeline_configs["cot_model"],
            cot=cot_config,
            prompt_gen_fn=cot_prompt_generator
        )
        icl_classifier.create_batch_prompts()
        icl_classifier.save_batches_as_jsonl()
        icl_classifier.upload_batches_to_gcs()
        icl_classifier.submit_batch_inference_job()
        icl_classifier.download_job_outputs_from_gcs()
        
        assert icl_classifier.batches is not None
        assert len(icl_classifier.batches) > 0
        assert icl_classifier.gcp_uri is not None
        assert icl_classifier.job is not None
        assert icl_classifier.destination_file_name is not None
        assert os.path.exists(icl_classifier.destination_file_name)
        
        results["icl_classifier"] = icl_classifier
    
    # evaluation of cot predictions test
    def test_step_8_evaluation_of_cot_predictions(
            self, 
            titanic_dataset: Dataset, 
            results: Dict[str, Any]
        ) -> None:
        """Step 8: Evaluation of COT Predictions"""

        # create evaluation of cot
        icl_classifier = results["icl_classifier"]
        
        # evaluate cot predictions
        eval_cot = Evaluator(
            prompting_strategy="xai-guided-cot",
            dataset=titanic_dataset,
            results_jsonl_path=icl_classifier.destination_file_name,
            postprocess_fn=parse_cot_llm_results
        )
        eval_cot.evaluate()
        results["cot"] = eval_cot.metrics
        
        assert "xgboost" in results["cot"]
        assert "xai-guided-cot" in results["cot"]
        assert "accuracy" in results["cot"]["xgboost"]
        assert "macro_f1_score" in results["cot"]["xgboost"]
        assert "accuracy" in results["cot"]["xai-guided-cot"]
        assert "macro_f1_score" in results["cot"]["xai-guided-cot"]
        assert 0.0 <= results["cot"]["xgboost"]["accuracy"] <= 1.0
        assert 0.0 <= results["cot"]["xai-guided-cot"]["accuracy"] <= 1.0
