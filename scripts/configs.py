# modules for validation and typing
from scripts.constants import (VALID_PROVIDERS, 
                               VALID_MODELS)
from typing import Callable, Dict, Any
from pydantic import (BaseModel, Field, StrictStr, 
                      StrictInt, StrictFloat,
                      field_validator)

# module for file operations
import os

# dataset configuration
class Dataset(BaseModel):
    """
    Configuration for a dataset used in the pipeline.

    Attributes:
        name (StrictStr): Name of the dataset.
        path (StrictStr): Path to the dataset CSV file.
        config_file_path (StrictStr): Path to the dataset configuration file.
        shap_vals_path (StrictStr): Path to store SHAP values.
        preprocess_fn (Callable): Preprocessing function for the dataset.
        target_col (StrictStr): Name of the target column.
        labels (Dict[StrictInt, StrictStr]): Mapping of class labels (must be binary).
    
    Raises:
        FileNotFoundError: If the dataset file does not exist at the specified path.
        ValueError: If the dataset path does not end with '.csv'.
        TypeError: If the preprocess_fn is not callable.
        ValueError: If labels do not contain exactly 2 entries for binary classification.

    """

    name: StrictStr
    path: StrictStr
    config_file_path: StrictStr
    shap_vals_path: StrictStr 
    preprocess_fn: Callable
    target_col: StrictStr
    labels: Dict[StrictInt, StrictStr] 

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    @field_validator("preprocess_fn")
    @classmethod
    def _validate_preprocess_fn(cls, v):
        if not callable(v):
            raise TypeError("preprocess_fn must be callable")
        return v
    
    @field_validator("path")
    @classmethod
    def _validate_path(cls, v):

        # check file extension
        if not v.endswith(".csv"):
            raise ValueError("Dataset path must end with '.csv'")
        
        # check file existence
        if not os.path.isfile(v):
            raise FileNotFoundError(f"Dataset file does not exist at path: {v}")
        return v
    
    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v):

        # check for 
        # binary classification
        if len(v) != 2:
            raise ValueError("Labels must have exactly 2 entries (binary classification)")
        return v

# model configuration
class Model(BaseModel):
    """
    Configuration for a language or ML model.

    Attributes:
        provider (StrictStr): Provider name.
        name (StrictStr): Model name.
        temperature (StrictFloat): Sampling temperature for generation.
        max_tokens (StrictInt): Maximum number of tokens to generate.
    
    Raises:
        ValueError: If provider is not in VALID_PROVIDERS.
        ValueError: If model name is not valid for the specified provider.

    """

    provider: StrictStr
    name: StrictStr
    temperature: StrictFloat
    max_tokens: StrictInt

    model_config = {
        "validate_assignment": True,
    }

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v):
        if v not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider: {v}. Must be one of {VALID_PROVIDERS}")
        return v
    
    @field_validator("name")
    @classmethod
    def _validate_name(cls, v, values):
        provider = values.data.get("provider")
        if provider and v not in VALID_MODELS.get(provider, []):
            raise ValueError(f"Invalid model name: {v} for provider: {provider}. Must be one of {VALID_MODELS[provider]}")
        return v

# chain-of-thought 
# configuration
class COT(BaseModel):
    """
    Configuration for chain-of-thought (CoT) reasoning.

    Attributes:
        num_examples_per_agent (StrictInt): Number of examples per agent.
        reasoning (Dict[StrictInt, StrictStr]): Reasoning steps or templates.
        thinking_budget (StrictInt): Budget for reasoning steps.

    """
    
    num_examples_per_agent: StrictInt
    reasoning: Dict[StrictInt, StrictStr] = dict()
    thinking_budget: StrictInt

    model_config = {
        "validate_assignment": True,
    }
    