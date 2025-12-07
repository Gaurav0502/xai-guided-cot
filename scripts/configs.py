from scripts.constants import VALID_PROVIDERS, VALID_MODELS
from typing import Callable, Dict, Any
from pydantic import (BaseModel, Field, StrictStr, 
                      StrictInt, StrictFloat,
                      field_validator)

class Dataset(BaseModel):
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

class Model(BaseModel):
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
class COT(BaseModel):
    num_examples_per_agent: StrictInt
    reasoning: Dict[StrictInt, StrictStr] = dict()
    thinking_budget: StrictInt

    model_config = {
        "validate_assignment": True,
    }
    