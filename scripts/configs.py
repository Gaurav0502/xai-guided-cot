from dataclasses import dataclass
from typing import Callable, Optional, Tuple

@dataclass
class Dataset:
    name: str
    path: str
    config_file_path: str
    shap_vals_path: str 
    preprocess_fn: Optional[Callable] = None
    target_col: Optional[str] = None

@dataclass
class Model:
    name: str
    temperature: int
    max_tokens: int
    