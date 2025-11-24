from dataclasses import dataclass
from typing import Callable, Optional, Tuple

@dataclass
class Dataset:
    name: str
    path: str
    preprocess_fn: Optional[Callable] = None
    target_col: Optional[str] = None
    