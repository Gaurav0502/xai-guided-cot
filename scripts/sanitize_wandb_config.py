# modules used for typing
from typing import Dict, Any

# sanitize best params
# fetched from wandb
def sanitize_wandb_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and standardizes a wandb configuration dictionary by removing private keys,
    unwrapping values, and converting parameter types to int or float as needed.

    Args:
        cfg (Dict[str, Any]): Raw configuration dictionary fetched from wandb.

    Returns:
        Dict[str, Any]: Sanitized configuration dictionary with appropriate types.
    """

    # required params
    int_params = {"n_estimators", "max_depth", "min_child_weight", 
                  "random_state", "num_boost_round"}
    
    # sanitized output
    out = {}
    for k, v in dict(cfg).items():

        # skip private params
        if str(k).startswith("_") or k == "wandb":
            continue

        # unwrap value if
        # inside a dict
        if isinstance(v, dict) and "value" in v:
            v = v["value"]

        # convert to int/float 
        # if needed
        if isinstance(v, str):
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v_float = float(v)
                    v = v_float
                except Exception:
                    pass

        # convert to int
        if k in int_params:
            try:
                v = int(v)
            except Exception:
                pass

        # add to output
        out[k] = v

    return out