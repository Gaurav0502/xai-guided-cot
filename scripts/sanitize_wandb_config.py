def sanitize_wandb_config(cfg: dict) -> dict:
    
    int_params = {"n_estimators", "max_depth", "min_child_weight", 
                  "random_state", "num_boost_round"}
    out = {}
    for k, v in dict(cfg).items():

        if str(k).startswith("_") or k == "wandb":
            continue

        if isinstance(v, dict) and "value" in v:
            v = v["value"]

        if isinstance(v, str):
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v_float = float(v)
                    v = v_float
                except Exception:
                    pass

        if k in int_params:
            try:
                v = int(v)
            except Exception:
                pass

        out[k] = v
    return out