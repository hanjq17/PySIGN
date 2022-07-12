import json
import yaml
from easydict import EasyDict


def load_params(param_path):
    """
    load the arguments from json/yml file.
    """
    if param_path.endswith(".yml") or param_path.endswith(".yaml"):
        load_func = yaml.safe_load
    elif param_path.endswith(".json"):
        load_func = json.load
    else:
        raise NotImplementedError(f"Unsupported parameter file: {param_path}.")

    with open(param_path, 'r') as f:
        params = load_func(f)
    params = EasyDict(params)

    return params
