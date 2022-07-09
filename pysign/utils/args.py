import json
import yaml


class Args:
    def __init__(self):
        pass


def get_default_args():
    default_args = Args()
    default_args_dict = {
        'exp_name': 'exp',
        'batch_size': 128,
        'epoch': 100,
        'seed': 1,
        'eval_epoch': 1,
        'lr': 5e-4,
        'hidden_dim': 64,
        'model': 'EGNN',
        'n_layers': 4,
        'dataset': 'qm9',
        'weight_decay': 1e-6,
        'data_dir': 'cached_datasets'
    }
    for k, v in default_args_dict.items():
        setattr(default_args, k, v)
    return default_args


def load_params(args, param_path):
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
    for k, v in params.items():
        setattr(args, k, v)
    return args
