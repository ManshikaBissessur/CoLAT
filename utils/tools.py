import torch
import numpy as np
import random
import argparse
import os
import ruamel.yaml as yaml
import warnings

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def load_conf(path: str = None, method: str = None, dataset: str = None):
    if path is None:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(config_dir, method, f"{method}_{dataset}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        conf_dict = yaml.YAML(typ='rt').load(f)
    return argparse.Namespace(**(conf_dict or {}))


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True