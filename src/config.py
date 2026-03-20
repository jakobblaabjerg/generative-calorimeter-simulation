import os, yaml
from types import SimpleNamespace 

def load_config(file_path):

    with open(file_path) as f:
        config = yaml.safe_load(f)

    config = dict_to_namespace(config)

    return config

def dict_to_namespace(d):
    return SimpleNamespace(**{
        k: dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })

def namespace_to_dict(ns):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
        for k, v in vars(ns).items()
    }

def save_config(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(namespace_to_dict(config), f)