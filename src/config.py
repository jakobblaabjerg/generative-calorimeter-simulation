import os, yaml, copy, random, json
from types import SimpleNamespace 
import numpy as np 





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


def set_nested(cfg, key, value):
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise AttributeError(f"{p} not found in config")
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def sample_config(cfg_base, search_space):

    if isinstance(search_space, list):
        merged = {}
        for space in search_space:
            overlap = merged.keys() & space.keys()
            if overlap:
                raise ValueError(f"Duplicate keys in search space: {overlap}")
            merged.update(space)
        search_space = merged

    cfg_version = copy.deepcopy(cfg_base)

    params = {k: sample_param(v) for k, v in search_space.items()}

    for key, value in params.items():
        set_nested(cfg_version, key, value)

    return cfg_version, params


def sample_param(spec):

    param_type = spec["type"]

    if param_type == "categorical":
        return random.choice(spec["values"])

    elif param_type == "uniform":
        return float(np.random.uniform(spec["min"], spec["max"]))

    elif param_type == "log_uniform":
        return float(np.exp(np.random.uniform(np.log(spec["min"]), np.log(spec["max"]))))

    elif param_type == "int":
        return int(random.randint(spec["min"], spec["max"]))
    
    else:
        raise ValueError(f"Unknown sampling type: {param_type}")
    


def get_search_space(load_dir, model, encoder, selected_space):

    # load full search space
    file_path = os.path.join(load_dir, "search_space.json")
    with open(file_path, "r") as f:
        search_space_full = json.load(f)

    if selected_space is None:
        
        if model == "cfm":
            selected_space = ["optim", "data_loader", "mlp"]
            
            if encoder is not None:
                selected_space.append("encoder")
        
        elif model == "mdn":
            selected_space = ["optim", "data_loader", "mlp", "mixture"] 
    
    else:
        selected_space = [s.strip() for s in selected_space.split(",")]

    search_space = []
    for k in selected_space:
        if k == "encoder" and encoder is not None:
            search_space.append(search_space_full[k][encoder])
        else:
            search_space.append(search_space_full[k])

    return search_space


def override_config(cfg, overrides):

    for k, v in overrides.items():
        if v is not None:
            parts = k.split(".")
            obj = cfg
            for p in parts[:-1]:
                if not hasattr(obj, p):
                    raise AttributeError(f"{p} not found in config")
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)