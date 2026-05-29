""" 
Utilities for configuration management and hyperparameter search. 

This module provides functionality for: 
- Loading and saving YAML configurations 
- Converting dictionaries to namespaces 
- Updating nested configuration values 
- Sampling hyperparameters from search spaces 

"""

import os, yaml, copy, random, json
from types import SimpleNamespace 
from typing import Any
import numpy as np 


def load_config(file_path: str) -> SimpleNamespace:
    """ 
    Load a YAML configuration file. 
    
    Parameters 
    ---------- 
    file_path : str 
        Path to YAML config file. 
        
    Returns 
    ------- 
    SimpleNamespace 
        Configuration object with attribute-style access. 
    """

    with open(file_path) as f:
        config = yaml.safe_load(f)

    return dict_to_namespace(config)




def save_config(config: SimpleNamespace, save_dir: str) -> None:
    """ 
    Save configuration to YAML file. 
    
    Parameters 
    ---------- 
    config : SimpleNamespace 
        Configuration object. 
    save_dir : str 
        Directory where config.yaml will be stored. 
    """

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yaml")
    
    with open(config_path, "w") as f:
        yaml.safe_dump(namespace_to_dict(config), f)




def dict_to_namespace(data: dict) -> SimpleNamespace:

    """ 
    Recursively convert a dictionary into a SimpleNamespace. 
    
    Parameters 
    ---------- 
    data : dict 
        Input dictionary. 
        
    Returns 
    ------- 
    SimpleNamespace 
        Namespace representation of dictionary. 
    """
    return SimpleNamespace(
        **{
            key: dict_to_namespace(value) 
            if isinstance(value, dict) 
            else value
            for key, value in data.items()
        }
    )




def namespace_to_dict(namespace: SimpleNamespace) -> dict:
    """ 
    Recursively convert a SimpleNamespace into a dictionary. 
    
    Parameters 
    ---------- 
    namespace : SimpleNamespace 
        Namespace object. 
    
    Returns 
    ------- 
    dict 
        Dictionary representation. 
    """

    return {
        key: namespace_to_dict(value) 
        if isinstance(value, SimpleNamespace) 
        else value
        for key, value in vars(namespace).items()
    }




def set_nested_attr(config, key, value):
    """ 
    Set a nested attribute using dot notation. 
    
    Example 
    ------- 
    set_nested_attr(config, "optimizer.lr", 1e-3) 
    
    Parameters 
    ---------- 
    config : SimpleNamespace 
        Configuration object. 
    key : str 
        Dot-separated attribute path. 
    value : Any 
        Value to assign. 
    """
    parts = key.split(".")
    obj = config
    
    for part in parts[:-1]:
    
        if not hasattr(obj, part):
            raise AttributeError(f"{part} not found in config")
        obj = getattr(obj, part)
    
    setattr(obj, parts[-1], value)




def override_config(config: SimpleNamespace, overrides: dict[str, Any]):

    """ 
    Override configuration values from dictionary. 
    
    Parameters 
    ---------- 
    
    config : SimpleNamespace 
        Base configuration. 
    overrides : dict[str, Any] 
        Dictionary of overrides using dot notation. 
    """

    for key, value in overrides.items():        
        if value is not None:            
            set_nested_attr(config, key, value)



def sample_param(spec: dict) -> Any:
    """ 
    Sample a parameter from a search-space specification. 
    
    Supported parameter types 
    ------------------------- 
    categorical 
    uniform 
    log_uniform 
    int 
    
    Parameters 
    ---------- 
    spec : dict 
        Parameter specification. 
        
    Returns 
    ------- 
    Any 
        Sampled parameter value. 
    """

    param_type = spec["type"]

    if param_type == "categorical":
        return random.choice(spec["values"])

    if param_type == "uniform":
        return float(np.random.uniform(spec["min"], spec["max"]))

    if param_type == "log_uniform":
        return float(
            np.exp(
                np.random.uniform(
                    np.log(spec["min"]), 
                    np.log(spec["max"])
                )
            )
        )

    if param_type == "int":
        return int(random.randint(spec["min"], spec["max"]))
    
    raise ValueError(f"Unknown sampling type: {param_type}")



def sample_config(
        config: SimpleNamespace, 
        search_space: dict | list[dict]
    ) -> tuple[SimpleNamespace, dict]:

    """ 
    Sample a new configuration from a hyperparameter search space. 
    
    Parameters 
    ---------- 
    config : SimpleNamespace 
        Base configuration.     
    search_space : dict | list[dict] 
        Search-space specification(s). 
    
    Returns 
    ------- 
    tuple 
        (sampled_config, sampled_parameters) 
    """

    # merge multiple spaces if needed
    if isinstance(search_space, list):
        merged = {}
        
        for space in search_space:
            overlap = merged.keys() & space.keys()
            
            if overlap:
                raise ValueError(f"Duplicate keys in search space: {overlap}")
            
            merged.update(space)
        
        search_space = merged

    config_sampled = copy.deepcopy(config)
    params = {key: sample_param(spec) for key, spec in search_space.items()}

    for key, value in params.items():
        set_nested_attr(config_sampled, key, value)

    return config_sampled, params




def get_search_space(
        load_dir: str, 
        model: str, 
        encoder: str | None, 
        selected_space: str | None
        ) -> list[dict]:


    """ 
    Load and select hyperparameter search spaces. 
    
    Parameters 
    ---------- 
    load_dir : str 
        Directory containing search_space.json. 
    model : str 
        Model name. 
    encoder : str | None 
        Encoder type. 
    selected_space : str | None 
        Comma-separated list of search-space groups. 
    
    Returns 
    ------- 
    list[dict] 
        Selected search-space dictionaries. 
    """

    # load full search space
    file_path = os.path.join(load_dir, "search_space.json")
    
    with open(file_path, "r") as f:
        search_space_full = json.load(f)


    if selected_space is None:

        defaults = {
            "cfm": ["optim", "data_loader", "mlp"],
            "mdnV1": ["optim", "data_loader", "mlp", "mixture"],
            "mdnV2": ["optim", "data_loader", "mlp", "mixture", "mdn_head", "poisson_head"]
        }

        selected_space = defaults.get(model, []).copy()

        if selected_space is None: 
            raise ValueError(f"Unknown model: {model}") 
        
        if encoder is not None and model == "cfm": 
            selected_space.append("encoder")
 
    else:
        selected_space = [s.strip() for s in selected_space.split(",")]

    search_space = []
    
    for key in selected_space:
        
        if key == "encoder" and encoder is not None:
            search_space.append(search_space_full[key][encoder])
        else:
            search_space.append(search_space_full[key])

    return search_space