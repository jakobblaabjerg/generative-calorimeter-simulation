import random 

from src.config import load_config, sample_config
from src.logger import Logger
from src.models import MixtureDensityNetwork, load_checkpoint
from src.datasets import get_data_loader
import torch
from train import run_train
import numpy as np
import pandas as pd


# class Evaluator:

#     def __init__(self, cfg):

#         pass


#     def evaluate(self, split="val"):
#         pass
#     def get_data_loader(self):
#         pass

def evaluate(cfg):

    model = MixtureDensityNetwork(**vars(cfg.model))
    device = torch.device(cfg.sampler.device)
    model.to(device)
    run_dir = cfg.run_dir
    load_checkpoint(run_dir, model, device)
    data_loader = get_data_loader(split="val", **vars(cfg.data_loader))

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
            loss = model(*batch)

            if not isinstance(loss, (tuple, list)):
                loss = (loss,)

            if batch_idx == 0:
                loss_list = [[] for _ in range(len(loss))]

            for i, l in enumerate(loss):
                loss_list[i].append(l.item())
            
    loss_components = [sum(l)/len(l) for l in loss_list]
    loss_total = sum(loss_components)
    n_params = sum(p.numel() for p in model.parameters())
    version = run_dir.split("_")[-1]

    return loss_total, n_params, version


def run_sweep(cfg_base, search_space, n_trials):
    
    leaderboard = []

    for _ in range(n_trials):
        
        cfg_version, params = sample_config(cfg_base, search_space)
        run_train(cfg_version)

        loss, n_params, version = evaluate(cfg_version)

        leaderboard.append({
            "version": version,
            "val_acc": loss,
            "parameters": n_params,
            **params
        })

        df = pd.DataFrame(leaderboard)
        df = df.sort_values("val_acc", ascending=True)
        df.to_csv(f"{cfg_version.logger.log_dir}/leaderboard.csv", index=False)


if __name__ == "__main__":

    search_space = {
        "optimizer.type": {
            "type": "categorical",
            "values": ["adam", "sgd"]
        },       
        "optimizer.lr": {
            "type": "log_uniform",
            "min": 1e-5,
            "max": 1e-2
        },
        "data_loader.batch_size": {
            "type": "categorical",
            "values": [64, 128, 256]
        },
        "model.hidden_layers": {
            "type": "categorical",
            "values": [[64, 64], [128, 128], [256, 256], [128, 128, 128]]
        },
        "model.layer_norm": {
            "type": "categorical",
            "values": [True, False]
        },
        "model.activation": {
            "type": "categorical",
            "values": ["relu","gelu","tanh","sigmoid","elu","selu","leaky_relu"]
        },
        "model.k": {
            "type": "int",
            "min": 2,
            "max": 6
        },

        "model.add_jacobian": {
            "type": "categorical",
            "values": [True, False]
        }

    }

    cfg_base = load_config("configs/base.yaml")
    cfg_base.logger.log_dir = "logs/sweep"
    run_sweep(cfg_base, search_space, n_trials=100)