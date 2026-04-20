from src.evaluator import Evaluator
from src.models import MixtureDensityNetwork
from src.config import load_config, sample_config
from train import run_train
import pandas as pd


def run_sweep(cfg_base, search_space, n_trials):
    
    leaderboard = []

    for _ in range(n_trials):
        
        try:
            cfg_version, params = sample_config(cfg_base, search_space)
            run_train(cfg_version)

            evaluator = Evaluator(model_cls=MixtureDensityNetwork, cfg=cfg_version)
            loss = evaluator.evaluate(split="val")
            num_params = evaluator.num_params()
            version = evaluator.version()

            leaderboard.append({
                "version": version,
                "split": "val",
                "loss": loss,
                "parameters": num_params,
                **params
            })

            df = pd.DataFrame(leaderboard)
            df = df.sort_values("acc", ascending=True)
            df.to_csv(f"{cfg_version.logger.log_dir}/leaderboard.csv", index=False)

        except Exception as e:
            print("Run failed:", e)
            continue


search_space_mdn = {
    "optimizer.type": {
        "type": "categorical",
        "values": ["adam", "adamw"]
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


search_space_cfm = {
    "optimizer.type": {
        "type": "categorical",
        "values": ["adam", "adamw"]
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
    "model.mlp.hidden_layers": {
        "type": "categorical",
        "values": [[64, 64], [128, 128], [256, 256], [128, 128, 128]]
    },
    "model.mlp.layer_norm": {
        "type": "categorical",
        "values": [True, False]
    },
    "model.mlp.activation": {
        "type": "categorical",
        "values": ["relu","gelu", "silu", "selu", "leaky_relu"]
    },

    "model.encoder.hidden_size": {
        "type": "categorical",
        "values": [64, 128]
    },

    "model.encoder.cell_type": {
        "type": "categorical",
        "values": ["gru", "lstm"]
    }

}




if __name__ == "__main__":

    MODEL = "cfm"
    cfg_base = load_config(f"configs/base_{MODEL}.yaml")
    cfg_base.logger.log_dir = "logs/sweep_cfm"
    run_sweep(cfg_base, search_space, n_trials=100)