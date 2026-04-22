from src.evaluator import Evaluator
from src.models import MODEL_REGISTRY
from src.config import load_config, sample_config
from src.logger import setup_logger
from train import run_train
import pandas as pd



def run_sweep(cfg_base, search_space, n_trials):
    
    leaderboard = []
    model_name = cfg_base.name
    log_dir = cfg_base.logger.log_dir
    
    logger = setup_logger(name="sweep_log", save_dir=log_dir)
    logger.info(f"Starting sweep with {n_trials} trials")

    for _ in range(n_trials):

        cfg_version = None
        
        try:
            cfg_version, params = sample_config(cfg_base, search_space)
            run_train(cfg_version)

            evaluator = Evaluator(model_cls=MODEL_REGISTRY[model_name], cfg=cfg_version)
            loss = evaluator.evaluate(split="val")
            num_params = evaluator.num_params()
            version = evaluator.version()

            logger.info(f"Finished version {version} | loss={loss:.5f}")

            leaderboard.append({
                "version": version,
                "split": "val",
                "loss": loss,
                "parameters": num_params,
                **params
            })

            df = pd.DataFrame(leaderboard)
            df = df.sort_values("loss", ascending=True)
            df.to_csv(f"{log_dir}/leaderboard.csv", index=False)

        except Exception as e:

            run_dir = getattr(cfg_version, "run_dir", None)
            version = run_dir.split("_")[-1] if run_dir else None
            logger.error(f"Something failed in version {version}: {e}")
            continue
    
    logger.info("Finished sweep!")


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


optim_space = {
    "optimizer.type": {
        "type": "categorical",
        "values": ["adam", "adamw"]
    },       
    "optimizer.lr": {
        "type": "log_uniform",
        "min": 1e-5,
        "max": 1e-2
    },
}

data_loader_space = {
    "data_loader.batch_size": {
        "type": "categorical",
        "values": [64, 128]
    },
}

mlp_space = {
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
}

encoder_space = {

    "sequence": {
        "model.encoder.output_size": {
            "type": "categorical",
            "values": [64, 128, 256]
        },
        "model.encoder.cell_type": {
            "type": "categorical",
            "values": ["gru", "lstm"]
        },
    },

    "pointnet": {
        "model.encoder.hidden_layers": {
            "type": "categorical",
            "values": [[64, 64], [128, 128], [256, 256], [128, 128, 128]]
        },
        "model.encoder.layer_norm": {
            "type": "categorical",
            "values": [True, False]
        },
        "model.encoder.activation": {
            "type": "categorical",
            "values": ["relu","gelu", "silu", "selu", "leaky_relu"]
        },

        "model.encoder.output_size": {
            "type": "categorical",
            "values": [64, 128, 256]
        },

        "model.encoder.pooling": {
            "type": "categorical",
            "values": ["mean", "max", "sum"]
        }
    },

    "deepsets": {},
    
}

if __name__ == "__main__":

    MODEL = "cfm"
    ENCODER = "deepsets"

    cfg_base = load_config(f"configs/base_{MODEL}.yaml")

    if ENCODER:
        cfg_encoder = load_config(f"configs/{ENCODER}_encoder.yaml")
        cfg_base.model.encoder = cfg_encoder

    search_space = [
        optim_space,
        data_loader_space,
        mlp_space,
        encoder_space[ENCODER] if ENCODER is not None else {},
    ]

    run_sweep(cfg_base, search_space, n_trials=100)