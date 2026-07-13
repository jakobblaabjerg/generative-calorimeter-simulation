from src.logger import setup_logger
from src.config import sample_config
from src.models.registry import MODEL_REGISTRY 

from .evaluation import evaluate_quality, evaluate_complexity
from .trainer import run_train

import torch
import pandas as pd


def run_sweep(cfg, search_space, num_trials, num_mc_samples):
    
    leaderboard = []
    seed = 123

    log_dir = cfg.logger.log_dir
    logger = setup_logger(name="sweep_log", save_dir=log_dir)
    logger.info(f"Starting sweep with {num_trials} trials")

    for _ in range(num_trials):

        cfg_version = None
        
        try:
            cfg_version, params = sample_config(cfg, search_space)
            run_train(cfg_version, seed)

        except Exception as e:
            run_dir = getattr(cfg_version, "run_dir", None) if cfg_version else None
            version = get_version(run_dir)
            logger.error(f"Something failed in version {version}: {e}")
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MODEL_REGISTRY[cfg_version.name](cfg_version.model)
        model.to(device)
        model.load_checkpoint(cfg_version.run_dir)
        
        metrics = {}
    
        metrics_complexity = evaluate_complexity(
            model=model, 
            num_mc_samples=num_mc_samples
        )
        metrics.update(metrics_complexity)

        metrics_quality = evaluate_quality(
            model=model, 
            cfg=cfg_version, 
            split="val", 
            num_mc_samples=num_mc_samples, 
            seed=seed
        )
        metrics.update(metrics_quality)

        version = get_version(cfg_version.run_dir)
        loss = metrics["loss_mean"]
        logger.info(f"Finished version {version} | loss={loss:.5f}")

        leaderboard.append(create_entry(version, metrics, params))
        save_leaderboard(leaderboard, log_dir)
    
    logger.info("Finished sweep!")

def get_version(run_dir):
    return run_dir.split("_")[-1] if run_dir else None

def create_entry(version, metrics, params):
    return {
        "version": version,
        **metrics,
        **params,
    }

def save_leaderboard(leaderboard, log_dir, sort_by="loss_mean"):
    
    df = pd.DataFrame(leaderboard)
    df = df.sort_values(sort_by, ascending=True)
    df.to_csv(f"{log_dir}/leaderboard.csv", index=False)