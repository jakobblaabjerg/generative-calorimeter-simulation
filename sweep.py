from src.config import load_config, sample_config, get_search_space
from src.logger import setup_logger
from src.trainer import run_eval
from train import run_train
import pandas as pd
import argparse


def run_sweep(cfg_base, search_space, num_trials, num_samples, debug=False):
    
    leaderboard = []

    log_dir = cfg_base.logger.log_dir
    logger = setup_logger(name="sweep_log", save_dir=log_dir)
    logger.info(f"Starting sweep with {num_trials} trials")

    for _ in range(num_trials):

        cfg_version = None
        
        try:
            cfg_version, params = sample_config(cfg_base, search_space)
            run_train(cfg_version, debug=debug)

        except Exception as e:
            run_dir = getattr(cfg_version, "run_dir", None) if cfg_version else None
            version = get_version(run_dir)
            logger.error(f"Something failed in version {version}: {e}")
            continue

        metrics = run_eval(cfg_version, split="val", num_samples=num_samples, seed=0) # rembember seed here !?
        version = get_version(cfg_version.run_dir)
        logger.info(f"Finished version {version} | loss={metrics["loss_mean"]:.5f}")

        leaderboard.append(make_leaderboard_entry(version, metrics, params))
        save_leaderboard(leaderboard, log_dir)
    
    logger.info("Finished sweep!")


def get_version(run_dir):
    return run_dir.split("_")[-1] if run_dir else None


def make_leaderboard_entry(version, metrics, params):
    return {
        "version": version,
        **metrics,
        **params,
    }


def save_leaderboard(leaderboard, log_dir, sort_by="loss_mean"):
    
    df = pd.DataFrame(leaderboard)
    df = df.sort_values(sort_by, ascending=True)
    df.to_csv(f"{log_dir}/leaderboard.csv", index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--space", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # load base config
    cfg_base = load_config(f"configs/base_{args.model}.yaml")

    # optional encoder
    if args.encoder and args.model == "cfm":
        cfg_encoder = load_config(f"configs/{args.encoder}_encoder.yaml")
        cfg_base.model.encoder = cfg_encoder


    print(args.debug)

    search_space = get_search_space("configs", args.model, args.encoder, args.space)
    run_sweep(cfg_base, search_space, args.trials, args.samples, args.debug)