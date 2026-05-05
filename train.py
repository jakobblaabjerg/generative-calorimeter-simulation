import argparse

from src.config import load_config, override_config
from src.trainer import run_train


def main():

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--cfg_file", type=str)

    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_files", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = load_config(f"configs/base_{args.model}.yaml")

    if args.encoder and args.model == "cfm":
        cfg_encoder = load_config(f"configs/{args.encoder}_encoder.yaml")
        cfg.model.encoder = cfg_encoder

    cfg.logger.log_dir = args.log_dir or f"logs/{cfg.name}"

    overrides = {
        "trainer.patience": args.patience,
        "trainer.epochs": args.epochs,
        "data_loader.load_dir": args.data_dir,
        "data_loader.num_files": args.num_files,
    }

    override_config(cfg, overrides)
    run_train(cfg=cfg, debug=args.debug)



if __name__ == "__main__":
    main()