from src.config import load_config, save_config
from src.logger import Logger
from src.datasets import get_data_loader
from src.models import MODEL_REGISTRY
from src.optimizers import create_optimizer
from src.trainer import Trainer
import argparse

# TO DO:
# retrain and training schedule

def run_train(cfg, debug=False):

    print("Loading configuration")
    logger = Logger(**vars(cfg.logger))
    run_dir = logger.get_run_dir()
    cfg.run_dir = run_dir
    save_config(cfg, run_dir)

    print("Setting up loaders")
    train_loader = get_data_loader(split="train", **vars(cfg.data_loader))
    val_loader = get_data_loader(split="val", **vars(cfg.data_loader))

    print("Initializing model")
    model = MODEL_REGISTRY[cfg.name](cfg.model)
    optimizer = create_optimizer(model, cfg.optimizer)

    print("Starting training")
    trainer = Trainer(model, optimizer, run_dir, **vars(cfg.trainer))
    trainer.fit(train_loader, val_loader, debug=debug)


if __name__ == "__main__":

    # user arguments 
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, default=None)
    group.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    # select model 
    if args.cfg_file is not None:
        cfg_base = load_config(args.cfg_file)
    else:
        cfg_base = load_config(f"configs/base_{args.model}.yaml")
    
    # optional default encoder
    if args.encoder and args.model == "cfm":
        cfg_encoder = load_config(f"configs/{args.encoder}_encoder.yaml")
        cfg_base.model.encoder = cfg_encoder

    # additional options 
    cfg_base.logger.log_dir = args.log_dir or f"logs/{cfg_base.name}"

    if args.patience is not None:
        cfg_base.trainer.patience = args.patience

    if args.epochs is not None:
        cfg_base.trainer.epochs = args.epochs

    # train model 
    run_train(cfg_base, debug=args.debug)