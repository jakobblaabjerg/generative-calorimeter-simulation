from src.config import load_config, save_config
from src.logger import Logger
from src.datasets import get_data_loader
from src.models import MODEL_REGISTRY
from src.optimizers import create_optimizer
from src.trainer import Trainer

import argparse

def run_train(cfg):

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
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    cfg_base = load_config(f"configs/base_{args.model}.yaml")
    run_train(cfg_base)