from src.config import load_config, save_config
from src.logger import Logger
from src.datasets import get_data_loader
from src.models import MixtureDensityNetwork
from src.optimizers import create_optimizer
from src.trainer import Trainer

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
    model = MixtureDensityNetwork(**vars(cfg.model))
    optimizer = create_optimizer(model, cfg.optimizer)

    print("Starting training")
    trainer = Trainer(model, optimizer, run_dir, **vars(cfg.trainer))
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":

    cfg_base = load_config("configs/base.yaml")
    run_train(cfg_base)
