from src.config import load_config, save_config
from src.logger import Logger
from src.datasets import get_data_loader
from src.models import GaussianMixtureModel
from src.optimizers import create_optimizer
from src.trainer import Trainer

print("Loading configuration")
config = load_config("configs/base.yaml")
logger = Logger(**vars(config.logger))
run_dir = logger.get_run_dir()
save_config(config, run_dir)


print("Setting up loaders")
train_loader = get_data_loader(split="train", **vars(config.dataset))
val_loader = get_data_loader(split="val", **vars(config.dataset))

print("Initializing model")
model = GaussianMixtureModel(**vars(config.model))
optimizer = create_optimizer(model, config.optimizer)

print("Starting training")
trainer = Trainer(model, optimizer, run_dir, **vars(config.trainer))
trainer.fit(train_loader, val_loader)

# python -m src.train