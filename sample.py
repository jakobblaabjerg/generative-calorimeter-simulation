from src.config import load_config
from src.datasets import get_data_loader, get_cond_loader
from src.models import MODEL_REGISTRY, load_checkpoint
from src.sampler import Sampler
import torch

#######################

VERSION = 5
MODEL = "cfm"

#######################


print("Loading configuration")

cfg_base = load_config(f"configs/{MODEL}_base.yaml")
cfg_filters = load_config("configs/filters.yaml")
cfg_sampling = load_config("configs/sampling.yaml")

run_dir = f"{cfg_base.logger.log_dir}/version_{str(VERSION)}"
cfg_version = load_config(f"{run_dir}/config.yaml")


print("Loading model")

device = torch.device(cfg_sampling.device)
model = MODEL_REGISTRY[MODEL](**vars(cfg_version.model)).to(device)
load_checkpoint(run_dir, model, device)


print("Sampling data")

sampler = Sampler(model, cfg_base.sampler) 
for _ in range(sampler.num_files):
    data_loader = get_cond_loader(**vars(cfg_base.sampler.data_loader))
    sampler.sample(data_loader)
    sampler.inverse_transform(cfg_filters)
    sampler.save_data()