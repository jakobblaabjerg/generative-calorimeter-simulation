from src.config import load_config
from src.datasets import get_data_loader
from src.models import MixtureDensityNetwork, load_checkpoint
from src.sampler import Sampler
import torch

# version = str(22)
# version = str(31)
# version = str(32)
version = str(7)

print("Loading configuration")
cfg_base = load_config("configs/base.yaml")
cfg_filters = load_config("configs/filters.yaml")
run_dir = f"{cfg_base.logger.log_dir}/version_{version}"
cfg_version = load_config(f"{run_dir}/config.yaml")

print("Loading model")
model = MixtureDensityNetwork(**vars(cfg_version.model))
device = torch.device(cfg_base.sampler.device)
model.to(device)
load_checkpoint(run_dir, model, device)

print("Sampling data")
sampler = Sampler(model, cfg_base.sampler) 
for _ in range(sampler.num_files):
    data_loader = get_data_loader(sample=True, **vars(sampler.cfg_sampler.data_loader))
    sampler.sample(data_loader)
    sampler.inverse_transform(cfg_filters)
    sampler.save_data()