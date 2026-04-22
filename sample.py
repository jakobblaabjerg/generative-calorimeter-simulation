from src.config import load_config
from src.datasets import get_data_loader, get_cond_loader
from src.models import MODEL_REGISTRY, load_checkpoint
from src.sampler import Sampler
import torch
from src.data_processing import DataProcessor

#######################

VERSION = 8
MODEL = "cfm"

#######################


print("Loading configuration")

cfg_base = load_config(f"configs/base_{MODEL}.yaml")
cfg_filters = load_config("configs/filters.yaml")
cfg_sampling = load_config("configs/sampling.yaml")

run_dir = f"{cfg_base.logger.log_dir}/version_{str(VERSION)}"
cfg_version = load_config(f"{run_dir}/config.yaml")


print("Loading model")

device = torch.device(cfg_sampling.device)
model = MODEL_REGISTRY[MODEL](cfg_version.model).to(device)
load_checkpoint(run_dir, model, device)


print("Sampling data")

standardize_vars = cfg_version.data_loader.standardize_vars
load_dir = "data/test6"
data_processor = DataProcessor(cfg_filters, output_dir=load_dir)

sampler = Sampler(model) 
for i in range(cfg_sampling.num_files):
    data_loader = get_cond_loader(standardize_vars=standardize_vars, load_dir=load_dir, **vars(cfg_sampling.data_loader))
    data, meta = sampler.sample(data_loader)
    data, meta = data_processor.inverse_transform(data, meta, standardize_vars)
    data_processor.save_data(data, meta, stage="sampled", file_idx=i+1)