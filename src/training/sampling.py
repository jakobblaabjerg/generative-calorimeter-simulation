from src.models import load_model
from src.config import load_config
from src.io import load_stats, save_data
from src.datasets import create_loader
from src.calosim import CaloSimDataset
from src.processing import postprocess
from src.utils import move_to_device

from tqdm import tqdm
import torch


def run_generation(model_dir, data_dir, save_dir, cfg_filters, cfg_sampling):

    cfg_version = load_config(f"{model_dir}/config.yaml")

    device = torch.device(cfg_sampling.device)
    model = load_model(cfg_version, device=device)

    stats = load_stats(load_dir=data_dir)
    standardize_vars = cfg_version.data_loader.standardize_vars

    print("Starting sampling")

    for i in range(cfg_sampling.num_files):
        
        loader = create_loader(
            standardize_vars=standardize_vars, 
            stats=stats, 
            **vars(cfg_sampling.data_loader)
            )
                
        dataset = generate_samples(model, loader, device)
        postprocess(dataset, stats, cfg_filters, standardize_vars)
        save_data(dataset, save_dir, stage="sampled", file_idx=i+1)

    print("Finished sampling")


def generate_samples(model, loader, device, return_outputs=True):

    dataset = CaloSimDataset()
    iterator = tqdm(loader,leave=False)
    model.eval()

    with torch.no_grad():

        for batch in iterator:
            
            batch = move_to_device(batch, device)
            dataset_b = model.sample(batch)

            if return_outputs:
                dataset.append(dataset_b)

    return dataset