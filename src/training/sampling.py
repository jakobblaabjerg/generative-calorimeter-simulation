from src.models.registry import MODEL_REGISTRY 
from src.models import cfm, mdn
from src.config import load_config
from src.io import load_stats, save_data
from src.data.datasets import create_loader
from src.calosim import CaloSimDataset
from src.utils import move_to_device
from src.processing import postprocess_data

from tqdm import tqdm
import torch


def run_generation(model_dir, data_dir, save_dir, cfg_dataset, cfg_sampling):

    cfg_version = load_config(f"{model_dir}/config.yaml")

    device = torch.device(cfg_sampling.device)
    model = MODEL_REGISTRY[cfg_version.name](cfg_version.model) 
    model.load_checkpoint(model_dir) # changed from run dir
    model.to(device)


    dataset_stats = load_stats(load_dir=data_dir)
    standardize_vars = cfg_version.data_loader.standardize_vars
    convert_to_voxel = getattr(getattr(cfg_version, "sampling", None), "convert_to_voxel", False)

    print("Starting sampling")

    for i in range(cfg_sampling.num_files):
        
        loader = create_loader(
            standardize_vars=standardize_vars, 
            stats=dataset_stats, 
            c_vars = cfg_version.model.input_vars.c_vars,
            **vars(cfg_sampling.data_loader)
            )
                  
        dataset = generate_samples(model, loader)
        postprocess_data(dataset, dataset_stats, cfg_dataset, standardize_vars, convert_to_voxel)
        save_data(dataset, save_dir, stage="sampled", file_name=f"file_{i+1}")

    print("Finished sampling")


def generate_samples(model, loader, return_outputs=True):

    device = model.device
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