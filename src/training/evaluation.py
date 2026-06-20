from src.config import load_config

from src.models.factory import create_model
from src.io import load_stats
from src.data.datasets import create_loader
from src.utils import set_seed, synchronize_cuda

from .loops import run_epoch
from .sampling import generate_samples

from tqdm import tqdm
import time
import torch





def run_eval(model_dir, data_dir, num_mc_samples, cfg_sampling, seed=None):

    cfg = load_config(f"{model_dir}/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg)
    model.to(device)
    model.load_checkpoint(cfg.run_dir)

    metrics = {}
    
    print("Evaluating complexity")
    metrics_complexity = evaluate_complexity(
        model=model, 
        num_mc_samples=num_mc_samples
    )
    metrics.update(metrics_complexity)

    print("Evaluating quality")
    metrics_quality = evaluate_quality(
        model=model, 
        cfg=cfg, 
        split="test", 
        num_mc_samples=num_mc_samples, 
        seed=seed
    )
    metrics.update(metrics_quality)
    
    print("Evaluating efficiency")
    metrics_efficiency = evaluate_efficiency(
        model=model, 
        cfg=cfg, 
        cfg_sampling=cfg_sampling, 
        data_dir=data_dir, 
        num_mc_samples=num_mc_samples,
        seed=seed
    )
    metrics.update(metrics_efficiency)
    
    return metrics

def evaluate_complexity(model, num_mc_samples):

    return {
        "num_mc_samples": num_mc_samples,
        "num_params": model.num_params
        }



def evaluate_quality(model, cfg, split, num_mc_samples, seed):
    
    loader = create_loader(split=split, **vars(cfg.data_loader))
    losses = []

    iterator = tqdm(range(num_mc_samples), leave=False)  # creates 2 iterators

    for i in iterator:
        if seed is not None:
            set_seed(seed=seed+i)
        loss = run_epoch(model, loader) 
        losses.append(sum(loss))

    return compute_mean_std(values=losses, prefix="loss")

def evaluate_efficiency(model, cfg, cfg_sampling, data_dir, num_mc_samples, seed):

    device = torch.device(cfg_sampling.device)
    model.to(device)

    stats = load_stats(load_dir=data_dir)
    standardize_vars = cfg.data_loader.standardize_vars
    loader = create_loader(
        standardize_vars=standardize_vars, 
        stats=stats,
        **vars(cfg_sampling.data_loader)
        )

    times = []
    iterator = tqdm(range(num_mc_samples), leave=False)

    for i in iterator:
        
        if seed is not None:
            set_seed(seed=seed+i)

        synchronize_cuda(device)
        start = time.time()
        generate_samples(model, loader, device, return_outputs=False)
        synchronize_cuda(device)
        end = time.time()
        times.append(end-start)

    return compute_mean_std(values=times, predix="time")

def compute_mean_std(values, prefix=""):

    mean = sum(values)/len(values)
    std = (sum((x - mean)**2 for x in values)/len(values))**0.5

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
    }
