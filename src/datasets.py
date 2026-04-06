import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_processing import load_stats, load_split, standardize_data, create_dataset, safe_underscore


class FullDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        split, 
        load_dir,
        spherical, 
        num_files, 
        standardize_vars,
        transforms,
        ):

        dataset = load_split(split, load_dir, num_files)
        stats = load_stats(load_dir)
        standardize_data(dataset, stats, standardize_vars) 

        self.transforms = transforms
        self.spherical = spherical
        self.pre_stack(dataset)


    def pre_stack(self, dataset):

        self.N = dataset["N"]

        transform_e = safe_underscore(self.transforms.e)
        transform_z_hat = safe_underscore(self.transforms.z_hat)

        if self.spherical:
            self.x_stack = np.stack([
                dataset["r_hat"], 
                dataset["z_hat"], 
                dataset["e"]
            ], axis=1).astype(np.float32)
            
            self.z_stack = np.stack([
                dataset["r_hat_norm"], 
                dataset[f"z_hat{transform_z_hat}_norm"], 
                dataset[f"e{transform_e}_norm"]
            ], axis=1).astype(np.float32)           


        else:
            self.x_stack = np.stack([
                dataset["x_hat"], 
                dataset["y_hat"], 
                dataset["z_hat"], 
                dataset["e"]
            ], axis=1).astype(np.float32)
            
            self.z_stack = np.stack([
                dataset["x_hat_norm"], 
                dataset["y_hat_norm"], 
                dataset[f"z_hat{transform_z_hat}_norm"], 
                dataset[f"e{transform_e}_norm"]
            ], axis=1).astype(np.float32)


        self.c_stack = np.stack([
            dataset["dir_x_norm"], 
            dataset["dir_y_norm"], 
            dataset["dir_z_norm"], 
            dataset["e_inc_norm"],
        ], axis=1).astype(np.float32)


    def __len__(self):
        return len(self.N)


    def __getitem__(self, idx):

        z = torch.from_numpy(self.z_stack[idx])
        x = torch.from_numpy(self.x_stack[idx])
        c = torch.from_numpy(self.c_stack[idx])
        N = torch.tensor(self.N[idx], dtype=torch.long)

        return z, N, c, x

        

class ContextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        n_samples,
        standardize_vars,
        load_dir,
        phi=None,
        theta=None, 
        e_inc=None,
        seed=None
        ):

        self.dataset = create_dataset(n_samples, phi, theta, e_inc, seed=seed)
        self.stats = load_stats(load_dir)
        self.standardize_vars = standardize_vars
        standardize_data(self.dataset, self.stats, self.standardize_vars) 
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        dir_x = self.dataset["dir_x_norm"][idx]
        dir_y = self.dataset["dir_y_norm"][idx]
        dir_z = self.dataset["dir_z_norm"][idx]
        e_inc = self.dataset[f"e_inc_norm"][idx] 

        c = np.array([dir_x, dir_y, dir_z, e_inc])
        c = torch.from_numpy(c) # (4,)

        return c
    

def get_data_loader(batch_size, split=None, sample=False, **kwargs):

    if sample:
        dataset = ContextDataset(**kwargs)        
    else:
        dataset = FullDataset(split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train")
    )