import os, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_processing import load_stats, load_split, standardize_data

class Step2PointDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        split, 
        load_dir, 
        num_files, 
        D, 
        vars_to_standardize,
        transform,
        ):

        self.D = D
        self.dataset = load_split(split, load_dir, num_files)
    
        self.stats = load_stats(load_dir)
        self.vars_to_standardize = vars_to_standardize
        standardize_data(self.dataset, self.stats, self.vars_to_standardize) 

        self.transform = transform

    def __len__(self):
        return len(self.dataset["eid"])


    def __getitem__(self, idx):

        # save split names change

        z_hat = self.dataset["z_hat_norm"][idx]
        e = self.dataset[f"e_{self.transform}_norm"][idx]

        if self.D == 4:
            x_hat = self.dataset["x_hat_norm"][idx]
            y_hat = self.dataset["y_hat_norm"][idx]
            x = np.array([x_hat, y_hat, z_hat, e])

        elif self.D == 3:
            r_hat = self.dataset["r_hat_norm"][idx]
            x = np.array([r_hat, z_hat, e])

        x = torch.from_numpy(x) # (D,)

        dir_x = self.dataset["dir_x_norm"][idx]
        dir_y = self.dataset["dir_y_norm"][idx]
        dir_z = self.dataset["dir_z_norm"][idx]
        e_incident = self.dataset[f"e_incident_norm"][idx] # maybe use self.transform

        c = np.array([dir_x, dir_y, dir_z, e_incident])
        c = torch.from_numpy(c) # (4,)

        return x, c


def get_data_loader(split, batch_size, **dataset_kwargs):

    dataset = Step2PointDataset(split=split, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train")
    )
