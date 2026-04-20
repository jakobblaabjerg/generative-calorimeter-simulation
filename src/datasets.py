import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_processing import load_stats, load_split, standardize_data, create_meta, safe_underscore, merge_dicts
from collections import defaultdict
from functools import partial


def setup_var_names(transforms, spherical):

    x_vars = ["z_hat", "e"]
    z_vars = [f"z_hat{safe_underscore(transforms.z_hat)}_norm", f"e{safe_underscore(transforms.e)}_norm"] 
    
    if spherical:
        x_vars = ["r_hat", *x_vars]
        z_vars = ["r_hat_norm", *z_vars]
    else:
        x_vars = ["x_hat", "y_hat", *x_vars]
        z_vars = ["x_hat_norm", "y_hat_norm", *z_vars] 

    c_vars = ["dir_x_norm", "dir_y_norm", "dir_z_norm", "e_inc_norm"]
    return x_vars, z_vars, c_vars


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            split,
            load_dir,
            num_files,
            transforms,
            spherical,
            standardize_vars,
            ):

        super().__init__()

        # load split data
        self.data, self.meta = load_split(split, load_dir, num_files)

        # standardize data 
        stats = load_stats(load_dir)
        standardize_data(self.meta, stats, standardize_vars) 
        standardize_data(self.data, stats, standardize_vars) 

        self.x_vars, self.z_vars, self.c_vars = setup_var_names(transforms, spherical)


class GroupedDataset(BaseDataset):

    def __init__(
            self, 
            split, 
            load_dir, 
            num_files, 
            transforms, 
            spherical, 
            standardize_vars,
            sort_by_time,
            ):
        
        super().__init__(split, load_dir, num_files, transforms, spherical, standardize_vars)

        if sort_by_time:
            self.sort_by_time()
        
        self.pre_compute_index()


    def sort_by_time(self):

        order = np.lexsort(keys=(self.data["t"], self.data["eid"]))

        for k, v in self.data.items():
            if k in self.z_vars + self.x_vars:
                self.data[k] = v[order]


    def pre_compute_index(self):

        self.indices = defaultdict(list)

        for i, eid in enumerate(self.data["eid"]):
            self.indices[eid].append(i)

        self.indices = {k: np.array(v) for k, v in self.indices.items()}


    def __len__(self):
        return len(self.meta["eid"])

    def __getitem__(self, idx):

        mask = self.indices[idx]

        x = np.column_stack([self.data[k][mask] for k in self.x_vars])
        z = np.column_stack([self.data[k][mask] for k in self.z_vars])
        c = np.array([self.meta[k][idx] for k in self.c_vars])
    
        z = torch.from_numpy(z.astype(np.float32))
        x = torch.from_numpy(x.astype(np.float32))
        c = torch.from_numpy(c.astype(np.float32)) 

        return x, z, c


def collate_padded(batch, max_seq_len):

    x_list, z_list, c_list = zip(*batch)

    batch_size = len(x_list)
    point_dim = z_list[0].size(1)

    num_points = [min(x.size(0), max_seq_len) for x in x_list]

    x_padded = torch.zeros(batch_size, max_seq_len, point_dim, dtype=x_list[0].dtype)
    z_padded = torch.zeros(batch_size, max_seq_len, point_dim, dtype=z_list[0].dtype)

    for i in range(batch_size):
        x_padded[i, :num_points[i]] = x_list[i][:num_points[i]]
        z_padded[i, :num_points[i]] = z_list[i][:num_points[i]]

    num_points = torch.tensor(num_points, dtype=torch.long)
    c = torch.stack(c_list)

    return x_padded, z_padded, c, num_points


def collate_sparse(batch):

    x_list, z_list, c_list = zip(*batch)

    x = torch.cat(x_list, dim=0)
    z = torch.cat(z_list, dim=0)

    num_points = [x.size(0) for x in x_list]
    num_points = torch.tensor(num_points, dtype=torch.long)
    c = torch.stack(c_list)

    return x, z, c, num_points


class FlatDataset(BaseDataset):

    def __init__(
            self, 
            split, 
            load_dir, 
            num_files, 
            transforms, 
            spherical, 
            standardize_vars,
            ):
        
        super().__init__(split, load_dir, num_files, transforms, spherical, standardize_vars)
        
        data = merge_dicts(self.data, self.meta)
        del self.data, self.meta
        self._pre_stack_data(data)



    def _pre_stack_data(self, data):

        self.x = np.stack([data[k] for k in self.x_vars], axis=1).astype(np.float32)
        self.z = np.stack([data[k] for k in self.z_vars], axis=1).astype(np.float32)
        self.c = np.stack([data[k] for k in self.c_vars], axis=1).astype(np.float32)
        self.num_points = data["N"]

        self.x = torch.from_numpy(self.x)
        self.z = torch.from_numpy(self.z)
        self.c = torch.from_numpy(self.c)
        self.num_points = torch.as_tensor(self.num_points, dtype=torch.long)

    def __len__(self):
        return len(self.num_points)

    def __getitem__(self, idx):
        return self.x[idx], self.z[idx], self.c[idx], self.num_points[idx]


def get_data_loader(batch_size, split, struc, collate=None, **kwargs):

    # loader used for traning 

    collate_fn = get_collate_fn(collate)

    if struc == "flat":
        dataset = FlatDataset(split=split, **kwargs) 

    elif struc == "grouped":
        dataset = GroupedDataset(split=split, **kwargs) 

    else:
        raise ValueError(f"Unknown structure: {struc}")


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )


def get_cond_loader(batch_size, **kwargs):

    # loader used sampling 

    return DataLoader(
        SampleDataset(**kwargs),
        batch_size=batch_size,
        shuffle=False,
    )


def get_collate_fn(cfg):

    if cfg is None:
        return None
    
    elif cfg.mode == "padded":
        return lambda batch: collate_padded(batch, max_seq_len=cfg.max_seq_len)

    elif cfg.mode == "sparse":
        return collate_sparse        

    else:
        raise ValueError(f"Unknown collate mode: {cfg.mode}")


class SampleDataset(torch.utils.data.Dataset):

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

        self.meta = create_meta(n_samples, phi, theta, e_inc, seed=seed)
        stats = load_stats(load_dir)
        standardize_data(self.meta, stats, standardize_vars) 
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        dir_x = self.meta["dir_x_norm"][idx]
        dir_y = self.meta["dir_y_norm"][idx]
        dir_z = self.meta["dir_z_norm"][idx]
        e_inc = self.meta["e_inc_norm"][idx] 

        c = np.array([dir_x, dir_y, dir_z, e_inc])
        c = torch.from_numpy(c) # (4,)

        return c