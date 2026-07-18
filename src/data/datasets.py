import numpy as np
import torch
from collections import defaultdict


from src.io import load_split
from src.calosim import CaloSimDataset

from .collate import COLLATE_REGISTRY
from .transforms import standardize_data, normalize_meta


class BaseTorchDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            split,
            load_dir,
            num_files,
            transforms,
            input_vars,
            standardize_vars,
            ):

        super().__init__()

        self.dataset, self.stats = load_split(split, load_dir, num_files)
        standardize_data(self.dataset, self.stats, standardize_vars) 
        self.x_vars, self.z_vars, self.c_vars = get_feature_names(input_vars, transforms)


    def create_collate_fn(self, name):
        return None


def get_feature_names(input_vars, transforms):

    """
    Create feature variable names used as model inputs and targets.

    Parameters
    ----------
    transforms : object
        Configuration specifying the transform applied to each feature.
    spherical : bool
        If True, use spherical coordinates (r_hat).
        Otherwise use Cartesian coordinates (x_hat, y_hat).

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Input variables (x_vars), transformed target variables (z_vars),
        and conditioning variables (c_vars).
    """
    x_vars = [var for var in input_vars.z_vars if hasattr(transforms, var)]
    z_vars = [feature_name(var, getattr(transforms, var, None)) for var in input_vars.z_vars]
    c_vars = [feature_name(var, getattr(transforms, var, None)) for var in input_vars.c_vars]
    
    return x_vars, z_vars, c_vars

def feature_name(name, transform=None):

    """
    Construct the standardized target variable name.

    Parameters
    ----------
    name : str
        Base feature name.
    transform : str or None, optional
        Name of the transform applied to the feature.

    Returns
    -------
    str
        Standardized variable name following the convention
        '<name>_<transform>_norm' or '<name>_norm'.
    """

    suffix = f"_{transform}" if transform else ""
    return f"{name}{suffix}_norm"



class EventTorchDataset(BaseTorchDataset):

    """
    Event-level dataset.

    Each sample corresponds to a complete event containing a variable
    number of points. Samples are returned as tensors of shape
    (num_points, point_dim) together with event-level conditioning variables.
    """


    def __init__(
            self, 
            split, 
            load_dir, 
            num_files, 
            transforms, 
            input_vars, 
            standardize_vars,
            is_ragged,
            sort_by_time=False,
            ):
        
        super().__init__(split, load_dir, num_files, transforms, input_vars, standardize_vars)
        self.is_ragged = is_ragged

        if sort_by_time:
            self.sort_by_time()

        if self.is_ragged:    
            self.create_index_map()

        else:
            self.create_fixed_index()


    def create_fixed_index(self):

        _, counts = np.unique(self.dataset.data["idx"], return_counts=True)
        size = counts[0]

        for k in self.x_vars + self.z_vars:
            self.dataset.data[k] = self.dataset.data[k].reshape(-1, size)

    def sort_by_time(self):

        order = np.lexsort(keys=(self.dataset.data["t"], self.dataset.data["idx"]))

        for key, value in self.dataset.data.items():
            if key in self.z_vars + self.x_vars:
                self.dataset.data[key] = value[order]


    def create_index_map(self):

        self.indices = defaultdict(list)

        for i, idx in enumerate(self.dataset.data["idx"]):
            self.indices[idx].append(i)

        self.indices = {key: np.array(value) for key, value in self.indices.items()}


    def __len__(self):
        return self.dataset.num_events

    def __getitem__(self, idx):


        if self.is_ragged:

            mask = self.indices[idx]

            x = np.column_stack([self.dataset.data[k][mask] for k in self.x_vars])
            z = np.column_stack([self.dataset.data[k][mask] for k in self.z_vars])
            
        
        else:
            # only works for energy as the only variable!
            x = self.dataset.data[self.x_vars[0]][idx]
            z = self.dataset.data[self.z_vars[0]][idx]


        c = np.array([self.dataset.meta[k][idx] for k in self.c_vars])

        z = torch.from_numpy(z.astype(np.float32))
        x = torch.from_numpy(x.astype(np.float32))
        c = torch.from_numpy(c.astype(np.float32)) 

        return x, z, c
    


    def create_collate_fn(self, name, **kwargs):
        return COLLATE_REGISTRY[name](**kwargs)     
    


    

class PointTorchDataset(BaseTorchDataset):

    """
    Point-level dataset.

    Expands event-level data into individual points. Samples are gathered
    through a custom collate function using index-based lookup.
    """

    def __init__(
            self, 
            split, 
            load_dir, 
            num_files, 
            transforms, 
            input_vars, 
            standardize_vars,
            ):
        
        super().__init__(split, load_dir, num_files, transforms, input_vars, standardize_vars)
        
        
        self.dataset.expand()      
        self.store = self._create_tensor_store(self.dataset.data)
        self.num_samples = self.store.x.shape[0]

    def create_collate_fn(self, name="index"):
        return COLLATE_REGISTRY[name](self.store)


    def _create_tensor_store(self, data):

        x = np.stack([data[k] for k in self.x_vars], axis=1).astype(np.float32)
        z = np.stack([data[k] for k in self.z_vars], axis=1).astype(np.float32)
        c = np.stack([data[k] for k in self.c_vars], axis=1).astype(np.float32)
        num_points = data["num_points"]

        return TensorStore(x, z, c, num_points)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Samples are gathered by collate_index from TensorStore.
        return idx


class TensorStore:

    """
    Container holding tensors used by PointTorchDataset.

    Attributes
    ----------
    x : torch.Tensor
        Raw features.
    z : torch.Tensor
        Transformed features.
    c : torch.Tensor
        Conditioning variables.
    num_points : torch.Tensor
        Number of points associated with each sample.
    """


    def __init__(self, x, z, c, num_points):
        self.x = torch.from_numpy(x)
        self.z = torch.from_numpy(z)
        self.c = torch.from_numpy(c)
        self.num_points = torch.from_numpy(num_points)


class ConditionalTorchDataset(torch.utils.data.Dataset):

    """
    Dataset of generated conditioning variables.

    Used for generation when sampling new detector conditions
    without requiring existing calorimeter observations.
    """

    def __init__(
        self,
        num_samples,
        standardize_vars,
        stats,
        conditions,
        sampling_specs,
        c_vars,
        seed,
        split=None
        ):

        super().__init__()

        self.dataset = create_meta(num_samples, conditions, sampling_specs, seed)
        normalize_meta(self.dataset, inverse=False)
        standardize_data(self.dataset, stats, standardize_vars)

        self.c_vars = [feature_name(var, None) for var in c_vars]
        

    def __len__(self):
        return len(self.dataset.meta["e_inc"]) 

    def __getitem__(self, idx):

        c = np.array([self.dataset.meta[k][idx] for k in self.c_vars])
        c = torch.from_numpy(c) # (4,)

        return c


  
def create_meta(num_samples: int, conditions, sampling_specs, seed=None):

    rng = np.random.default_rng(seed)
    meta = {}

    for key, spec in vars(sampling_specs).items():

        value = getattr(conditions, key, None)
        
        if value is None:
            
            if spec.distribution == "uniform":        
                samples = rng.uniform(
                    spec.min,
                    spec.max,
                    num_samples,
                    )

            elif spec.distribution == "log_uniform":
                samples = np.exp(
                    rng.uniform(
                        np.log(spec.min),
                        np.log(spec.max),
                        num_samples,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown distribution '{spec.distribution}' for '{key}'"
                )
            
            meta[key] = samples.astype(np.float32)

        else:
            meta[key] = np.full(
                num_samples,
                value,
                dtype=np.float32,
            )

    return CaloSimDataset(meta=meta)



def create_loader(batch_size: int, data_view: str, batch_mode: dict | None = None, split: str = None, **kwargs):

    dataset_cls = DATASET_REGISTRY[data_view]
    dataset = dataset_cls(split=split, **kwargs)

    collate_fn = None

    if batch_mode is not None:
        collate_fn = dataset.create_collate_fn(**vars(batch_mode))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=="train"),
        collate_fn=collate_fn,
    )

DATASET_REGISTRY = {
    "point": PointTorchDataset,
    "event": EventTorchDataset,
    "conditional": ConditionalTorchDataset,
}