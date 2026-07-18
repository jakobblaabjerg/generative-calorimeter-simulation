import numpy as np
from src.geometry import compute_detector_distances
from src.voxel import create_voxel_grid, clip_voxels, compute_voxel_id, cartesian_to_cylindrical
from src.operations import remove_unused_data, aggregate_data
from types import SimpleNamespace

def standardize_data(dataset, stats, standardize_vars, inverse=False):

    for var in standardize_vars:

        for container in [dataset.data, dataset.meta]:
            if container is None or var not in container:
                continue

            mean = stats[var]["mean"]
            std = stats[var]["std"] 

            if inverse:
                container[var] = container[var] * std + mean
            else:
                container[var] = (container[var] - mean) / std


def dequantize(voxels, binning, inverse=False):

    bins = np.array([binning.z, binning.a, binning.r])

    if not inverse:
        voxels = voxels.astype(np.float32)
        voxels += np.random.rand(*voxels.shape)
        voxels /= bins
        
    else:
        voxels = np.floor(voxels * bins)
        voxels = voxels.astype(np.int32)

    return voxels 



def voxel_to_point(dataset, binning):

    num_voxels = binning.r * binning.a * binning.z
    
    events = dataset.data["e"].copy().reshape(-1, num_voxels) # shape(num_events, num_voxels)
    nonzero = np.argwhere(events>0)

    vid = nonzero[:,1] # voxel id
    dataset.data["e"] = events[nonzero[:,0], nonzero[:,1]]
    dataset.data["idx"] = nonzero[:,0]

    voxel_grid = create_voxel_grid(binning)
    voxels = voxel_grid[vid]
    voxels = dequantize(voxels, binning) # [z, a, r] is now [0, 1]

    a = voxels[:, 1] * 2 * np.pi - np.pi # [-pi, pi]
    r = voxels[:, 2]
    dataset.data["x_hat_norm"] = np.cos(a) * r # [-1 ,1]
    dataset.data["y_hat_norm"] = np.sin(a) * r # [-1, 1]
    dataset.data["z_hat_norm"] = voxels[:, 0]
    dataset.data["r_hat_norm"] = r



def point_to_voxel(dataset, binning):

    print("Converting point into voxel")

    z = dataset.data["z_hat_norm"]

    if {"x_hat_norm", "y_hat_norm"} <= dataset.data.keys():
        x = dataset.data["x_hat_norm"]
        y = dataset.data["y_hat_norm"]
        a, r = cartesian_to_cylindrical(x, y)

    elif {"a_hat_norm", "r_hat_norm"} <= dataset.data.keys():
        a = dataset.data["a_hat_norm"]
        r = dataset.data["r_hat_norm"]
    
    else:
        raise ValueError(
            "point_to_voxel requires either "
            "('x_hat_norm', 'y_hat_norm') or "
            "('a_hat_norm', 'r_hat_norm')."
        )

    voxels_sparse = np.stack([z, a, r], axis=-1)

    voxels_sparse = dequantize(voxels_sparse, binning, inverse=True)
    voxels_sparse = clip_voxels(voxels_sparse, binning)

    # aggegrate data 
    dataset.data["z"] = voxels_sparse[:, 0]
    dataset.data["a"] = voxels_sparse[:, 1]
    dataset.data["r"] = voxels_sparse[:, 2]

    keepvars = ["idx", "z", "a", "r", "e", "e_inc"]
    remove_unused_data(dataset, keepvars)

    keys = ["idx", "z", "a", "r"]
    operations = {"e": "sum"}
    operations = SimpleNamespace(**operations)
    default = "group"
    aggregate_data(dataset, keys, operations, default) 

    vid = compute_voxel_id(dataset, binning)
    num_events = dataset.num_events
    num_voxels = binning.r * binning.a * binning.z
    e_dense = np.zeros(num_events*num_voxels, dtype=np.float32)
    e_dense[vid] = dataset.data["e"]
    dataset.data["e"] = e_dense
    dataset.data["idx"] = np.repeat(np.arange(num_events), num_voxels)

    keepvars = ["idx", "e", "e_inc"]
    remove_unused_data(dataset, keepvars)



class Transform:
  
    requires = {
        "meta": set(),
        "data": set()
        }
    
    produces = {
        "meta": set(),
        "data": set()
    }

    def can_apply(self, dataset, inverse=False):

        requirements = self.produces if inverse else self.requires

        for container_name, required_keys in requirements.items():
            container = getattr(dataset, container_name)
            if not required_keys <= container.keys():
                return False
        return True

class IncidentDirectionTransform(Transform):

    def __init__(self):
    
        self.requires = {
            "meta": {"theta", "phi"}
            }                
        self.produces = {
            "meta": {"dir_x_norm", "dir_y_norm", "dir_z_norm"}
            }

    def forward(self, dataset):

        if not self.can_apply(dataset):
            return

        theta = dataset.meta["theta"]
        phi = dataset.meta["phi"]
        
        dataset.meta["dir_x_norm"] = np.sin(theta) * np.cos(phi)
        dataset.meta["dir_y_norm"] = np.sin(theta) * np.sin(phi)
        dataset.meta["dir_z_norm"] = np.cos(theta)


    def inverse(self, dataset):

        if not self.can_apply(dataset, inverse=True):
            return

        dir_x_norm = dataset.meta["dir_x_norm"]
        dir_y_norm = dataset.meta["dir_y_norm"]
        dir_z_norm = dataset.meta["dir_z_norm"]
        
        dataset.meta["theta"] = np.arccos(dir_z_norm)
        dataset.meta["phi"] = np.arctan2(dir_y_norm, dir_x_norm)


class IncidentEnergyTransform(Transform):

    def __init__(self):

        self.requires = {
            "meta": {"e_inc"}
            }
        self.produces = {
            "meta": {"e_inc_norm"}
            }

    def forward(self, dataset):

        if not self.can_apply(dataset):
            return
        
        dataset.meta["e_inc_norm"] = np.log1p(dataset.meta["e_inc"])


    def inverse(self, dataset):

        if not self.can_apply(dataset, inverse=True):
            return
        
        dataset.meta["e_inc"] = np.expm1(dataset.meta["e_inc_norm"])

class XYPositionTransform(Transform):


    def __init__(self, scale_xy):

        self.requires = {
            "data": {"x_hat", "y_hat"}
            }
        
        self.produces = {
            "data": {"x_hat_norm", "y_hat_norm"}
            }

        self.scale_xy = scale_xy

    def forward(self, dataset):

        if not self.can_apply(dataset):
            return

        # normalize x/y position to [-1, 1]
        dataset.data["x_hat_norm"] = dataset.data["x_hat"] / self.scale_xy
        dataset.data["y_hat_norm"] = dataset.data["y_hat"] / self.scale_xy
    
    def inverse(self, dataset):

        if not self.can_apply(dataset, inverse=True):
            return

        # x/y position to [-scale, scale]
        dataset.data["x_hat"] = dataset.data["x_hat_norm"] * self.scale_xy
        dataset.data["y_hat"] = dataset.data["y_hat_norm"] * self.scale_xy


class ZPositionTransform(Transform):


    def __init__(self):

        self.requires = {
            "data": {"z_hat"},
            "meta": {"exit_dist", "entry_dist"}
        }

    def forward(self, dataset):

        if not self.can_apply(dataset):
            return

        # z distance to [0, ~1.2]
        _, counts = np.unique(dataset.data["idx"], return_counts=True)
        max_dist = np.repeat(dataset.meta["exit_dist"] - dataset.meta["entry_dist"], counts)
        z_hat_linear_norm = dataset.data["z_hat"] / max_dist
        dataset.data["z_hat_linear_norm"] = z_hat_linear_norm
        dataset.data["z_hat_log_norm"] = np.log(z_hat_linear_norm + 1e-6)
        dataset.data["z_hat_sqrt_norm"] = np.sqrt(z_hat_linear_norm)


    def inverse(self, dataset):

        compute_detector_distances(dataset)
        _, counts = np.unique(dataset.data["idx"], return_counts=True)
        max_dist = np.repeat(dataset.meta["exit_dist"] - dataset.meta["entry_dist"], counts)

        if "z_hat_linear_norm" in dataset.data:
            dataset.data["z_hat"] = dataset.data["z_hat_linear_norm"] * max_dist
        elif "z_hat_log_norm" in dataset.data:
            dataset.data["z_hat"] = (np.exp(dataset.data["z_hat_log_norm"]) -1e-6) * max_dist
        elif "z_hat_sqrt_norm" in dataset.data:
            z_hat_sqrt_norm = np.clip(dataset.data["z_hat_sqrt_norm"], -0.05, None)
            dataset.data["z_hat"] = (z_hat_sqrt_norm**2) * max_dist
        else:
            raise ValueError("No normalized z position found.")

class RadialPositionTransform(Transform):

    def __init__(self, scale_xy):

        self.requires = {
            "data": {"r_hat"}
        }

        self.produces = {
            "data": {"r_hat_norm"}
        }

        self.r_max = np.sqrt(2 * scale_xy**2)

    def forward(self, dataset): 

        if not self.can_apply(dataset):
            return
            
        # radial distance to [0, 1]
        r_hat = dataset.data["r_hat"]
        dataset.data["r_hat_norm"] = r_hat / self.r_max

    def inverse(self, dataset):

        if not self.can_apply(dataset, inverse=True):
            return

        # radial distance to [0, r_max]
        dataset.data["r_hat"] = dataset.data["r_hat_norm"] * self.r_max


class DepositedEnergyTransform(Transform):
    
    def __init__(self, scale_e=1):
        
        self.scale_e = scale_e

        self.requires = {
            "data": {"e"}
        }

    def forward(self, dataset):

        if not self.can_apply(dataset):
            return

        e_scaled = dataset.data["e"] / self.scale_e
        dataset.data["e_log_norm"] = np.log(e_scaled)
        dataset.data["e_sqrt_norm"] = np.sqrt(e_scaled)

    def inverse(self, dataset):

        if "e_log_norm" in dataset.data:
            e_scaled = np.exp(dataset.data["e_log_norm"])
        elif "e_sqrt_norm" in dataset.data:
            e_scaled = dataset.data["e_sqrt_norm"] ** 2
        else:
            raise ValueError("No normalized deposited energy found.") 
        
        dataset.data["e"] = e_scaled * self.scale_e
        

META_TRANSFORMS = {
    "direction": IncidentDirectionTransform,
    "e_inc": IncidentEnergyTransform,
}


DATA_TRANSFORMS = {
    "xy_position": XYPositionTransform,
    "z_position": ZPositionTransform,
    "r_position": RadialPositionTransform,
    "e_deposit": DepositedEnergyTransform,
}


def apply_transforms(dataset, transforms, inverse=False):

    method = "inverse" if inverse else "forward"
    for transform in transforms:
        getattr(transform, method)(dataset)


def normalize_meta(dataset, transforms=None, inverse=False):

    if transforms is None:
        transforms = ["e_inc", "direction"]

    transforms_objects = [
        META_TRANSFORMS[name]()
        for name in transforms
    ]
    apply_transforms(dataset, transforms_objects, inverse)


def normalize_data(dataset, config, inverse=False):

    print("Normalizing data")

    normalize_meta(dataset, config.transforms.meta, inverse)

    transform_objects = []
    transforms = config.transforms.data

    for name in transforms:

        if name == "xy_position":
            scale_xy = config.filter_params.retention.box_size / 2
            transform_objects.append(XYPositionTransform(scale_xy))

        elif name == "r_position":
            scale_xy = config.filter_params.retention.box_size / 2
            transform_objects.append(RadialPositionTransform(scale_xy))

        elif name == "e_deposit":
            try:
                scale_e = config.filter_params.energy.threshold
            except AttributeError:
                scale_e = 1

            transform_objects.append(DepositedEnergyTransform(scale_e=scale_e))

        else:
            transform_objects.append(
                DATA_TRANSFORMS[name]()
            )

    apply_transforms(dataset, transform_objects, inverse)