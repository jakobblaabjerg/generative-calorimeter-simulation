import numpy as np 

def create_voxel_grid(binning):
    return np.array([
        [z, a, r]
        for z in range(binning.z)
        for a in range(binning.a)
        for r in range(binning.r)
    ])


def clip_voxels(voxels, binning):

    mins = np.array([0, 0, 0])
    maxs = np.array([binning.z - 1, binning.a - 1, binning.r - 1])
    return np.clip(voxels, mins, maxs)


def cartesian_to_cylindrical(x, y):

    a = (np.atan2(y, x) + np.pi) / (2 * np.pi)
    r = np.sqrt(x**2 + y**2)  

    return a, r


def compute_voxel_id(dataset, binning):

    num_voxels = binning.r * binning.a * binning.z

    return (
        dataset.data["idx"] * num_voxels
        + dataset.data["z"] * (binning.a * binning.r)
        + dataset.data["a"] * binning.r
        + dataset.data["r"]
    )
