import numpy as np
from src.geometry import compute_detector_distances

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


def normalize_meta(dataset, inverse):

    if not inverse:
        dataset.meta["e_inc_norm"] = np.log1p(dataset.meta["e_inc"])
        theta = dataset.meta["theta"]
        phi = dataset.meta["phi"]
        dataset.meta["dir_x_norm"] = np.sin(theta) * np.cos(phi)
        dataset.meta["dir_y_norm"] = np.sin(theta) * np.sin(phi)
        dataset.meta["dir_z_norm"] = np.cos(theta)
    
    else:
        dataset.meta["e_inc"] = np.expm1(dataset.meta["e_inc_norm"])
        dir_x_norm = dataset.meta["dir_x_norm"]
        dir_y_norm = dataset.meta["dir_y_norm"]
        dir_z_norm = dataset.meta["dir_z_norm"]
        dataset.meta["theta"] = np.arccos(dir_z_norm)
        dataset.meta["phi"] = np.arctan2(dir_y_norm, dir_x_norm)


def normalize_data(dataset, config, inverse=False):

    print("Normalizing data")

    normalize_meta(dataset, inverse)
    _, counts = np.unique(dataset.data["eid"], return_counts=True)
    scale_xy = config.retention.box_size / 2
    scale_e = config.energy.threshold
    r_max = np.sqrt(2 * scale_xy**2)

    if not inverse:

        # normalize x/y position to [-1, 1]
        dataset.data["x_hat_norm"] = dataset.data["x_hat"] / scale_xy
        dataset.data["y_hat_norm"] = dataset.data["y_hat"] / scale_xy

        # normalize radial distance to [0, 1]
        r_hat = dataset.data["r_hat"]
        dataset.data["r_hat_norm"] = r_hat / r_max

        # normalize z distance to [0, ~1.2]
        max_dist = np.repeat(dataset.meta["exit_dist"] - dataset.meta["entry_dist"], counts)
        dataset.data["z_hat_norm"] = dataset.data["z_hat"] / max_dist
        dataset.data["z_hat_log_norm"] = np.log((dataset.data["z_hat_norm"]) + 1e-6)
        dataset.data["z_hat_sqrt_norm"] = np.sqrt(dataset.data["z_hat_norm"])

        # log/sqrt transform of scaled energy
        e_scaled = dataset.data["e"] / scale_e
        dataset.data["e_log_norm"] = np.log(e_scaled)
        dataset.data["e_sqrt_norm"] = np.sqrt(e_scaled)

    else:
        compute_detector_distances(dataset.meta)

        # denormalize x/y position to [-scale, scale]
        if "x_hat_norm" in dataset.data:
            dataset.data["x_hat"] = dataset.data["x_hat_norm"] * scale_xy
            dataset.data["y_hat"] = dataset.data["y_hat_norm"] * scale_xy

        # denormalize radial distance to [0, r_max]
        if "r_hat_norm" in dataset.data:
            dataset.data["r_hat"] = dataset.data["r_hat_norm"] * r_max

        # denormalize z distance to [0, max_distance]
        max_dist = np.repeat(dataset.meta["exit_dist"] - dataset.meta["entry_dist"], counts)

        if "z_hat_norm" in dataset.data:
            dataset.data["z_hat"] = dataset.data["z_hat_norm"] * max_dist

        elif "z_hat_log_norm" in dataset.data:
            dataset.data["z_hat"] = (np.exp(dataset.data["z_hat_log_norm"]) -1e-6) * max_dist

        elif "z_hat_sqrt_norm" in dataset.data:
            z_hat_sqrt_norm = np.clip(dataset.data["z_hat_sqrt_norm"], -0.05, None)
            dataset.data["z_hat"] = (z_hat_sqrt_norm**2) * max_dist

        # inverse transform energy
        if "e_log_norm" in dataset.data:
            e_scaled = np.exp(dataset.data["e_log_norm"])
        elif "e_sqrt_norm" in dataset.data:
            e_scaled = dataset.data["e_sqrt_norm"] ** 2
        dataset.data["e"] = e_scaled * scale_e