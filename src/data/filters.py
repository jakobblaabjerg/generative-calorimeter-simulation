from src.geometry import compute_centroids, compute_misalignment, compute_retention
from src.reporting import FilterReport
from src.utils import filter_dict
import numpy as np






def mask_time(dataset, threshold):
    level = "data"
    return dataset.data["t"] <= threshold, level


def mask_energy(dataset, threshold):
    level = "data"
    return dataset.data["e"] >= threshold, level


def mask_misalignment(dataset, threshold, method):
    level = "meta"
    compute_centroids(dataset) 
    compute_misalignment(dataset)
    return dataset.meta[f"{method}_misalign"] < threshold, level


def mask_subdet(dataset, subdets):
    level = "data"
    return np.isin(dataset.data["subdet"], subdets), level


def mask_z_hat(dataset, threshold):
    level = "data"
    return dataset.data["z_hat"] >= threshold, level


def mask_rentention(dataset, threshold, box_size):
    level = "meta"
    compute_retention(dataset, box_size=box_size)
    return dataset.meta["retention_pct"] >= threshold, level


def mask_eid(dataset, eid):
    level = "meta"
    return dataset.meta["eid"] == eid, level
    

def mask_xy_box(dataset, box_size):
    level = "data"
    x_hat = np.abs(dataset.data["x_hat"])
    y_hat = np.abs(dataset.data["y_hat"])
    return (x_hat <= box_size/2) & (y_hat <= box_size/2), level


def apply_filter(dataset, mask_fn, **params):

    before = dataset.state()
    mask, level = mask_fn(dataset, **params)

    if level == "data":
        dataset.data = filter_dict(dataset.data, mask)
    elif level == "meta":
        dataset.meta = filter_dict(dataset.meta, mask)

    dataset.sync()
    dataset.reindex()

    return FilterReport(
        name=mask_fn.__name__.removeprefix("mask_"),
        before=before,
        after=dataset.state(),
        params=params,
    )


FILTER_REGISTRY = {
    "time": mask_time,
    "energy": mask_energy,
    "misalignment": mask_misalignment,
    "subdet": mask_subdet,
    "z_hat": mask_z_hat,
    "retention": mask_rentention,
    "eid": mask_eid,
    "xy_box": mask_xy_box,
}