from src.filters import apply_filter, FILTER_REGISTRY
from src.reporting import FilterReport
from src.calosim import CaloSimDataset

import numpy as np

def filter_data(dataset, filter_name, params):

    print(f"Filtering by {filter_name}")
    mask_fn = FILTER_REGISTRY[filter_name]
    report = apply_filter(dataset, mask_fn, **vars(params))

    return report

def aggregate_data(dataset: CaloSimDataset, keys: list, operations: dict, default: str):

    print("Aggregating data")

    before = dataset.state()

    key_arrays = [dataset.data[key] for key in keys]
    names = (", ").join(keys)
    group_keys = np.rec.fromarrays(key_arrays, names=names)

    unique, first, inverse, counts = np.unique(group_keys, return_index=True, return_inverse=True, return_counts=True)

    for key in dataset.data.keys():

        agg_op = "group" if key in keys else getattr(operations, key, default)
        values = dataset.data[key]

        if agg_op == "group":
            dataset.data[key] = unique[key]

        elif agg_op == "first":
            dataset.data[key] = values[first]

        elif agg_op == "sum":
            dataset.data[key] = np.bincount(inverse, weights=values)

        elif agg_op == "mean":
            dataset.data[key] = np.bincount(inverse, weights=values)/counts 

    return FilterReport(
        name="aggregation",
        before=before,
        after=dataset.state(),
    )

def remove_unused_data(dataset, keepvars, keep_suffix=None):

    print("Removing unsused data")
    
    dataset.data = {
        key: value for key, value in dataset.data.items()
        if key in keepvars or (keep_suffix and key.endswith(keep_suffix))
    }
    
    dataset.meta = {
        key: value for key, value in dataset.meta.items()
        if key in keepvars or (keep_suffix and key.endswith(keep_suffix))
    }