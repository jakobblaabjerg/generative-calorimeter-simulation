from src.utils import filter_dict
from src.io import load_raw, get_file_paths, save_data, get_file_name
from src.geometry import compute_geometric_features
from src.filters import apply_filter, FILTER_REGISTRY
from src.data.transforms import normalize_data, standardize_data
from src.calosim import CaloSimDataset
from src.statistics import DatasetStats
from src.reporting import FilterReport, DatasetReport

import numpy as np 


def filter_data(dataset, filter_name, params):

    print(f"Filtering by {filter_name}")
    mask_fn = FILTER_REGISTRY[filter_name]
    report = apply_filter(dataset, mask_fn, **vars(params))

    return report

def aggregate_data(dataset):

    print("Aggregating data")

    before = dataset.state()

    keys = [dataset.data["idx"], dataset.data["pid"], dataset.data["cid"]]
    keys = np.rec.fromarrays(keys, names="idx, pid, cid")

    unique, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)

    operations = {
        "idx": "group",
        "pid": "group",
        "cid": "group",
        "eid": "first",
        "subdet": "first",
        "e": "sum",
    }

    for key in dataset.data.keys():

        agg_op = operations.get(key, "mean")
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

def remove_unused_data(dataset, keepvars):

    print("Removing unsused data")
    
    dataset.data = {
        key: value for key, value in dataset.data.items()
        if key in keepvars or key.endswith("norm")
    }
    
    dataset.meta = {
        key: value for key, value in dataset.meta.items()
        if key in keepvars or key.endswith("norm")
    }

def preprocess_data(dataset, config, save_dir, file_name, debug):

    if not debug:

        dataset_report = DatasetReport()

        for filter_name in config.filters:

            if filter_name == "aggregate":
                report = aggregate_data(dataset)

            else:
                params = getattr(config.filter_params, filter_name)
                report = filter_data(dataset, filter_name, params)
            
            dataset_report.add(report)
        
        save_data(dataset, save_dir, "filtered", file_name) # create checkpoint
        dataset_report.write(save_dir, file_name)

    if config.normalize:
        normalize_data(dataset, config)

    remove_unused_data(dataset, config.keepvars)

def postprocess_data(dataset, stats, config, standardize_vars):
        standardize_data(dataset, stats, standardize_vars, inverse=True)
        normalize_data(dataset, config, inverse=True)
        compute_geometric_features(dataset, inverse=True)

def create_splits(dataset, ratios, seed=42):

    """
    Create random train, validation and test event splits.

    Parameters
    ----------
    ratios : tuple[float, float, float]
        Split fractions for train, validation and test sets.

    Returns
    -------
    dict[str, ndarray]
        Mapping from split name to event indices.
    """

    idxs = dataset.meta["idx"].copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)

    num_events = dataset.num_events
    num_train = int(ratios[0] * num_events)
    num_val = int(ratios[1] * num_events)

    return {
        "train": idxs[:num_train], 
        "val": idxs[num_train:num_train+num_val], 
        "test": idxs[num_train+num_val:]
    }
    
def split_data(dataset, config, stats, save_dir, file_name):

        print("Splitting data")

        if stats is None:
            keys = dataset.data.keys() | dataset.meta.keys()
            stats = DatasetStats(keys=keys)  

        splits = create_splits(dataset, config.split_ratios)

        for split_name, idxs in splits.items():

            data_mask = np.isin(dataset.data["idx"], idxs)
            data_filtered = filter_dict(dataset.data, data_mask)
            
            meta_mask = np.isin(dataset.meta["idx"], idxs)
            meta_filtered = filter_dict(dataset.meta, meta_mask)
            
            dataset_split = CaloSimDataset(data_filtered, meta_filtered)
            dataset_split.reindex()
            
            save_data(dataset_split, save_dir, split_name, file_name)

            if split_name == "train":
                stats.update(dataset_split)
        
        return stats

def build_dataset(load_dir, save_dir, config, debug=False):

    stage = "raw" if debug == False else "filtered"
    files = get_file_paths(root_dir=load_dir, stage=stage)
    stats = None

    for path in files:

        f_name = get_file_name(path)

        if not debug:
            dataset = load_raw(path, config.name)
        else:
            dataset = CaloSimDataset.from_npz(path)
   
        preprocess_data(dataset, config, save_dir, f_name, debug)
        stats = split_data(dataset, config, stats, save_dir, f_name)

    stats.save(save_dir)