from src.utils import filter_dict
from src.io import load_raw, get_file_paths, save_data, get_file_name
from src.geometry import compute_geometric_features

from src.data.transforms import normalize_data, standardize_data, voxel_to_point, point_to_voxel
from src.calosim import CaloSimDataset
from src.statistics import DatasetStats
from src.reporting import DatasetReport
from src.operations import aggregate_data, filter_data, remove_unused_data

import numpy as np 

def preprocess_data(dataset, config, save_dir, file_name, debug):

    if not debug:

        dataset_report = DatasetReport()

        for filter_name in config.filters:

            params = getattr(config.filter_params, filter_name)

            if filter_name == "aggregate":
                report = aggregate_data(dataset, **vars(params))

            else:
                report = filter_data(dataset, filter_name, params)
            
            dataset_report.add(report)
        
        save_data(dataset, save_dir, "filtered", file_name) # create checkpoint
        dataset_report.write(save_dir, file_name)

    if config.normalize:
        normalize_data(dataset, config)

    remove_unused_data(dataset, config.keepvars, "norm")

def postprocess_data(dataset, stats, config, standardize_vars, convert_to_voxel):
    
    standardize_data(dataset, stats, standardize_vars, inverse=True)
    normalize_data(dataset, config, inverse=True)
    
    if convert_to_voxel:
        point_to_voxel(dataset, config.binning)

    if "z_hat_norm" in dataset.data:
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
   
        if config.view == "point" and dataset.view == "voxel":
            voxel_to_point(dataset, config.binning)

        preprocess_data(dataset, config, save_dir, f_name, debug)
        stats = split_data(dataset, config, stats, save_dir, f_name)

    stats.save(save_dir)