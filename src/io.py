import os
import h5py
import numpy as np 
import json 

from src.calosim import CaloSimDataset


def get_file_names(root_dir: str, stage: str) -> list:
    """
    Return dataset files for a given processing stage.

    Parameters
    ----------
    root_dir : str
        Dataset root directory.

    stage : str
        Dataset stage:
        {"raw", "filtered", "train", "val", "test"}.

    Returns
    -------
    list[str]
        Sorted file names.
    """

    if stage == "raw":
        suffix = ".h5"
        search_dir = root_dir

    else:
        suffix = "data.npz"
        search_dir = os.path.join(root_dir, stage)

    return sorted(
        (f for f in os.listdir(search_dir) if f.endswith(suffix)),
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )








#!!!!!!!!!!

def load_split(split: str, load_dir: str, num_files: int) -> CaloSimDataset:

    """
    Load a dataset split from disk and merge multiple files into a single CaloSimDataset.

    Parameters
    ----------
    split : str
        Dataset split to load (e.g. "train", "val", "test").
    load_dir : str
        Root directory containing the saved dataset files.
    num_files : int
        Number of files to load from the split directory.

    Returns
    -------
    CaloSimDataset

    Notes
    -----
    - Event IDs are shifted inside `merge()` to ensure global uniqueness.
    """

    dataset = CaloSimDataset()
    files = get_file_names(root_dir=load_dir, stage=split)

    for f in files[:num_files]:

        file_name = f.rsplit("_", 1)[0]
        file_path = os.path.join(load_dir, split, file_name)
        other = CaloSimDataset.from_npz(file_path)
        dataset.append(other)

    stats = load_stats(load_dir)

    return dataset, stats


def save_data(dataset: CaloSimDataset, save_dir: str, stage: str, file_idx: int) -> None:

    """
    Save a CaloSimDataset split to disk as separate NPZ files for data and meta.

    Parameters
    ----------
    dataset : CaloSimDataset
    save_dir : str
    stage : str
        Dataset split name (e.g. "filtered", "train", "val", "test"). Determines subfolder name.
    file_idx : int
        File index used for naming saved files (e.g. file{file_idx}_data.npz).
    """

    save_dir = os.path.join(save_dir, stage)
    os.makedirs(save_dir, exist_ok=True)    
    file_path = os.path.join(save_dir, f"file{file_idx}")
    dataset.to_npz(file_path)


def load_stats(load_dir):

    """
    Load dataset statistics from a JSON file.

    Parameters
    ----------
    load_dir : str
        Directory containing the `stats.json` file.

    Returns
    -------
    dict
        Dictionary containing saved statistics (e.g. mean, std, count per feature).
    """
    
    file_path = os.path.join(load_dir, "stats.json")
    
    with open(file_path, "r") as f:
        stats = json.load(f)
    
    return stats


def load_data(file_dir: str, file_name: str, file_type: str) -> CaloSimDataset:

    """
    Load a raw HDF5 calorimeter simulation file and convert it
    into a structured CaloSimDataset.

    Parameters
    ----------
    file_dir : str
        Directory containing the HDF5 file.
    file_name : str
        Name of the file to load.
    file_type : str
    
    Returns
    -------
    CaloSimDataset
        Structured dataset containing:
        - data: step-level features
        - meta: event-level features
    """

    file_path = os.path.join(file_dir, file_name)
    
    if file_type == "h5":
        dataset = CaloSimDataset.from_h5(file_path)

    elif file_type == "npz":
        dataset = CaloSimDataset.from_npz(file_path)

    return dataset