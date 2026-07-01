import os
import json 

from src.calosim import CaloSimDataset
from src.step2point import Step2Point
from src.calochallenge import CaloChallenge


def get_file_name(file_path):

    file_name = os.path.basename(file_path)
    return os.path.splitext(file_name)[0]


def get_file_paths(root_dir: str, stage: str) -> list:
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
        File names.
    """

    if stage == "raw":
        valid_suffixes = (".h5", ".hdf5")
        search_dir = root_dir

    else:
        valid_suffixes = ("data.npz",)
        search_dir = os.path.join(root_dir, stage)


    paths = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith(valid_suffixes)
    ]
    return [x.removesuffix("_data.npz") for x in paths]


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
    files = get_file_paths(root_dir=load_dir, stage=split)
    
    for path in files[:num_files]:
        
        other = CaloSimDataset.from_npz(path)
        dataset.append(other)

    stats = load_stats(load_dir)

    return dataset, stats


def save_data(dataset: CaloSimDataset, save_dir: str, stage: str, file_name: str) -> None:

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

    print("Saving data")

    save_dir = os.path.join(save_dir, stage)
    os.makedirs(save_dir, exist_ok=True)    
    file_path = os.path.join(save_dir, file_name)
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


def load_raw(file_path: str, dataset_name: str) -> CaloSimDataset:

    """
    Load a raw HDF5 calorimeter simulation file and convert it
    into a structured CaloSimDataset.
 
    Returns
    -------
    CaloSimDataset
        Structured dataset containing:
        - data: step-level features
        - meta: event-level features
    """

    print("Loading raw data")

    dataset_classes = {
        "calochallenge": CaloChallenge,
        "step2point": Step2Point,
    }

    try:
        dataset_cls = dataset_classes[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'.")

    return dataset_cls.from_h5(file_path)
