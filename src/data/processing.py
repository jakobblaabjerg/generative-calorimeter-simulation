from src.utils import filter_dict
from src.io import load_data, get_file_names, save_data
from src.geometry import compute_geometric_features
from src.data.filters import apply_filter, FILTER_REGISTRY
from src.data.transforms import normalize_data, standardize_data
from src.calosim import CaloSimDataset
from src.statistics import DatasetStats
from src.reporting import FilterReport, DatasetReport

import numpy as np 
import os 

class DatasetBuilder:

    """
    Build train/validation/test datasets from raw calorimeter simulation files.

    The builder handles:

    - Loading raw or filtered datasets.
    - Applying configured filters.
    - Aggregating duplicate points.
    - Normalizing features.
    - Cleaning unused variables.
    - Splitting into train/validation/test subsets.
    - Computing dataset statistics.
    """

    def __init__(self, config): 

        """
        Initialize dataset builder.

        Parameters
        ----------
        config : object
            Configuration object containing dataset, filter,
            normalization and splitting settings.
        """
        self.config = config       
        self.dataset = None
        self.rng = np.random.default_rng(42)


    def load_raw(self, load_dir, file_name):

        """
        Load a raw dataset file and compute geometric features.

        Parameters
        ----------
        load_dir : str
            Directory containing raw input files.
        file_name : str
            Name of the file to load.
        """

        print("Loading data")
        
        self.dataset = load_data(load_dir, file_name, file_type="h5")        
        compute_geometric_features(self.dataset)
    

    def load_filtered(self, load_dir, file_name):

        """
        Load a previously filtered dataset file.

        Parameters
        ----------
        load_dir : str
            Root dataset directory.
        file_name : str
            Name of the filtered file to load.
        """

        print("Loading data")
        
        load_dir = os.path.join(load_dir, "filtered")
        self.dataset = load_data(load_dir, file_name, file_type="npz")   


    def remove_data(self, keepvars):


        """
        Remove variables that are not required for training.

        All normalized variables (ending in ``_norm``) are retained.

        Parameters
        ----------
        keepvars : list[str]
            Variables to preserve in the dataset.
        """

        print("Pruning data")
        self.dataset.data = {
            key: value for key, value in self.dataset.data.items()
            if key in keepvars or key.endswith("norm")
        }
        self.dataset.meta = {
            key: value for key, value in self.dataset.meta.items()
            if key in keepvars or key.endswith("norm")
        }


    def filter_data(self, filter_name):

        """
        Apply a configured filter to the dataset.

        Parameters
        ----------
        filter_name : str
            Name of the filter in ``FILTER_REGISTRY``.

        Returns
        -------
        FilterReport
            Summary of the filtering operation.
        """


        print(f"Filtering by {filter_name}")
        mask_fn = FILTER_REGISTRY[filter_name]
        params = vars(getattr(self.config, filter_name))
        report = apply_filter(self.dataset, mask_fn, **params)

        return report


    def aggregate_data(self):

        """
        Aggregate duplicate points.

        Points sharing the same event, particle and cell identifiers
        are merged according to predefined aggregation rules.

        Returns
        -------
        FilterReport
            Summary of the aggregation step.
        """


        print("Aggregating data")

        before = self.dataset.state()

        keys = [self.dataset.data["idx"], self.dataset.data["pid"], self.dataset.data["cid"]]
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

        for key in self.dataset.data.keys():

            agg_op = operations.get(key, "mean")
            values = self.dataset.data[key]

            if agg_op == "group":
                self.dataset.data[key] = unique[key]

            elif agg_op == "first":
                self.dataset.data[key] = values[first]

            elif agg_op == "sum":
                self.dataset.data[key] = np.bincount(inverse, weights=values)

            elif agg_op == "mean":
                self.dataset.data[key] = np.bincount(inverse, weights=values)/counts 

        return FilterReport(
            name="aggregation",
            before=before,
            after=self.dataset.state(),
        )


    def preprocess_file(self, load_dir, save_dir, file_name, file_idx):


        """
        Preprocess a single raw dataset file.

        The preprocessing pipeline consists of:

        1. Loading the raw dataset.
        2. Applying configured filters.
        3. Aggregating duplicate hits (optional).
        4. Saving the filtered dataset.
        5. Writing a preprocessing report.

        Parameters
        ----------
        load_dir : str
            Directory containing raw input files.
        save_dir : str
            Output directory.
        file_name : str
            File to process.
        file_idx : int
            File index used when saving outputs.

        Returns
        -------
        DatasetReport
            Report describing all preprocessing operations.
        """


        dataset_report = DatasetReport()
        
        self.load_raw(load_dir, file_name)
        
        for filter_name in self.config.dataset.filters: 
            filter_report = self.filter_data(filter_name)
            dataset_report.add(filter_report)

        if self.config.dataset.aggregate:
            filter_report = self.aggregate_data()
            dataset_report.add(filter_report)

        dataset_report.write(save_dir, file_idx)

        print("Saving filtered data")
        save_data(self.dataset, save_dir, stage="filtered", file_idx=file_idx)

        return dataset_report
        

    def build(self, load_dir, save_dir, debug=False):


        """
        Build train, validation and test datasets.

        For each file, the dataset is optionally preprocessed,
        normalized, cleaned, split into subsets and saved.
        Dataset statistics are accumulated from the training split.

        Parameters
        ----------
        load_dir : str
            Input dataset directory.
        save_dir : str
            Output dataset directory.
        debug : bool, default=False
            If True, load pre-filtered files instead of processing
            raw files.
        """

        stage = "filtered" if debug else "raw"
        file_names = get_file_names(root_dir=load_dir, stage=stage)
        stats = None

        for i, file_name in enumerate(file_names):

            if debug:
                file_name = file_name.split("_")[0] 
                self.load_filtered(load_dir, file_name)
            else:
                self.preprocess_file(load_dir, save_dir, file_name, file_idx=i+1)



            if self.config.dataset.normalize:
                normalize_data(self.dataset, self.config)

            self.remove_data(keepvars=self.config.dataset.keepvars)

            datasets = self.split_data()
            for split_name, dataset in datasets.items():
                save_data(dataset, save_dir, stage=split_name, file_idx=i+1)

            train_dataset = datasets["train"]
            if not stats:
                keys = train_dataset.data.keys() | train_dataset.meta.keys()
                stats = DatasetStats(keys=keys)  
            stats.update(train_dataset)

        stats.save(save_dir)


    def create_split_indices(self, ratios):

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

        idx_all = self.dataset.meta["idx"].copy()
        self.rng.shuffle(idx_all)
        num_events = self.dataset.num_events

        num_train = int(ratios[0] * num_events)
        num_val = int(ratios[1] * num_events)

        idx_train = idx_all[:num_train]
        idx_val = idx_all[num_train:num_train+num_val]
        idx_test = idx_all[num_train+num_val:]

        return {
            "train": idx_train, 
            "val": idx_val, 
            "test": idx_test
        }

    
    def split_data(self):

        """
        Split the current dataset into train, validation and test sets.

        Returns
        -------
        dict[str, CaloSimDataset]
            Dataset subsets keyed by split name.
        """

        print("Splitting data")

        splits = self.create_split_indices(self.config.dataset.split_ratios)
        datasets = {}

        for split_name, idxs in splits.items():

            mask = np.isin(self.dataset.data["idx"], idxs)
            data_filtered = filter_dict(self.dataset.data, mask)
            mask = np.isin(self.dataset.meta["idx"], idxs)
            meta_filtered = filter_dict(self.dataset.meta, mask)
            dataset_split = CaloSimDataset(data_filtered, meta_filtered)
            dataset_split.reindex()
            datasets[split_name] = dataset_split 

        return datasets


def postprocess(dataset, stats, cfg_filters, standardize_vars):

    standardize_data(dataset, stats, standardize_vars, inverse=True)
    normalize_data(dataset, cfg_filters, inverse=True)
    compute_geometric_features(dataset, inverse=True)







# def get_file_idx(file_name):
#     return int(''.join(c for c in os.path.splitext(file_name)[0] if c.isdigit()))

    
#     def inverse_transform(self, output, stats, standardize_vars):

#         standardize_data(output["data"], stats, standardize_vars, inverse=True)
#         standardize_data(output["meta"], stats, standardize_vars, inverse=True)

#         normalize_data(output["data"], output["meta"], self.cfg, inverse=True)
#         compute_static_features(output["data"], output["meta"], inverse=True)

        


# # utils

# def safe_underscore(name):
#     return f"_{name}" if name else ""

# def create_meta(num_samples, phi=None, theta=None, e_inc=None, normalize=True, seed=None):

#     rng = np.random.default_rng(seed)
#     meta = {}
    
#     specs = {
#         "phi": (-np.pi, np.pi, phi),
#         "theta": (6 * np.pi / 180, 174 * np.pi / 180, theta),
#         "e_inc": (0.1, 100, e_inc)
#     }
    
#     for key, (min_val, max_val, val) in specs.items():
#         if val is None:
#             u = rng.uniform(size=num_samples).astype(np.float32)
#             meta[key] = u * (max_val - min_val) + min_val
#         else:
#             meta[key] = np.repeat(val, num_samples).astype(np.float32)
    
#     if normalize:
#         normalize_meta(meta, inverse=False)

#     return meta


# def concat_dict(dict1, dict2):

#     for k, v in dict2.items():
        
#         if k in dict1:
           
#             if k == "eid":
#                 eids = dict1[k]
#                 offset = len(np.unique(eids))
#                 v = v + offset
#             dict1[k] = np.concatenate((dict1[k], v))
        
#         else:
#             dict1[k] = v.copy()  
    
#     return dict1





# def filter_by_eid(data, meta, eid):

#     # maybe original?

#     mask = data["eid"] == eid
#     data_filtered = filter_dict(data, mask)   
    
#     mask = meta["eid"] == eid
#     meta_filtered = filter_dict(meta, mask)

#     return data_filtered, meta_filtered







