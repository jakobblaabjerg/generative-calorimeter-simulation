import numpy as np
from dataclasses import dataclass, field
from src.utils import filter_dict

@dataclass 
class CaloSimDataset: 

    data: dict = field(default_factory=dict)           
    meta: dict = field(default_factory=dict)

    @property 
    def num_events(self):
        """
        Number of events in the dataset.

        Returns
        -------
        int
            Number of unique event entries stored in `meta`.
        """
        return len(self.meta.get("idx", []))
    

    @property
    def num_steps(self):
        return len(self.data.get("idx", []))


    @property
    def unique_events(self):
        return self.meta.get("eid", [])
    

    def state(self):

        return {
            "num_steps": self.num_steps,
            "num_events": self.num_events,
            "unique_events": self.unique_events
            }


    def expand(self):

        _, num_points = np.unique(self.data["idx"], return_counts=True)
        self.meta["num_points"] = num_points

        for key in self.meta.keys():
            if key not in self.data.keys():            
                self.data[key] = np.repeat(self.meta[key], repeats=self.meta["num_points"])


    def append(self, other) -> None:

        """
        Merge another dataset into this dataset.

        Event indices from the incoming dataset are shifted to ensure
        uniqueness before concatenation.

        Parameters
        ----------
        other : CaloSimDataset
            Dataset to append.
        """

        if not self.data:
            self.data = {key: value.copy() for key, value in other.data.items()}
            self.meta = {key: value.copy() for key, value in other.meta.items()}

        else:
            offset = self.meta["idx"].max()+1
            other = other.copy()
            other.data["idx"] += offset 
            other.meta["idx"] += offset 

            for key in self.data:
                if key not in other.data:
                        raise KeyError(f"Missing key in other.data: {key}")
                else:
                    self.data[key] = np.concatenate([self.data[key], other.data[key]])
            
            for key in self.meta:
                if key not in other.meta:
                        raise KeyError(f"Missing key in other.meta: {key}")
                else:
                    self.meta[key] = np.concatenate([self.meta[key], other.meta[key]])


    def reindex(self) -> None:
        """
        Reindex event identifiers to a contiguous range.

        Existing values in `idx` are replaced with integers
        in the range `[0, num_events)` while preserving the
        correspondence between `data` and `meta`.
        """

        # assert they have same unique idxs

        idx_old = np.unique(self.meta["idx"])
        idx_new = {old: new for new, old in enumerate(idx_old)}
    
        self.meta["idx"] = np.array([idx_new[eid] for eid in self.meta["idx"]])
        self.data["idx"] = np.array([idx_new[eid] for eid in self.data["idx"]])


    def sync(self) -> None:
        """
        Synchronize event indices between `data` and `meta`.

        Only event indices present in both dictionaries are retained.
        Entries with unmatched indices are removed.
        """

        shared = np.intersect1d(self.data["idx"], self.meta["idx"])
        self.data = filter_dict(self.data, np.isin(self.data["idx"], shared))
        self.meta = filter_dict(self.meta, np.isin(self.meta["idx"], shared))


    @classmethod
    def from_npz(cls, file_path):
        """
        Load a dataset from NPZ files.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        CaloSimDataset
        """

        data = {k: v for k, v in np.load(f"{file_path}_data.npz").items()}
        meta = {k: v for k, v in np.load(f"{file_path}_meta.npz").items()}

        return cls(data=data, meta=meta)


    def to_npz(self, file_path):
        """
        Save the dataset as NPZ files.

        Parameters
        ----------
        file_path : str
        """

        np.savez(f"{file_path}_data.npz", **self.data)
        np.savez(f"{file_path}_meta.npz", **self.meta)

    
    def copy(self):
        return CaloSimDataset(
            data={key: value.copy() for key, value in self.data.items()},
            meta={key: value.copy() for key, value in self.meta.items()},
        )
    
