import numpy as np
import os
import json

class DatasetStats:

    def __init__(self, keys):
   
        EXCLUDE_KEYS = {"idx", "eid", "subdet"}
        self.stats = {}
   
        for key in keys:
            if key not in EXCLUDE_KEYS:
                self.stats[key] = RunningStats()

    def update(self, dataset):
        
        for key, stat in self.stats.items():
            
            if hasattr(dataset, "data") and key in dataset.data:
                stat.update(dataset.data[key])
            
            elif hasattr(dataset, "meta") and key in dataset.meta:
                stat.update(dataset.meta[key])

    def to_dict(self):

        return {
            key: {
                "mean": float(stat.mean),
                "std": float(stat.std()),
                "n": int(stat.n),
            }
            for key, stat in self.stats.items()
        }

    def save(self, save_dir):

        file_path = os.path.join(save_dir, "stats.json")
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class RunningStats:

    def __init__(self):
        self.mean = 0.0
        self.M2 = 0.0
        self.n = 0

    def update(self, vals):

        n = len(vals)
        mean = vals.mean()
        variance = vals.var(ddof=1)

        M2 = variance * (n - 1)

        delta = mean - self.mean

        n_new = self.n + n

        self.mean += delta * n / n_new
        self.M2 += M2 + delta**2 * self.n * n / n_new
        self.n = n_new

    def std(self):
        if self.n <= 1:
            return 0.0
        return np.sqrt(self.M2 / (self.n - 1))