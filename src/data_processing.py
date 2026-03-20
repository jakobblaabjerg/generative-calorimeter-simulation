import h5py
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split


def merge_dicts(dict1, dict2):
    """
    merges dict2 into dict1 if keys are not in dict1
    """
    _, counts = np.unique(dict1["eid"], return_counts=True)

    for k in dict2.keys():
        if k not in dict1.keys():            
            dict1[k] = np.repeat(dict2[k], repeats=counts)

    return dict1

def load_stats(load_dir):

    file_path = os.path.join(load_dir, "stats.json")

    with open(file_path, "r") as f:
        stats = json.load(f)

    return stats


def load_split_file(split, load_dir, file_name):

    data = np.load(os.path.join(load_dir, split, f"{file_name}_data.npz"))
    meta = np.load(os.path.join(load_dir, split, f"{file_name}_meta.npz"))
    
    return dict(data), dict(meta)


def load_split(split, load_dir, num_files):

    ## only works for with merge of data and meta
    files = sorted(
        ["_".join(f.split("_")[:-1]) for f in os.listdir(os.path.join(load_dir, split)) if f.endswith("data.npz")],
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )

    combined = {}

    for i in range(num_files):

        file_name = files[i]
        data, meta = load_split_file(split, load_dir, file_name)
        dataset = merge_dicts(data, meta)

        for k, v in dataset.items():

            if k not in combined:
                combined[k] = []

            combined[k].append(v)
    
    return {k: np.concatenate(v) for k, v in combined.items()}


def standardize_data(data, stats, vars_to_standardize, inverse=False):


    for var in vars_to_standardize:
        mean = stats[var]["mean"]
        std = stats[var]["std"]
        
        if inverse:
            data[var] = (data[var] * std) + mean
        else:
            data[var] = (data[var] - mean) / std



def filter_dict(data, mask):
    return {k: v[mask] for k, v in data.items()}


def filter_by_eid(data, meta, eid):

    # maybe original?

    mask = data["eid"] == eid
    data_filtered = filter_dict(data, mask)   
    
    mask = meta["eid"] == eid
    meta_filtered = filter_dict(meta, mask)

    return data_filtered, meta_filtered


def remove_eids(ref, to_filter):
    
    # remove missing eids from to_filter   
    mask = np.isin(to_filter["eid"], ref["eid"])

    if not mask.all():
        to_filter = filter_dict(to_filter, mask) 
    
    return to_filter 
  

def reindex_eid(data, meta):
   
    # reindex eid for data and meta 
    eids_old = np.unique(meta["eid"])
    eids_new = {old: new for new, old in enumerate(eids_old)}
    
    meta["eid"] = np.array([eids_new[eid] for eid in meta["eid"]])
    data["eid"] = np.array([eids_new[eid] for eid in data["eid"]])

    return data, meta


def compute_angles(meta):
    
    p_x = meta["p_x"]
    p_y = meta["p_y"]
    p_z = meta["p_z"]

    denom = (p_x**2 + p_y**2 + p_z**2)**0.5
    cos_theta = p_z / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    meta["theta"] = np.arccos(cos_theta)
    meta["phi"] = np.atan2(p_y, p_x)



def compute_energy(meta):

    p_x = meta["p_x"]
    p_y = meta["p_y"]
    p_z = meta["p_z"]

    meta["e_incident"] = (p_x**2 + p_y**2 + p_z**2)**0.5



def compute_centroids(data, meta):

    eids = data["eid"] 
    unique_eids, inverse = np.unique(eids, return_inverse=True)

    x = data["x"]
    y = data["y"]
    z = data["z"]
    e = data["e"]

    e_tot = np.bincount(inverse, weights=e)
    xe_sum = np.bincount(inverse, weights=x*e)
    ye_sum = np.bincount(inverse, weights=y*e)
    ze_sum = np.bincount(inverse, weights=z*e)

    meta["x_c"] = xe_sum / e_tot
    meta["y_c"] = ye_sum / e_tot
    meta["z_c"] = ze_sum / e_tot




def compute_misalignment(meta):

    # actual energy deposit
    x_c = meta["x_c"]
    y_c = meta["y_c"]
    z_c = meta["z_c"]

    # predicted direction based on incident particle
    theta = meta["theta"]
    phi = meta["phi"]
    r = np.linalg.norm([x_c, y_c, z_c], axis=0) 
    x_true = r * np.sin(theta) * np.cos(phi)
    y_true = r * np.sin(theta) * np.sin(phi)
    z_true = r * np.cos(theta)

    # compute angle and distance between predicted and actual direction
    dot = x_c*x_true + y_c*y_true + z_c*z_true
    norm_c = np.linalg.norm([x_c, y_c, z_c], axis=0)
    norm_pred = np.linalg.norm([x_true, y_true, z_true], axis=0)
    cos_angle = dot / (norm_c * norm_pred)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    meta["err_angle"] =  angle*180/np.pi
    meta["err_dist"] = np.linalg.norm([x_true-x_c, y_true-y_c, z_true-z_c], axis=0)
    



def compute_basis(meta): 

    theta = meta["theta"]
    phi = meta["phi"]

    z_hat = np.stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ], axis=1) 

    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(theta), 1))
    mask = np.abs(np.sum(z_hat * ref, axis=1)) > 0.99
    ref[mask] = [0.0, 1.0, 0.0] 

    x_hat = np.cross(ref, z_hat, axis=1)
    x_hat /= np.linalg.norm(x_hat, axis=1, keepdims=True)
    y_hat = np.cross(z_hat, x_hat, axis=1)

    basis = np.stack([x_hat, y_hat, z_hat], axis=2)

    unit_length, orthogonal = check_basis(basis)

    assert np.all(unit_length), "Some vectors are not unit length"
    assert np.all(orthogonal), "Some vectors are not orthogonal"

    meta["basis"] = basis

    

def check_basis(basis, tol=1e-6):

    n_samples, vec_dim, n_vectors = basis.shape

    norms = np.linalg.norm(basis, axis=1)
    unit_length = np.all(np.abs(norms - 1) < tol, axis=1) 

    orthogonal = np.ones(n_samples, dtype=bool)
    for i in range(n_vectors-1):
        for j in range(i+1, n_vectors):
            dot = np.sum(basis[:,:,i] * basis[:,:,j], axis=1)  
            orthogonal &= np.abs(dot) < tol

    return unit_length, orthogonal


def project_coordinates(data, meta):

    eids, counts = np.unique(data["eid"], return_counts=True)
    basis = np.repeat(meta["basis"], counts, axis=0)

    coords = np.stack([data["x"], data["y"], data["z"]], axis=1)

    data["x_hat"] = np.sum(coords * basis[:,:,0], axis=1)
    data["y_hat"] = np.sum(coords * basis[:,:,1], axis=1)
    data["z_hat"] = np.sum(coords * basis[:,:,2], axis=1)


def compute_retainment(data, meta, box_size):
    
    events_before, steps_before = np.unique(data["eid"], return_counts=True)
    data_filtered = filter_by_xy_box(data, box_size)

    idx, steps_after = np.unique(data_filtered["eid"], return_counts=True)

    steps_after_full = np.zeros(len(steps_before))
    steps_after_full[idx] = steps_after

    retained_pct = steps_after_full/steps_before*100 
    meta["retainment_steps"] = retained_pct


def compute_r_hat(data):

    x_hat = data["x_hat"]
    y_hat = data["y_hat"]

    r_hat = np.sqrt(x_hat**2 + y_hat**2)
    data["r_hat"] = r_hat


def compute_static_features(data, meta):

    compute_angles(meta) 
    compute_energy(meta)
    compute_basis(meta)
    compute_detector_distance(meta)
    project_coordinates(data, meta)
    center_z_hat(data, meta)
    compute_r_hat(data)

def filter_by_xy_box(data, box_size):

    x_hat = np.abs(data["x_hat"])
    y_hat = np.abs(data["y_hat"])

    mask = (x_hat <= box_size/2) & (y_hat <= box_size/2)
    data_filtered = filter_dict(data, mask)

    return data_filtered


def compute_detector_distance(meta):

    # determines which region is hit first by particle

    # barrel geometry 
    r_barrel = [1250, 1500] # mm, inner and outer radius 
    z_barrel = [0, 3050] # mm, start and end of barrel
    
    # endcap geometry 
    r_endcap = [315, 1500] # mm, inner and outer radius
    z_endcap = [3200, 3450] # mm, start and end of endcap

    # transition angle, from barrel to endcap front
    theta_1 = np.atan(r_barrel[0]/z_barrel[1]) 
    # transition angle, from endcap front to endcap inner surface  
    theta_2 = np.atan(r_endcap[0]/z_endcap[0])

    theta = meta["theta"]

    # fold detector symmetry
    theta = np.abs(theta)
    theta = np.where(theta > np.pi/2, np.pi - theta, theta)

    mask_barrel = theta >= theta_1
    mask_endcap_front = (theta >= theta_2) & (theta < theta_1)
    mask_endcap_inside = (theta < theta_2)    

    # mask_barrel = (theta >= theta_1) & (theta <= np.pi-theta_1)
    # mask_endcap_front = ((theta >= theta_2) & (theta < theta_1)) | ((theta > np.pi-theta_1) & (theta <= np.pi-theta_2))
    # mask_endcap_inside = (theta < theta_2) | (theta > np.pi-theta_2)


    entry_distance = np.zeros_like(theta)

    entry_distance[mask_barrel] = r_barrel[0] / np.sin(theta[mask_barrel])
    # entry_distance[mask_endcap_front] = z_endcap[0] / np.abs(np.cos(theta[mask_endcap_front]))
    entry_distance[mask_endcap_front] = z_endcap[0] / np.cos(theta[mask_endcap_front])
    entry_distance[mask_endcap_inside] = r_endcap[0] / np.sin(theta[mask_endcap_inside])

    meta["entry_distance"] = entry_distance

    exit_distance = np.minimum(r_barrel[1] / np.sin(theta), z_endcap[1] / np.cos(theta))
    meta["exit_distance"] = exit_distance



def center_z_hat(data, meta):

    z_hat = data["z_hat"]
    eids = data["eid"]

    eids_unique, counts = np.unique(eids, return_counts=True)

    dist = meta["entry_distance"]
    dist = np.repeat(dist, counts)

    z_hat = z_hat-dist
    data["z_hat"] = z_hat


def compute_energy_sum(data, meta):

    e = data["e"]
    eids = data["eid"]

    meta["e_sum"] = np.bincount(eids, weights=e)


def load_h5py_file(file_dir, file_name):

    file_path = os.path.join(file_dir, file_name)

    with h5py.File(file_path, "r") as f:
        
        steps = f["steps"]
        primary = f["primary"]
        metadata = f["metadata"]

        # extract relevant data 
        data = {
            "eid": steps["event_id"][:],
            "x": steps["position"][:,0],
            "y": steps["position"][:,1],
            "z": steps["position"][:,2],
            "e": steps["energy"][:],
            "t": steps["time"][:],
            "pid": steps["mcparticle_id"][:],
            "cid": steps["cell_id"][:],
            "d": steps["subdetector"][:],
        }

        sd_names = metadata["subdetector_names"][:]
        sd_names = [name[:-10] for name in sd_names]
        sd_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in sd_names]
        data["d"] = np.array(sd_names)[data["d"]]

        meta = {
            "eid": primary["event_id"][:],
            "x": primary["vertex"][:,0],
            "y": primary["vertex"][:,1],
            "z": primary["vertex"][:,2],
            "p_x": primary["momentum"][:,0],
            "p_y": primary["momentum"][:,1],
            "p_z": primary["momentum"][:,2],
            "pdg": primary["pdg"][:],
        }

    meta = remove_eids(data, meta) 
    data, meta = reindex_eid(data, meta)

    data["eid_original"] = data["eid"].copy()
    meta["eid_original"] = meta["eid"].copy()

    return data, meta   



class DataProcessor():

    def __init__(self, file_dir, save_dir):

        self.file_dir = file_dir
        self.save_dir = save_dir

        self.rng = np.random.default_rng(42)

        self.files = sorted(
            (f for f in os.listdir(file_dir) if f.endswith(".h5")),
            key=lambda f: int("".join(filter(str.isdigit, f)))
        )

        self.filter_map = {
            "time": self.filter_by_time,
            "energy": self.filter_by_energy,
            "misalignment": self.filter_by_misalignment,
            "retainment": self.filter_by_retainment,
            "z_hat": self.filter_by_z_hat,
            "detector": self.filter_by_detector
        }


    def load_raw(self, file_num=0):

        self.data_filtered, self.meta_filtered = None, None

        file_name = self.files[file_num]
        self.current = os.path.splitext(file_name)[0].split("_")[-1]
        self.data, self.meta = load_h5py_file(self.file_dir, file_name)
        compute_static_features(self.data, self.meta)
        

    def load_filtered(self, file_num=0):

        self.data, self.meta = None, None
        self.filters = {}

        file_name = self.files[file_num]
        self.current = os.path.splitext(file_name)[0].split("_")[-1]

        load_dir = os.path.join(self.save_dir, "filtered")

        self.data_filtered = dict(np.load(os.path.join(load_dir, f"{self.current}_data.npz")))
        self.meta_filtered = dict(np.load(os.path.join(load_dir, f"{self.current}_meta.npz")))


    def filter_data(self, config, filters=None, reset=True):

        if reset:
            self.filters = {}
            
            self.data_filtered = {k: v.copy() for k, v in self.data.items()}
            self.meta_filtered = {k: v.copy() for k, v in self.meta.items()}

        if filters is None:
            filters = ["time", "misalignment", "retainment", "z_hat"]

        eids, counts = np.unique(self.data["eid"], return_counts=True)
        self.before = dict(zip(eids, counts))

        for filter_name in filters:

            print(f"Filtering on {filter_name}")
            params = vars(getattr(config, filter_name))
            self.filter_map[filter_name](reindex=True, **params)

        eids_after, steps_after = np.unique(self.data_filtered["eid_original"], return_counts=True)
        self.after = dict(zip(eids_after, steps_after))

    def get_raw_data(self):
        return self.data, self.meta

    def get_filtered_data(self):
        return self.data_filtered, self.meta_filtered


    def get_key(self, filter):

        suffixes = []

        for k in self.filters.keys():
            if k.startswith(f"{filter}_"):
                suffixes.append(int(k.split("_")[-1]))

        return f"{filter}_{max(suffixes, default=0) + 1}"


    def filter_by_time(self, threshold=200, reindex=False):

        params = {"threshold": threshold}
        key = self.get_key(filter="time")
        self.filters[key] = {"params": params}

        data_mask = self.data_filtered["t"] <= threshold
        self._apply_filter(filter_name=key, data_mask=data_mask, reindex=reindex)


    def filter_by_energy(self, threshold=0.00001, reindex=False):

        params = {"threshold": threshold}
        key = self.get_key(filter="energy")
        self.filters[key] = {"params": params}

        data_mask = self.data_filtered["e"] >= threshold
        self._apply_filter(filter_name=key, data_mask=data_mask, reindex=reindex)


    def filter_by_misalignment(self, threshold=6, method="angle", reindex=False):

        params = {"threshold": threshold, "method": method}
        key = self.get_key(filter="misalignment")
        self.filters[key] = {"params": params}

        # remember to recompute centroids before using.        
        compute_centroids(self.data_filtered, self.meta_filtered) 
        compute_misalignment(self.meta_filtered)

        meta_mask = self.meta_filtered[f"err_{method}"] < threshold
        self._apply_filter(filter_name=key, meta_mask=meta_mask, reindex=reindex)

    
    def filter_by_detector(self, membership, reindex=False):

        params = {"membership": membership}
        key = self.get_key(filter="detector")
        self.filters[key] = {"params": params}

        data_mask = np.isin(self.data_filtered["d"], membership)
        self._apply_filter(filter_name=key, data_mask=data_mask, reindex=reindex)


    def filter_by_z_hat(self, threshold=0, reindex=False):

        params = {"threshold": threshold}
        key = self.get_key(filter="z_hat")
        self.filters[key] = {"params": params}

        data_mask = self.data_filtered["z_hat"] >= threshold
        self._apply_filter(filter_name=key, data_mask=data_mask, reindex=reindex)


    def filter_by_retainment(self, box_size, threshold=90, reindex=False):

        params = {"threshold": threshold, "box_size": box_size}
        key = self.get_key(filter="retainment")
        self.filters[key] = {"params": params}

        compute_retainment(self.data_filtered, self.meta_filtered, box_size=box_size)

        meta_mask = self.meta_filtered["retainment_steps"] >= threshold
        self._apply_filter(filter_name=key, meta_mask=meta_mask, reindex=reindex)

        data_mask = (
            (np.abs(self.data_filtered["x_hat"]) <= box_size / 2) &
            (np.abs(self.data_filtered["y_hat"]) <= box_size / 2)
        )
        self._apply_filter(filter_name=key, data_mask=data_mask, reindex=reindex)


    def _apply_filter(self, filter_name, data_mask=None, meta_mask=None, reindex=False):

        data_filtered = self.data_filtered
        meta_filtered = self.meta_filtered

        eids_before, steps_before = np.unique(data_filtered["eid_original"], return_counts=True)

        # step-level filtering
        if data_mask is not None: 
            data_filtered = filter_dict(data_filtered, data_mask)
            meta_filtered = remove_eids(data_filtered, meta_filtered)

        # event-level filtering
        if meta_mask is not None:
            meta_filtered = filter_dict(meta_filtered, meta_mask)
            data_filtered = remove_eids(meta_filtered, data_filtered)

        eids_after, steps_after = np.unique(data_filtered["eid_original"], return_counts=True)
        eids_removed = np.setdiff1d(eids_before, eids_after)

        steps_removed = steps_before.sum() - steps_after.sum()
        events_removed = len(eids_removed)


        self.filters[filter_name]["n_steps"] = self.filters[filter_name].get("n_steps", 0) + steps_removed
        self.filters[filter_name]["n_events"] = self.filters[filter_name].get("n_events", 0) + events_removed
        self.filters[filter_name].setdefault("eids", []).extend(eids_removed.tolist())

        # reindex
        if reindex and events_removed > 0:
            data_filtered, meta_filtered = reindex_eid(data_filtered, meta_filtered)

        self.data_filtered = data_filtered
        self.meta_filtered = meta_filtered


    def save_data(self, status_file=True): # save_filtered
        
        print("Saving data")

        save_dir = os.path.join(self.save_dir, "filtered")
        os.makedirs(save_dir, exist_ok=True)

        np.savez(os.path.join(save_dir, f"{self.current}_data.npz"), **self.data_filtered)
        np.savez(os.path.join(save_dir, f"{self.current}_meta.npz"), **self.meta_filtered)

        if status_file:
            self._write_status_file(save_dir)

        print("Save complete")


    def _write_status_file(self, save_dir):

        status_path = os.path.join(save_dir, f"{self.current}_status.txt")

        with open(status_path, "w") as f:

            f.write(f"Dataset name: {self.current}\n")
            f.write("=" * 40 + "\n\n")

            for k in self.filters.keys():

                filter_name = "_".join(k.split("_")[:-1])
            
                f.write(f"Filter: {filter_name}\n")
                f.write(f"  Steps removed:  {self.filters[k]['n_steps']}\n")
                f.write(f"  Events removed: {self.filters[k]['n_events']}\n")
                f.write(f"  Parameters:\n")
                
                for param, val in self.filters[k]["params"].items():
                    f.write(f"    - {param}: {val}\n")

                f.write("-" * 30 + "\n")

            f.write("\nFinal dataset summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Remaining events: {len(np.unique(self.data_filtered['eid']))}\n")
            f.write(f"Remaining steps:  {len(self.data_filtered['eid'])}\n")   




    def build_dataset(self, config, filters, normalize=True, debug=False):

        self.stats = {}

        # first loop
        for i in range(len(self.files)):

            # load -> filter -> save -> clean -> normalize -> split -> save

            if debug:
                print("Load filtered data")
                self.load_filtered()
            else:
                print("Load raw data")
                self.load_raw(file_num=i)
                self.filter_data(config=config, filters=filters, reset=True)
                self.save_data()

            print("Normalizing data")
            if normalize:
                normalize_data(self.data_filtered, self.meta_filtered, box_size=config.retainment.box_size)

            print("Cleaning data")

            self.clean_data(keepvars=config.dataset.keepvars)

            if not self.stats:
                self.init_stats()

            print("Splitting data")
            self.split_data(**vars(config.dataset))

        self.save_stats()


    def clean_data(self, keepvars):
       
        self.data_filtered = {
            k: v for k, v in self.data_filtered.items() 
            if k in keepvars or k.endswith("norm")
            }
        
        self.meta_filtered = {
            k: v for k, v in self.meta_filtered.items()
             if k in keepvars or k.endswith("norm")
             }

    def split_data(self, split=[0.8, 0.1, 0.1], keepvars=None, merge=True):

        data = self.data_filtered
        meta = self.meta_filtered

        eids_all = meta["eid"].copy()

        self.rng.shuffle(eids_all)

        n = len(eids_all)

        n_train = int(split[0] * n)
        n_val = int(split[1] * n)

        eids_train = eids_all[:n_train]
        eids_val = eids_all[n_train:n_train+n_val]
        eids_test = eids_all[n_train+n_val:]

        eids_split = {
            "train": eids_train, 
            "val": eids_val, 
            "test": eids_test
            }

        for split_name, eids in eids_split.items():

            data_mask = np.isin(data["eid"], eids)
            meta_mask = np.isin(meta["eid"], eids)

            data_filtered = filter_dict(data, data_mask)
            meta_filtered = filter_dict(meta, meta_mask)
            
            if split_name == "train":

                self.update_stats(data_filtered, meta_filtered)

            self.save_split(data_filtered, meta_filtered, split=split_name)

    def init_stats(self):

        for k in self.data_filtered | self.meta_filtered:
            if k not in ["eid", "eid_original"]:
                self.stats[k] = RunningStats()

    def update_stats(self, data, meta):

        # make sure no key are the same in data and meta
        for k, stat in self.stats.items():

            vals = data.get(k)

            if vals is None:
                vals = meta.get(k)
            
            if vals is None:
                continue

            stat.update(vals)

    def save_stats(self):

        stats_out = {}

        for k, stat in self.stats.items():

            stats_out[k] = {
                "mean": float(stat.mean),
                "std": float(stat.std()),
                "n": int(stat.n)
            }

        save_path = os.path.join(self.save_dir, "stats.json")

        with open(save_path, "w") as f:
            json.dump(stats_out, f, indent=4)

    def save_split(self, data, meta, split):

        save_dir = os.path.join(self.save_dir, split)
        os.makedirs(save_dir, exist_ok=True)

        np.savez(os.path.join(save_dir, f"{self.current}_data.npz"), **data)
        np.savez(os.path.join(save_dir, f"{self.current}_meta.npz"), **meta)


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
            return 1.0
        return np.sqrt(self.M2 / (self.n - 1))



def normalize_data(data, meta, box_size, inverse=False):

    if inverse:
        _denormalize(data, meta, box_size)
    else:
        _normalize(data, meta, box_size)


def _normalize(data, meta, box_size):

    # energy
    meta["e_incident_norm"] = np.log1p(meta["e_incident"])

    # direction
    theta = meta["theta"]
    phi = meta["phi"]

    meta["dir_x_norm"] = np.sin(theta) * np.cos(phi)
    meta["dir_y_norm"] = np.sin(theta) * np.sin(phi)
    meta["dir_z_norm"] = np.cos(theta)

    # event counts
    _, counts = np.unique(data["eid"], return_counts=True)

    # x/y position to [-1, 1]
    scale = box_size / 2
    data["x_hat_norm"] = data["x_hat"] / scale
    data["y_hat_norm"] = data["y_hat"] / scale

    # radial distance to [0, 1]
    r_max = np.sqrt(2 * scale**2)
    r_hat = data["r_hat"]
    data["r_hat_norm"] = r_hat / r_max
    data["r_log_norm"] = np.log1p(r_hat) / np.log1p(r_max)


    # z distance to [0, ~1.2]
    max_distance = np.repeat(
        meta["exit_distance"] - meta["entry_distance"],
        counts
    )

    data["z_hat_norm"] = data["z_hat"] / max_distance

    e_incident = np.repeat(meta["e_incident"], counts)
    e_rel = data["e"] / e_incident
    data["e_log_norm"] = np.log1p(e_rel)
    data["e_sqrt_norm"] = np.sqrt(e_rel)



def _denormalize(data, meta, box_size):

    # energy
    meta["e_incident"] = np.expm1(meta["e_incident_norm"])

    # direction
    dir_x_norm = meta["dir_x_norm"]
    dir_y_norm = meta["dir_y_norm"]
    dir_z_norm = meta["dir_z_norm"]
    # dir_z_norm = np.clip(meta["dir_z_norm"], -1, 1)

    meta["theta"] = np.arccos(dir_z_norm)
    meta["phi"] = np.arctan2(dir_y_norm, dir_x_norm)

    compute_detector_distance(meta)

    _, counts = np.unique(data["eid"], return_counts=True)
    scale = box_size / 2

    data["x_hat"] = data["x_hat_norm"] * scale
    data["y_hat"] = data["y_hat_norm"] * scale


    r_max = np.sqrt(2 * scale**2)

    if "r_hat_norm" in data:
        data["r_hat"] = data["r_hat_norm"] * r_max

    elif "r_log_norm" in data:
        data["r_hat"] = np.expm1(data["r_log_norm"] * np.log1p(r_max))


    max_distance = np.repeat(
        meta["exit_distance"] - meta["entry_distance"],
        counts
    )

    data["z_hat"] = data["z_hat_norm"] * max_distance

    # energy
    e_incident = np.repeat(meta["e_incident"], counts)

    if "e_log_norm" in data:
        e_rel = np.expm1(data["e_log_norm"])

    elif "e_sqrt_norm" in data:
        e_rel = data["e_sqrt_norm"] ** 2

    data["e"] = e_rel * e_incident