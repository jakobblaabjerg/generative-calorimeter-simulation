import h5py
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split


def safe_underscore(name):
    return f"_{name}" if name else ""


def create_meta(n_samples, phi=None, theta=None, e_inc=None, normalize=True, seed=None):

    rng = np.random.default_rng(seed)
    meta = {}
    
    specs = {
        "phi": (-np.pi, np.pi, phi),
        "theta": (6 * np.pi / 180, 174 * np.pi / 180, theta),
        "e_inc": (0.1, 100, e_inc)
    }
    
    for key, (min_val, max_val, val) in specs.items():
        if val is None:
            u = rng.uniform(size=n_samples).astype(np.float32)
            meta[key] = u * (max_val - min_val) + min_val
        else:
            meta[key] = np.repeat(val, n_samples).astype(np.float32)
    
    if normalize:
        normalize_meta(meta, inverse=False)

    return meta


def concat_dict(dict1, dict2):

    for k, v in dict2.items():
        
        if k in dict1:
            
            if k == "eid":
                eids = dict1[k]
                offset = len(np.unique(eids))
                v = v + offset
            dict1[k] = np.concatenate((dict1[k], v))
        
        else:
            dict1[k] = v.copy()  
    
    return dict1


def merge_dicts(dict1, dict2):
    """
    merges dict2 into dict1 if keys are not in dict1
    """
    _, counts = np.unique(dict1["eid"], return_counts=True)

    for k in dict2.keys():
        if k not in dict1.keys():            
            dict1[k] = np.repeat(dict2[k], repeats=counts)

    dict1["N"] = np.repeat(counts, repeats=counts)

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


def append_to_dict(data, combined):

    for k, v in data.items():
        if k not in combined:
            combined[k] = []
        combined[k].append(v)


def load_split(split, load_dir, num_files):

    data_combined = {}
    meta_combined = {}

    files = get_file_names(load_dir, split)

    offset = 0 

    for i in range(num_files):

        file_name = "_".join(files[i].split("_")[:-1])
        data, meta = load_split_file(split, load_dir, file_name)

        data["eid"] += offset
        meta["eid"] += offset

        offset = meta["eid"].max()+1

        append_to_dict(data, data_combined)
        append_to_dict(meta, meta_combined)

    assert len(np.unique(meta["eid"])) == len(meta["eid"])
        
    data_combined = {k: np.concatenate(v) for k, v in data_combined.items()}
    meta_combined = {k: np.concatenate(v) for k, v in meta_combined.items()}
    
    return data_combined, meta_combined 


def standardize_data(data, stats, standardize_vars, inverse=False):


    for var in standardize_vars:

        if not var in data:
            continue

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

    meta["e_inc"] = (p_x**2 + p_y**2 + p_z**2)**0.5



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


def project_coordinates(data, meta, inverse=False):

    eids, counts = np.unique(data["eid"], return_counts=True)
    basis = np.repeat(meta["basis"], counts, axis=0)

    if not inverse:
        coords = np.stack([data["x"], data["y"], data["z"]], axis=1)
        data["x_hat"] = np.sum(coords * basis[:,:,0], axis=1)
        data["y_hat"] = np.sum(coords * basis[:,:,1], axis=1)
        data["z_hat"] = np.sum(coords * basis[:,:,2], axis=1)

    else:
        coords_hat = np.stack([data["x_hat"], data["y_hat"], data["z_hat"]], axis=1)
        coords = (coords_hat[:,0,None] * basis[:,:,0] + coords_hat[:,1,None] * basis[:,:,1] + coords_hat[:,2,None] * basis[:,:,2])
        
        data["x"] = coords[:,0]
        data["y"] = coords[:,1]
        data["z"] = coords[:,2]



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


def compute_static_features(data, meta, inverse=False):

    if not inverse:

        compute_angles(meta) 
        compute_energy(meta)
        compute_basis(meta)
        compute_detector_distance(meta)
        project_coordinates(data, meta)
        center_z_hat(data, meta)
        compute_r_hat(data)
    
    else:
        data["d"] = np.repeat(str(0), len(data["eid"]))
        compute_basis(meta)
        center_z_hat(data, meta, inverse=True)
        project_coordinates(data, meta, inverse=True)
        center_z_hat(data, meta)
        compute_centroids(data, meta)
        compute_misalignment(meta)


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



def center_z_hat(data, meta, inverse=False):

    z_hat = data["z_hat"]
    eids = data["eid"]

    dist = meta["entry_distance"][eids]

    sign = 1 if inverse else -1
    z_hat = z_hat + sign * dist
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

def get_file_idx(file_name):
    return int(''.join(c for c in os.path.splitext(file_name)[0] if c.isdigit()))


def get_file_names(file_dir, folder_name):

    if folder_name == "raw":
        files = sorted(
            (f for f in os.listdir(file_dir) if f.endswith(".h5")),
            key=lambda f: int("".join(filter(str.isdigit, f)))
        )

    else:
        files = sorted(
            (f for f in os.listdir(os.path.join(file_dir, folder_name)) if f.endswith("data.npz")),
            key=lambda f: int("".join(filter(str.isdigit, f)))
        )

    return files



class DataProcessor():

    def __init__(
            self, 
            cfg, 
            output_dir,
            input_dir=None, 
            ):


        self.cfg = cfg
        self.output_dir = output_dir 
        self.input_dir = input_dir 

        self.stats = self.load_stats() # load it if exists 
        
        self.rng = np.random.default_rng(42)

        self.filter_map = {
            "time":         self.filter_by_time,
            "energy":       self.filter_by_energy,
            "misalignment": self.filter_by_misalignment,
            "retainment":   self.filter_by_retainment,
            "z_hat":        self.filter_by_z_hat,
            "detector":     self.filter_by_detector
        }

    def load_raw(self, load_dir, file_name):
        """
        load_dir: directory to load raw files from 
        file_num: file idx of file to use 
        """               
        print(file_name)
        data, meta = load_h5py_file(load_dir, file_name)
        compute_static_features(data, meta)

        return data, meta
        

    def load_data(self, stage, file_idx):

        # reset filters 
        self.filters = {}
       
        file_name = f"file{file_idx}"
        file_dir = os.path.join(self.output_dir, stage)

        data = dict(np.load(os.path.join(file_dir, f"{file_name}_data.npz")))
        meta = dict(np.load(os.path.join(file_dir, f"{file_name}_meta.npz")))

        return data, meta


    def filter_data(self, data, meta, filters=None, reset=True):

        if reset:
            self.filters = {}
        
        # copy input
        data_filtered = {k: v.copy() for k, v in data.items()}
        meta_filtered = {k: v.copy() for k, v in meta.items()}

        # default filters
        if filters is None:
            filters = ["time", "misalignment", "retainment", "z_hat"]

        # loop through filters
        for filter_name in filters:
            print(f"Filtering on {filter_name}")
            params = vars(getattr(self.cfg, filter_name))
            data_filtered, meta_filtered = self.filter_map[filter_name](data_filtered, meta_filtered, **params)

        # reindex only once after all filters
        data_filtered, meta_filtered = reindex_eid(data_filtered, meta_filtered)

        return data_filtered, meta_filtered

    def aggregate_data(self, data, reset=False):

        if reset:
            self.filters = {}
        
        data_agg = {k: v.copy() for k, v in data.items()} 
       
        _, steps_before = np.unique(data_agg["eid_original"], return_counts=True)
        keys = np.rec.fromarrays([data_agg["eid"], data_agg["pid"], data_agg["cid"]], names="eid, pid, cid")
        unique_keys, first_idx, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True, return_index=True)

        for k in data_agg.keys():

            if k in ["eid", "pid", "cid"]:
                data_agg[k] = unique_keys[k]
            elif k in ["eid_original", "d"]:
                # first occurence
                data_agg[k] = data_agg[k][first_idx]
            elif k == "e":
                # compute sum
                data_agg[k] = np.bincount(inverse, weights=data_agg[k])             
            else:
                # compute averages
                data_agg[k] = np.bincount(inverse, weights=data_agg[k])/counts 

        _, steps_after = np.unique(data_agg["eid_original"], return_counts=True)
        steps_removed = steps_before.sum() - steps_after.sum()


        # make this cleaner !? 
        self.filters.setdefault("aggregation", {})

        self.filters["aggregation"]["n_steps"] = steps_removed
        self.filters["aggregation"]["n_events"] = 0
        self.filters["aggregation"]["eids"] = []

        return data_agg 

    def get_key(self, filter_name):

        suffixes = []
        for k in self.filters.keys():
            if k.startswith(f"{filter_name}_"):
                suffixes.append(int(k.split("_")[-1]))
        return f"{filter_name}_{max(suffixes, default=0) + 1}"

    def filter_by_time(self, data, meta, threshold=200, reindex=True):

        params = {"threshold": threshold}
        key = self.get_key(filter_name="time")
        self.filters[key] = {"params": params}

        mask = data["t"] <= threshold
        return self._apply_filter(data, meta, filter_name=key, mask=mask, level="data", reindex=reindex)

    def filter_by_energy(self, data, meta, threshold=0.00001, reindex=True):

        params = {"threshold": threshold}
        key = self.get_key(filter_name="energy")
        self.filters[key] = {"params": params}

        mask = data["e"] >= threshold
        return self._apply_filter(data, meta, filter_name=key, mask=mask, level="data", reindex=reindex)


    def filter_by_misalignment(self, data, meta, threshold=6, method="angle", reindex=True):

        params = {"threshold": threshold, "method": method}
        key = self.get_key(filter_name="misalignment")
        self.filters[key] = {"params": params}

        # remember to recompute centroids before using.        
        compute_centroids(data, meta) 
        compute_misalignment(meta)

        mask = meta[f"err_{method}"] < threshold
        return self._apply_filter(data, meta, filter_name=key, mask=mask, level="meta", reindex=reindex)

    
    def filter_by_detector(self, data, meta, membership, reindex=True):

        params = {"membership": membership}
        key = self.get_key(filter_name="detector")
        self.filters[key] = {"params": params}

        mask = np.isin(data["d"], membership)
        return self._apply_filter(data, meta, filter_name=key, mask=mask, level="data", reindex=reindex)


    def filter_by_z_hat(self, data, meta, threshold=0, reindex=True):

        params = {"threshold": threshold}
        key = self.get_key(filter_name="z_hat")
        self.filters[key] = {"params": params}

        mask = data["z_hat"] >= threshold
        return self._apply_filter(data, meta, filter_name=key, mask=mask, level="data", reindex=reindex)


    def filter_by_retainment(self, data, meta, box_size, threshold=90, reindex=True):

        params = {"threshold": threshold, "box_size": box_size}
        key = self.get_key(filter_name="retainment")
        self.filters[key] = {"params": params}

        compute_retainment(data, meta, box_size=box_size)

        mask1 = meta["retainment_steps"] >= threshold
        data, meta = self._apply_filter(data, meta, filter_name=key, mask=mask1, level="meta", reindex=reindex)

        mask2 = (
            (np.abs(data["x_hat"]) <= box_size / 2) &
            (np.abs(data["y_hat"]) <= box_size / 2)
        )
        return self._apply_filter(data, meta, filter_name=key, mask=mask2, level="data", reindex=reindex)



    def _apply_filter(self, data, meta, filter_name, mask, level, reindex):

        eids_before, steps_before = np.unique(data["eid_original"], return_counts=True)

        # step-level filtering
        if level == "data": 
            data_filtered = filter_dict(data, mask)
            meta_filtered = remove_eids(data_filtered, meta)

        # event-level filtering
        elif level == "meta":
            meta_filtered = filter_dict(meta, mask)
            data_filtered = remove_eids(meta_filtered, data)

        eids_after, steps_after = np.unique(data_filtered["eid_original"], return_counts=True)

        eids_removed = np.setdiff1d(eids_before, eids_after)
        steps_removed = steps_before.sum() - steps_after.sum()
        events_removed = len(eids_removed)

        if reindex and events_removed > 0:
            data_filtered, meta_filtered = reindex_eid(data_filtered, meta_filtered)

        self.filters[filter_name]["n_steps"] = self.filters[filter_name].get("n_steps", 0) + steps_removed
        self.filters[filter_name]["n_events"] = self.filters[filter_name].get("n_events", 0) + events_removed
        self.filters[filter_name].setdefault("eids", []).extend(eids_removed.tolist())

        return data_filtered, meta_filtered



    def save_data(self, data, meta, stage, file_idx): 

        print("Saving data")
        save_dir = os.path.join(self.output_dir, stage)
        os.makedirs(save_dir, exist_ok=True)

        np.savez(os.path.join(save_dir, f"file{file_idx}_data.npz"), **data)
        np.savez(os.path.join(save_dir, f"file{file_idx}_meta.npz"), **meta)

        if stage == "filtered":
            self._write_status_file(save_dir, data, file_idx)

        print("Save complete")


    def _write_status_file(self, save_dir, data, file_idx):

        file_path = os.path.join(save_dir, f"file{file_idx}_status.txt")

        with open(file_path, "w") as f:

            f.write(f"Dataset name: {file_idx}\n")
            f.write("=" * 40 + "\n\n")

            for k in self.filters.keys():

                if k == "aggregation":
                    filter_name = k
                else:
                    filter_name = "_".join(k.split("_")[:-1])
                
                f.write(f"Filter: {filter_name}\n")
                f.write(f"  Steps removed:  {self.filters[k]['n_steps']}\n")
                f.write(f"  Events removed: {self.filters[k]['n_events']}\n")

                if "params" in self.filters[k]:

                    f.write(f"  Parameters:\n")
                    
                    for param, val in self.filters[k]["params"].items():
                        f.write(f"    - {param}: {val}\n")

                f.write("-" * 30 + "\n")

            f.write("\nFinal dataset summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Remaining events: {len(np.unique(data['eid']))}\n")
            f.write(f"Remaining steps:  {len(data['eid'])}\n")   


    def build_dataset(self, debug=False):

        self.stats = {}
        file_names = get_file_names(self.output_dir, "filtered") if debug else get_file_names(self.input_dir, "raw")

        for i in range(len(file_names)):

            if debug:
                print("Load filtered data") 
                file_idx = get_file_idx(file_names[i])       
                data_filtered, meta_filtered = self.load_data(stage="filtered", file_idx=file_idx)

            else:
                print("Load raw data")
                data, meta = self.load_raw(self.input_dir, file_names[i])
                filters = self.cfg.dataset.filters
                data_filtered, meta_filtered = self.filter_data(data, meta, filters=filters, reset=True)

                # aggregate data                 
                if self.cfg.dataset.aggregate:
                    print("Aggregating data")
                    data_filtered = self.aggregate_data(data_filtered, reset=False)

                # save data
                self.save_data(data_filtered, meta_filtered, stage="filtered", file_idx=i+1)

            # normalize data
            if self.cfg.dataset.normalize:
                print("Normalizing data")
                normalize_data(data_filtered, meta_filtered, self.cfg)

            print("Cleaning data")
            data_filtered = self.clean_data(data_filtered)
            meta_filtered = self.clean_data(meta_filtered)

            if not self.stats:
                self.init_stats(data_filtered, meta_filtered) # not very clean!?

            print("Splitting data")
            split_ratios = self.cfg.dataset.split_ratios
            self.split_and_save(data_filtered, meta_filtered, split_ratios, i+1)

        self.save_stats()


    def clean_data(self, dict1):
        return {
            k: v for k, v in dict1.items() 
            if k in self.cfg.dataset.keepvars or k.endswith("norm")
            }


    def split_and_save(self, data, meta, split_ratios, file_idx):

        eids_all = meta["eid"].copy()

        self.rng.shuffle(eids_all)

        n = len(eids_all)

        n_train = int(split_ratios[0] * n)
        n_val = int(split_ratios[1] * n)

        eids_train = eids_all[:n_train]
        eids_val = eids_all[n_train:n_train+n_val]
        eids_test = eids_all[n_train+n_val:]

        splits = {
            "train": eids_train, 
            "val": eids_val, 
            "test": eids_test
            }

        for split, eids in splits.items():

            data_mask = np.isin(data["eid"], eids)
            meta_mask = np.isin(meta["eid"], eids)

            data_filtered = filter_dict(data, data_mask)
            meta_filtered = filter_dict(meta, meta_mask)
            
            data_filtered, meta_filtered = reindex_eid(data_filtered, meta_filtered)

            if split == "train":
                self.update_stats(data_filtered, meta_filtered)

            self.save_split(data_filtered, meta_filtered, split, file_idx)


    def save_split(self, data, meta, split, file_idx):

        save_dir = os.path.join(self.output_dir, split)
        os.makedirs(save_dir, exist_ok=True)

        np.savez(os.path.join(save_dir, f"file{file_idx}_data.npz"), **data)
        np.savez(os.path.join(save_dir, f"file{file_idx}_meta.npz"), **meta)


    def init_stats(self, data, meta):

        for k in data | meta:
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

        save_path = os.path.join(self.output_dir, "stats.json")

        with open(save_path, "w") as f:
            json.dump(stats_out, f, indent=4)

    
    def load_stats(self):
        
        try: 
            stats = load_stats(load_dir=self.output_dir)
            return stats
        
        except Exception as e:
            return None 
        
    
    def inverse_transform(self, data, meta, standardize_vars):

        standardize_data(data, self.stats, standardize_vars, inverse=True)
        standardize_data(meta, self.stats, standardize_vars, inverse=True)

        normalize_data(data, meta, self.cfg, inverse=True)
        compute_static_features(data, meta, inverse=True)

        return data, meta 

        



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


def normalize_meta(meta, inverse):

    if not inverse:
        meta["e_inc_norm"] = np.log1p(meta["e_inc"])
        theta = meta["theta"]
        phi = meta["phi"]

        meta["dir_x_norm"] = np.sin(theta) * np.cos(phi)
        meta["dir_y_norm"] = np.sin(theta) * np.sin(phi)
        meta["dir_z_norm"] = np.cos(theta)
    
    else:
        meta["e_inc"] = np.expm1(meta["e_inc_norm"])
        dir_x_norm = meta["dir_x_norm"]
        dir_y_norm = meta["dir_y_norm"]
        dir_z_norm = meta["dir_z_norm"]
        # dir_z_norm = np.clip(meta["dir_z_norm"], -1, 1)

        meta["theta"] = np.arccos(dir_z_norm)
        meta["phi"] = np.arctan2(dir_y_norm, dir_x_norm)


def normalize_data(data, meta, cfg, inverse=False):

    normalize_meta(meta, inverse)
    _, counts = np.unique(data["eid"], return_counts=True)
    scale_xy = cfg.retainment.box_size / 2
    scale_e = cfg.energy.threshold
    r_max = np.sqrt(2 * scale_xy**2)

    if not inverse:

        # normalize x/y position to [-1, 1]
        data["x_hat_norm"] = data["x_hat"] / scale_xy
        data["y_hat_norm"] = data["y_hat"] / scale_xy

        # normalize radial distance to [0, 1]
        r_hat = data["r_hat"]
        data["r_hat_norm"] = r_hat / r_max

        # normalize z distance to [0, ~1.2]
        max_distance = np.repeat(meta["exit_distance"] - meta["entry_distance"], counts)
        data["z_hat_norm"] = data["z_hat"] / max_distance
        data["z_hat_log_norm"] = np.log((data["z_hat_norm"]) + 1e-6)
        data["z_hat_sqrt_norm"] = np.sqrt(data["z_hat_norm"])

        # log/sqrt transform of scaled energy
        e_scaled = data["e"] / scale_e
        data["e_log_norm"] = np.log(e_scaled)
        data["e_sqrt_norm"] = np.sqrt(e_scaled)

    else:
        compute_detector_distance(meta)

        # denormalize x/y position to [-scale, scale]
        if "x_hat_norm" in data:
            data["x_hat"] = data["x_hat_norm"] * scale_xy
            data["y_hat"] = data["y_hat_norm"] * scale_xy

        # denormalize radial distance to [0, r_max]
        if "r_hat_norm" in data:
            data["r_hat"] = data["r_hat_norm"] * r_max

        # denormalize z distance to [0, max_distance]
        max_distance = np.repeat(meta["exit_distance"] - meta["entry_distance"], counts)

        if "z_hat_norm" in data:
            data["z_hat"] = data["z_hat_norm"] * max_distance

        elif "z_hat_log_norm" in data:
            data["z_hat"] = (np.exp(data["z_hat_log_norm"]) -1e-6) * max_distance

        elif "z_hat_sqrt_norm" in data:
            z_hat_sqrt_norm = np.clip(data["z_hat_sqrt_norm"], -0.05, None)
            data["z_hat"] = (z_hat_sqrt_norm**2) * max_distance

        # inverse transform energy
        if "e_log_norm" in data:
            e_scaled = np.exp(data["e_log_norm"])
        elif "e_sqrt_norm" in data:
            e_scaled = data["e_sqrt_norm"] ** 2
        data["e"] = e_scaled * scale_e