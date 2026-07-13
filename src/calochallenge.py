import h5py 
import numpy as np 

from src.calosim import CaloSimDataset

class CaloChallenge:

    @classmethod
    def from_h5(cls, file_path):

        with h5py.File(file_path, "r") as f:
            data = cls._extract_showers(f)
            meta = cls._extract_incident_energies(f)
            dataset = CaloSimDataset(data=data, meta=meta, view="voxel")
        
        return dataset
    
    @staticmethod
    def _extract_incident_energies(f):

        e_inc = f["incident_energies"][:].astype(np.float32).reshape(-1)
        idx = np.arange(len(e_inc))

        return {
            "idx": idx,
            "e_inc": e_inc
            }

    @staticmethod 
    def _extract_showers(f):
        
        e = f["showers"][:, :].astype(np.float32)
        num_events, num_voxels = e.shape
        e = e.flatten()         
        idx = np.repeat(np.arange(num_events), num_voxels)
        
        return {
            "idx": idx,
            "e": e
        }