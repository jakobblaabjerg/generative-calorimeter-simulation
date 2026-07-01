import h5py 
import numpy as np 

from src.calosim import CaloSimDataset
from src.geometry import compute_geometric_features


class Step2Point:
    
    @classmethod
    def from_h5(cls, file_path):
        """
        Load a dataset from a raw HDF5 simulation file.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        CaloSimDataset
        """
        with h5py.File(file_path, "r") as f:
            data = cls._extract_steps(f)
            data = cls._decode_subdetector(f, data)
            meta = cls._extract_primary(f)
            dataset = CaloSimDataset(data=data, meta=meta)

        dataset.sync()
        dataset.reindex()

        compute_geometric_features(dataset)

        return dataset

    @staticmethod
    def _extract_particles(f):

        return {
            "eid": f["particles"]["event_id"],
            "pid": f["particles"]["id"],
            "pid_src": f["particles"]["parent_id"]
        }
        

    @staticmethod
    def _extract_steps(f):

        """
        Extract step-level (point-level) information from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file containing a "steps" group.

        Returns
        -------
        dict
            Dictionary containing per-step arrays:
            - eid: event identifier
            - idx: event identifier (internal index, currently identical to eid)
            - x, y, z: spatial coordinates of hits
            - e: deposited energy per step
            - t: time of step
            - pid: particle ID
            - cid: detector cell ID
            - subdet: subdetector index (to be decoded later)
        """

        pos = f["steps"]["position"][:]

        return {
            "eid": f["steps"]["event_id"][:],
            "idx": f["steps"]["event_id"][:],
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "e": f["steps"]["energy"][:],
            "t": f["steps"]["time"][:],
            "pid": f["steps"]["mcparticle_id"][:],
            "cid": f["steps"]["cell_id"][:],
            "subdet": f["steps"]["subdetector"][:],
        }

    @staticmethod
    def _extract_primary(f):
        """
        Extract event-level (primary particle) information from an HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file containing a "primary" group.

        Returns
        -------
        dict
            Dictionary containing event-level arrays:
            - eid: event identifier
            - idx: event identifier (internal index, currently identical to eid)
            - p_x, p_y, p_z: incident momentum components
            - pdg: PDG particle ID
        """

        momentum = f["primary"]["momentum"][:]

        return {
            "eid": f["primary"]["event_id"][:],
            "idx": f["primary"]["event_id"][:],
            "p_x": momentum[:,0],
            "p_y": momentum[:,1],
            "p_z": momentum[:,2],
            "pdg": f["primary"]["pdg"][:],
            }


    @staticmethod
    def _decode_subdetector(f, data):
        """
        Decode subdetector indices into human-readable names.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file containing metadata with subdetector names.
        data : dict

        Returns
        -------
        dict
            Updated data dictionary where 'subdet' is replaced with
            decoded string labels instead of integer indices.
        """

        names = f["metadata"]["subdetector_names"][:]
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]
        data["subdet"] = np.asarray(names)[data["subdet"]]
        return data