import numpy as np 
from src.calosim import CaloSimDataset
from dataclasses import dataclass
from src.utils import filter_dict


# remove this
def filter_by_xy_box(data, box_size):

    x_hat = np.abs(data["x_hat"])
    y_hat = np.abs(data["y_hat"])

    mask = (x_hat <= box_size/2) & (y_hat <= box_size/2)
    data_filtered = filter_dict(data, mask)

    return data_filtered



def compute_spherical_coords(dataset: CaloSimDataset) -> None:
    """ 
    Compute spherical coordinates from incident momentum vectors. 
    
    Parameters 
    ---------- 
    dataset : CaloSimDataset 
        Dataset containing momentum components in 
        dataset.meta["p_x"], dataset.meta["p_y"], dataset.meta["p_z"]. 
        
    Requires 
    -------- 
    dataset.events["p_x"] 
    dataset.events["p_y"] 
    dataset.events["p_z"] 
    
    Mutates 
    ------- 
    dataset.events["theta"] 
    dataset.events["phi"] 
    
    Notes 
    ----- 
    Theta and phi are computed in radians. 
    """

    p_x = dataset.meta["p_x"]
    p_y = dataset.meta["p_y"]
    p_z = dataset.meta["p_z"]

    denom = (p_x**2 + p_y**2 + p_z**2)**0.5
    cos_theta = p_z / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    dataset.meta["theta"] = np.arccos(cos_theta)
    dataset.meta["phi"] = np.atan2(p_y, p_x)



def compute_incident_energy(dataset: CaloSimDataset) -> None:
    """
    Compute incident particle energy from momentum components.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing momentum components in
        dataset.meta["p_x"], dataset.meta["p_y"], dataset.meta["p_z"].

    Requires
    --------
    dataset.meta["p_x"]
    dataset.meta["p_y"]
    dataset.meta["p_z"]

    Mutates
    -------
    dataset.meta["e_inc"]

    Notes
    -----
    This assumes natural units where |p| corresponds to energy
    (e.g., c = 1), as is standard in particle physics simulations.
    """

    p_x = dataset.meta["p_x"]
    p_y = dataset.meta["p_y"]
    p_z = dataset.meta["p_z"]

    dataset.meta["e_inc"] = (p_x**2 + p_y**2 + p_z**2)**0.5



def compute_centroids(dataset: CaloSimDataset) -> None:
    """
    Compute energy-weighted centroids of event.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing spatial coordinates and energies.

    Requires
    --------
    dataset.data["eid"]
    dataset.data["x"]
    dataset.data["y"]
    dataset.data["z"]
    dataset.data["e"]

    Mutates
    -------
    dataset.meta["x_c"]
    dataset.meta["y_c"]
    dataset.meta["z_c"]

    Notes
    -----
    This assumes:
    - energies are positive or physically meaningful weights
    - all steps belonging to an event share the same eid grouping
    """

    idxs = dataset.data["idx"] 
    _, inverse = np.unique(idxs, return_inverse=True)

    x = dataset.data["x"]
    y = dataset.data["y"]
    z = dataset.data["z"]
    e = dataset.data["e"]

    e_tot = np.bincount(inverse, weights=e)
    
    x_weighted_sum = np.bincount(inverse, weights=x*e)
    y_weighted_sum = np.bincount(inverse, weights=y*e)
    z_weighted_sum = np.bincount(inverse, weights=z*e)

    dataset.meta["x_c"] = x_weighted_sum / e_tot
    dataset.meta["y_c"] = y_weighted_sum / e_tot
    dataset.meta["z_c"] = z_weighted_sum / e_tot




def compute_misalignment(dataset: CaloSimDataset) -> None:

    """
    Compute geometric misalignment between centroid direction
    and expected direction from incident particle kinematics.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing centroids and incident angles.

    Requires
    --------
    dataset.meta["x_c"], ["y_c"], ["z_c"]
        Energy-weighted centroids.
    dataset.meta["theta"], ["phi"]
        Incident particle direction in spherical coordinates.

    Mutates
    -------
    dataset.meta["ang_misalign"]
        Angular misalignment between predicted and reconstructed shower axis (degrees).
    dataset.meta["pos_misalign"]
        Euclidean distance between predicted and reconstructed centroid positions.

    Method
    ------
    1. Reconstruct ideal direction from incident angles:
       x_true = r sin(theta) cos(phi)
       y_true = r sin(theta) sin(phi)
       z_true = r cos(theta)

    2. Compare reconstructed centroid vector (x_c, y_c, z_c)
       with predicted vector (x_true, y_true, z_true)

    3. Compute:
       - angular misalignment via dot product
       - spatial misalignment via Euclidean distance

    Notes
    -----
    - Angles are assumed in radians.
    - Output angle is converted to degrees.
    - Numerical stability ensured via clipping of cosine values.
    """

    x_c = dataset.meta["x_c"]
    y_c = dataset.meta["y_c"]
    z_c = dataset.meta["z_c"]


    # expected direction based on incident particle kinematics
    theta = dataset.meta["theta"]
    phi = dataset.meta["phi"]
    r = np.linalg.norm([x_c, y_c, z_c], axis=0) 
    
    x_pred = r * np.sin(theta) * np.cos(phi)
    y_pred = r * np.sin(theta) * np.sin(phi)
    z_pred = r * np.cos(theta)


    # compute angle and distance between predicted and actual direction
    norm_c = np.linalg.norm([x_c, y_c, z_c], axis=0)
    norm_pred = np.linalg.norm([x_pred, y_pred, z_pred], axis=0)

    dot = x_c*x_pred + y_c*y_pred + z_c*z_pred
    cos_angle = dot / (norm_c * norm_pred)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    dataset.meta["ang_misalign"] = angle*180/np.pi
    dataset.meta["pos_misalign"] = np.linalg.norm([x_pred-x_c, y_pred-y_c, z_pred-z_c], axis=0)
    



def compute_basis(dataset: CaloSimDataset) -> None: 

    """
    Construct an orthonormal local coordinate basis aligned with the
    incident particle direction.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing incident particle spherical coordinates.

    Requires
    --------
    dataset.meta["theta"]
    dataset.meta["phi"]

    Mutates
    -------
    dataset.meta["basis"]
        Local orthonormal basis vectors per event with shape (N, 3, 3),
        where columns correspond to (x_hat, y_hat, z_hat).

    Method
    ------
    1. Construct z_hat from spherical coordinates:
       z_hat = (sinθ cosφ, sinθ sinφ, cosθ)

    2. Build a stable reference vector orthogonal to z_hat.

    3. Compute:
       x_hat = normalize(ref × z_hat)
       y_hat = z_hat × x_hat

    4. Stack basis as:
       basis = [x_hat, y_hat, z_hat]

    """

    theta = dataset.meta["theta"]
    phi = dataset.meta["phi"]

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

    unit_length, orthogonal = _check_basis(basis)

    assert np.all(unit_length), "Some vectors are not unit length"
    assert np.all(orthogonal), "Some vectors are not orthogonal"

    dataset.meta["basis"] = basis

    

def _check_basis(basis, tol=1e-6):

    """
    Validate orthonormality of a batch of 3D basis vectors.

    Parameters
    ----------
    basis : np.ndarray
        Array of shape (N, 3, 3) containing basis vectors.
    tol : float
        Numerical tolerance for validation.

    Returns
    -------
    unit_length : np.ndarray
        Boolean mask indicating unit-norm vectors.
    orthogonal : np.ndarray
        Boolean mask indicating orthogonal bases.
    """

    n_samples, vec_dim, n_vectors = basis.shape

    norms = np.linalg.norm(basis, axis=1)
    unit_length = np.all(np.abs(norms - 1) < tol, axis=1) 

    orthogonal = np.ones(n_samples, dtype=bool)

    for i in range(n_vectors-1):
        for j in range(i+1, n_vectors):
            dot = np.sum(basis[:,:,i] * basis[:,:,j], axis=1)  
            orthogonal &= np.abs(dot) < tol

    return unit_length, orthogonal


# check this

def project_coordinates(dataset: CaloSimDataset, inverse: bool = False) -> None:

    """
    Project Cartesian coordinates into a shower-centric basis,
    or reconstruct Cartesian coordinates from the projected space.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing either Cartesian coordinates or projected coordinates.
    inverse : bool, default=False
        If False, projects (x, y, z) → (x_hat, y_hat, z_hat).
        If True, reconstructs (x_hat, y_hat, z_hat) → (x, y, z).

    Requires
    --------
    If inverse = False:
        dataset.data["x"], ["y"], ["z"]
        dataset.meta["basis"]

    If inverse = True:
        dataset.data["x_hat"], ["y_hat"], ["z_hat"]
        dataset.meta["basis"]

    Mutates
    -------
    If inverse = False:
        dataset.data["x_hat"]
        dataset.data["y_hat"]
        dataset.data["z_hat"]

    If inverse = True:
        dataset.data["x"]
        dataset.data["y"]
        dataset.data["z"]

    Method
    ------
    Uses orthonormal basis vectors:

        forward:
            x_hat = x · b_x
            y_hat = x · b_y
            z_hat = x · b_z

        inverse:
            x_hat b_x + y_hat b_y + z_hat b_z

    """

    _, counts = np.unique(dataset.data["idx"], return_counts=True) # likely this?
    basis = np.repeat(dataset.meta["basis"], counts, axis=0)

    if not inverse:

        x = dataset.data["x"]
        y = dataset.data["y"]
        z = dataset.data["z"]
        coords = np.stack([x, y, z], axis=1)

        dataset.data["x_hat"] = np.sum(coords * basis[:,:,0], axis=1)
        dataset.data["y_hat"] = np.sum(coords * basis[:,:,1], axis=1)
        dataset.data["z_hat"] = np.sum(coords * basis[:,:,2], axis=1)

    else:

        x_hat = dataset.data["x_hat"]
        y_hat = dataset.data["y_hat"]
        z_hat = dataset.data["z_hat"]

        coords_hat = np.stack([x_hat, y_hat, z_hat], axis=1)
        coords = (coords_hat[:,0,None] * basis[:,:,0] + coords_hat[:,1,None] * basis[:,:,1] + coords_hat[:,2,None] * basis[:,:,2])
        
        dataset.data["x"] = coords[:,0]
        dataset.data["y"] = coords[:,1]
        dataset.data["z"] = coords[:,2]



def compute_retention(dataset: CaloSimDataset, box_size: int) -> None:
    
    """
    Compute step retention per event after applying an XY geometric cut.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset. Must include:
        - dataset.data["idx"]
        - dataset.data["x_hat"]
        - dataset.data["y_hat"]

    box_size : int
        Side length of the square selection window in (x_hat, y_hat) space.
        The cut is defined as:
            |x_hat| <= box_size / 2
            |y_hat| <= box_size / 2

    Requires
    --------
    dataset.data["idx"]
    dataset.data["x_hat"], dataset.data["y_hat"]

    Mutates
    -------
    dataset.meta["retention_pct"]
        Percentage of steps retained per event after applying the XY cut.
        Defined as:
            retained_steps / original_steps * 100

    Notes
    -----
    - This metric is step-based, not energy-weighted.
    """
    
    _, steps_before = np.unique(dataset.data["idx"], return_counts=True)




    data_filtered = filter_by_xy_box(dataset.data, box_size)




    idxs, steps_after = np.unique(data_filtered["idx"], return_counts=True)

    steps_after_full = np.zeros(len(steps_before))
    steps_after_full[idxs] = steps_after
    retention_pct = steps_after_full/steps_before*100 

    dataset.meta["retention_pct"] = retention_pct




def compute_transverse_radius(dataset: CaloSimDataset) -> None:

    """
    Compute transverse radial distance in projected shower-centric coordinate system.

    Parameters
    ----------
    dataset : CaloSimDataset
        Dataset containing projected coordinates.

    Requires
    --------
    dataset.data["x_hat"] 
    dataset.data["y_hat"]

    Mutates
    -------
    dataset.data["r_hat"]
    """

    x_hat = dataset.data["x_hat"]
    y_hat = dataset.data["y_hat"]
    r_hat = np.sqrt(x_hat**2 + y_hat**2)
    dataset.data["r_hat"] = r_hat


@dataclass(frozen=True)
class DetectorGeometry:
    # barrel (mm)
    r_barrel_inner: float = 1250
    r_barrel_outer: float = 1500
    z_barrel_min: float = 0
    z_barrel_max: float = 3050

    # endcap (mm)
    r_endcap_inner: float = 315
    r_endcap_outer: float = 1500
    z_endcap_min: float = 3200
    z_endcap_max: float = 3450



def classify_impact_regions(theta, geometry):

    """
    Classify detector entry region from indicent particle direction.

    Parameters
    ----------
    theta : np.ndarray
    geometry : DetectorGeometry
        Detector geometry definition.

    Returns
    -------
    in_barrel : np.ndarray
        Boolean mask for particles entering through the barrel surface.
    in_endcap_front : np.ndarray
        Boolean mask for particles entering through the endcap front face.
    in_endcap_inner : np.ndarray
        Boolean mask for particles entering through the endcap inner radius.

    Method
    ------
    1. Fold detector symmetry around pi/2:
       theta -> min(theta, pi - theta)

    2. Compute transition angles:
       theta_1 = atan(r_barrel_inner / z_barrel_max)
       theta_2 = atan(r_endcap_inner / z_endcap_min)

    3. Classify each trajectory according to the first detector
       surface intersected by the particle.

    """

    # change arctan to atan

    theta_1 = np.atan(geometry.r_barrel_inner / geometry.z_barrel_max) # from barrel to endcap front
    theta_2 = np.atan(geometry.r_endcap_inner / geometry.z_endcap_min) # from endcap front to endcap inner surface

    in_barrel = theta >= theta_1
    in_endcap_front = (theta >= theta_2) & (theta < theta_1)
    in_endcap_inner = theta < theta_2

    return in_barrel, in_endcap_front, in_endcap_inner


# check this

def compute_detector_distances(dataset: CaloSimDataset) -> None:

    """
    Compute detector entry and exit distances along the incident particle trajectory.

    Parameters
    ----------
    dataset : CaloSimDataset
    geometry : DetectorGeometry
        Detector geometry definition.

    Requires
    --------
    dataset.meta["theta"]
        Incident particle polar angle (radians).

    Mutates
    -------
    dataset.meta["entry_dist"]
        Distance from the origin to the detector entry surface (mm).
    dataset.meta["exit_dist"]
        Distance from the origin to the detector exit surface (mm).

    Notes
    -----
    Numerical stability is ensured near theta = pi/2 by avoiding
    division by very small cosine values.
    """

    geometry = DetectorGeometry()
    theta = dataset.meta["theta"]
    theta = np.abs(theta)
    theta = np.where(theta > np.pi / 2, np.pi - theta, theta)

    in_barrel, in_endcap_front, in_endcap_inner = classify_impact_regions(theta, geometry)

    entry = np.zeros_like(theta)

    entry[in_barrel] = geometry.r_barrel_inner / np.sin(theta[in_barrel])
    entry[in_endcap_front] = geometry.z_endcap_min / np.cos(theta[in_endcap_front])
    entry[in_endcap_inner] = geometry.r_endcap_inner / np.sin(theta[in_endcap_inner])

    dataset.meta["entry_dist"] = entry

    eps = 1e-7
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    barrel_exit = geometry.r_barrel_outer / sin_theta
    endcap_exit = np.full_like(theta, np.inf)

    valid = np.abs(cos_theta) >= eps
    endcap_exit[valid] = geometry.z_endcap_max / cos_theta[valid]

    dataset.meta["exit_dist"] = np.minimum(barrel_exit, endcap_exit)



# check this

def shift_z_hat(dataset: CaloSimDataset, inverse: bool = False) -> None:


    """
    Shift longitudinal coordinates relative to the detector entry point.

    Parameters
    ----------
    dataset : CaloSimDataset
    inverse : bool, default=False
        If False, shift coordinates such that the detector entry point is
        located at z_hat = 0. If True, restore the original coordinates.

    Requires
    --------
    dataset.data["z_hat"]
    dataset.data["idx"]
    dataset.meta["entry_dist"]

    Mutates
    -------
    dataset.data["z_hat"]

    """
    z_hat = dataset.data["z_hat"]
    idxs = dataset.data["idx"]

    dist = dataset.meta["entry_dist"][idxs]

    sign = 1 if inverse else -1
    z_hat = z_hat + sign * dist
    dataset.data["z_hat"] = z_hat


def compute_energy_sum(dataset: CaloSimDataset) -> None:

    """
    Compute total deposited energy per event.

    Parameters
    ----------
    dataset : CaloSimDataset

    Requires
    --------
    dataset.data["e"]
    dataset.data["eid"]

    Mutates
    -------
    dataset.meta["e_sum"]
        Total deposited energy for each event.

    """

    e = dataset.data["e"]
    idxs = dataset.data["idx"]

    dataset.meta["e_sum"] = np.bincount(idxs, weights=e)



def compute_geometric_features(dataset: CaloSimDataset, inverse: bool = False) -> None:

    """
    Compute or reconstruct geometric features.

    Parameters
    ----------
    dataset : CaloSimDataset
    inverse : bool, default=False
        If False, compute derived geometric features from incident
        particle kinematics. If True, reconstruct global coordinates
        from transformed coordinates.

    Requires
    --------
    Forward mode:
        dataset.meta["p_x"], ["p_y"], ["p_z"]
        dataset.data["x"], ["y"], ["z"]

    Inverse mode:
        dataset.meta["theta"], ["phi"]
        dataset.data["x_hat"], ["y_hat"], ["z_hat"]

    Mutates
    -------
    Forward mode:
        dataset.meta["theta"]
        dataset.meta["phi"]
        dataset.meta["e_inc"]
        dataset.meta["basis"]
        dataset.meta["entry_dist"]
        dataset.meta["exit_dist"]

        dataset.data["x_hat"]
        dataset.data["y_hat"]
        dataset.data["z_hat"]
        dataset.data["r_hat"]

    Inverse mode:
        dataset.data["x"]
        dataset.data["y"]
        dataset.data["z"]
        dataset.meta["basis"]

        dataset.meta["x_c"]
        dataset.meta["y_c"]
        dataset.meta["z_c"]
        dataset.meta["ang_misalign"]
        dataset.meta["pos_misalign"]
    """


    if not inverse:
        compute_spherical_coords(dataset) 
        compute_incident_energy(dataset)
        compute_basis(dataset)
        compute_detector_distances(dataset)
        project_coordinates(dataset)
        shift_z_hat(dataset)
        compute_transverse_radius(dataset)

    else:
        dataset.data["d"] = np.repeat(str(0), len(dataset.data["eid"])) # change this 
        compute_basis(dataset)
        shift_z_hat(dataset, inverse=True)
        project_coordinates(dataset, inverse=True)
        shift_z_hat(dataset)
        compute_centroids(dataset)
        compute_misalignment(dataset)
