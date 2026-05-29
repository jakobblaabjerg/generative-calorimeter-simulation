import numpy as np 
from src.data_processing import CalorimeterSimDataset, filter_by_xy_box


def compute_spherical_coords(dataset: CalorimeterSimDataset) -> None:
    """ 
    Compute spherical coordinates from incident momentum vectors. 
    
    Parameters 
    ---------- 
    dataset : CalorimeterSimDataset 
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



def compute_incident_energy(dataset: CalorimeterSimDataset) -> None:
    """
    Compute incident particle energy from momentum components.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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



def compute_centroids(dataset: CalorimeterSimDataset) -> None:
    """
    Compute energy-weighted centroids of event.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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

    eids = dataset.data["eid"] 
    _, inverse = np.unique(eids, return_inverse=True)

    x = dataset.data["x"]
    y = dataset.data["y"]
    z = dataset.data["z"]
    E = dataset.data["e"]

    E_tot = np.bincount(inverse, weights=E)
    
    x_weighted_sum = np.bincount(inverse, weights=x*E)
    y_weighted_sum = np.bincount(inverse, weights=y*E)
    z_weighted_sum = np.bincount(inverse, weights=z*E)

    dataset.meta["x_c"] = x_weighted_sum / E_tot
    dataset.meta["y_c"] = y_weighted_sum / E_tot
    dataset.meta["z_c"] = z_weighted_sum / E_tot




def compute_misalignment(dataset: CalorimeterSimDataset) -> None:

    """
    Compute geometric misalignment between centroid direction
    and expected direction from incident particle kinematics.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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
    



def compute_basis(dataset: CalorimeterSimDataset) -> None: 

    """
    Construct an orthonormal local coordinate basis aligned with the
    incident particle direction.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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


def project_coordinates(dataset: CalorimeterSimDataset, inverse: bool = False) -> None:

    """
    Project Cartesian coordinates into a shower-centric basis,
    or reconstruct Cartesian coordinates from the projected space.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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

    _, counts = np.unique(dataset.data["eid"], return_counts=True)
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



def compute_retention(dataset: CalorimeterSimDataset, box_size: int) -> None:
    
    """
    Compute step retention per event after applying an XY geometric cut.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
        Dataset. Must include:
        - dataset.data["eid"]
        - dataset.data["x_hat"]
        - dataset.data["y_hat"]

    box_size : int
        Side length of the square selection window in (x_hat, y_hat) space.
        The cut is defined as:
            |x_hat| <= box_size / 2
            |y_hat| <= box_size / 2

    Requires
    --------
    dataset.data["eid"]
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
    
    _, steps_before = np.unique(dataset.data["eid"], return_counts=True)
    data_filtered = filter_by_xy_box(dataset.data, box_size)

    eids, steps_after = np.unique(data_filtered["eid"], return_counts=True)

    steps_after_full = np.zeros(len(steps_before))
    steps_after_full[eids] = steps_after
    retention_pct = steps_after_full/steps_before*100 

    dataset.meta["retention_pct"] = retention_pct




def compute_transverse_radius(dataset: CalorimeterSimDataset) -> None:

    """
    Compute transverse radial distance in projected shower-centric coordinate system.

    Parameters
    ----------
    dataset : CalorimeterSimDataset
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