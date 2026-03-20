import matplotlib.pyplot as plt
import numpy as np 
from src.data_processing import filter_by_eid, filter_by_xy_box


def plot_hist(data, var, bins=200):

    plt.hist(data[var], bins=bins, histtype="step")
    plt.xlabel(var)
    plt.show()

def plot_retainment(data, start=20, stop=220, step=20):

    avg_pct_steps, q5_pct_steps = [], [] 
    avg_pct_e, q5_pct_e = [], []

    box_sizes = []

    eids = data["eid"]
    eids_unique, steps_before = np.unique(eids, return_counts=True)

    e = data["e"]
    e_tot_before = np.bincount(eids, weights=e)

    x_hat = np.abs(data["x_hat"])
    y_hat = np.abs(data["y_hat"])

    for box_size in range(start, stop, step):

        # create mask
        mask = (x_hat <= box_size/2) & (y_hat <= box_size/2)
        idx, steps_after = np.unique(eids[mask], return_counts=True)

        # compute steps retained
        steps_after_full = np.zeros(len(steps_before))
        steps_after_full[idx] = steps_after
        steps_retained_pct = steps_after_full/steps_before*100

        # compute energy retained
        e_tot_after = np.bincount(eids, weights=e*mask) 
        e_retained_pct = e_tot_after/e_tot_before*100
     
        avg_pct_steps.append(np.mean(steps_retained_pct))
        q5_pct_steps.append(np.percentile(steps_retained_pct, 5))       
        avg_pct_e.append(np.mean(e_retained_pct))
        q5_pct_e.append(np.percentile(e_retained_pct, 5))        
        box_sizes.append(box_size)

    plt.figure(figsize=(10, 6))

    plt.plot(box_sizes, q5_pct_steps, label="5% quantile (points)", marker="o")
    plt.plot(box_sizes, q5_pct_e, label="5% quantile (energy)", marker="o")

    plt.plot(box_sizes, avg_pct_steps, label="avg (points)", marker="o")
    plt.plot(box_sizes, avg_pct_e, label="avg (energy)", marker="o")
        

    plt.hlines(y=90, xmin=box_sizes[0], xmax=box_sizes[-1], colors="red", linestyles=":", linewidth=2, label="y=90%")

    # plt.xlim([box_sizes[0], box_sizes[-1]])

    plt.xlabel("size [mm]", fontsize=16)
    plt.ylabel("retention (per event) [%]", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()



def plot_shower_3d(data, meta, eid):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)

    x_c = meta_filtered["x_c"][0]
    y_c = meta_filtered["y_c"][0]
    z_c = meta_filtered["z_c"][0]

    x = data_filtered["x"]
    y = data_filtered["y"]
    z = data_filtered["z"]

    theta = meta_filtered["theta"][0]
    phi = meta_filtered["phi"][0]
    e_incident = meta_filtered["e_incident"][0]

    r = np.linalg.norm([x_c, y_c, z_c])
    x_true = r * np.sin(theta) * np.cos(phi)
    y_true = r * np.sin(theta) * np.sin(phi)
    z_true = r * np.cos(theta)

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')

    # plot origin
    ax.scatter(0, 0, 0, color='red', s=100, label='Origin')

    # plot energy-weighted position
    ax.scatter(x_c, y_c, z_c, color='blue', s=100, label='Centroid')

    # plot steps
    sd = data_filtered["sd"]
    unique_sd = np.unique(sd)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sd)))

    for detector, color in zip(unique_sd, colors):
        mask = sd == detector
        ax.scatter(x[mask], y[mask], z[mask], color=color, s=1, alpha=0.6, label=detector)
    
    # plot direction 
    ax.plot([0, x_c], [0, y_c], [0, z_c], color='blue', linestyle='--', label="pred")
    ax.plot([0, x_true], [0, y_true], [0, z_true], color='green', linestyle='--', label="true")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(rf"Event {eid} (e={e_incident:.2f}, $\theta$={theta* 180/np.pi:.2f}, $\phi$={phi* 180/np.pi:.2f})")

    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    # ax.legend()
    plt.show()


def plot_shower_2d(data, meta, box_size=100, events=6, prj="XY", color_by="energy", seed=42):

    prj = [coord for coord in prj.lower()]

    if type(events)==int:
        np.random.seed(seed)
        unique_events = np.unique(data["eid"])
        selected_events = np.random.choice(unique_events, size=events, replace=False)

    elif type(events)==list:
        selected_events = events
        events = len(selected_events)

    cols = 3
    rows = (events+(cols-1)) // cols
 
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flatten()

    for i in range(len(selected_events)):

        eid = selected_events[i]
        ax = axes[i]

        data_filtered, meta_filtered = filter_by_eid(data, meta, eid)
        data_filtered = filter_by_xy_box(data_filtered, box_size)

        x_hat = data_filtered[f"{prj[0]}_hat"]
        y_hat = data_filtered[f"{prj[1]}_hat"]
        e = data_filtered["e"]


        theta = meta_filtered["theta"][0] * 180/np.pi
        phi = meta_filtered["phi"][0] * 180/np.pi
        e_incident = meta_filtered["e_incident"][0]

        if color_by == "energy":            
            sc = ax.scatter(x_hat, y_hat, c=e, cmap="viridis", s=20)
            fig.colorbar(sc, ax=ax, label="Energy [GeV]", shrink=0.8)


        elif color_by == "detector":

            sd = data_filtered["sd"]
            unique_sd = np.unique(sd)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sd)))

            for detector, color in zip(unique_sd, colors):
                mask = sd == detector
                ax.scatter(x_hat[mask], y_hat[mask], color=color, s=20, label=detector)

            ax.legend()
        
        else:
            raise ValueError

        ax.set_title(rf"Event {eid} (e={e_incident:.2f}, $\theta$={theta:.2f}, $\phi$={phi:.2f})")

        if prj[0] != "z":
            ax.set_xlim([-box_size/2, box_size/2])
        
        if prj[1] != "z":
            ax.set_ylim([-box_size/2, box_size/2])
        
        ax.set_xlabel(f"{prj[0].upper()} [mm]")
        ax.set_ylabel(f"{prj[1].upper()} [mm]")
        ax.set_aspect("equal")
        

   
    for ax in axes[len(selected_events):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_title(meta):

    eid = meta["eid"][0]
    theta = meta["theta"][0] * 180/np.pi
    phi = meta["phi"][0] * 180/np.pi
    e_incident = meta["e"][0]

    return rf"Event {eid} (e={e_incident:.2f}, $\theta$={theta:.2f}, $\phi$={phi:.2f})"


def plot_time_vs_energy(data, meta, eid, grouped_by_pid=False):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)

    title = get_title(meta_filtered)

    plt.figure(figsize=(10, 6))

    time = data_filtered["t"]
    time_shifted = time - np.min(time)


    if grouped_by_pid:

        pids = data_filtered["pid"]
        unique_pids, inverse = np.unique(pids, return_inverse=True)

        energy = np.zeros(len(unique_pids))
        np.add.at(energy, inverse, data_filtered["e"])


        time_min = np.full(len(unique_pids), np.inf)
        np.minimum.at(time_min, inverse, time_shifted)
        time = time_min

    else: 
        energy = data_filtered["e"]
        time = time_shifted


    esum = energy.sum()
    efrac = energy/esum*100


    plt.scatter(time, efrac)

    plt.xlabel("time [ns]")
    plt.ylabel("energy fraction [%]")
    plt.title(title)

    plt.show()




def plot_time_vs_z(data, meta, eid):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)

    title = get_title(meta_filtered)

    plt.figure(figsize=(10, 6))

    time = data_filtered["t"]
    time_shifted = time - np.min(time)

    z_hat = data_filtered["z_hat"]

    plt.scatter(time_shifted, z_hat)

    plt.xlabel("time [ns]")
    plt.ylabel("Z [mm]")
    plt.title(title)

    plt.show()


def plot_unique_pid_hist(data, bins=100, density=True):

    eids = data["eid"]
    pids = data["pid"]

    pairs = np.vstack((eids, pids)).T 
    unique_pairs, counts = np.unique(pairs, return_counts=True, axis=0)

    unique_pids_count = np.bincount(unique_pairs[:,0])
    unique_eids, step_count = np.unique(eids, return_counts=True)

    frac = unique_pids_count/step_count*100

    plt.figure(figsize=(8, 5))
    plt.hist(frac, bins=bins, alpha=0.75, density=density)

    plt.xlabel("Unique pid fraction (per event) [%]")
    
    if density:
        plt.ylabel("Density")
    else:
        plt.ylabel("Count")
    
    plt.show()


def plot_radial_profile(data, meta, eid, bins=100, density=False, normalize=False):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)
    title = get_title(meta_filtered)

    x_hat = data_filtered["x_hat"]
    y_hat = data_filtered["y_hat"]

    r = np.sqrt(x_hat**2 + y_hat**2)
    energy = data_filtered["e"]*1000 # unit check!!

    if normalize:
        r = (r - r.min()) / (r.max() - r.min())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].hist(r, bins=bins, alpha=0.75, density=density)
    axes[0].set_xlabel("r [%]" if normalize else "r [mm]")
    axes[0].set_ylabel("Density" if density else "Count")

    # --- Right: Energy-weighted histogram ---
    axes[1].hist(r, bins=bins, alpha=0.75, density=density, weights=energy)
    axes[1].set_xlabel("r [%]" if normalize else "r [mm]")
    axes[1].set_ylabel("Density" if density else "Energy [MeV]")

    plt.show()



def plot_longitudinal_profile(data, meta, eid, bins=100, density=False, normalize=False):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)
    title = get_title(meta_filtered)
    
    z_hat = data_filtered["z_hat"]
    energy = data_filtered["e"]*1000

    if normalize:
        z_hat = (z_hat - z_hat.min()) / (z_hat.max() - z_hat.min())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].hist(z_hat, bins=bins, alpha=0.75, density=density)
    axes[0].set_xlabel("z [%]" if normalize else "z [mm]")
    axes[0].set_ylabel("Density" if density else "Count")

    # --- Right: Energy-weighted histogram ---
    axes[1].hist(z_hat, bins=bins, alpha=0.75, density=density, weights=energy)
    axes[1].set_xlabel("z [%]" if normalize else "z [mm]")
    axes[1].set_ylabel("Density" if density else "Energy [MeV]")

    plt.show()

def plot_radial_longitudinal_profile(data, meta, eid, bins=100, density=False, normalize=False):

    data_filtered, meta_filtered = filter_by_eid(data, meta, eid)
    title = get_title(meta_filtered)
    
    x_hat = data_filtered["x_hat"]
    y_hat = data_filtered["y_hat"]
    z_hat = data_filtered["z_hat"]
    energy = data_filtered["e"]*1000

    r = np.sqrt(x_hat**2 + y_hat**2)


    if normalize:
        z_hat = (z_hat - z_hat.min()) / (z_hat.max() - z_hat.min())
        r = (r - r.min()) / (r.max() - r.min())
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)


    h1 = axes[0].hist2d(z_hat, r, bins=bins, density=density)

    axes[0].set_ylabel("r [%]" if normalize else "r [mm]")
    axes[0].set_xlabel("z [%]" if normalize else "z [mm]")

    fig.colorbar(h1[3], ax=axes[0], label="Density" if density else "Count")

    # --- Right: Energy-weighted histogram ---
    h2 = axes[1].hist2d(z_hat, r, bins=bins, density=density, weights=energy)

    axes[1].set_ylabel("r [%]" if normalize else "r [mm]")
    axes[1].set_xlabel("z [%]" if normalize else "z [mm]")

    fig.colorbar(h2[3], ax=axes[1], label="Density" if density else "Energy [MeV]")

    plt.show()