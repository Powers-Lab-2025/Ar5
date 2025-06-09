import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import DBSCAN
import MDAnalysis as mda


def guess_column_positions(coms, box, eps=2, min_samples=5, verbose=False, return_noise=False):
    xy_coords = coms[:, :2]
    box_xy = box[:2]
    xy_wrapped = xy_coords % box_xy

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_wrapped)
    labels = db.labels_

    n_total = len(labels)
    n_noise = np.sum(labels == -1)

    if verbose:
        if n_noise > 0:
            print(f"{n_noise} out of {n_total} COMs were not assigned to any column (DBSCAN noise).")
        else:
            print("All COMs were assigned to columns.")

    column_anchors = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_points = xy_wrapped[labels == label]
        anchor = np.mean(cluster_points, axis=0)
        column_anchors.append(anchor)

    if verbose:
        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for color, label in zip(colors, unique_labels):
            mask = labels == label
            if label == -1:
                plt.scatter(xy_wrapped[mask, 0], xy_wrapped[mask, 1], c='gray', label='Noise', alpha=0.3)
            else:
                plt.scatter(xy_wrapped[mask, 0], xy_wrapped[mask, 1], c=[color], label=f'Column {label + 1}', alpha=0.6)

        for i, (x, y) in enumerate(column_anchors):
            plt.scatter(x, y, c='black', s=80, marker='x')
            plt.text(x + 0.3, y + 0.3, f'C{i+1}', fontsize=10, color='black')

        plt.title("Predicted Column Locations from COM Clustering")
        plt.xlabel("X A")
        plt.ylabel("Y A")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if return_noise:
        return np.array(column_anchors), labels
    else:
        return np.array(column_anchors)



def assign_fragments_to_columns(coms, column_anchors, box, verbose=False):
    xy_coms = coms[:, :2]
    box_xy = box[:2]
    xy_wrapped = xy_coms % box_xy

    assigned_columns = []
    for com in xy_wrapped:
        dists = np.linalg.norm((column_anchors - com + box_xy / 2) % box_xy - box_xy / 2, axis=1)
        assigned_columns.append(np.argmin(dists))

    assigned_columns = np.array(assigned_columns)

    if verbose:
        counts = np.bincount(assigned_columns)
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(counts)), counts, color='skyblue', edgecolor='black')
        plt.xlabel("Column Index")
        plt.ylabel("Number of Fragments")
        plt.title("Fragment Assignment to Columns")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    nearest_dists = []
    for i, anchor in enumerate(column_anchors):
        dists = np.linalg.norm((column_anchors - anchor + box_xy / 2) % box_xy - box_xy / 2, axis=1)
        dists[i] = np.inf
        nearest_dists.append(np.min(dists))

    r = np.mean(nearest_dists)

    if verbose:
        print(f"Average inter-column spacing r = {r:.3f} A")

    return assigned_columns, r


def detect_column_hops(filepaths, anchors, home_assignments, r, box):
    hop_events = []
    box_xy = box[:2]
    cutoff = 0.45 * r

    hopped_fragments = set()
    bar = tqdm.tqdm(filepaths, desc="Analyzing trajectory frames")

    for frame_num, filepath in enumerate(bar):
        u = mda.Universe(filepath, format="TXYZ")
        fragments = list(u.atoms.fragments)
        coms = np.array([frag.center_of_mass() for frag in fragments])
        xy_coms = coms[:, :2]

        for i, (com, home_idx) in enumerate(zip(xy_coms, home_assignments)):
            if i in hopped_fragments:
                continue

            home_anchor = anchors[home_idx]
            d_home = np.linalg.norm((com - home_anchor + box_xy / 2) % box_xy - box_xy / 2)

            for j, anchor in enumerate(anchors):
                if j == home_idx:
                    continue
                d_other = np.linalg.norm((com - anchor + box_xy / 2) % box_xy - box_xy / 2)
                if d_other < d_home and d_other < cutoff:
                    hop_events.append({
                        "fragment_index": i,
                        "frame": frame_num,
                        "from_column": home_idx,
                        "to_column": j,
                        "distance_to_new": round(d_other, 3),
                        "distance_to_home": round(d_home, 3),
                        "file": os.path.basename(filepath)  # store filename for compatibility
                    })
                    hopped_fragments.add(i)
                    break

        bar.set_postfix(hops=len(hop_events))

    return hop_events


def traj_com(filepaths):
    com_traj = []
    bar = tqdm.tqdm(filepaths, desc="Reading COMs")

    for filepath in bar:
        u = mda.Universe(filepath, format="TXYZ")
        fragments = list(u.atoms.fragments)
        coms = [frag.center_of_mass() for frag in fragments]
        com_traj.append(coms)

    return np.array(com_traj)


def compute_orientational_order_parameter(frags, vec_out=False, verbose=False):
    axes = [frag.principal_axes()[0] for frag in frags]
    ni = np.array(axes).T
    A = sum(3 * np.outer(ni[:, i], ni[:, i]) - np.identity(3)
            for i in range(ni.shape[1])) / (2 * ni.shape[1])
    evals, evecs = np.linalg.eigh(A)
    idx = np.argmax(evals)
    director = evecs[:, idx]
    if vec_out:
        upQ = np.mean(np.sign(ni.T @ director))
        director = director * (1 if upQ >= 0 else -1)
        if verbose: print(director)
        return director
    S = evals[idx]
    if verbose: print(f"{S:.4f}")
    return S