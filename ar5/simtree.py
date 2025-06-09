import os
import re
import numpy as np
import tqdm
import MDAnalysis as mda

from .analysis import guess_column_positions, compute_orientational_order_parameter

def analyze_simtree_dynamic_fixed(simtree_root="/Users/alforest/Desktop/Documents/SomeSeeds/SimTree", eps=5, min_samples=12, verbose=False):

    hopper_results = []

    mol_dirs = sorted([d for d in os.listdir(simtree_root) if d.startswith("F") and os.path.isdir(os.path.join(simtree_root, d))])

    for mol in mol_dirs:
        mol_path = os.path.join(simtree_root, mol)
        temp_dirs = sorted([t for t in os.listdir(mol_path) if os.path.isdir(os.path.join(mol_path, t))])

        for temp in temp_dirs:
            data_path = os.path.join(mol_path, temp, "data")
            if not os.path.isdir(data_path):
                continue

            iter_dirs = sorted([i for i in os.listdir(data_path) if i.isdigit()], key=lambda x: int(x))
            if not iter_dirs:
                continue

            # Compute S of last frame
            xyz_last = os.path.join(data_path, iter_dirs[-1], f"{mol}_ramp_dync.xyz")
            if not os.path.exists(xyz_last):
                continue

            u_last = mda.Universe(xyz_last, format="TXYZ")
            S = compute_orientational_order_parameter(u_last, vec_out=False, verbose=verbose)
            if S < 0.7:
                if verbose:
                    print(f"Skipping {mol} @ {temp}K (S={S:.3f})")
                continue

            fragment_column_hist = {}
            fragment_noise_frames = {}
            fragment_unwrapped_xy = {}

            bar = tqdm.tqdm(iter_dirs, desc=f"{mol} @ {temp}K", position=0, leave=True)

            prev_xy_coms = None
            box_xy = None

            for frame_num, iter_dir in enumerate(bar):
                xyz_path = os.path.join(data_path, iter_dir, f"{mol}_ramp_dync.xyz")
                if not os.path.exists(xyz_path):
                    continue

                u = mda.Universe(xyz_path, format="TXYZ")
                fragments = list(u.atoms.fragments)
                box = u.dimensions
                box_xy = box[:2]

                coms = np.array([frag.center_of_mass() for frag in fragments])
                xy_coms = coms[:, :2]

                if prev_xy_coms is None:
                    prev_xy_coms = xy_coms.copy()
                    for frag_idx in range(len(fragments)):
                        fragment_unwrapped_xy[frag_idx] = [xy_coms[frag_idx].copy()]
                else:
                    for frag_idx in range(len(fragments)):
                        delta = xy_coms[frag_idx] - prev_xy_coms[frag_idx]
                        delta -= box_xy * np.round(delta / box_xy)
                        new_pos = fragment_unwrapped_xy[frag_idx][-1] + delta
                        fragment_unwrapped_xy[frag_idx].append(new_pos)
                    prev_xy_coms = xy_coms.copy()

                anchors, labels = guess_column_positions(coms, box, eps=eps, min_samples=min_samples, verbose=False, return_noise=True)

                for frag_idx in range(len(fragments)):
                    label = labels[frag_idx]

                    if frag_idx not in fragment_column_hist:
                        fragment_column_hist[frag_idx] = []
                    if frag_idx not in fragment_noise_frames:
                        fragment_noise_frames[frag_idx] = set()

                    if label == -1:
                        fragment_noise_frames[frag_idx].add(frame_num)
                        fragment_column_hist[frag_idx].append(None)
                    else:
                        fragment_column_hist[frag_idx].append(label)

            for frag_idx in fragment_column_hist:
                seq = fragment_column_hist[frag_idx]
                first_col = None
                after_noise_col = None
                seen_noise = False

                for label in seq:
                    if label is not None and first_col is None:
                        first_col = label
                    elif label is None and first_col is not None:
                        seen_noise = True
                    elif label is not None and seen_noise:
                        after_noise_col = label
                        break

                if first_col is not None and after_noise_col is not None and after_noise_col != first_col:
                    unwrapped_xy = np.array(fragment_unwrapped_xy[frag_idx])
                    delta_xy = np.diff(unwrapped_xy, axis=0)
                    distances = np.linalg.norm(delta_xy, axis=1)
                    xy_distance_travelled = np.sum(distances)

                    hopper_results.append({
                        "molecule": mol,
                        "temperature": temp,
                        "fragment_index": frag_idx,
                        "noise_frames": sorted(list(fragment_noise_frames[frag_idx])),
                        "column_sequence": seq,
                        "xy_distance_travelled": xy_distance_travelled
                    })

    return hopper_results