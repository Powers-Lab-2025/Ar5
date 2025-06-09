import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from .analysis import guess_column_positions

def plot_xy_trajectories(com_traj, fragment_indices, box, verbose=False):
    if not verbose:
        return

    box_xy = np.array(box[:2])
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab10", len(fragment_indices))

    for i, frag_idx in enumerate(fragment_indices):
        xy = com_traj[:, frag_idx, :2].copy()

        unwrapped = [xy[0]]
        for j in range(1, len(xy)):
            delta = xy[j] - xy[j - 1]
            delta -= box_xy * np.round(delta / box_xy)
            unwrapped.append(unwrapped[-1] + delta)

        unwrapped = np.array(unwrapped)

        plt.plot(unwrapped[:, 0], unwrapped[:, 1], label=f"Fragment {frag_idx}", color=colors(i))
        plt.scatter(unwrapped[0, 0], unwrapped[0, 1], color=colors(i), marker='o', s=40, label=f"Start {frag_idx}")
        plt.scatter(unwrapped[-1, 0], unwrapped[-1, 1], color=colors(i), marker='x', s=60, label=f"End {frag_idx}")

    plt.xlabel("X A")
    plt.ylabel("Y A")
    plt.title("XY Trajectories of Selected Fragments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_single_fragment_trajectory_with_anchors(com_traj, frag_idx, box, anchors, verbose=False):
    if not verbose:
        return

    box_xy = np.array(box[:2])
    xy = com_traj[:, frag_idx, :2].copy()

    unwrapped = [xy[0]]
    for j in range(1, len(xy)):
        delta = xy[j] - xy[j - 1]
        delta -= box_xy * np.round(delta / box_xy)
        unwrapped.append(unwrapped[-1] + delta)
    unwrapped = np.array(unwrapped)

    origin = unwrapped[0]
    shifted = unwrapped - origin
    shifted_anchors = (anchors - origin + box_xy / 2) % box_xy - box_xy / 2

    n_frames = len(shifted)
    cmap = cm.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=0, vmax=n_frames - 1)
    colors = [cmap(norm(i)) for i in range(n_frames)]

    plt.figure(figsize=(8, 6))

    for i in range(n_frames - 1):
        plt.plot(shifted[i:i+2, 0], shifted[i:i+2, 1], color=colors[i], linewidth=2)

    plt.scatter(0, 0, color='lime', marker='o', s=60, label="Start")
    plt.scatter(shifted[-1, 0], shifted[-1, 1], color='red', marker='x', s=60, label="End")

    for i, (x, y) in enumerate(shifted_anchors):
        plt.scatter(x, y, c='black', marker='x', s=80)
        plt.text(x + 0.3, y + 0.3, f'C{i+1}', fontsize=9, color='black')

    plt.title(f"Trajectory of Fragment {frag_idx}!")
    plt.xlabel("Relative X A")
    plt.ylabel("Relative Y A")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()