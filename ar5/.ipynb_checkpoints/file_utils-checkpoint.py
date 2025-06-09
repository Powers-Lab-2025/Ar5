import os
import re
import numpy as np
import tqdm


def dyn_to_xyz(dyn_path, ref_prefix, verbose=False):
    with open(dyn_path, 'r') as file:
        lines = file.readlines()
    if len(lines) < 6:
        return -1

    tmp = lines[1].split()
    num_atoms = int(tmp[0])
    title = lines[1][len(tmp[0])+1:] if len(tmp) > 1 else ""

    if verbose:
        print(f'num_atoms: {num_atoms}\n{title.strip()}\n')

    sdims = lines[3].split()
    DIMS = []
    for s in sdims:
        base, exp = s.split('D')
        DIMS.append(float(base) * 10**float(exp))

    angs = []
    for s in lines[4].split():
        base, exp = s.split('D')
        angs.append(float(base) * 10**float(exp))

    coords = np.zeros((3, num_atoms), dtype=np.float64)
    for ai in range(num_atoms):
        line = lines[6 + ai].split()
        for i in range(3):
            base, exp = line[i].split('D')
            coords[i, ai] = float(base) * 10**float(exp)

    base_dir = os.path.dirname(dyn_path)
    ref_path = os.path.join(base_dir, f"{ref_prefix}.xyz")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference .xyz file not found: {ref_path}")

    with open(ref_path, 'r') as ref_file:
        ref_lines = ref_file.readlines()
    ref_num_atoms = int(ref_lines[0].strip().split()[0])
    if ref_num_atoms != num_atoms:
        raise ValueError(f"Atom count mismatch: dyn={num_atoms}, ref_xyz={ref_num_atoms}")

    header = ref_lines[1]
    atom_lines = ref_lines[2:2 + num_atoms]

    out_path = os.path.join(base_dir, f"{ref_prefix}_dync.xyz")
    with open(out_path, 'w') as out:
        out.write(f"{num_atoms}\n{header}")
        for i in range(num_atoms):
            parts = atom_lines[i].strip().split()
            index = parts[0]
            atom_name = parts[1]
            mol_id = parts[5]
            atom_type = parts[6]
            bonds = parts[7:]
            bond_str = "  " + "  ".join(bonds) if bonds else ""
            out.write(f"{index:>5s}  {atom_name:<2s}  {coords[0,i]:12.6f}  {coords[1,i]:12.6f}  {coords[2,i]:12.6f}  {mol_id:>5s}  {atom_type:>5s}{bond_str}\n")

    if verbose:
        print(f"Wrote TXYZ file with bonds to {out_path}")

    return num_atoms, DIMS

def batch_convert_dyn_to_xyz(simtree_root, verbose=False):

    dyn_paths = []

    for mol in sorted(os.listdir(simtree_root)):
        mol_path = os.path.join(simtree_root, mol)
        if not os.path.isdir(mol_path) or not mol.startswith("F"):
            continue
        for temp in sorted(os.listdir(mol_path)):
            temp_path = os.path.join(mol_path, temp, "data")
            if not os.path.isdir(temp_path):
                continue
            for iteration in sorted(os.listdir(temp_path)):
                iter_path = os.path.join(temp_path, iteration)
                if not os.path.isdir(iter_path):
                    continue
                dyn_file = os.path.join(iter_path, f"{mol}_ramp.dyn")
                if os.path.exists(dyn_file):
                    dyn_paths.append((dyn_file, mol))

    bar = tqdm.tqdm(dyn_paths, desc="Converting .dyn to .xyz")

    for dyn_file, mol in bar:
        try:
            dyn_to_xyz(dyn_path=dyn_file, ref_prefix=f"{mol}_ramp", verbose=verbose)
        except Exception as e:
            print(f"Failed to convert: {dyn_file}\n{e}")

def convert_dync_atom_labels(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_dync.xyz"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] in {"CA", "HA"}:
                        if parts[1] == "CA":
                            parts[1] = "C"
                        elif parts[1] == "HA":
                            parts[1] = "H"
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)

                with open(file_path, "w") as f:
                    f.writelines(new_lines)

                print(f"Converted: {file_path}")


def combine_xyz_frames(simtree_root="/Users/alforest/Desktop/Documents/SomeSeeds/SimTree", use_dync=True, verbose=False):

    mol_dirs = sorted([d for d in os.listdir(simtree_root) if d.startswith("F") and os.path.isdir(os.path.join(simtree_root, d))])

    for mol in tqdm.tqdm(mol_dirs, desc="Combining", position=0, leave=True):
        mol_path = os.path.join(simtree_root, mol)
        temp_dirs = sorted([t for t in os.listdir(mol_path) if os.path.isdir(os.path.join(mol_path, t))])

        for temp in temp_dirs:
            data_path = os.path.join(mol_path, temp, "data")
            if not os.path.isdir(data_path):
                continue

            iter_dirs = sorted([i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, i))])

            combined_lines = []
            n_atoms = None

            for frame_num, iter_dir in enumerate(iter_dirs):

                xyz_basename = f"{mol}_ramp_dync.xyz" if use_dync else f"{mol}_ramp.xyz"
                xyz_path = os.path.join(data_path, iter_dir, xyz_basename)
                if not os.path.exists(xyz_path):
                    continue

                with open(xyz_path, 'r') as infile:
                    lines = infile.readlines()

                n_atoms_in_file = int(lines[0].strip().split()[0])
                if n_atoms is None:
                    n_atoms = n_atoms_in_file
                else:
                    if n_atoms_in_file != n_atoms:
                        raise ValueError(f"Inconsistent atom count in {xyz_path}")

                combined_lines.append(f"{n_atoms}\nFrame {frame_num + 1}\n")
                for line in lines[2:2 + n_atoms]:
                    parts = line.strip().split()
                    atom_name = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    combined_lines.append(f"{atom_name} {x:.6f} {y:.6f} {z:.6f}\n")


            out_name = f"{mol}_ramp_combined_dync2.xyz" if use_dync else f"{mol}_ramp_combined.xyz"
            out_path = os.path.join(mol_path, temp, out_name)

            with open(out_path, 'w') as outfile:
                outfile.writelines(combined_lines)

            if verbose:
                print(f"Wrote combined XYZ to {out_path}")