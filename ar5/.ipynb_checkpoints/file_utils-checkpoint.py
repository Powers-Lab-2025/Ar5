import os
import re
import numpy as np
import tqdm


def _dyn_to_xyz(dyn_path, ref_prefix, verbose=False):
    """
    Internal function to convert one .dyn file to _dync.xyz.
    """
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


def convert_dyn_to_xyz(directories, ref_prefix_func=None, verbose=False):
    """
    Batch convert .dyn files to _dync.xyz for a list of iteration directories.

    Parameters:
    - directories: list of iteration directories (containing the .dyn file)
    - ref_prefix_func: function(dir_path) → str, returns ref_prefix to use for that directory
      If None, defaults to taking mol name from parent directory.
    - verbose: bool
    """
    dyn_paths = []

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            print(f"Skipping non-directory: {dir_path}")
            continue

        dyn_file = None
        # Find a .dyn file in this directory:
        for f in os.listdir(dir_path):
            if f.endswith(".dyn"):
                dyn_file = os.path.join(dir_path, f)
                break

        if dyn_file:
            dyn_paths.append(dyn_file)

    bar = tqdm.tqdm(dyn_paths, desc="Converting .dyn to .xyz")

    for dyn_path in bar:
        try:
            # Determine ref_prefix
            if ref_prefix_func is not None:
                ref_prefix = ref_prefix_func(os.path.dirname(dyn_path))
            else:
                # Default → assume mol is grandparent dir, name it as <mol>_ramp
                base_dir = os.path.dirname(dyn_path)
                mol_dir = os.path.basename(os.path.dirname(os.path.dirname(base_dir)))
                ref_prefix = f"{mol_dir}_ramp"

            _dyn_to_xyz(dyn_path=dyn_path, ref_prefix=ref_prefix, verbose=verbose)

        except Exception as e:
            print(f"Failed to convert: {dyn_path}\n{e}")



def convert_dync_atom_labels(directories):
    """
    Convert atom labels CA → C and HA → H in _dync.xyz files within given directories.

    Parameters:
    - directories: list of directory paths to process
    """
    for dir_path in directories:
        if not os.path.isdir(dir_path):
            print(f"Skipping non-directory: {dir_path}")
            continue

        for file in os.listdir(dir_path):
            if file.endswith("_dync.xyz"):
                file_path = os.path.join(dir_path, file)
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



def combine_xyz_frames(directory, prefix, type="xyz", use_dync=True, verbose=False):
    """
    Combine XYZ frames into one XYZ file.

    Parameters:
    - directory: str, root directory
    - prefix: str, file prefix to match
    - type: str, one of "xyz", "numbered", "simtree"
    - use_dync: bool, for simtree type, whether to use _dync.xyz or not
    - verbose: bool
    """
    if type in {"xyz", "numbered"}:
        # Simple case → just combine the list
        filepaths = find_xyz_files(directory, prefix, type=type)

        combined_lines = []
        n_atoms = None

        for frame_num, filepath in enumerate(tqdm.tqdm(filepaths, desc="Combining", position=0, leave=True)):
            with open(filepath, 'r') as infile:
                lines = infile.readlines()

            n_atoms_in_file = int(lines[0].strip().split()[0])
            if n_atoms is None:
                n_atoms = n_atoms_in_file
            else:
                if n_atoms_in_file != n_atoms:
                    raise ValueError(f"Inconsistent atom count in {filepath}")

            combined_lines.append(f"{n_atoms}\nFrame {frame_num + 1}\n")
            for line in lines[2:2 + n_atoms]:
                parts = line.strip().split()
                atom_name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                combined_lines.append(f"{atom_name} {x:.6f} {y:.6f} {z:.6f}\n")

        # Save output
        out_name = f"{prefix}_combined.xyz"
        out_path = os.path.join(directory, out_name)
        with open(out_path, 'w') as outfile:
            outfile.writelines(combined_lines)

        if verbose:
            print(f"Wrote combined XYZ to {out_path}")

    elif type == "simtree":
        # Simtree → per mol/temp
        simtree_results = find_xyz_files(directory, prefix, type="simtree")

        for (mol, temp), filepaths in tqdm.tqdm(simtree_results.items(), desc="Combining SimTree", position=0, leave=True):
            combined_lines = []
            n_atoms = None

            for frame_num, filepath in enumerate(filepaths):
                with open(filepath, 'r') as infile:
                    lines = infile.readlines()

                n_atoms_in_file = int(lines[0].strip().split()[0])
                if n_atoms is None:
                    n_atoms = n_atoms_in_file
                else:
                    if n_atoms_in_file != n_atoms:
                        raise ValueError(f"Inconsistent atom count in {filepath}")

                combined_lines.append(f"{n_atoms}\nFrame {frame_num + 1}\n")
                for line in lines[2:2 + n_atoms]:
                    parts = line.strip().split()
                    atom_name = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    combined_lines.append(f"{atom_name} {x:.6f} {y:.6f} {z:.6f}\n")

            # Save per mol/temp
            out_name = f"{mol}_ramp_combined_dync2.xyz" if use_dync else f"{mol}_ramp_combined.xyz"
            out_path = os.path.join(directory, mol, temp, out_name)
            with open(out_path, 'w') as outfile:
                outfile.writelines(combined_lines)

            if verbose:
                print(f"Wrote combined XYZ to {out_path}")

    else:
        raise ValueError(f"Unsupported type: {type}. Must be 'xyz', 'numbered', or 'simtree'.")



def find_xyz_files(directory, prefix, type="xyz"):
    """
    Find XYZ files in various directory structures.

    Parameters:
    - directory: str, path to the root directory
    - prefix: str, file prefix to match (e.g. "F7_ramp")
    - type: str, one of:
        - "xyz": matches F7_ramp_1.xyz, F7_ramp_2.xyz, etc.
        - "numbered": matches F7_ramp.001, F7_ramp.002, etc.
        - "simtree": returns a dict { (mol, temp): list of filepaths }

    Returns:
    - If type != "simtree": list of filepaths (sorted)
    - If type == "simtree": dict { (mol, temp): list of filepaths }
    """
    if type == "xyz":
        files = sorted(
            f for f in os.listdir(directory)
            if re.match(rf"^{re.escape(prefix)}_\d+\.xyz$", f)
        )
        filepaths = [os.path.join(directory, f) for f in files]
        return filepaths

    elif type == "numbered":
        files = sorted(
            f for f in os.listdir(directory)
            if re.match(rf"^{re.escape(prefix)}\.\d+$", f)
        )
        filepaths = [os.path.join(directory, f) for f in files]
        return filepaths

    elif type == "simtree":
        simtree_results = {}
        mol_dirs = sorted([d for d in os.listdir(directory) if d.startswith("F") and os.path.isdir(os.path.join(directory, d))])

        for mol in mol_dirs:
            mol_path = os.path.join(directory, mol)
            temp_dirs = sorted([t for t in os.listdir(mol_path) if os.path.isdir(os.path.join(mol_path, t))])

            for temp in temp_dirs:
                data_path = os.path.join(mol_path, temp, "data")
                if not os.path.isdir(data_path):
                    continue

                iter_dirs = sorted([i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, i))])

                filepaths = []
                for iter_dir in iter_dirs:
                    xyz_path = os.path.join(data_path, iter_dir, f"{mol}_ramp_dync.xyz")
                    if os.path.exists(xyz_path):
                        filepaths.append(xyz_path)

                simtree_results[(mol, temp)] = filepaths

        return simtree_results

    else:
        raise ValueError(f"Unsupported type: {type}. Must be 'xyz', 'numbered', or 'simtree'.")
